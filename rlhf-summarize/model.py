from transformers import AutoModel
from torch import nn

import torch.nn.functional as F

import torch
import copy



class LLMBackbone(nn.Module):
    """Loads LLM backbone from Huggingface AutoModel"""
    def __init__(self, model_name, data_type, device) -> None:
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name).to(dtype=data_type).to(device=device)

    def forward(self, input_ids, attention_mask):
        return self.backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state 


class PolicyHead(nn.Module):
    """Policy head"""
    def __init__(self, hidden, vocab_size) -> None:
        super().__init__()
        self.lin = nn.Linear(hidden, vocab_size)
        
    def forward(self, x):
        return self.lin(x)

    def log_probs_for_tokens(self, x, token_ids):
        log_probs = F.log_softmax(self.forward(x), dim=-1)
        return torch.gather(log_probs, dim=-1, index=token_ids.unsqueeze(-1)).squeeze(-1)


class ValueHead(nn.Module):
    """Value head"""
    def __init__(self, hidden) -> None:
        super().__init__()
        self.lin = nn.Linear(hidden, 1)

    def forward(self, x):
        return self.lin(x).squeeze(-1)


class SupervisedFineTuningModel(nn.Module):
    """Supervised Fine-Tuning Model"""
    def __init__(self, model_name, data_type, device) -> None:
        super().__init__()
        self.backbone = LLMBackbone(model_name, data_type, device)
        hidden = self.backbone.backbone.config.hidden_size
        vocab_size = self.backbone.backbone.config.vocab_size
        self.policy_head = PolicyHead(hidden, vocab_size).to(dtype=data_type, device=device)

    def forward(self, input_ids, attention_mask):
        hidden_states = self.backbone(input_ids, attention_mask)
        return self.policy_head(hidden_states)

    def sft_loss(self, input_ids, attention_mask, labels):
        hidden_states = self.backbone(input_ids, attention_mask)
        shift_hidden = hidden_states[:, :-1, :]
        shift_labels = labels[:, 1:]
        mask = shift_labels != -100
        logits = self.policy_head.lin(shift_hidden[mask])
        return F.cross_entropy(logits, shift_labels[mask])


class RewardModel(nn.Module):
    """Reward Model"""
    def __init__(self, backbone, device, dtype) -> None:
        super().__init__()
        self.backbone = backbone
        hidden = backbone.backbone.config.hidden_size
        self.value_head = ValueHead(hidden).to(dtype=dtype, device=device)

    def forward(self, input_ids, attention_mask):
        hidden_states = self.backbone(input_ids, attention_mask)
        per_token_values = self.value_head(hidden_states)
        
        position_ids = torch.arange(attention_mask.size(1), device=attention_mask.device)
        last_idx = (attention_mask * position_ids).argmax(dim=1)
        rewards = torch.gather(per_token_values, dim=1, index=last_idx.unsqueeze(-1)).squeeze(-1)

        return rewards

    @classmethod
    def from_sft_model(cls, sft_model, device, dtype):
        backbone = copy.deepcopy(sft_model.backbone)
        return cls(backbone, device, dtype)


class PPOModel(nn.Module):
    """PPO"""
    def __init__(self, backbone, policy_head, device, dtype):
        super().__init__()
        self.backbone = backbone
        hidden = self.backbone.backbone.config.hidden_size
        self.policy_head = policy_head
        self.value_head = ValueHead(hidden).to(dtype=dtype, device=device)
    
    def forward(self, input_ids, attention_mask, token_ids=None, last_token_only=False):
        hidden_states = self.backbone(input_ids, attention_mask)

        if last_token_only:
            last_hidden = hidden_states[:, -1, :]
            logits = self.policy_head.lin(last_hidden)
            values = self.value_head.lin(last_hidden).squeeze(-1)
            if token_ids is None:
                return logits, values
            log_probs = F.log_softmax(logits, dim=-1).gather(1, token_ids.unsqueeze(-1)).squeeze(-1)
            return logits, values, log_probs

        logits = self.policy_head(hidden_states)
        values = self.value_head(hidden_states)

        if token_ids is None:
            return logits, values
        log_probs = self.policy_head.log_probs_for_tokens(hidden_states, token_ids)
        return logits, values, log_probs

    def pretrain_mix_loss(self, input_ids, attention_mask):
        hidden_states = self.backbone(input_ids, attention_mask)
        shift_hidden = hidden_states[:, :-1, :]
        shift_labels = input_ids[:, 1:]
        shift_mask = attention_mask[:, 1:]
        logits = self.policy_head.lin(shift_hidden[shift_mask.bool()])
        return F.cross_entropy(logits, shift_labels[shift_mask.bool()])


    @classmethod
    def from_sft_model(cls, sft_model, device, dtype):
        backbone = copy.deepcopy(sft_model.backbone)
        policy_head = copy.deepcopy(sft_model.policy_head)
        return cls(backbone, policy_head, device, dtype)