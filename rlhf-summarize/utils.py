from pathlib import Path
from model import LLMBackbone, RewardModel, SupervisedFineTuningModel

import torch

from data import _pad_sequences, _tokenize_query_summary


def save_checkpoint(model, save_dir, filename):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / filename
    torch.save(model.state_dict(), save_path)
    print(f"saved checkpoint to {save_path}")
    return save_path


def load_sft_model(model_name, data_type, device, checkpoint_dir, checkpoint_file="sft.pt"):

    checkpoint_path = Path(checkpoint_dir) / checkpoint_file
    model = SupervisedFineTuningModel(model_name, data_type, device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    model.eval()
    return model


def load_rm_model(model_name, data_type, device, checkpoint_dir, checkpoint_file="rm.pt"):
    checkpoint_path = Path(checkpoint_dir) / checkpoint_file
    backbone = LLMBackbone(model_name, data_type, device)
    model = RewardModel(backbone, device, data_type)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    model.eval()
    return model


def compute_rm_offset(reward_model, sft_data, tokenizer, device, max_length, batch_size=8):
    pad_id = tokenizer.pad_token_id
    scores = []
    examples = list(sft_data)
    for start in range(0, len(examples), batch_size):
        batch = examples[start : start + batch_size]
        ids_list = []
        for example in batch:
            ids = _tokenize_query_summary(
                example["query"], example["reference_summary"], tokenizer, max_length
            )
            if ids is not None:
                ids_list.append(ids)
        if not ids_list:
            continue
        input_ids, attention_mask = _pad_sequences(ids_list, pad_id, device)
        with torch.no_grad():
            scores.append(reward_model(input_ids, attention_mask))
    if not scores:
        raise ValueError("no reference summaries available for RM offset")
    return torch.cat(scores).mean().item()


def load_rm_offset(rm_save_path, offset_file, device):
    offset_path = Path(rm_save_path) / offset_file
    if not offset_path.exists():
        return None
    return torch.load(offset_path, map_location=device, weights_only=True)["offset"]


def save_rm_offset(offset, rm_save_path, offset_file):
    offset_path = Path(rm_save_path) / offset_file
    offset_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"offset": offset}, offset_path)
    print(f"saved RM offset {offset:.4f} to {offset_path}")
    return offset_path
