import torch
from torch.distributions import Categorical


def select_action(model, input_ids, attention_mask):
    logits, values = model(input_ids, attention_mask, last_token_only=True)
    dist = Categorical(logits=logits)
    actions = dist.sample()
    log_probs = dist.log_prob(actions)
    
    return actions, log_probs, values


def ref_log_prob_last(ref_model, input_ids, attention_mask, actions):
    hidden = ref_model.backbone(input_ids, attention_mask)
    logits = ref_model.policy_head.lin(hidden[:, -1, :])

    return Categorical(logits=logits).log_prob(actions)


def compute_generalized_advantages(rewards_T, values_T, gamma, lam, horizon, batch_size):
    rewards_T = rewards_T.transpose(0, 1)
    values_T = values_T.transpose(0, 1)
    device = rewards_T.device
    gae = torch.zeros(horizon, batch_size, device=device)
    running_gae = torch.zeros(batch_size, device=device)
    returns = torch.zeros(horizon, batch_size, device=device)
    for t in range(horizon - 1, -1, -1):
        delta = rewards_T[t, :] + gamma * values_T[t + 1, :] - values_T[t, :]
        running_gae = delta + gamma * lam * running_gae
        gae[t, :] = running_gae
        returns[t, :] = running_gae + values_T[t, :]

    return gae, returns


def build_prefix_minibatch(prompt_ids, actions_T, n_idx, t_idx, pad_id, device):
    seqs = []
    for n, t in zip(n_idx.tolist(), t_idx.tolist()):
        if t == 0:
            seqs.append(prompt_ids[n])
        else:
            seqs.append(torch.cat([prompt_ids[n], actions_T[n, :t]]))
    seq_len = max(s.size(0) for s in seqs)
    input_ids_mb = torch.full((len(seqs), seq_len), pad_id, dtype=torch.long, device=device)
    attention_mask_mb = torch.zeros((len(seqs), seq_len), dtype=torch.long, device=device)
    for j, s in enumerate(seqs):
        input_ids_mb[j, : s.size(0)] = s
        attention_mask_mb[j, : s.size(0)] = 1

    return input_ids_mb, attention_mask_mb


def train_step(model, optimizer, input_ids_mb, attention_mask_mb, actions_mb, gae_mb, values_target_mb, log_prob_old_mb, clip_eps, c1, pretrain_input_ids=None, pretrain_attention_mask=None, pretrain_gamma=0.0):
    logits, values = model(input_ids_mb, attention_mask_mb, last_token_only=True)
    dist = Categorical(logits=logits)
    log_prob_new = dist.log_prob(actions_mb)
    r = (log_prob_new - log_prob_old_mb).exp()
    L_clip = torch.min(r * gae_mb, torch.clamp(r, 1.0 - clip_eps, 1.0 + clip_eps) * gae_mb)
    L_val = (values - values_target_mb) ** 2
    loss = -L_clip.mean() + c1 * L_val.mean() + pretrain_gamma * model.pretrain_mix_loss(pretrain_input_ids, pretrain_attention_mask)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.detach()
