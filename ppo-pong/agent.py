import torch
from torch.distributions import Categorical



def select_action(model, states_t):
    '''forward pass, select from stochastic policy for each parallel env.'''
    policy_logits, values_t = model(states_t)
    dist = Categorical(logits = policy_logits)
    actions = dist.sample()
    log_prob = dist.log_prob(actions)

    return actions, log_prob, values_t.squeeze(-1)


def compute_return_advantage(rewards_T, values_T, terminals_T, cfg):
    '''generalized advantage estimates (A_t) and returns (A_t + V(s_t))'''
    gae = torch.zeros(cfg['HORIZON'], cfg['TOTAL_AGENTS']).cuda()
    running_gae = torch.zeros(cfg['TOTAL_AGENTS']).cuda()
    returns = torch.zeros(cfg['HORIZON'], cfg['TOTAL_AGENTS']).cuda()
    for t in range(cfg['HORIZON'] - 1, -1, -1):
        delta = rewards_T[t, :] + cfg['GAMMA'] * values_T[t + 1, :] * (1 - terminals_T[t, :]) - values_T[t, :]
        running_gae = delta + cfg['GAMMA'] * cfg['LAMBDA'] * running_gae * (1 - terminals_T[t, :])
        gae[t] = running_gae
        returns[t] = running_gae + values_T[t, :]
    
    return gae, returns


def train_step(model, optimizer, states_T, actions_T, t_idx, n_idx, gae, values_target, log_prob_old, cfg):
    '''optimize surrogate (clipping) wrt theta using minibatch'''
    states_mb = states_T[t_idx, n_idx]
    actions_mb = actions_T[t_idx, n_idx]
    gae_mb = gae[t_idx, n_idx]
    values_target_mb = values_target[t_idx, n_idx]
    log_prob_old_mb = log_prob_old[t_idx, n_idx]

    #calculate L clip = min(r * A, clip(r, 1-eps, 1+eps) * A)
    policy_logits, values = model(states_mb)
    dist = Categorical(logits = policy_logits)
    log_prob_new = dist.log_prob(actions_mb)
    r = (log_prob_new - log_prob_old_mb).exp() 
    L_clip = torch.min(r * gae_mb, torch.clamp(r, 1.0 - cfg['EPS_ALPHA'], 1.0 + cfg['EPS_ALPHA']) * gae_mb) #TODO: eps alpha * alpha   
    #calculate L val
    L_val = (values.squeeze(-1) - values_target_mb) ** 2
    #entropy S
    entropy = dist.entropy()

    #adam step
    loss = -L_clip.mean() + cfg['C1'] * L_val.mean() - cfg['C2'] * entropy.mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.detach()
