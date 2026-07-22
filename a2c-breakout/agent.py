import torch
from torch.distributions import Categorical
from torch.distributions.utils import logits_to_probs



def select_action(model_p, model_v, states):
    '''forward pass actor to get action, pi(a|s), compute v(s)'''
    logits = model_p(states)
    dist = Categorical(logits=logits)
    actions = dist.sample()
    log_probs = dist.log_prob(actions)

    return actions, log_probs, model_v(states).squeeze(-1)


def train_step(optimizer_p, optimizer_v, values, advantages, returns, log_probs):
    '''compute losses, batch update'''

    loss_p = log_probs * (returns - values.detach())
    optimizer_p.zero_grad()
    loss_p.backward()
    optimizer_p.step()

    loss_v = advantages ** 2
    optimizer_v.zero_grad()
    loss_v.backward()
    optimizer_v.step()
   
    return loss_p, loss_v


def compute_return_advantage(model, rewards, values_bootstrap, values, terminals, cfg):
    '''returns for value target, advantages for update'''

    returns = torch.zeros(cfg['T_MAX'], cfg['TOTAL_AGENTS'], device=rewards.device)
    running = values_bootstrap
    for t in range(cfg['T_MAX']-1, -1, -1):
        running = rewards[t] + cfg['GAMMA'] * running * (1.0 - terminals[t])
        returns[t] = running

    advantages =  returns - values
    return advantages, returns