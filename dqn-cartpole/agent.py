import torch
import torch.nn.functional


def select_action(qnet, phi, epsilon, num_actions, total_agents):
    '''forward pass, eps-greedy action for each parallel env.'''
    with torch.no_grad():
        q_val = qnet(phi)
        greedy = q_val.argmax(dim=1)
        random_actions = torch.randint(0, num_actions, (total_agents,), device=phi.device)
        explore = torch.rand(total_agents, device=phi.device) < epsilon
        actions = torch.where(explore, random_actions, greedy)
    return actions


def train_step(qnet, q_target, optimizer, phi, actions, rewards, phi_n, terminals, gamma):
    '''one qnet update on sampled replay minibatch'''
    q = qnet(phi).gather(1, actions.unsqueeze(1)).squeeze(1)
    with torch.no_grad():
        q_n = q_target(phi_n).max(dim=1)[0]
        y = rewards + gamma * q_n * (1 - terminals)
    loss = torch.nn.functional.mse_loss(q, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.detach()
