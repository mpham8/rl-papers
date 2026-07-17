import torch


class ReplayBuffer:
    def __init__(self, capacity, phi_dim, device='cuda'):
        self.capacity = capacity
        self.device = device
        self.phi = torch.zeros(capacity, phi_dim, device=device)
        self.phi_n = torch.zeros(capacity, phi_dim, device=device)
        self.actions = torch.zeros(capacity, dtype=torch.long, device=device)
        self.rewards = torch.zeros(capacity, device=device)
        self.terminals = torch.zeros(capacity, device=device)
        self.ptr = 0
        self.size = 0

    def add_batch(self, phi, actions, rewards, phi_n, terminals):
        n = phi.shape[0]
        if n >= self.capacity:
            phi = phi[-self.capacity:]
            actions = actions[-self.capacity:]
            rewards = rewards[-self.capacity:]
            phi_n = phi_n[-self.capacity:]
            terminals = terminals[-self.capacity:]
            n = self.capacity

        end = self.ptr + n
        if end <= self.capacity:
            sl = slice(self.ptr, end)
            self.phi[sl] = phi
            self.phi_n[sl] = phi_n
            self.actions[sl] = actions
            self.rewards[sl] = rewards
            self.terminals[sl] = terminals
        else:
            first = self.capacity - self.ptr
            second = n - first
            self.phi[self.ptr:] = phi[:first]
            self.phi[:second] = phi[first:]
            self.phi_n[self.ptr:] = phi_n[:first]
            self.phi_n[:second] = phi_n[first:]
            self.actions[self.ptr:] = actions[:first]
            self.actions[:second] = actions[first:]
            self.rewards[self.ptr:] = rewards[:first]
            self.rewards[:second] = rewards[first:]
            self.terminals[self.ptr:] = terminals[:first]
            self.terminals[:second] = terminals[first:]

        self.ptr = end % self.capacity
        self.size = min(self.size + n, self.capacity)

    def sample(self, batch_size):
        idx = torch.randint(0, self.size, (batch_size,), device=self.device)
        return (
            self.phi[idx],
            self.actions[idx],
            self.rewards[idx],
            self.phi_n[idx],
            self.terminals[idx],
        )