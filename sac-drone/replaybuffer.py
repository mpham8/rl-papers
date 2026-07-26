import torch


class ReplayBuffer:
    def __init__(self, capacity, state_dim, action_dim, device='cuda'):
        self.capacity = capacity
        self.device = device
        self.states = torch.zeros(capacity, state_dim, device=device)
        self.states_n = torch.zeros(capacity, state_dim, device=device)
        self.actions = torch.zeros(capacity, action_dim, device=device)
        #kept 2d so they broadcast against the (batch, 1) critic outputs
        self.rewards = torch.zeros(capacity, 1, device=device)
        self.terminals = torch.zeros(capacity, 1, device=device)
        self.ptr = 0
        self.size = 0

    def add_batch(self, states, actions, rewards, states_n, terminals):
        rewards = rewards.reshape(-1, 1)
        terminals = terminals.reshape(-1, 1)

        n = states.shape[0]
        if n >= self.capacity:
            states = states[-self.capacity:]
            actions = actions[-self.capacity:]
            rewards = rewards[-self.capacity:]
            states_n = states_n[-self.capacity:]
            terminals = terminals[-self.capacity:]
            n = self.capacity

        end = self.ptr + n
        if end <= self.capacity:
            sl = slice(self.ptr, end)
            self.states[sl] = states
            self.states_n[sl] = states_n
            self.actions[sl] = actions
            self.rewards[sl] = rewards
            self.terminals[sl] = terminals
        else:
            first = self.capacity - self.ptr
            second = n - first
            self.states[self.ptr:] = states[:first]
            self.states[:second] = states[first:]
            self.states_n[self.ptr:] = states_n[:first]
            self.states_n[:second] = states_n[first:]
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
            self.states[idx],
            self.actions[idx],
            self.rewards[idx],
            self.states_n[idx],
            self.terminals[idx],
        )


class NStepAccumulator:
    '''Buffers the last n transitions per agent and emits n-step ones:
    (s_t, a_t, sum_{i<n} gamma^i r_{t+i}, s_{t+n}, done_within_n).
    The reward sum stops at the first terminal (the env auto-resets, so later
    rewards belong to the next episode); done=1 then also masks the bootstrap,
    so s_{t+n} being post-reset is never used. Bootstrap in the TD target must
    use gamma**n to match.'''
    def __init__(self, n, gamma, replay):
        self.n = n
        self.gamma = gamma
        self.replay = replay
        self.pending = []

    def add_batch(self, states, actions, rewards, states_n, terminals):
        self.pending.append((
            states, actions,
            rewards.reshape(-1, 1), terminals.reshape(-1, 1),
            states_n,
        ))
        if len(self.pending) < self.n:
            return

        s0, a0 = self.pending[0][0], self.pending[0][1]
        ret = torch.zeros_like(self.pending[0][2])
        done = torch.zeros_like(ret)
        for i in range(self.n):
            ret += (self.gamma ** i) * self.pending[i][2] * (1 - done)
            done = torch.maximum(done, self.pending[i][3])
        s_n = self.pending[-1][4]

        self.replay.add_batch(s0, a0, ret, s_n, done)
        self.pending.pop(0)

    @property
    def size(self):
        return self.replay.size
