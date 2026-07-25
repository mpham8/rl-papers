import torch

import torch.nn as nn
from torch.distributions import Normal


class SoftValueFunction(nn.Module):
    def __init__(self, num_states, hidden_layer) -> None:
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(num_states, hidden_layer),
            nn.ReLU(),
            nn.Linear(hidden_layer, hidden_layer),
            nn.ReLU(),
            nn.Linear(hidden_layer, 1)
        )

    def forward(self, x):
        return self.fc(x)


class SoftQFunction(nn.Module):
    def __init__(self, num_states, num_actions, hidden_layer) -> None:
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(num_states + num_actions, hidden_layer),
            nn.ReLU(),
            nn.Linear(hidden_layer, hidden_layer),
            nn.ReLU(),
            nn.Linear(hidden_layer, 1)
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim = -1)
        return self.fc(x)


class PolicyFunction(nn.Module):
    def __init__(self, num_states, num_actions, hidden_layer, log_std_min, log_std_max) -> None:
        super().__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        self.fc = nn.Sequential(
            nn.Linear(num_states, hidden_layer),
            nn.ReLU(),
            nn.Linear(hidden_layer, hidden_layer),
            nn.ReLU(),
        )
        self.mean = nn.Linear(hidden_layer, num_actions)
        self.std = nn.Linear(hidden_layer, num_actions)

    def forward(self, x):
        x = self.fc(x)

        mean = self.mean(x)
        log_std = self.std(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)

        return mean, log_std

    def sample(self, x):
        mean, log_std = self.forward(x)
        std = log_std.exp() #log std to enforce non negativity

        #appendix c
        epsilon = torch.randn_like(mean)
        u = epsilon * std + mean
        a = torch.tanh(u)

        normal = Normal(mean, std)
        log_prob_u = normal.log_prob(u)
        log_prob = (log_prob_u - torch.log(1 - a.pow(2) + 1e-6)).sum(dim=-1, keepdim=True)

        mean_action = torch.tanh(mean)

        return a, log_prob, mean_action

