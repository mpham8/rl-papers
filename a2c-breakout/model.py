import torch.nn as nn


class PolicyFunctionApproximation(nn.Module):
    def __init__(self, input_dim, hidden, num_actions) -> None:
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_actions)
        )
        
    def forward(self, x):
        return self.fc(x)


class ValueFunctionApproximation(nn.Module):
    def __init__(self, input_dim, hidden) -> None:
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )

    def forward(self, x):
        return self.fc(x)



