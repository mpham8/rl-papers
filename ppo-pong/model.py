import torch.nn as nn


class L_clip_vf_s(nn.Module):
    def __init__(self, inputs, num_actions, d) -> None:
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(inputs, d),
            nn.ReLU(),
            nn.Linear(d, d),
            nn.ReLU()
        )
        self.policy_head = nn.Linear(d, num_actions)
        self.value_head = nn.Linear(d, 1)

    def forward(self, x):
        x = self.fc(x)
        return self.policy_head(x), self.value_head(x)



