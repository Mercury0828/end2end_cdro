from __future__ import annotations

import torch
import torch.nn as nn


class RhoNet(nn.Module):
    def __init__(self, in_dim=2, hidden_dim=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
            nn.Softplus(),
        )

    def forward(self, obs):
        return self.net(obs)
