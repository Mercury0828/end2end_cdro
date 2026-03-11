from __future__ import annotations

import numpy as np
import torch
from .base import BaseController
from src.optimization.robust_layer import ScalarRobustLayer
from src.learning.rho_net import RhoNet


class E2ECDROController(BaseController):
    name = "e2e_cdro"

    def __init__(self, alpha, beta, gamma, u_max, lambda_penalty, hidden_dim=16, lr=1e-2):
        self.layer = ScalarRobustLayer(alpha, beta, gamma, u_max, lambda_penalty, xi_nominal=0.2)
        self.net = RhoNet(in_dim=2, hidden_dim=hidden_dim)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr)

    def reset(self):
        return None

    def rho_from_obs(self, x, z):
        obs = torch.tensor([[x, z]], dtype=torch.float32)
        with torch.no_grad():
            return float(self.net(obs).item())

    def act(self, x, z, tamb, xi_hat=None):
        rho = self.rho_from_obs(x, z)
        out = self.layer.act(x, tamb, rho)
        return out.u, rho, out

    def update_supervised(self, batch_obs, target_rho):
        obs = torch.tensor(batch_obs, dtype=torch.float32)
        trg = torch.tensor(target_rho, dtype=torch.float32).reshape(-1, 1)
        pred = self.net(obs)
        loss = ((pred - trg) ** 2).mean()
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        return float(loss.item())
