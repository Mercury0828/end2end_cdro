from __future__ import annotations

from .base import BaseController


class PPOController(BaseController):
    """Optional integration hook. Keeps pipeline runnable when PPO deps are absent."""
    name = "ppo"

    def __init__(self, *args, **kwargs):
        self.enabled = False

    def reset(self):
        return None

    def act(self, x, z, tamb, xi_hat=None):
        # conservative fallback behavior
        return 1.0 if z < 0.5 else 4.0, 0.0, None
