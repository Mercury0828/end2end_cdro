from __future__ import annotations

from abc import ABC, abstractmethod


class BaseController(ABC):
    name = "base"

    @abstractmethod
    def reset(self):
        ...

    @abstractmethod
    def act(self, x: float, z: float, tamb: float, xi_hat: float | None = None):
        ...
