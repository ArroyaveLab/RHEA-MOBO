"""Cost-aware acquisition function wrapper and elemental cost model."""

from collections.abc import Callable

import torch
from botorch.acquisition.acquisition import AcquisitionFunction

from .config import ELEMENT_COST


class CostAwareEHVI(AcquisitionFunction):
    """Acquisition function that divides a base acquisition value by predicted cost."""

    def __init__(self, base_acqf: AcquisitionFunction, cost_model: Callable[[torch.Tensor], torch.Tensor]) -> None:
        """Store the base acquisition function and the cost model used to scale it."""
        super().__init__(model=base_acqf.model)
        self.base_acqf = base_acqf
        self.cost_model = cost_model

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Return the base acquisition value divided by the (clamped) predicted cost."""
        return self.base_acqf(X) / self.cost_model(X).clamp(min=1e-6)


def cost_model(x: torch.Tensor) -> torch.Tensor:
    """Compute the elemental dollar cost of each composition row in ``x``."""
    return (x @ ELEMENT_COST).unsqueeze(-1)
