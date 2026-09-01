"""Tests for the elemental cost model and cost-aware acquisition wrapper."""

from typing import cast

import torch
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.models.model import Model

from rhea_mobo.config import ELEMENT_COST
from rhea_mobo.cost import CostAwareEHVI, cost_model


class _ConstantAcqf(AcquisitionFunction):
    """Acquisition function stub that returns a fixed value for every input."""

    def __init__(self, model: Model, value: float) -> None:
        """Store the stub's constant return value."""
        super().__init__(model=model)
        self.value = value

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Return the constant value, broadcast to one entry per batch row."""
        return torch.full((X.shape[0],), self.value, dtype=torch.double)


def test_cost_model_matches_manual_dot_product() -> None:
    """cost_model should match a manual dot product against ELEMENT_COST."""
    x = torch.tensor([[0.2, 0.2, 0.2, 0.2, 0.1, 0.1]], dtype=torch.double)
    expected = (x @ ELEMENT_COST).unsqueeze(-1)
    assert torch.allclose(cost_model(x), expected)


def test_cost_aware_ehvi_divides_base_value_by_cost() -> None:
    """forward() should return the base acquisition value divided by the predicted cost."""
    base = _ConstantAcqf(model=cast("Model", None), value=10.0)
    acqf = CostAwareEHVI(base_acqf=base, cost_model=lambda X: torch.full((X.shape[0],), 2.0, dtype=torch.double))

    out = acqf(torch.zeros(3, 1, 6, dtype=torch.double))

    assert torch.allclose(out, torch.full((3,), 5.0, dtype=torch.double))


def test_cost_aware_ehvi_clamps_cost_to_avoid_division_by_zero() -> None:
    """forward() should clamp near-zero predicted cost to 1e-6 before dividing."""
    base = _ConstantAcqf(model=cast("Model", None), value=1.0)
    acqf = CostAwareEHVI(base_acqf=base, cost_model=lambda X: torch.zeros(X.shape[0], dtype=torch.double))

    out = acqf(torch.zeros(2, 1, 6, dtype=torch.double))

    assert torch.allclose(out, torch.full((2,), 1.0 / 1e-6, dtype=torch.double))


def test_cost_aware_ehvi_stores_base_acqf_and_underlying_model() -> None:
    """__init__ should keep a reference to the base acqf and expose its underlying model."""
    sentinel_model = cast("Model", object())
    base = _ConstantAcqf(model=sentinel_model, value=1.0)
    acqf = CostAwareEHVI(base_acqf=base, cost_model=lambda X: torch.ones(X.shape[0], dtype=torch.double))

    assert acqf.base_acqf is base
    assert acqf.model is sentinel_model
