"""Tests for GP model fitting and multi-objective acquisition-function construction."""

import pytest
import torch
from botorch.acquisition.multi_objective.analytic import ExpectedHypervolumeImprovement
from botorch.models import SingleTaskGP
from botorch.models.model_list_gp_regression import ModelListGP

from rhea_mobo.gp_utils import build_model, fit_single_task_model, get_acquisition


def _toy_data(n: int = 6, d: int = 2, m: int = 2) -> tuple[torch.Tensor, torch.Tensor]:
    """Build small deterministic synthetic training data for GP fitting."""
    generator = torch.Generator().manual_seed(0)
    train_x = torch.rand(n, d, generator=generator, dtype=torch.double)
    train_y = torch.rand(n, m, generator=generator, dtype=torch.double)
    return train_x, train_y


def test_fit_single_task_model_returns_fitted_gp() -> None:
    """fit_single_task_model should return a SingleTaskGP usable for posterior prediction."""
    train_x, train_y = _toy_data(m=1)

    model = fit_single_task_model(train_x, train_y)

    assert isinstance(model, SingleTaskGP)
    posterior = model.posterior(train_x[:1])
    assert posterior.mean.shape == (1, 1)


def test_build_model_drops_rows_with_nan_and_fits_one_gp_per_objective() -> None:
    """build_model should drop any row with a NaN objective and fit one GP per remaining column."""
    train_x, train_y = _toy_data(n=8, m=2)
    train_y[0, 0] = float("nan")

    model = build_model(train_x, train_y)

    assert isinstance(model, ModelListGP)
    assert len(model.models) == 2


def test_build_model_raises_when_every_row_is_nan() -> None:
    """build_model should raise a RuntimeError when no usable training rows remain."""
    train_x, train_y = _toy_data(n=4, m=2)
    train_y = torch.full_like(train_y, float("nan"))

    with pytest.raises(RuntimeError, match="All training outputs are NaN."):
        build_model(train_x, train_y)


def test_get_acquisition_builds_ehvi_over_pareto_front() -> None:
    """get_acquisition should return an EHVI acquisition function that yields finite values."""
    train_x, train_y = _toy_data(n=6, m=2)
    model = build_model(train_x, train_y)
    ref_point = train_y.min(dim=0).values - 1.0

    acqf = get_acquisition(model, train_y, ref_point)

    assert isinstance(acqf, ExpectedHypervolumeImprovement)
    value = acqf(train_x[:1].unsqueeze(1))
    assert torch.isfinite(value).all()
