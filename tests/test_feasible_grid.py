"""Tests for the constraint-feasible composition grid."""

import torch

from rhea_mobo.feasible_grid import xs_feasible


def test_feasible_points_sum_to_one() -> None:
    """Every feasible composition's mole fractions should sum to one."""
    assert torch.allclose(xs_feasible.sum(dim=1), torch.ones(xs_feasible.shape[0]), atol=1e-8)


def test_feasible_points_have_equal_co_and_hf() -> None:
    """Every feasible composition should have equal Co and Hf mole fractions."""
    assert torch.allclose(xs_feasible[:, 4], xs_feasible[:, 5], atol=1e-8)


def test_feasible_points_are_all_positive() -> None:
    """Every feasible composition should have strictly positive mole fractions."""
    assert (xs_feasible > 0).all()
