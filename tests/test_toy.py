"""Structural checks for the license-free synthetic benchmark."""

import torch

from rhea_mobo.toy import composition_grid, feasible, random_candidates, synthetic_objectives


def test_grid_and_synthetic_objectives() -> None:
    grid = composition_grid()
    assert grid.shape == (1716, 6)
    assert torch.unique(grid, dim=0).shape == grid.shape
    assert torch.allclose(grid.sum(-1), torch.ones(len(grid)))
    assert (grid >= 0.05).all()
    assert torch.equal(grid[:, 4], grid[:, 5])
    assert feasible(grid).any() and not feasible(grid).all()
    values = synthetic_objectives(grid)
    assert values.shape == (1716, 5)
    assert torch.isfinite(values).all()
    # The endpoint utilities must prefer different compositions.
    assert values[:, 0].argmax() != values[:, 4].argmax()


def test_random_baseline_is_paired_unique_and_rng_independent() -> None:
    grid = composition_grid()
    initial = grid[feasible(grid)][:8]
    before = torch.random.get_rng_state().clone()
    selected = random_candidates(grid, initial, 20, 43)
    assert torch.equal(before, torch.random.get_rng_state())
    assert torch.equal(selected, random_candidates(grid, initial, 20, 43))
    assert not torch.equal(selected, random_candidates(grid, initial, 20, 44))
    combined = torch.cat([initial, selected])
    assert torch.unique(combined, dim=0).shape == (28, 6)
    assert feasible(combined).all()
