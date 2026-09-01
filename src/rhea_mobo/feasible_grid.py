"""Discretized, constraint-feasible grid of six-element compositions."""

import torch

from .config import discrete_choices

X1, X2, X3, X4, X5, X6 = torch.meshgrid(*discrete_choices, indexing="ij")
xs: torch.Tensor = torch.stack([X1.flatten(), X2.flatten(), X3.flatten(), X4.flatten(), X5.flatten(), X6.flatten()], dim=-1)

mask_sum: torch.Tensor = torch.isclose(xs.sum(dim=1), torch.ones(xs.shape[0]), atol=1e-8)
mask_equal: torch.Tensor = torch.isclose(xs[:, 4], xs[:, 5], atol=1e-8)
mask_positive: torch.Tensor = (xs > 0).all(dim=1)

mask_total: torch.Tensor = mask_sum & mask_equal & mask_positive
xs_feasible: torch.Tensor = xs[mask_total]
