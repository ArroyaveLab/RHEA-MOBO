import torch

from config import discrete_choices

# --------------------
# Feasible Grid
# --------------------
# --------------------
# Feasible Grid
# --------------------
X1, X2, X3, X4, X5, X6 = torch.meshgrid(*discrete_choices, indexing="ij")
xs = torch.stack([X1.flatten(), X2.flatten(), X3.flatten(), X4.flatten(), X5.flatten(), X6.flatten()], dim=-1)

# Constraint 1: sum to 1
mask_sum = torch.isclose(xs.sum(dim=1), torch.ones(xs.shape[0]), atol=1e-8)

# Constraint 2: X5 == X6
mask_equal = torch.isclose(xs[:, 4], xs[:, 5], atol=1e-8)

# Constraint 3: All values > 0
mask_positive = (xs > 0).all(dim=1)

# Combine masks (before ML-based filtering)
mask_total = mask_sum & mask_equal & mask_positive
xs_filtered = xs[mask_total]

# Final feasible set
xs_feasible = xs_filtered
