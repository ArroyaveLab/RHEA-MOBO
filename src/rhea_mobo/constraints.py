"""Linear inequality constraints on the six-element composition simplex.

Each constraint is expressed as the ``(indices, coefficients, rhs)`` triple
expected by botorch's discrete acquisition optimizers, encoding
``coefficients @ x[indices] >= rhs``.
"""

import torch

idx: torch.Tensor = torch.tensor([0, 1, 2, 3, 4, 5], dtype=torch.long)
coef1: torch.Tensor = torch.tensor([1.0] * 6, dtype=torch.double)
coef2: torch.Tensor = torch.tensor([-1.0] * 6, dtype=torch.double)
coef3: torch.Tensor = torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0, -1.0], dtype=torch.double)
coef4: torch.Tensor = torch.tensor([0.0, 0.0, 0.0, 0.0, -1.0, 1.0], dtype=torch.double)
coef5: torch.Tensor = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.double)
coef6: torch.Tensor = torch.tensor([0.0, 1.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.double)
coef7: torch.Tensor = torch.tensor([0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=torch.double)
coef8: torch.Tensor = torch.tensor([0.0, 0.0, 0.0, 1.0, 0.0, 0.0], dtype=torch.double)
coef9: torch.Tensor = torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=torch.double)

rhs1: float = 1.0
rhs2: float = -1.0
rhs3: float = 0.0
rhs4: float = 0.0
rhs5: float = 0.05
rhs6: float = 0.05
rhs7: float = 0.05
rhs8: float = 0.05
rhs9: float = 0.05

inequality_constraints: list[tuple[torch.Tensor, torch.Tensor, float]] = [
    (idx, coef1, rhs1),
    (idx, coef2, rhs2),
    (idx, coef3, rhs3),
    (idx, coef4, rhs4),
    (idx, coef5, rhs5),
    (idx, coef6, rhs6),
    (idx, coef7, rhs7),
    (idx, coef8, rhs8),
    (idx, coef9, rhs9),
]
