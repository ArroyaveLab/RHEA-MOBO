import torch

# --------------------
# Inequality Constraints
# --------------------
idx = torch.tensor([0, 1, 2, 3, 4, 5], dtype=torch.long)
coef1 = torch.tensor([1.0]*6, dtype=torch.double)
coef2 = torch.tensor([-1.0]*6, dtype=torch.double)
coef3 = torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0, -1.0], dtype=torch.double)
coef4 = torch.tensor([0.0, 0.0, 0.0, 0.0, -1.0, 1.0], dtype=torch.double)
coef5 = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.double)
coef6 = torch.tensor([0.0, 1.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.double)
coef7 = torch.tensor([0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=torch.double)
coef8 = torch.tensor([0.0, 0.0, 0.0, 1.0, 0.0, 0.0], dtype=torch.double)
coef9 = torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=torch.double)

rhs1 = 1.0
rhs2 = -1.0
rhs3 = 0.0
rhs4 = 0.0
rhs5 = 0.05
rhs6 = 0.05
rhs7 = 0.05
rhs8 = 0.05
rhs9 = 0.05

inequality_constraints = [
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
