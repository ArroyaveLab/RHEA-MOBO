import torch

from config import ELEMENT_COST

# --------------------
# Cost-Aware EHVI Wrapper
# --------------------
class CostAwareEHVI(torch.nn.Module):
    def __init__(self, base_acqf, cost_model):
        super().__init__()
        self.base_acqf = base_acqf
        self.cost_model = cost_model

    def forward(self, X):
        return self.base_acqf(X) / self.cost_model(X).clamp(min=1e-6)


def cost_model(x):
    return (x @ ELEMENT_COST).unsqueeze(-1)
