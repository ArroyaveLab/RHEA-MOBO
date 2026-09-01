"""Global configuration constants for the RHEA-MOBO optimization pipeline."""

import torch

torch.set_default_dtype(torch.double)

num_input: int = 6
num_obj: int = 5
iterations: int = 200
initial_samples: int = 50

num_restart: int = 20
raw_samples: int = 50

grid_points_per_dim: int = 21
discrete_choices: list[torch.Tensor] = [torch.linspace(0.0, 1.0, grid_points_per_dim) for _ in range(num_input)]

elements: list[str] = ["Mo", "Nb", "Ta", "W", "Co", "Hf"]
ELEMENT_COST: torch.Tensor = torch.tensor([6.65, 4.92, 10.94, 10.53, 3.99, 5.95], dtype=torch.double)
