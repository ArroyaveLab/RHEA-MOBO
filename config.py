import torch

torch.set_default_dtype(torch.double)

# --------------------
# Configuration
# --------------------
num_input = 6
num_obj = 5
iterations = 200
initial_samples = 50

num_restart = 20
raw_samples = 50

grid_points_per_dim = 21
discrete_choices = [
    torch.linspace(0.0, 1.0, grid_points_per_dim)
    for _ in range(num_input)
]

elements = ['Mo', 'Nb', 'Ta', 'W', 'Co', 'Hf']
ELEMENT_COST = torch.tensor([6.65, 4.92, 10.94, 10.53, 3.99, 5.95], dtype=torch.double)
