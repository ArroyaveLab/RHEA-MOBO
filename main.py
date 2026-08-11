import matplotlib.pyplot as plt

from config import iterations, initial_samples
from optimization import run_optimization

# --------------------
# Run and Plot
# --------------------
hypervolumes, final_train_x, final_train_y, final_pareto_x, final_pareto_y = run_optimization(
    num_queries=iterations,
    init_points=initial_samples,
)


plt.figure(figsize=(8, 5))
plt.plot(hypervolumes.cpu().numpy(), marker="o")
plt.xlabel("Iteration")
plt.ylabel("Hypervolume")
plt.title("Cost-aware EHVI Optimization")
plt.grid(True)
plt.tight_layout()
plt.show()
