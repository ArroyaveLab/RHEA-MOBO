"""Entry point: run the optimization loop and plot hypervolume progress."""

import matplotlib.pyplot as plt

from .config import initial_samples, iterations
from .optimization import run_optimization


def main() -> None:
    """Run cost-aware EHVI optimization and plot the hypervolume trace."""
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


if __name__ == "__main__":
    main()
