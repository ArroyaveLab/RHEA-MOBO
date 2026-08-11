import random
import torch

from botorch.optim.optimize import optimize_acqf_discrete_local_search

# --------------------
# Local Search
# --------------------
def stochastic_local_search(
    acq_function, discrete_choices, inequality_constraints, X_avoid,
    batch_initial_conditions, num_restart=20, raw_samples=512, exploration_prob=0.05,
):
    best_val = float("-inf")
    best_x = None

    for _ in range(num_restart):
        try:
            candidate, acq_val_tensor = optimize_acqf_discrete_local_search(
                acq_function=acq_function,
                discrete_choices=discrete_choices,
                q=1,
                num_restarts=1,
                raw_samples=raw_samples,
                inequality_constraints=inequality_constraints,
                X_avoid=X_avoid,
                batch_initial_conditions=batch_initial_conditions,
                unique=True,
            )
            val = acq_val_tensor.item()

            if val > 1.05 * best_val:
                best_val = val
                best_x = candidate
            elif random.random() < exploration_prob:
                best_val = val
                best_x = candidate
        except (IndexError, RuntimeError):
            continue

    return best_x, torch.tensor([best_val])
