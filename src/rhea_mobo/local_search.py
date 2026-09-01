"""Stochastic-restart local search over a discrete acquisition function."""

import random

import torch
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.optim.optimize import optimize_acqf_discrete_local_search


def stochastic_local_search(
    acq_function: AcquisitionFunction,
    discrete_choices: list[torch.Tensor],
    inequality_constraints: list[tuple[torch.Tensor, torch.Tensor, float]],
    X_avoid: torch.Tensor,
    batch_initial_conditions: torch.Tensor,
    num_restart: int = 20,
    raw_samples: int = 512,
    exploration_prob: float = 0.05,
) -> tuple[torch.Tensor | None, torch.Tensor]:
    """Run repeated discrete local searches and keep the best (or an exploratory) candidate.

    Each restart calls ``optimize_acqf_discrete_local_search`` independently; a
    candidate replaces the current best whenever its acquisition value clears a
    5% improvement margin, or at random with probability ``exploration_prob`` to
    avoid getting stuck at an early local optimum. Restarts that fail to produce
    a candidate are skipped.

    Returns:
        A tuple of the best candidate found (or ``None`` if every restart
        failed) and its acquisition value as a length-1 tensor.
    """
    best_val = float("-inf")
    best_x: torch.Tensor | None = None

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

            if val > 1.05 * best_val or random.random() < exploration_prob:
                best_val = val
                best_x = candidate
        except (IndexError, RuntimeError):
            continue

    return best_x, torch.tensor([best_val])
