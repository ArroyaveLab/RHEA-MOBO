"""License-free synthetic exercise; no alloy properties or historical results."""

import argparse
import csv
import itertools
import json
import random
from pathlib import Path

import torch
from botorch.utils.multi_objective.hypervolume import Hypervolume
from botorch.utils.multi_objective.pareto import is_non_dominated

from .config import ELEMENT_COST, elements
from .constraints import inequality_constraints
from .cost import CostAwareEHVI, cost_model
from .gp_utils import build_model, get_acquisition
from .local_search import stochastic_local_search


def composition_grid() -> torch.Tensor:
    """Enumerate the 1,716 positive 5 mol% compositions without a 21**6 tensor."""
    rows = []
    for co in range(1, 9):
        for mo, nb, ta in itertools.product(range(1, 20), repeat=3):
            w = 20 - 2 * co - mo - nb - ta
            if w >= 1:
                rows.append([mo, nb, ta, w, co, co])
    return torch.tensor(sorted(rows), dtype=torch.double) / 20


def synthetic_objectives(x: torch.Tensor) -> torch.Tensor:
    """Five maximized, dimensionless utilities with competing preferred Mo fractions.

    y_j = 1 - (t-c_j)^2 - p, t=(x_Mo-0.05)/0.70,
    c=(0,0.25,0.5,0.75,1), p=2*((x_Ta-.05)^2+(x_W-.05)^2+(x_Co-.05)^2).
    These are analytical test functions, not physical property predictions.
    """
    t = (x[..., 0] - 0.05) / 0.70
    centers = torch.linspace(0, 1, 5, dtype=x.dtype, device=x.device)
    penalty = 2 * ((x[..., [2, 3, 4]] - 0.05) ** 2).sum(dim=-1)
    return 1 - (t.unsqueeze(-1) - centers) ** 2 - penalty.unsqueeze(-1)


def feasible(x: torch.Tensor) -> torch.Tensor:
    """Synthetic screening rule, unrelated to the physical Pugh ratio."""
    return x[..., 0] + x[..., 3] <= 0.65 + 1e-10


def random_candidates(grid: torch.Tensor, initial_x: torch.Tensor, count: int, seed: int) -> torch.Tensor:
    """Uniform feasible sampling without replacement using an independent RNG."""
    available = grid[feasible(grid) & ~torch.isclose(grid[:, None], initial_x[None], atol=1e-9).all(-1).any(-1)]
    if count < 0 or count > len(available):
        raise ValueError("Random budget exceeds remaining feasible candidates")
    generator = torch.Generator().manual_seed(seed)
    return available[torch.randperm(len(available), generator=generator)[:count]]


def run(output: Path, iterations: int = 3, initial: int = 8, seed: int = 42) -> dict:  # noqa: PLR0912
    """Run bounded BO and compare accepted evaluations with an exhaustive oracle.

    Uses the production GP/EHVI/local-search code. Toy-specific orchestration
    excludes rejected proposals and bounds retries; it does not test TC-Python.
    """
    if iterations < 1 or initial < 2:
        raise ValueError("iterations must be positive and initial must be at least two")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Choose an empty output directory to preserve previous runs")
    torch.manual_seed(seed)
    random.seed(seed)
    torch.set_num_threads(1)
    grid = composition_grid()
    valid = feasible(grid)
    oracle_x = grid[valid]
    oracle_y = synthetic_objectives(oracle_x)
    if initial + iterations > len(oracle_x):
        raise ValueError("Requested more evaluations than feasible candidates")
    oracle_front = is_non_dominated(oracle_y)
    reference = torch.full((5,), -1.0, dtype=torch.double)
    hv = Hypervolume(reference)
    oracle_hv = hv.compute(oracle_y[oracle_front])
    train_x = oracle_x[torch.randperm(len(oracle_x))[:initial]]
    train_y = synthetic_objectives(train_x)
    random_seed = seed + 1
    random_x = torch.cat([train_x.clone(), random_candidates(grid, train_x, iterations, random_seed)])
    random_y = synthetic_objectives(random_x)
    initial_hv = hv.compute(train_y[is_non_dominated(train_y)])
    comparison = [
        {
            "accepted_additions": 0,
            "total_evaluations": initial,
            "bo_hypervolume": initial_hv,
            "random_hypervolume": initial_hv,
            "bo_oracle_fraction": initial_hv / oracle_hv,
            "random_oracle_fraction": initial_hv / oracle_hv,
        }
    ]
    avoided = train_x.clone()
    history = []
    progress = []
    output.mkdir(parents=True, exist_ok=True)

    def save() -> None:
        with (output / "evaluations.csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["evaluation", "stage", *elements, *[f"synthetic_y{i}" for i in range(1, 6)]])
            for i, (x, y) in enumerate(zip(train_x.tolist(), train_y.tolist(), strict=True)):
                writer.writerow([i, "initial" if i < initial else "sequential", *x, *y])
        with (output / "random_evaluations.csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["evaluation", "stage", *elements, *[f"synthetic_y{i}" for i in range(1, 6)]])
            for i in range(len(train_x)):
                writer.writerow([i, "initial" if i < initial else "sequential", *random_x[i].tolist(), *random_y[i].tolist()])
        with (output / "comparison.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(comparison[0]))
            writer.writeheader()
            writer.writerows(comparison)
        (output / "progress.json").write_text(json.dumps(progress, indent=2))
        (output / "proposals.json").write_text(json.dumps(history, indent=2))

    save()
    for step in range(iterations):
        model = build_model(train_x, train_y)
        base = get_acquisition(model, train_y, reference)
        acquisition = CostAwareEHVI(base, cost_model)
        # Exercise real batched weighting independently of optimizer behavior.
        probe = grid[:3].unsqueeze(1)
        with torch.no_grad():
            actual = acquisition(probe)
            expected = base(probe) / (grid[:3] @ ELEMENT_COST)
        if actual.shape != (3,) or not torch.allclose(actual, expected):
            raise RuntimeError("Acquisition must return one correctly weighted score per candidate")
        for attempt in range(50):
            available = grid[~torch.isclose(grid[:, None], avoided[None], atol=1e-9).all(-1).any(-1)]
            starts = available[torch.randperm(len(available))[:4]].unsqueeze(1)
            candidate, _ = stochastic_local_search(
                acq_function=acquisition,
                discrete_choices=[torch.linspace(0, 1, 21)] * 6,
                inequality_constraints=inequality_constraints,
                X_avoid=avoided,
                batch_initial_conditions=starts,
                num_restart=2,
                raw_samples=8,
            )
            if candidate is None:
                raise RuntimeError("Production local search returned no candidate")
            if not torch.isclose(grid, candidate, atol=1e-9).all(-1).any():
                raise RuntimeError("Local search returned an off-grid composition")
            if torch.isclose(avoided, candidate, atol=1e-9).all(-1).any():
                raise RuntimeError("Local search repeated an excluded composition")
            avoided = torch.cat([avoided, candidate])
            accepted = bool(feasible(candidate).item())
            history.append({"step": step, "attempt": attempt, "composition": candidate.tolist()[0], "accepted": accepted})
            if accepted:
                break
        else:
            raise RuntimeError("Synthetic feasibility screening exceeded 50 attempts")
        train_x = torch.cat([train_x, candidate])
        train_y = torch.cat([train_y, synthetic_objectives(candidate)])
        attained = hv.compute(train_y[is_non_dominated(train_y)])
        if progress and attained < progress[-1]["hypervolume"] - 1e-9:
            raise RuntimeError("Hypervolume decreased")
        if attained > oracle_hv + 1e-8:
            raise RuntimeError("Attained hypervolume exceeds exhaustive feasible oracle")
        progress.append({"accepted_additions": step + 1, "hypervolume": attained, "oracle_fraction": attained / oracle_hv})
        random_values = random_y[: len(train_x)]
        random_hv = hv.compute(random_values[is_non_dominated(random_values)])
        comparison.append(
            {
                "accepted_additions": step + 1,
                "total_evaluations": len(train_x),
                "bo_hypervolume": attained,
                "random_hypervolume": random_hv,
                "bo_oracle_fraction": attained / oracle_hv,
                "random_oracle_fraction": random_hv / oracle_hv,
            }
        )
        save()
        print(f"Random: hypervolume/oracle = {random_hv / oracle_hv:.4f}", flush=True)
        print(f"Step {step + 1}: hypervolume/oracle = {attained / oracle_hv:.4f}", flush=True)
    with (output / "oracle_pareto.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow([*elements, *[f"synthetic_y{i}" for i in range(1, 6)]])
        writer.writerows(torch.cat([oracle_x[oracle_front], oracle_y[oracle_front]], dim=-1).tolist())
    summary = {
        "synthetic": True,
        "seed": seed,
        "grid_count": len(grid),
        "feasible_count": len(oracle_x),
        "oracle_pareto_count": int(oracle_front.sum()),
        "oracle_hypervolume": oracle_hv,
        "reference_point": reference.tolist(),
        "initial": initial,
        "iterations": iterations,
        "final_oracle_fraction": progress[-1]["oracle_fraction"],
        "random_seed": random_seed,
        "random_final_oracle_fraction": comparison[-1]["random_oracle_fraction"],
        "comparison_budget": "accepted objective evaluations; screening effort is not matched",
        "torch_version": torch.__version__,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    """Run the license-free example from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("toy-output"))
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--initial", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run(args.output, args.iterations, args.initial, args.seed)


if __name__ == "__main__":
    main()
