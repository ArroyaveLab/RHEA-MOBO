import torch
from tqdm import tqdm
from tc_python import *
from botorch.utils.multi_objective.pareto import is_non_dominated
from botorch.utils.multi_objective.hypervolume import Hypervolume

from config import discrete_choices, num_restart, raw_samples
from constraints import inequality_constraints
from cost import CostAwareEHVI, cost_model
from feasible_grid import xs_feasible
from gp_utils import build_model, get_acquisition
from local_search import stochastic_local_search
from objective import objective

# --------------------
# Main BO Loop with Progress Bar
# --------------------
def run_optimization(num_queries, init_points):
    # ----- Linspace initial sampling -----
    if init_points > len(xs_feasible):
        raise ValueError("init_points is larger than the number of available feasible points.")
    indices = torch.linspace(0, len(xs_feasible) - 1, init_points).long()
    train_x = xs_feasible[indices]
    # Evaluate objectives sequentially for each initial sample
    train_y_list = []
    for i in range(init_points):
        yi = objective(train_x[i].unsqueeze(0))
        train_y_list.append(yi)
    train_y = torch.cat(train_y_list, dim=0)
    ref_point = torch.tensor([0, 15, -3.5, -3.5, 15], dtype=torch.double)
    hypervolumes = []

    pbar = tqdm(total=num_queries, desc="Cost-aware EHVI Optimization")

    while pbar.n < num_queries:
        model = build_model(train_x, train_y)
        base_acqf = get_acquisition(model, train_y, ref_point)
        acq_function = CostAwareEHVI(base_acqf, cost_model)

        perm = torch.randperm(len(xs_feasible))[:num_restart]
        batch_initial_conditions = xs_feasible[perm].unsqueeze(1)

        candidate, _ = stochastic_local_search(
            acq_function=acq_function,
            discrete_choices=discrete_choices,
            inequality_constraints=inequality_constraints,
            batch_initial_conditions=batch_initial_conditions,
            X_avoid=train_x,
            num_restart=num_restart,
            raw_samples=raw_samples,
            exploration_prob=0.05,
        )

        if candidate is None:
            print(f"[Iteration {pbar.n+1}] Warning: No valid candidate found. Retrying...")
            continue

        # === Apply Ductility constraint ===
        x_new = candidate[:, :6].numpy()
        alloy_new = Composition(f"Mo{x_new[:, 0]}Nb{x_new[:, 1]}Ta{x_new[:, 2]}W{x_new[:, 3]}Co{x_new[:, 4]}Hf{x_new[:, 5]}")
        # Generate a supercell with SQS
        sqs_res = sqs_generator.generate(composition=alloy_new, crystal_structure="bcc", supercell_size=(10, 10, 10))
        bcc_MoNbTaWCoHf = sqs_res["structure"]
        # Calculate the elastic constants
        elas_res = elastic_analyzer.calculate(bcc_MoNbTaWCoHf)
        pugh_ratio = elas_res['pugh_ratio']
        predicted_D_new = 1/pugh_ratio

        if predicted_D_new <= 2.5:
            print(f"[Iteration {pbar.n+1}] Candidate rejected (predicted D={predicted_D_new:.2f} ≤ 2.5)")
            continue

        next_y = objective(candidate)

        if torch.isnan(next_y).any():
            print(f"[Iteration {pbar.n+1}] Candidate skipped due to NaN in objective.")
            continue

        train_x = torch.cat([train_x, candidate], dim=0)
        train_y = torch.cat([train_y, next_y], dim=0)
        pareto_x = train_x[is_non_dominated(train_y)]
        pareto_y = train_y[is_non_dominated(train_y)]
        hv = 100 * Hypervolume(ref_point).compute(pareto_y)
        hypervolumes.append(hv)
        pbar.update(1)

    pbar.close()
    return torch.tensor(hypervolumes), train_x, train_y, pareto_x, pareto_y
