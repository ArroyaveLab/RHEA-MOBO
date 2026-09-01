"""Standalone monolithic reference implementation (see src/rhea_mobo for the modular version).

Requires tc_python and materialsframework[matgl,sqsgen] installed manually; not part of the
installable rhea-mobo package, so run it directly with `uv run python scripts/bo_script_full.py`.
"""

import random
from collections.abc import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.acquisition.multi_objective.analytic import ExpectedHypervolumeImprovement
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.optim.optimize import optimize_acqf_discrete_local_search
from botorch.utils.multi_objective.box_decompositions.non_dominated import FastNondominatedPartitioning
from botorch.utils.multi_objective.hypervolume import Hypervolume
from botorch.utils.multi_objective.pareto import is_non_dominated
from gpytorch.mlls import ExactMarginalLogLikelihood
from materialsframework.analysis import CubicElasticConstantsAnalyzer
from materialsframework.transformations import SqsgenTransformation
from pymatgen.core import Composition
from sklearn.neighbors import KernelDensity
from tc_python import *
from tqdm import tqdm

torch.set_default_dtype(torch.double)

num_input = 6
num_obj = 5
iterations = 200
initial_samples = 50

num_restart = 20
raw_samples = 50

grid_points_per_dim = 21
discrete_choices = [torch.linspace(0.0, 1.0, grid_points_per_dim) for _ in range(num_input)]

elements = ["Mo", "Nb", "Ta", "W", "Co", "Hf"]
ELEMENT_COST = torch.tensor([6.65, 4.92, 10.94, 10.53, 3.99, 5.95], dtype=torch.double)


class CostAwareEHVI(AcquisitionFunction):
    """Acquisition function that divides a base acquisition value by predicted cost."""

    def __init__(self, base_acqf: AcquisitionFunction, cost_model: Callable[[torch.Tensor], torch.Tensor]) -> None:
        """Store the base acquisition function and the cost model used to scale it."""
        super().__init__(model=base_acqf.model)
        self.base_acqf = base_acqf
        self.cost_model = cost_model

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Return the base acquisition value divided by the (clamped) predicted cost."""
        return self.base_acqf(X) / self.cost_model(X).clamp(min=1e-6)


def cost_model(x: torch.Tensor) -> torch.Tensor:
    """Compute the elemental dollar cost of each composition row in ``x``."""
    return (x @ ELEMENT_COST).unsqueeze(-1)


def objective(x: torch.Tensor) -> torch.Tensor:
    """Evaluate the five RHEA design objectives for a single composition.

    Runs, in order: printability (Thermo-Calc Scheil/AM meltpool simulation and
    KDE-based defect-probability estimate), yield strength, crack susceptibility,
    BCC/B2 lattice misfit, and SQS-based Young's modulus. Each objective is
    computed in its own try/except block; a failure in any of the first four
    aborts evaluation and returns an all-NaN result, while a failure in the
    fifth returns an all-NaN result of shape ``(1, 5)``.

    Args:
        x: A ``(1, 6)`` tensor of mole fractions for Mo, Nb, Ta, W, Co, Hf.

    Returns:
        A ``(1, num_obj)`` tensor of objective values (each to be maximized),
        or an all-NaN tensor of the same shape if evaluation failed.
    """
    obj1 = obj2 = obj3 = obj4 = obj5 = float("nan")

    try:
        # --- Objective 1: Printability ---
        power_values = np.linspace(100, 400, 6)  # W
        speed_values = np.linspace(50, 1000, 6)  # mm/s

        grid_data = [{"Power (W)": p, "Speed (mm/s)": s} for p in power_values for s in speed_values]

        df = pd.DataFrame(grid_data)
        df[["Width (um)", "Length (um)", "Depth (um)"]] = np.nan

        with TCPython(logging_policy=LoggingPolicy.SCREEN) as session:
            composition = {
                "Mo": x[:, 0].item(),
                "Nb": x[:, 1].item(),
                "Ta": x[:, 2].item(),
                "W": x[:, 3].item(),
                "Co": x[:, 4].item(),
            }

            dependent_element = "Hf"
            database = "TCHEA7"

            elements = list(composition.keys())

            system = session.select_database_and_elements(database, [dependent_element] + elements).get_system()

            scheil_calc = (
                system.with_scheil_calculation()
                .set_start_temperature(4000.0)
                .set_composition_unit(CompositionUnit.MOLE_FRACTION)
                .with_options(
                    ScheilOptions()
                    .calculate_from_start_temperature()
                    .calculate_to_temperature_below_solidus()
                    .enable_evaporation_property_calculation()
                )
            )

            for el, frac in composition.items():
                scheil_calc = scheil_calc.set_composition(el, frac)

            mp = MaterialProperties.from_scheil_result(scheil_calc.calculate())

            heat_source = HeatSource.gaussian_with_constant_absorptivity().set_absorptivity(60.0).set_beam_radius(40.0e-6)

            am_calc = (
                session.with_additive_manufacturing()
                .with_steady_state_calculation()
                .with_mesh(Mesh().coarse())
                .with_material_properties(mp)
                .with_numerical_options(NumericalOptions().set_number_of_cores(10))
                .disable_fluid_flow_marangoni()
                .set_layer_thickness(30.0e-6)
                .with_heat_source(heat_source)
            )

            for i, row in df.iterrows():
                p = row["Power (W)"]
                s = row["Speed (mm/s)"]

                print(f"Running: Power = {p} W, Speed = {s} mm/s")

                try:
                    heat_source.set_power(p)

                    heat_source.set_scanning_speed(s / 1000.0)  # mm/s -> m/s

                    result = am_calc.calculate()

                    df.at[i, "Width (um)"] = result.get_meltpool_width() * 1e6

                    df.at[i, "Length (um)"] = result.get_meltpool_length() * 1e6

                    df.at[i, "Depth (um)"] = result.get_meltpool_depth() * 1e6

                except Exception as e:
                    print(f"Simulation failed at P={p} W, V={s} mm/s: {e}")

        df_clean = df.dropna(subset=["Width (um)", "Length (um)", "Depth (um)"])

        coords = df_clean[["Length (um)", "Width (um)", "Depth (um)"]].values

        kde = KernelDensity(kernel="gaussian", bandwidth=2.0)

        kde.fit(coords)

        samples = kde.sample(n_samples=50000, random_state=0)

        L = samples[:, 0]
        W = samples[:, 1]
        D = samples[:, 2]

        A = (3 * W - L) <= 0
        B = (2 * W - 3 * D) <= 0
        C = D <= 30

        p_A = A.mean()
        p_B = B.mean()
        p_C = C.mean()

        p_AB = np.logical_and(A, B).mean()
        p_BC = np.logical_and(B, C).mean()
        p_CA = np.logical_and(C, A).mean()

        p_ABC = np.logical_and.reduce([A, B, C]).mean()

        p_union = p_A + p_B + p_C - p_AB - p_BC - p_CA + p_ABC

        printability = 1.0 - p_union

        obj1 = printability

    except Exception:
        return torch.full((1, num_obj), float("nan"), dtype=torch.double)

    try:
        # --- Objective 2: Yield Strength ---
        temp = 1300 + 273  # in K
        composition = {"Mo": x[:, 0], "Nb": x[:, 1], "Ta": x[:, 2], "W": x[:, 3], "Co": x[:, 4]}
        dependent_element = "Hf"

        with TCPython() as session:
            system = (
                session.select_database_and_elements("TCHEA7", ["Mo", "Nb", "Ta", "W", "Co", "Hf"])
                .without_default_phases()
                .select_phase("BCC_B2#1")
                .select_phase("BCC_B2#2")
                .get_system()
            )

            calc = system.with_property_model_calculation("Yield Strength")

            (calc.set_temperature(temp).set_composition_unit(CompositionUnit.MOLE_FRACTION))

            for element in composition:
                calc.set_composition(element, composition[element])

            result = calc.set_argument("Matrix", "BCC_B2#1").set_argument("Precipitate-1", "BCC_B2#2").calculate()

            YS = result.get_value_of("Total yield strength")
        obj2 = YS / 10

    except Exception as e:
        print(f"Yield Strength calc failed: {e}")
        return torch.full((1, num_obj), float("nan"), dtype=torch.double)

    try:
        # --- Objective 3: Crack Susceptibility ---
        with TCPython() as session:
            session.disable_caching()
            active_el = ["Mo", "Nb", "Ta", "W", "Co", "Hf"]
            crack_calc = (
                session.select_database_and_elements("TCHEA7", active_el)
                .get_system()
                .with_property_model_calculation("Crack Susceptibility Coefficient")
                .set_temperature(1300 + 273.15)
                .set_composition_unit(CompositionUnit.MOLE_FRACTION)
                .set_argument("Start temperature", 4500)
                .set_composition("Mo", x[:, 0])
                .set_composition("Nb", x[:, 1])
                .set_composition("Ta", x[:, 2])
                .set_composition("W", x[:, 3])
                .set_composition("Co", x[:, 4])
            )
            crack_result = crack_calc.calculate()
            crack_coefficient = crack_result.get_value_of("Crack Susceptibility Coefficient")
        obj3 = -crack_coefficient

    except Exception as e:
        print(f"Crack Susceptibility calc failed: {e}")
        return torch.full((1, num_obj), float("nan"), dtype=torch.double)

    try:
        # --- Objective 4: Lattice Misfit ---
        with TCPython() as start:
            calculation = (
                start.select_database_and_elements("TCHEA7", ["Mo", "Nb", "Ta", "W", "Co", "Hf"])
                .get_system()
                .with_single_equilibrium_calculation()
                .set_condition(ThermodynamicQuantity.temperature(), 1573)
                .set_condition(ThermodynamicQuantity.mole_fraction_of_a_component("Mo"), x[:, 0])
                .set_condition(ThermodynamicQuantity.mole_fraction_of_a_component("Nb"), x[:, 1])
                .set_condition(ThermodynamicQuantity.mole_fraction_of_a_component("Ta"), x[:, 2])
                .set_condition(ThermodynamicQuantity.mole_fraction_of_a_component("W"), x[:, 3])
                .set_condition(ThermodynamicQuantity.mole_fraction_of_a_component("Co"), x[:, 4])
            )

            property_diagram = calculation.calculate()
            bcc_Vm = property_diagram.get_value_of(ThermodynamicQuantity.molar_volume_of_phase("BCC_B2#1"))
            b2_Vm = property_diagram.get_value_of(ThermodynamicQuantity.molar_volume_of_phase("BCC_B2#2"))
            a_b2 = (2 * b2_Vm / 6.02e23) ** (1 / 3)
            a_bcc = (2 * bcc_Vm / 6.02e23) ** (1 / 3)
            misfit = abs(200 * (a_b2 - a_bcc) / (a_bcc + a_b2))
        obj4 = -misfit

    except Exception as e:
        print(f"Misfit calc failed: {e}")
        return torch.full((1, num_obj), float("nan"), dtype=torch.double)

    try:
        # --- Objective 5: Young's Modulus ---
        sqs_generator = SqsgenTransformation()
        elastic_analyzer = CubicElasticConstantsAnalyzer()

        alloy = Composition(f"Mo{x[:, 0]}Nb{x[:, 1]}Ta{x[:, 2]}W{x[:, 3]}Co{x[:, 4]}Hf{x[:, 5]}")
        sqs_res = sqs_generator.generate(composition=alloy, crystal_structure="bcc", supercell_size=(10, 10, 10))
        bcc_MoNbTaWCoHf = sqs_res["structure"]

        elas_res = elastic_analyzer.calculate(bcc_MoNbTaWCoHf)

        youngs_modulus = elas_res["youngs_modulus"]
        obj5 = youngs_modulus / 10
    except Exception as e:
        print(f"Young Modulus prediction failed: {e}")
        return torch.full((1, 5), float("nan"), dtype=torch.double)

    return torch.tensor([[obj1, obj2, obj3, obj4, obj5]], dtype=torch.double)


def fit_single_task_model(train_x: torch.Tensor, train_y: torch.Tensor) -> SingleTaskGP:
    """Fit a single-output GP to one objective's training data."""
    model = SingleTaskGP(train_x, train_y)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)
    return model


def build_model(train_x: torch.Tensor, train_y: torch.Tensor) -> ModelListGP:
    """Fit one GP per objective column, dropping rows with any NaN objective value.

    Raises:
        RuntimeError: If every row has at least one NaN objective, leaving no
            usable training data.
    """
    mask = ~torch.isnan(train_y).any(dim=1)
    clean_x = train_x[mask]
    clean_y = train_y[mask]
    if clean_x.shape[0] == 0:
        raise RuntimeError("All training outputs are NaN.")
    models = [fit_single_task_model(clean_x, clean_y[:, i : i + 1]) for i in range(clean_y.shape[-1])]
    return ModelListGP(*models)


def get_acquisition(model: ModelListGP, train_y: torch.Tensor, ref_point: torch.Tensor) -> ExpectedHypervolumeImprovement:
    """Build an Expected Hypervolume Improvement acquisition function over the current Pareto front."""
    pareto_y = train_y[is_non_dominated(train_y)]
    partitioning = FastNondominatedPartitioning(ref_point=ref_point, Y=pareto_y)
    return ExpectedHypervolumeImprovement(model=model, ref_point=ref_point.tolist(), partitioning=partitioning)


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


idx = torch.tensor([0, 1, 2, 3, 4, 5], dtype=torch.long)
coef1 = torch.tensor([1.0] * 6, dtype=torch.double)
coef2 = torch.tensor([-1.0] * 6, dtype=torch.double)
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

X1, X2, X3, X4, X5, X6 = torch.meshgrid(*discrete_choices, indexing="ij")
xs = torch.stack([X1.flatten(), X2.flatten(), X3.flatten(), X4.flatten(), X5.flatten(), X6.flatten()], dim=-1)

mask_sum = torch.isclose(xs.sum(dim=1), torch.ones(xs.shape[0]), atol=1e-8)
mask_equal = torch.isclose(xs[:, 4], xs[:, 5], atol=1e-8)
mask_positive = (xs > 0).all(dim=1)

mask_total = mask_sum & mask_equal & mask_positive
xs_feasible = xs[mask_total]

sqs_generator = SqsgenTransformation()
elastic_analyzer = CubicElasticConstantsAnalyzer()


def run_optimization(
    num_queries: int, init_points: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run cost-aware EHVI Bayesian optimization over the feasible composition grid.

    Seeds training data from an evenly spaced subset of the feasible grid, then
    iteratively fits a GP per objective, proposes a candidate via cost-weighted
    EHVI local search, rejects it if it fails a predicted-ductility check or
    yields a NaN objective, and otherwise adds it to the training set.

    Args:
        num_queries: Number of accepted candidates to collect before stopping.
        init_points: Number of initial design points sampled from the feasible
            grid before the optimization loop begins.

    Returns:
        A tuple ``(hypervolumes, train_x, train_y, pareto_x, pareto_y)``: the
        running hypervolume after each accepted candidate, the full training
        inputs and objective values, and the inputs/objectives on the final
        Pareto front.

    Raises:
        ValueError: If `init_points` exceeds the number of feasible grid points.
    """
    if init_points > len(xs_feasible):
        raise ValueError("init_points is larger than the number of available feasible points.")
    indices = torch.linspace(0, len(xs_feasible) - 1, init_points).long()
    train_x = xs_feasible[indices]
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
            print(f"[Iteration {pbar.n + 1}] Warning: No valid candidate found. Retrying...")
            continue

        # === Apply Ductility constraint ===
        x_new = candidate[:, :6].numpy()
        alloy_new = Composition(f"Mo{x_new[:, 0]}Nb{x_new[:, 1]}Ta{x_new[:, 2]}W{x_new[:, 3]}Co{x_new[:, 4]}Hf{x_new[:, 5]}")
        sqs_res = sqs_generator.generate(composition=alloy_new, crystal_structure="bcc", supercell_size=(10, 10, 10))
        bcc_MoNbTaWCoHf = sqs_res["structure"]
        elas_res = elastic_analyzer.calculate(bcc_MoNbTaWCoHf)
        pugh_ratio = elas_res["pugh_ratio"]
        predicted_D_new = 1 / pugh_ratio

        if predicted_D_new <= 2.5:
            print(f"[Iteration {pbar.n + 1}] Candidate rejected (predicted D={predicted_D_new:.2f} ≤ 2.5)")
            continue

        next_y = objective(candidate)

        if torch.isnan(next_y).any():
            print(f"[Iteration {pbar.n + 1}] Candidate skipped due to NaN in objective.")
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
