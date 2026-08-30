# -*- coding: utf-8 -*-
import random
import numpy as np
import torch
import gpytorch
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
from tc_python import *
from scipy.stats import norm, multivariate_normal
from sklearn.neighbors import KernelDensity
from pymatgen.core import Composition
from materialsframework.analysis import CubicElasticConstantsAnalyzer
from materialsframework.transformations import SqsgenTransformation

from botorch.models import SingleTaskGP
from botorch.models.model_list_gp_regression import ModelListGP
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition.multi_objective.analytic import ExpectedHypervolumeImprovement
from botorch.utils.multi_objective.pareto import is_non_dominated
from botorch.utils.multi_objective.box_decompositions.non_dominated import FastNondominatedPartitioning
from botorch.utils.multi_objective.hypervolume import Hypervolume
from botorch.optim.optimize import optimize_acqf_discrete_local_search

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

# --------------------
# Objective Function
# --------------------

def objective(x):
    obj1 = obj2 = obj3 = obj4 = obj5 = float("nan")

    try:
        # --- Objective 1: Printability ---
    
        # 1. Set up the power-speed grid
        power_values = np.linspace(100, 400, 6)      # W
        speed_values = np.linspace(50, 1000, 6)      # mm/s
    
        grid_data = [
            {"Power (W)": p, "Speed (mm/s)": s}
            for p in power_values for s in speed_values
        ]
    
        df = pd.DataFrame(grid_data)
        df[["Width (um)", "Length (um)", "Depth (um)"]] = np.nan
    
        # 2. Thermo-Calc session
        with TCPython(logging_policy=LoggingPolicy.SCREEN) as session:
    
            composition = {
                "Mo": x[:, 0].item(),
                "Nb": x[:, 1].item(),
                "Ta": x[:, 2].item(),
                "W": x[:, 3].item(),
                "Co": x[:, 4].item()
            }
    
            dependent_element = "Hf"
            database = "TCHEA7"
    
            # Build system and material properties
            elements = list(composition.keys())
    
            system = session.select_database_and_elements(
                database,
                [dependent_element] + elements
            ).get_system()
    
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
    
            mp = MaterialProperties.from_scheil_result(
                scheil_calc.calculate()
            )
    
            # Set up AM calculation
            heat_source = (
                HeatSource.gaussian_with_constant_absorptivity()
                .set_absorptivity(60.0)
                .set_beam_radius(40.0e-6)
            )
    
            am_calc = (
                session.with_additive_manufacturing()
                .with_steady_state_calculation()
                .with_mesh(Mesh().coarse())
                .with_material_properties(mp)
                .with_numerical_options(
                    NumericalOptions().set_number_of_cores(10)
                )
                .disable_fluid_flow_marangoni()
                .set_layer_thickness(30.0e-6)
                .with_heat_source(heat_source)
            )
    
            # 3. Loop over power-speed grid
            for i, row in df.iterrows():
    
                p = row["Power (W)"]
                s = row["Speed (mm/s)"]
    
                print(
                    f"Running: Power = {p} W, "
                    f"Speed = {s} mm/s"
                )
    
                try:
                    heat_source.set_power(p)
    
                    heat_source.set_scanning_speed(
                        s / 1000.0
                    )  # mm/s -> m/s
    
                    result = am_calc.calculate()
    
                    df.at[i, "Width (um)"] = (
                        result.get_meltpool_width() * 1e6
                    )
    
                    df.at[i, "Length (um)"] = (
                        result.get_meltpool_length() * 1e6
                    )
    
                    df.at[i, "Depth (um)"] = (
                        result.get_meltpool_depth() * 1e6
                    )
    
                except Exception as e:
                    print(
                        f"Simulation failed at "
                        f"P={p} W, V={s} mm/s: {e}"
                    )
    
        # 4. Clean results
        df_clean = df.dropna(
            subset=[
                "Width (um)",
                "Length (um)",
                "Depth (um)"
            ]
        )
    
        coords = df_clean[
            [
                "Length (um)",
                "Width (um)",
                "Depth (um)"
            ]
        ].values
    
        # 5. KDE sampling
        kde = KernelDensity(
            kernel="gaussian",
            bandwidth=2.0
        )
    
        kde.fit(coords)
    
        samples = kde.sample(
            n_samples=50000,
            random_state=0
        )
    
        # 6. Defect criteria
        L = samples[:, 0]
        W = samples[:, 1]
        D = samples[:, 2]
    
        A = (3 * W - L) <= 0
        B = (2 * W - 3 * D) <= 0
        C = D <= 30
    
        # 7. Inclusion-exclusion calculation
        p_A = A.mean()
        p_B = B.mean()
        p_C = C.mean()
    
        p_AB = np.logical_and(A, B).mean()
        p_BC = np.logical_and(B, C).mean()
        p_CA = np.logical_and(C, A).mean()
    
        p_ABC = np.logical_and.reduce(
            [A, B, C]
        ).mean()
    
        p_union = (
            p_A
            + p_B
            + p_C
            - p_AB
            - p_BC
            - p_CA
            + p_ABC
        )
    
        printability = 1.0 - p_union
    
        obj1 = printability
    
        # 8. Save melt-pool results
        comp_id = "-".join(
            f"{v.item():.2f}"
            for v in x[0]
        )
    
    except Exception as e:
        return torch.full(
            (1, num_obj),
            float("nan"),
            dtype=torch.double
        )

    try:
        # --- Objective 2: Yield Strength ---
        # [Yield strength block...]
        temp = 1300+273  # in K
        composition = {"Mo": x[:, 0], "Nb": x[:, 1], "Ta": x[:, 2], "W": x[:, 3], "Co": x[:, 4]}
        dependent_element = "Hf"

        with TCPython() as session:
            system = (session
                .select_database_and_elements("TCHEA7", ["Mo", "Nb", "Ta", "W", "Co", "Hf"])
                .without_default_phases()
                .select_phase("BCC_B2#1")
                .select_phase("BCC_B2#2")
                .get_system())

            calc = system.with_property_model_calculation("Yield Strength")

            (calc
                .set_temperature(temp)
                .set_composition_unit(CompositionUnit.MOLE_FRACTION))

            for element in composition:
                calc.set_composition(element, composition[element])

            result = (calc
                .set_argument("Matrix", "BCC_B2#1")
                .set_argument("Precipitate-1", "BCC_B2#2")
                .calculate())

            YS = result.get_value_of("Total yield strength")
        obj2 = YS/10

    except Exception as e:
        print(f"Yield Strength calc failed: {e}")
        return torch.full((1, num_obj), float("nan"), dtype=torch.double)

    try:
        # --- Objective 3: Crack Susceptibility ---
        # [Crack calc block...]
        with TCPython() as session:
            session.disable_caching()
            active_el = ['Mo', 'Nb', 'Ta', 'W', 'Co', 'Hf']
            crack_calc = (
                    session.select_database_and_elements('TCHEA7', active_el)
                    .get_system()
                    .with_property_model_calculation('Crack Susceptibility Coefficient')
                    .set_temperature(1300 + 273.15)
                    .set_composition_unit(CompositionUnit.MOLE_FRACTION)
                    .set_argument('Start temperature', 4500)
                    .set_composition('Mo', x[:, 0])
                    .set_composition('Nb', x[:, 1])
                    .set_composition('Ta', x[:, 2])
                    .set_composition('W', x[:, 3])
                    .set_composition('Co', x[:, 4]))
            crack_result = crack_calc.calculate()
            crack_coefficient = crack_result.get_value_of('Crack Susceptibility Coefficient')
        obj3 = - crack_coefficient

    except Exception as e:
        print(f"Crack Susceptibility calc failed: {e}")
        return torch.full((1, num_obj), float("nan"), dtype=torch.double)

    try:
        # --- Objective 4: Lattice Misfit ---
        # [lattice misfit block...]

        with TCPython() as start:
            calculation = (
                start.select_database_and_elements("TCHEA7", ["Mo", "Nb", "Ta", "W", "Co", "Hf"]).get_system().
                    with_single_equilibrium_calculation().
                    set_condition(ThermodynamicQuantity.temperature(), 1573).
                    set_condition(ThermodynamicQuantity.mole_fraction_of_a_component("Mo"), x[:, 0]).
                    set_condition(ThermodynamicQuantity.mole_fraction_of_a_component("Nb"), x[:, 1]).
                    set_condition(ThermodynamicQuantity.mole_fraction_of_a_component("Ta"), x[:, 2]).
                    set_condition(ThermodynamicQuantity.mole_fraction_of_a_component("W"), x[:, 3]).
                    set_condition(ThermodynamicQuantity.mole_fraction_of_a_component("Co"), x[:, 4])
            )

            property_diagram = calculation.calculate()
            bcc_Vm = property_diagram.get_value_of(ThermodynamicQuantity.molar_volume_of_phase('BCC_B2#1'))
            b2_Vm = property_diagram.get_value_of(ThermodynamicQuantity.molar_volume_of_phase('BCC_B2#2'))
            a_b2 = (2*b2_Vm/6.02e23)**(1/3)
            a_bcc = (2*bcc_Vm/6.02e23)**(1/3)
            misfit = abs(200 * (a_b2 - a_bcc) / (a_bcc + a_b2))
        obj4 = - misfit

    except Exception as e:
        print(f"Misfit calc failed: {e}")
        return torch.full((1, num_obj), float("nan"), dtype=torch.double)

    try:
        # --- Objective 5: Young's Modulus ---
        sqs_generator = SqsgenTransformation()
        elastic_analyzer = CubicElasticConstantsAnalyzer()

        # Define the composition
        alloy = Composition(f"Mo{x[:, 0]}Nb{x[:, 1]}Ta{x[:, 2]}W{x[:, 3]}Co{x[:, 4]}Hf{x[:, 5]}")

        # Generate a supercell with SQS
        sqs_res = sqs_generator.generate(composition=alloy, crystal_structure="bcc", supercell_size=(10, 10, 10))
        bcc_MoNbTaWCoHf = sqs_res["structure"]

        # Calculate the elastic constants
        elas_res = elastic_analyzer.calculate(bcc_MoNbTaWCoHf)

        youngs_modulus = elas_res['youngs_modulus']
        obj5 = youngs_modulus/10
    except Exception as e:
        print(f"Young Modulus prediction failed: {e}")
        return torch.full((1, 5), float("nan"), dtype=torch.double)

    return torch.tensor([[obj1, obj2, obj3, obj4, obj5]], dtype=torch.double)

# --------------------
# GP‐fitting utilities
# --------------------
def fit_single_task_model(train_x, train_y):
    model = SingleTaskGP(train_x, train_y)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)
    return model

def build_model(train_x, train_y):
    mask = ~torch.isnan(train_y).any(dim=1)
    clean_x = train_x[mask]
    clean_y = train_y[mask]
    if clean_x.shape[0] == 0:
        raise RuntimeError("All training outputs are NaN.")
    models = [fit_single_task_model(clean_x, clean_y[:, i : i + 1]) for i in range(clean_y.shape[-1])]
    return ModelListGP(*models)

def get_acquisition(model, train_y, ref_point):
    pareto_y = train_y[is_non_dominated(train_y)]
    partitioning = FastNondominatedPartitioning(ref_point=ref_point, Y=pareto_y)
    return ExpectedHypervolumeImprovement(model=model, ref_point=ref_point.tolist(), partitioning=partitioning)

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

# --------------------
# Inequality Constraints
# --------------------
idx = torch.tensor([0, 1, 2, 3, 4, 5], dtype=torch.long)
coef1 = torch.tensor([1.0]*6, dtype=torch.double)
coef2 = torch.tensor([-1.0]*6, dtype=torch.double)
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

# --------------------
# Feasible Grid
# --------------------
# --------------------
# Feasible Grid
# --------------------
X1, X2, X3, X4, X5, X6 = torch.meshgrid(*discrete_choices, indexing="ij")
xs = torch.stack([X1.flatten(), X2.flatten(), X3.flatten(), X4.flatten(), X5.flatten(), X6.flatten()], dim=-1)

# Constraint 1: sum to 1
mask_sum = torch.isclose(xs.sum(dim=1), torch.ones(xs.shape[0]), atol=1e-8)

# Constraint 2: X5 == X6
mask_equal = torch.isclose(xs[:, 4], xs[:, 5], atol=1e-8)

# Constraint 3: All values > 0
mask_positive = (xs > 0).all(dim=1)

# Combine masks (before ML-based filtering)
mask_total = mask_sum & mask_equal & mask_positive
xs_filtered = xs[mask_total]

# Final feasible set
xs_feasible = xs_filtered


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

