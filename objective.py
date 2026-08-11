import numpy as np
import pandas as pd
import torch
from scipy.stats import norm, multivariate_normal
from tc_python import *
from scipy.stats import t, multivariate_t
from sklearn.neighbors import KernelDensity
from pymatgen.core import Composition

from materialsframework.analysis import CubicElasticConstantsAnalyzer
from materialsframework.transformations import SqsgenTransformation
warnings.simplefilter("ignore")


from config import num_obj

# --------------------
# Objective Function
# --------------------

def objective(x):
    obj1 = obj2 = obj3 = obj4 = obj5 = float("nan")
    
    try:
        # --- Objective 1: Printability ---
    
        # 1. Set up the power-speed grid
        power_values = np.linspace(100, 400, 3)      # W
        speed_values = np.linspace(50, 1000, 4)      # mm/s
    
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
                    set_condition(ThermodynamicQuantity.temperature(), 1573.).
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
