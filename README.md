# RHEA-MOBO

Multi-objective Bayesian optimization framework for composition design
and property optimization of **refractory high-entropy alloys (RHEAs)**.

This repository implements a cost-aware multi-objective Bayesian
optimization (MOBO) workflow for exploring the composition space of the
six-component alloy system:

**Mo--Nb--Ta--W--Co--Hf**

The optimization combines Gaussian-process surrogate models, expected
hypervolume improvement, discrete composition constraints, material cost
considerations, and physics-based property calculations.

------------------------------------------------------------------------

## Overview

The framework searches a constrained RHEA composition space and
optimizes multiple material-property objectives simultaneously.

The current implementation includes:

-   constrained six-component alloy composition design;
-   Gaussian-process surrogate modeling using BoTorch and GPyTorch;
-   multi-objective Bayesian optimization using Expected Hypervolume
    Improvement (EHVI);
-   material-cost-aware acquisition;
-   discrete stochastic local search;
-   Pareto-front identification;
-   hypervolume tracking;
-   thermodynamic and additive-manufacturing calculations using
    Thermo-Calc / TC-Python;
-   statistical and materials-property calculations using SciPy,
    scikit-learn, pymatgen, and related utilities;
-   post-processing and correlation analysis of predicted alloy
    properties.

The six design variables are the atomic/mole fractions of:

``` text
Mo, Nb, Ta, W, Co, Hf
```

The default configuration uses five optimization objectives.

------------------------------------------------------------------------

## Repository Structure

``` text
RHEA-MOBO/
│
├── __init__.py
├── main.py
├── config.py
├── constraints.py
├── cost.py
├── feasible_grid.py
├── gp_utils.py
├── local_search.py
├── objective.py
├── optimization.py
├── bo_script_full.py
├── predicted_correlation.py
├── predictions.csv
└── LICENSE
```

### `main.py`

Main entry point for the modular implementation.

It calls the optimization routine defined in `optimization.py` and plots
the hypervolume as a function of Bayesian-optimization iteration.

### `config.py`

Contains the principal optimization settings, including:

-   number of input variables;
-   number of objectives;
-   number of BO iterations;
-   number of initial samples;
-   acquisition-function restart settings;
-   discrete composition grid;
-   alloy element names;
-   elemental cost values.

The current default configuration is:

``` python
num_input = 6
num_obj = 5
iterations = 200
initial_samples = 50
num_restart = 20
raw_samples = 50
grid_points_per_dim = 21
```

with:

``` python
elements = ["Mo", "Nb", "Ta", "W", "Co", "Hf"]
```

### `constraints.py`

Defines the linear constraints imposed on the alloy composition.

These include composition normalization, equality/inequality
relationships, and lower bounds on selected elemental fractions.

### `feasible_grid.py`

Constructs the discrete composition grid and filters it according to the
imposed feasibility constraints.

The resulting feasible composition set is stored as:

``` python
xs_feasible
```

### `cost.py`

Implements the composition-dependent material-cost model and the
cost-aware acquisition-function wrapper.

The underlying acquisition value is divided by the estimated elemental
cost of the candidate alloy composition.

### `gp_utils.py`

Contains utilities for:

-   fitting `SingleTaskGP` Gaussian-process models;
-   constructing multi-output models;
-   fitting GP marginal likelihoods;
-   identifying non-dominated solutions;
-   constructing multi-objective EHVI acquisition functions.

The Bayesian optimization framework is based primarily on **BoTorch,
GPyTorch, and PyTorch**.

### `local_search.py`

Performs stochastic discrete local optimization of the acquisition
function using BoTorch's discrete local-search utilities.

### `objective.py`

Defines the physics/statistics-based objective function evaluated for
each alloy composition.

This module uses several external scientific packages and calls
Thermo-Calc through TC-Python.

Among other calculations, the current implementation performs
Thermo-Calc calculations for the Mo--Nb--Ta--W--Co--Hf system using the:

``` text
TCHEA7
```

thermodynamic database.

It also uses Thermo-Calc Scheil solidification and
additive-manufacturing functionality.

### `optimization.py`

Implements the primary sequential Bayesian-optimization loop.

This module:

1.  initializes the training dataset;
2.  evaluates initial alloy compositions;
3.  builds Gaussian-process models;
4.  constructs the multi-objective acquisition function;
5.  applies the cost-aware acquisition wrapper;
6.  searches the discrete feasible composition space;
7.  evaluates the selected alloy;
8.  updates the surrogate model;
9.  calculates the Pareto front;
10. tracks dominated hypervolume.

### `bo_script_full.py`

A monolithic version of the Bayesian-optimization workflow.

This file is useful as a standalone reference implementation, while the
other modules provide a more modular organization of the same overall
workflow.

### `predicted_correlation.py`

Post-processing script for analyzing relationships among predicted alloy
properties.

It reads:

``` text
predictions.csv
```

and uses statistical/machine-learning tools, including random forests,
to analyze correlations and relationships among predicted properties.

------------------------------------------------------------------------

## Software Requirements

### Python

A **Python 3** installation is required.

Because the workflow depends on TC-Python, the Python version must be
compatible with the installed Thermo-Calc/TC-Python release.

Users should consult the compatibility requirements for their specific
Thermo-Calc installation when selecting a Python version.

------------------------------------------------------------------------

## Python Dependencies

The source code directly imports the following third-party Python
packages:

  -----------------------------------------------------------------------
  Package                             Purpose
  ----------------------------------- -----------------------------------
  `torch`                             Tensor operations and numerical
                                      backend

  `botorch`                           Bayesian optimization and
                                      acquisition functions

  `gpytorch`                          Gaussian-process modeling

  `numpy`                             Numerical array operations

  `pandas`                            Tabular data handling

  `scipy`                             Statistical distributions and
                                      numerical utilities

  `scikit-learn`                      Kernel-density estimation, random
                                      forests, and ML utilities

  `matplotlib`                        Plotting and visualization

  `tqdm`                              Optimization progress bars

  `pymatgen`                          Materials composition
                                      representation and utilities

  `materialsframework`                Materials-analysis and
                                      transformation utilities

  `tc_python`                         Thermo-Calc Python API
  -----------------------------------------------------------------------

Standard-library modules such as `random` and `warnings` are also used
and do not require separate installation.

> **Note:** Exact package versions are not specified here because a
> verified working environment with pinned versions is not currently
> included in the repository. Version numbers should not be inferred or
> guessed.

------------------------------------------------------------------------

## Thermo-Calc / TC-Python Requirement

Running the complete optimization requires a working **Thermo-Calc
installation and license**.

The code imports the Thermo-Calc Python API using:

``` python
from tc_python import *
```

and uses Thermo-Calc calculations inside the optimization objectives.

The current implementation explicitly uses the:

``` text
TCHEA7
```

thermodynamic database for the Mo--Nb--Ta--W--Co--Hf alloy system.

Therefore, running the complete objective calculations requires:

1.  a compatible Thermo-Calc installation;
2.  access to TC-Python;
3.  an appropriate Thermo-Calc license;
4.  access to the required Thermo-Calc database and functionality used
    by the calculations; and
5.  `tc_python` available in the Python environment used to run the
    repository.

TC-Python should be installed and configured according to the
documentation corresponding to the user's specific Thermo-Calc release.
