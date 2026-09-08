<div align="center">

# RHEA-MOBO

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
![Python](https://img.shields.io/badge/python-3.12-blue)

[![Tests](https://github.com/ArroyaveLab/2026-islam-rhea-mobo/actions/workflows/tests.yml/badge.svg)](https://github.com/ArroyaveLab/2026-islam-rhea-mobo/actions/workflows/tests.yml)
[![Lint](https://github.com/ArroyaveLab/2026-islam-rhea-mobo/actions/workflows/lint.yml/badge.svg)](https://github.com/ArroyaveLab/2026-islam-rhea-mobo/actions/workflows/lint.yml)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22258223.svg)](https://doi.org/10.5281/zenodo.22258223)

`rhea-mobo` is a supply-risk-aware multi-objective Bayesian optimization (MOBO) framework for composition design of refractory high-entropy alloys (RHEAs) in the six-component Mo-Nb-Ta-W-Co-Hf system. It combines Gaussian-process surrogate models, Expected Hypervolume Improvement (EHVI), discrete composition constraints, supply-risk-aware acquisition, and physics-based objectives computed with Thermo-Calc/TC-Python and [MaterialsFramework](https://github.com/dogusariturk/MaterialsFramework).

</div>

---

## Associated paper

**Bayesian Multi-Objective Co-Optimization of Performance and Manufacturability
in BCC–B2 Refractory High Entropy Alloys**, Md Shafiqul Islam et al. (2026).

[Paper, PDF, and BibTeX](https://github.com/ArroyaveLab/publications/tree/main/papers/2026-islam-rhea-mobo)

This repository was renamed from `ArroyaveLab/RHEA-MOBO` to
`ArroyaveLab/2026-islam-rhea-mobo` to follow the group's
`YYYY-leadauthor-shortkeyword` publication naming convention. The Python package
and command remain `rhea-mobo`; existing GitHub repository links redirect here.
The manuscript and the software have separate citation and reuse terms.

## Key features

- Constrained six-component alloy composition search over a discrete feasible grid
- Gaussian-process surrogate modeling and multi-objective EHVI acquisition via `gp_utils` (BoTorch/GPyTorch)
- Supply-risk-aware acquisition
- Discrete stochastic local search over the feasible grid
- Five physics-based objectives (printability, yield strength, crack susceptibility, lattice misfit, Young's modulus) computed via Thermo-Calc/TC-Python (`TCHEA7`) and `materialsframework` SQS/elastic-constant prediction
- Pareto-front identification and hypervolume tracking across BO iterations
- Post-processing/correlation analysis of predicted properties (`scripts/predicted_correlation.py`)

---

## Data availability and reproducibility

**The original per-composition objective evaluations and sequential evaluation
history from the reported Bayesian-optimization campaign are unavailable.**
The original run used 50 initial compositions and 200 accepted sequential
additions. This repository does not contain an archive of those 250 evaluations,
their selection order, or the corresponding iteration-by-iteration model states.

`data/predictions.csv` contains a property-prediction table for 1,716 candidate
compositions. It is not the original evaluation log and cannot be used to
reconstruct which compositions were evaluated at each iteration or the exact
hypervolume trajectory reported in the paper.

The code is provided to make the computational workflow available for inspection,
reuse, and new optimization campaigns. It is not an exact replay package for the
reported run. No fixed random seed was set in the original notebook calculation;
rerunning the workflow is therefore not expected to reproduce the same selection
sequence or identical final non-dominated set. Setting a seed for a new run cannot
recover the unavailable original history. Reproduction of broader trends must be
assessed through new calculations rather than assumed from the supplied code.

---

## Supply-risk weighting

The acquisition weighting represents **elemental supply risk**, not purchase price
or dollar cost. The fixed scores used by this implementation are dimensionless:

| Element | Supply-risk score |
| --- | ---: |
| Mo | 6.65 |
| Nb | 4.92 |
| Ta | 10.94 |
| W | 10.53 |
| Co | 3.99 |
| Hf | 5.95 |

For mole fractions `x_i`, the alloy index is `R_alloy = sum(x_i * R_i)`.
Candidate selection uses `EHVI / R_alloy`: larger supply-risk scores reduce the
acquisition value at equal expected hypervolume improvement. Supply risk is an
acquisition modifier, not a sixth optimization objective, a probability of
supply failure, or a forecast of availability or price.

The accompanying manuscript attributes the metric to the twelve-indicator
framework of Helbig et al., *Supply Risk Considerations for the Elements in
Nickel-Based Superalloys*, Resources 9 (2020), 106,
[doi:10.3390/resources9090106](https://doi.org/10.3390/resources9090106).
As described in the manuscript, available indicators are averaged within four
categories (supply reduction, demand increase, market concentration, and political
risk), and the available category scores are then averaged with equal weights.
Missing indicator values are omitted from the corresponding averages.

The repository stores the final six elemental scores; it does not currently
include the underlying indicator table, its data vintage, or the normalization
inputs needed to reconstruct those scores independently. The numerical scores
above reproduce the `Risk` column of `data/predictions.csv` to floating-point
precision for all 1,716 rows when weighted by the named elemental mole fractions.
This checks implementation consistency, not independent provenance of the
underlying indicator data.

The identifiers `ELEMENT_COST`, `cost_model`, `CostAwareEHVI`, and the module
`cost.py` are retained for compatibility with existing callers. They refer to
supply-risk weighting throughout this repository. The numerical values and
acquisition arithmetic have not been changed by this terminology clarification.

---

## Crack-susceptibility calculation

Both implementations explicitly configure the manuscript's crack-susceptibility
objective using TCHEA7, a 4000 K Scheil start temperature, Classic Scheil, and the
Clyne and Davies criterion. The relaxation interval uses liquid fractions
0.6 to 0.1; the vulnerability interval uses 0.1 to 0.01.
The system selects `LIQUID` and `BCC_B2#1` (the matrix phase used in the
manuscript), with default phases disabled. The `BCC_B2#2` precipitate set is
not selected for this calculation. The separate yield-strength and equilibrium
calculations retain their own phase selections.

These settings align the implementation with the manuscript; they have not
been used to regenerate the checked-in predictions. A licensed TC-Python/TCHEA7
run is still required to verify the resulting phase path and numerical values.
In particular, selecting a matrix composition set is not by itself a check of
its ordering state or of any additional composition sets generated by the solver.

---

## Scripts

- `scripts/bo_script_full.py`: a standalone, monolithic reference implementation of the full BO loop. It isn't part of the installable `rhea-mobo` package: install `tc_python` and `materialsframework[matgl,sqsgen]` manually, then run it with `uv run python scripts/bo_script_full.py`.
- `scripts/predicted_correlation.py`: plots feature-importance and objective-correlation figures from `data/predictions.csv`. Run it with `uv run scripts/predicted_correlation.py`, no project install needed.

---

## License-free synthetic example

A small analytical example demonstrates the production GP fitting, five-objective
EHVI, supply-risk wrapper, and stochastic local search without importing
TC-Python, MaterialsFramework, or any physical property evaluator.

From a checkout, create an isolated environment with only the toy dependencies:

```sh
uv venv --python 3.12 .venv-toy
uv pip install --python .venv-toy/bin/python -r requirements-toy.txt
PYTHONPATH=src .venv-toy/bin/python -m rhea_mobo.toy --output toy-output
```

The default run uses eight initial feasible samples, three accepted additions,
and seed 42. Options `--initial`, `--iterations`, `--seed`, and `--output` control
these choices. Select an empty output directory for each run. Seeded results
should be compared within the same software environment; cross-version or
cross-platform bitwise identity is not guaranteed.

The example enumerates all 1,716 positive 5 mol% compositions with equal Co and
Hf fractions. All five synthetic objectives are maximized:
`y_j = 1 - (t - c_j)^2 - p`, where `t = (x_Mo - 0.05)/0.70`,
`c = (0, 0.25, 0.5, 0.75, 1)`, and
`p = 2*((x_Ta - 0.05)^2 + (x_W - 0.05)^2 + (x_Co - 0.05)^2)`.
The synthetic feasibility rule is `x_Mo + x_W <= 0.65`.
These dimensionless functions and constraint are not predictions of alloy
performance, printability, or ductility.

An exhaustive evaluation supplies the exact feasible discrete Pareto front
(up to floating-point comparisons) and reference hypervolume, using reference
point `(-1, -1, -1, -1, -1)`. The oracle is used for scoring, not GP training.
The run checks real batched acquisition weights, grid membership, feasibility
of accepted proposals, duplicate exclusion, and nondecreasing hypervolume
bounded by the oracle.

The toy initializes from feasible points, uses a smaller local-search budget
(four initialization points and two wrapper calls), records rejected proposals,
and excludes them from subsequent searches, with at most 50 attempts per addition.
These safeguards belong to the toy driver; the example does not validate the
licensed production objective evaluator or every control path of the production
optimization loop.

A uniform random-sampling baseline starts from exactly the same initial design
and receives the same number of accepted objective evaluations. It samples
feasible grid points without replacement, excluding the initial design, with
an independent random generator seeded by `seed + 1`; adding this baseline does
not change the BO random stream. This comparison matches objective evaluations,
not feasibility-screening effort, runtime, or cumulative supply risk.
`comparison.csv` records both hypervolumes and fractions of the oracle value,
including the shared initial design. `random_evaluations.csv` records the baseline
compositions and objectives. This comparison demonstrates that the license-free
pipeline runs and records both selection strategies. It is not a statistical
validation of the study and makes no claim that BO outperforms random sampling.

Outputs include `evaluations.csv`, `proposals.json`, `progress.json`,
`oracle_pareto.csv`, and `summary.json`. Evaluation and proposal histories are
saved after each accepted addition. All outputs describe the synthetic run,
not the original manuscript campaign. No claim of convergence is made from the
short default exercise.

---

## Installation

Install straight from GitHub with [uv](https://docs.astral.sh/uv/):

```sh
uv add git+https://github.com/ArroyaveLab/2026-islam-rhea-mobo.git
```

or with `pip`:

```sh
pip install git+https://github.com/ArroyaveLab/2026-islam-rhea-mobo.git
```

`tc_python` is not managed by PyPI. It ships with a licensed Thermo-Calc installation and cannot be redistributed, so install it into your environment manually after the above.

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE).

---

## Citation

The accompanying manuscript is prepared for resubmission to *Materials & Design*
following transfer from *Acta Materialia*. It does not yet have a journal DOI.
See the [publication entry](https://github.com/ArroyaveLab/publications/tree/main/papers/2026-islam-rhea-mobo)
for the manuscript PDF, full author list, abstract, and paper BibTeX.

The DOI below identifies all versions of the software archive, not the article.
If you use `rhea-mobo`, please cite the software:

> Islam, M. S., & Sarıtürk, D. (2026). RHEA-MOBO. Zenodo. https://doi.org/10.5281/zenodo.22258223

BibTeX:

```bibtex
@software{islam_2026_22258223,
  author    = {Islam, Md. Shafiqul and Sarıtürk, Doğuhan},
  title     = {RHEA-MOBO},
  year      = 2026,
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.22258223},
  url       = {https://doi.org/10.5281/zenodo.22258223},
}
```
