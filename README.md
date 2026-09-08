<div align="center">

# RHEA-MOBO

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
![Python](https://img.shields.io/badge/python-3.12-blue)

[![Tests](https://github.com/ArroyaveLab/RHEA-MOBO/actions/workflows/tests.yml/badge.svg)](https://github.com/ArroyaveLab/RHEA-MOBO/actions/workflows/tests.yml)
[![Lint](https://github.com/ArroyaveLab/RHEA-MOBO/actions/workflows/lint.yml/badge.svg)](https://github.com/ArroyaveLab/RHEA-MOBO/actions/workflows/lint.yml)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22258223.svg)](https://doi.org/10.5281/zenodo.22258223)

`rhea-mobo` is a supply-risk-aware multi-objective Bayesian optimization (MOBO) framework for composition design of refractory high-entropy alloys (RHEAs) in the six-component Mo-Nb-Ta-W-Co-Hf system. It combines Gaussian-process surrogate models, Expected Hypervolume Improvement (EHVI), discrete composition constraints, supply-risk-aware acquisition, and physics-based objectives computed with Thermo-Calc/TC-Python and [MaterialsFramework](https://github.com/dogusariturk/MaterialsFramework).

</div>

---

## Key features

- Constrained six-component alloy composition search over a discrete feasible grid
- Gaussian-process surrogate modeling and multi-objective EHVI acquisition via `gp_utils` (BoTorch/GPyTorch)
- Supply-risk-aware acquisition
- Discrete stochastic local search over the feasible grid
- Five physics-based objectives (printability, yield strength, crack susceptibility, lattice misfit, Young's modulus) computed via Thermo-Calc/TC-Python (`TCHEA7`) and `materialsframework` SQS/elastic-constant prediction
- Pareto-front identification and hypervolume tracking across BO iterations
- Post-processing/correlation analysis of predicted properties (`scripts/predicted_correlation.py`)

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

## Scripts

- `scripts/bo_script_full.py`: a standalone, monolithic reference implementation of the full BO loop. It isn't part of the installable `rhea-mobo` package: install `tc_python` and `materialsframework[matgl,sqsgen]` manually, then run it with `uv run python scripts/bo_script_full.py`.
- `scripts/predicted_correlation.py`: plots feature-importance and objective-correlation figures from `data/predictions.csv`. Run it with `uv run scripts/predicted_correlation.py`, no project install needed.

---

## Installation

Install straight from GitHub with [uv](https://docs.astral.sh/uv/):

```sh
uv add git+https://github.com/ArroyaveLab/RHEA-MOBO.git
```

or with `pip`:

```sh
pip install git+https://github.com/ArroyaveLab/RHEA-MOBO.git
```

`tc_python` is not managed by PyPI. It ships with a licensed Thermo-Calc installation and cannot be redistributed, so install it into your environment manually after the above.

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE).

---

## Citation

We are currently preparing a manuscript for publication. If you use `rhea-mobo` in your research, please cite the following:

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
