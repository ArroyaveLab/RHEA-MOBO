<div align="center">

# RHEA-MOBO

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
![Python](https://img.shields.io/badge/python-3.12-blue)

[![Tests](https://github.com/ArroyaveLab/RHEA-MOBO/actions/workflows/tests.yml/badge.svg)](https://github.com/ArroyaveLab/RHEA-MOBO/actions/workflows/tests.yml)
[![Lint](https://github.com/ArroyaveLab/RHEA-MOBO/actions/workflows/lint.yml/badge.svg)](https://github.com/ArroyaveLab/RHEA-MOBO/actions/workflows/lint.yml)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22258223.svg)](https://doi.org/10.5281/zenodo.22258223)

`rhea-mobo` is a cost-aware multi-objective Bayesian optimization (MOBO) framework for composition design of refractory high-entropy alloys (RHEAs) in the six-component Mo-Nb-Ta-W-Co-Hf system. It combines Gaussian-process surrogate models, Expected Hypervolume Improvement (EHVI), discrete composition constraints, material-cost-aware acquisition, and physics-based objectives computed with Thermo-Calc/TC-Python and [MaterialsFramework](https://github.com/dogusariturk/MaterialsFramework).

</div>

---

## Key features

- Constrained six-component alloy composition search over a discrete feasible grid
- Gaussian-process surrogate modeling and multi-objective EHVI acquisition via `gp_utils` (BoTorch/GPyTorch)
- Material-cost-aware acquisition
- Discrete stochastic local search over the feasible grid
- Five physics-based objectives (printability, yield strength, crack susceptibility, lattice misfit, Young's modulus) computed via Thermo-Calc/TC-Python (`TCHEA7`) and `materialsframework` SQS/elastic-constant prediction
- Pareto-front identification and hypervolume tracking across BO iterations
- Post-processing/correlation analysis of predicted properties (`scripts/predicted_correlation.py`)

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
