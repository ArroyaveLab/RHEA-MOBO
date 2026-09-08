# Changelog

## v0.1.2 — 2026-09-08

- Rename the repository to `ArroyaveLab/2026-islam-rhea-mobo` and link the
  manuscript entry in the ArroyaveLab publications index.
- Fix composition construction using explicit elemental fractions.
- Correct acquisition score shape for candidate batches while retaining
  sequential candidate selection (`q=1`).
- Describe the acquisition modifier consistently as dimensionless supply risk.
- Align the crack-susceptibility configuration with the manuscript.
- Document that the original 250 evaluations and their sequential history are
  unavailable and that licensed physical calculations have not been rerun.
- Add a license-free synthetic BO example with an exhaustive feasible oracle
  and a random-sampling baseline sharing the initial design and evaluation budget.
  The example demonstrates the pipeline; it is not statistical validation.
- Update package and citation metadata to v0.1.2. The citation DOI identifies
  the software's all-versions archive.

Validation: 15 focused tests and lint passed for the synthetic example and core
optimization helpers. The example also completed in the pinned Python 3.12
and Torch 2.13.0 environment. No new Thermo-Calc or experimental results are
claimed by this release.
