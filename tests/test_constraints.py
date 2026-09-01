"""Tests for the composition-simplex inequality constraints."""

from rhea_mobo.constraints import inequality_constraints


def test_inequality_constraints_reference_all_six_components() -> None:
    """Every constraint should index all six composition components."""
    for idx, _coef, _rhs in inequality_constraints:
        assert idx.tolist() == [0, 1, 2, 3, 4, 5]
