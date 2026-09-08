"""Exercise the real composition expressions without launching licensed solvers."""

import ast
from pathlib import Path

import pytest
import torch
from pymatgen.core import Composition


@pytest.mark.parametrize(
    ("source", "expected_calls"),
    [
        ("src/rhea_mobo/objective.py", 1),
        ("src/rhea_mobo/optimization.py", 1),
        ("scripts/bo_script_full.py", 2),
    ],
)
def test_compositions_preserve_element_fractions(source: str, expected_calls: int) -> None:
    """Check every production Composition call with distinct fractions for all elements.

    Extract only the calls so this regression requires neither TC-Python nor
    MaterialsFramework initialization, SQS generation, or elastic calculations.
    """
    path = Path(__file__).resolve().parents[1] / source
    tree = ast.parse(path.read_text())
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "Composition"
    ]
    assert len(calls) == expected_calls
    x = torch.tensor([[0.10, 0.15, 0.20, 0.25, 0.12, 0.18]], dtype=torch.double)
    expected = {"Mo": 0.10, "Nb": 0.15, "Ta": 0.20, "W": 0.25, "Co": 0.12, "Hf": 0.18}
    for call in calls:
        expression = compile(ast.Expression(body=call), str(path), "eval")
        alloy = eval(expression, {"Composition": Composition, "x": x, "x_new": x.numpy()})
        assert isinstance(alloy, Composition)
        assert alloy.as_dict() == pytest.approx(expected)
        assert alloy.num_atoms == pytest.approx(1.0)
