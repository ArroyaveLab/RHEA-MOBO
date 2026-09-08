"""Guard the manuscript's CSC settings without invoking a licensed solver."""

import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch


@pytest.mark.parametrize("source", ["src/rhea_mobo/objective.py", "scripts/bo_script_full.py"])
def test_crack_configuration_matches_manuscript(source: str) -> None:
    """Exercise the production builder chain and inspect its scientific settings."""
    path = Path(__file__).resolve().parents[1] / source
    tree = ast.parse(path.read_text())
    assignments = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "crack_calc" for target in node.targets)
    ]
    assert len(assignments) == 1
    builder = MagicMock()
    methods = (
        "select_database_and_elements", "without_default_phases", "select_phase",
        "get_system", "with_property_model_calculation", "set_temperature",
        "set_composition_unit", "set_argument", "set_composition",
    )
    for method in methods:
        getattr(builder, method).return_value = builder
    elements = ["Mo", "Nb", "Ta", "W", "Co", "Hf"]
    namespace = {
        "session": builder, "active_el": elements,
        "CompositionUnit": SimpleNamespace(MOLE_FRACTION="mole_fraction"),
        "x": torch.tensor([[0.2, 0.2, 0.2, 0.2, 0.1, 0.1]]),
    }
    exec(compile(ast.Module(body=assignments, type_ignores=[]), str(path), "exec"), namespace)
    builder.select_database_and_elements.assert_called_once_with("TCHEA7", elements)
    builder.without_default_phases.assert_called_once_with()
    phases = [call.args[0] for call in builder.select_phase.call_args_list]
    assert phases == ["LIQUID", "BCC_B2#1"]
    args = dict(call.args for call in builder.set_argument.call_args_list)
    assert args["Start temperature"] == 4000
    assert args["CSC Model"] == "Clyne and Davies"
    assert args["Scheil calculation type"] == "Classic"
    assert args["Liquid fraction for start of relaxation"] == 0.6
    assert args["Liquid fraction for start of vulnerability"] == 0.1
    assert args["Liquid fraction smallest for vulnerability"] == 0.01
