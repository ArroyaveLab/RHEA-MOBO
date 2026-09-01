"""Tests for stochastic-restart local search over a discrete acquisition function."""

from typing import TYPE_CHECKING, cast

import pytest
import torch

from rhea_mobo import local_search as local_search_module
from rhea_mobo.local_search import stochastic_local_search

if TYPE_CHECKING:
    from botorch.acquisition.acquisition import AcquisitionFunction

# The acquisition function is never invoked directly in these tests: the botorch
# optimizer that would call it is replaced with a fake via _queue_restarts, so a
# type-only placeholder stands in for a real AcquisitionFunction.
_ACQF_STUB = cast("AcquisitionFunction", None)


def _queue_restarts(monkeypatch: pytest.MonkeyPatch, results: list) -> None:
    """Patch optimize_acqf_discrete_local_search to yield each of ``results`` in turn.

    Each entry is either a ``(candidate, value)`` pair or an exception instance to raise.
    """
    calls = iter(results)

    def fake_optimize(**_kwargs: object) -> tuple[torch.Tensor, torch.Tensor]:
        result = next(calls)
        if isinstance(result, Exception):
            raise result
        candidate, value = result
        return candidate, torch.tensor([value])

    monkeypatch.setattr(local_search_module, "optimize_acqf_discrete_local_search", fake_optimize)


def _candidate(value: float) -> torch.Tensor:
    """Build a distinguishable 1x1 candidate tensor tagged with ``value``."""
    return torch.tensor([[value]])


def test_keeps_candidate_that_clears_five_percent_margin(monkeypatch: pytest.MonkeyPatch) -> None:
    """A later candidate should only replace the best once it clears the 5% improvement margin."""
    _queue_restarts(
        monkeypatch,
        [
            (_candidate(0.0), 1.0),  # first restart always becomes the initial best
            (_candidate(1.0), 1.5),  # 1.5 > 1.05 * 1.0 -> replaces best
            (_candidate(2.0), 1.2),  # 1.2 < 1.05 * 1.5 -> does not replace best
        ],
    )
    monkeypatch.setattr(local_search_module.random, "random", lambda: 1.0)  # never explore

    best_x, best_val = stochastic_local_search(
        acq_function=_ACQF_STUB,
        discrete_choices=[],
        inequality_constraints=[],
        X_avoid=torch.empty(0, 1),
        batch_initial_conditions=torch.empty(0, 1, 1),
        num_restart=3,
        exploration_prob=0.0,
    )

    assert best_x is not None
    assert torch.equal(best_x, _candidate(1.0))
    assert best_val.item() == pytest.approx(1.5)


def test_exploration_can_replace_a_better_candidate(monkeypatch: pytest.MonkeyPatch) -> None:
    """When the exploration draw succeeds, the current restart's candidate wins even if worse."""
    _queue_restarts(
        monkeypatch,
        [
            (_candidate(0.0), 5.0),
            (_candidate(1.0), 0.1),  # much worse, but exploration forces a swap
        ],
    )
    monkeypatch.setattr(local_search_module.random, "random", lambda: 0.0)  # always explore

    best_x, best_val = stochastic_local_search(
        acq_function=_ACQF_STUB,
        discrete_choices=[],
        inequality_constraints=[],
        X_avoid=torch.empty(0, 1),
        batch_initial_conditions=torch.empty(0, 1, 1),
        num_restart=2,
        exploration_prob=1.0,
    )

    assert best_x is not None
    assert torch.equal(best_x, _candidate(1.0))
    assert best_val.item() == pytest.approx(0.1)


def test_failed_restarts_are_skipped(monkeypatch: pytest.MonkeyPatch) -> None:
    """IndexError and RuntimeError from a restart should be swallowed and the search continued."""
    _queue_restarts(
        monkeypatch,
        [
            IndexError("no feasible neighbor"),
            RuntimeError("optimizer failed"),
            (_candidate(3.0), 2.0),
        ],
    )
    monkeypatch.setattr(local_search_module.random, "random", lambda: 1.0)

    best_x, best_val = stochastic_local_search(
        acq_function=_ACQF_STUB,
        discrete_choices=[],
        inequality_constraints=[],
        X_avoid=torch.empty(0, 1),
        batch_initial_conditions=torch.empty(0, 1, 1),
        num_restart=3,
        exploration_prob=0.0,
    )

    assert best_x is not None
    assert torch.equal(best_x, _candidate(3.0))
    assert best_val.item() == pytest.approx(2.0)


def test_returns_none_when_every_restart_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    """If every restart raises, the function should return None and a -inf acquisition value."""
    _queue_restarts(monkeypatch, [RuntimeError("fail"), IndexError("fail")])

    best_x, best_val = stochastic_local_search(
        acq_function=_ACQF_STUB,
        discrete_choices=[],
        inequality_constraints=[],
        X_avoid=torch.empty(0, 1),
        batch_initial_conditions=torch.empty(0, 1, 1),
        num_restart=2,
    )

    assert best_x is None
    assert best_val.item() == float("-inf")


def test_other_exceptions_are_not_swallowed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Exception types other than IndexError/RuntimeError should propagate to the caller."""
    _queue_restarts(monkeypatch, [ValueError("unexpected")])

    with pytest.raises(ValueError, match="unexpected"):
        stochastic_local_search(
            acq_function=_ACQF_STUB,
            discrete_choices=[],
            inequality_constraints=[],
            X_avoid=torch.empty(0, 1),
            batch_initial_conditions=torch.empty(0, 1, 1),
            num_restart=1,
        )
