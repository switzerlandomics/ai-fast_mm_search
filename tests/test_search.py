from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from fast_mm.config import (
    MetricsConfig,
    OutputConfig,
    ProblemConfig,
    ReconstructionConfig,
    RunConfig,
    SearchConfig,
)
from fast_mm.decomposition import pack_parameters
from fast_mm.metrics import MetricsRecorder
from fast_mm.search import RestartDecision, objective_and_gradient, run_search
from fast_mm.tensor import matrix_multiplication_tensor


def _config(restarts: int = 3) -> RunConfig:
    return RunConfig(
        problem=ProblemConfig(matrix_size=2, rank=7),
        search=SearchConfig(
            restarts=restarts,
            max_iterations=5,
            candidate_tolerance=1.0,
            seed=1,
        ),
        reconstruction=ReconstructionConfig(enabled=False),
        metrics=MetricsConfig(enabled=False, live=False, plots=False),
        output=OutputConfig(root="runs"),
    )


def test_analytic_gradient_matches_finite_difference() -> None:
    rng = np.random.default_rng(3)
    matrix_size = 2
    rank = 3
    target = matrix_multiplication_tensor(matrix_size)
    parameters = rng.normal(size=3 * rank * matrix_size * matrix_size)

    _, gradient = objective_and_gradient(
        parameters,
        matrix_size,
        rank,
        target,
    )

    epsilon = 1e-6
    indices = [0, 5, 11, parameters.size - 1]
    for index in indices:
        forward = parameters.copy()
        backward = parameters.copy()
        forward[index] += epsilon
        backward[index] -= epsilon
        value_forward, _ = objective_and_gradient(
            forward,
            matrix_size,
            rank,
            target,
        )
        value_backward, _ = objective_and_gradient(
            backward,
            matrix_size,
            rank,
            target,
        )
        numerical = (value_forward - value_backward) / (2 * epsilon)
        assert np.isclose(gradient[index], numerical, rtol=1e-5, atol=1e-6)


def test_numerical_threshold_does_not_stop_search(monkeypatch, strassen_decomposition) -> None:
    parameters = pack_parameters(strassen_decomposition)

    def fake_minimize(*args, **kwargs):
        return SimpleNamespace(
            x=parameters.copy(),
            nit=1,
            nfev=1,
            success=True,
            message="mock",
            fun=0.0,
        )

    monkeypatch.setattr("fast_mm.search.minimize", fake_minimize)
    outcome = run_search(
        _config(restarts=3),
        MetricsRecorder(None, False, False, 10),
    )

    assert len(outcome.restart_summaries) == 3
    assert outcome.termination_reason == "configured_restarts_completed"
    assert all(
        summary["candidate_threshold_met"]
        for summary in outcome.restart_summaries
    )


def test_verified_callback_can_stop_later_restart(
    monkeypatch,
    strassen_decomposition,
) -> None:
    exact = pack_parameters(strassen_decomposition)
    worse = exact.copy()
    worse[0] += 0.05
    sequence = [exact, worse, worse]
    call_index = 0

    def fake_minimize(*args, **kwargs):
        nonlocal call_index
        x = sequence[call_index]
        call_index += 1
        return SimpleNamespace(
            x=x.copy(),
            nit=1,
            nfev=1,
            success=True,
            message="mock",
            fun=0.0,
        )

    monkeypatch.setattr("fast_mm.search.minimize", fake_minimize)

    def callback(restart, candidate, summary):
        if restart == 1:
            return RestartDecision(
                stop=True,
                reason="verified_identity",
                metadata={"reconstruction_status": "VERIFIED"},
            )
        return RestartDecision(
            metadata={"reconstruction_status": "NOT_VERIFIED"}
        )

    outcome = run_search(
        _config(restarts=3),
        MetricsRecorder(None, False, False, 10),
        on_restart_complete=callback,
    )

    assert len(outcome.restart_summaries) == 2
    assert outcome.termination_reason == "verified_identity"
    assert outcome.best_restart == 0
    assert outcome.restart_summaries[1]["reconstruction_status"] == "VERIFIED"

