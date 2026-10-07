from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
from scipy.optimize import minimize

from fast_mm.config import RunConfig
from fast_mm.decomposition import Decomposition, unpack_parameters
from fast_mm.metrics import MetricsRecorder
from fast_mm.tensor import matrix_multiplication_tensor, residual_statistics


@dataclass(frozen=True)
class RestartDecision:
    stop: bool = False
    reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchOutcome:
    best: Decomposition
    best_restart: int
    best_statistics: dict[str, float]
    restart_summaries: list[dict[str, Any]]
    termination_reason: str


def objective_and_gradient(
    parameters: np.ndarray,
    matrix_size: int,
    rank: int,
    target: np.ndarray,
) -> tuple[float, np.ndarray]:
    decomposition = unpack_parameters(parameters, matrix_size, rank)
    reconstructed = np.einsum(
        "ra,rb,rc->abc",
        decomposition.left,
        decomposition.right,
        decomposition.output,
        optimize=True,
    )
    difference = reconstructed - target
    objective = 0.5 * float(np.sum(difference * difference))

    left_gradient = np.einsum(
        "abc,rb,rc->ra",
        difference,
        decomposition.right,
        decomposition.output,
        optimize=True,
    )
    right_gradient = np.einsum(
        "abc,ra,rc->rb",
        difference,
        decomposition.left,
        decomposition.output,
        optimize=True,
    )
    output_gradient = np.einsum(
        "abc,ra,rb->rc",
        difference,
        decomposition.left,
        decomposition.right,
        optimize=True,
    )
    gradient = np.concatenate(
        [
            left_gradient.ravel(),
            right_gradient.ravel(),
            output_gradient.ravel(),
        ]
    )
    return objective, gradient


def _initial_parameters(
    matrix_size: int,
    rank: int,
    scale: float,
    rng: np.random.Generator,
) -> np.ndarray:
    dimension = matrix_size * matrix_size
    return rng.normal(0.0, scale, size=3 * rank * dimension)


def run_search(
    config: RunConfig,
    recorder: MetricsRecorder,
    on_restart_complete: Callable[
        [int, Decomposition, dict[str, Any]],
        RestartDecision | None,
    ]
    | None = None,
) -> SearchOutcome:
    matrix_size = config.problem.matrix_size
    rank = config.problem.rank
    target = matrix_multiplication_tensor(matrix_size)
    seed_sequence = np.random.SeedSequence(config.search.seed)
    restart_sequences = seed_sequence.spawn(config.search.restarts)

    best: Decomposition | None = None
    best_restart = -1
    best_statistics = {
        "residual_norm": float("inf"),
        "relative_residual": float("inf"),
        "max_abs_residual": float("inf"),
    }
    restart_summaries: list[dict[str, Any]] = []
    termination_reason = "configured_restarts_completed"

    for restart, child_sequence in enumerate(restart_sequences):
        rng = np.random.default_rng(child_sequence)
        parameters = _initial_parameters(
            matrix_size,
            rank,
            config.search.initial_scale,
            rng,
        )
        iteration = 0

        recorder.emit(
            "restart_start",
            restart=restart,
            restart_count=config.search.restarts,
            seed_state=child_sequence.generate_state(4).tolist(),
        )

        def callback(current_parameters: np.ndarray) -> None:
            nonlocal iteration
            iteration += 1
            if iteration % config.metrics.interval != 0:
                return
            current = unpack_parameters(current_parameters, matrix_size, rank)
            statistics = residual_statistics(current, target)
            current_best = min(
                best_statistics["relative_residual"],
                statistics["relative_residual"],
            )
            recorder.emit(
                "iteration",
                restart=restart,
                iteration=iteration,
                residual_norm=statistics["residual_norm"],
                relative_residual=statistics["relative_residual"],
                max_abs_residual=statistics["max_abs_residual"],
                best_relative_residual=current_best,
            )
            recorder.progress(
                restart,
                config.search.restarts,
                iteration,
                statistics["relative_residual"],
                current_best,
            )

        result = minimize(
            objective_and_gradient,
            parameters,
            args=(matrix_size, rank, target),
            method="L-BFGS-B",
            jac=True,
            callback=callback,
            options={
                "maxiter": config.search.max_iterations,
                "ftol": config.search.ftol,
                "gtol": config.search.gtol,
                "maxls": config.search.max_line_search_steps,
            },
        )

        candidate = unpack_parameters(result.x, matrix_size, rank)
        statistics = residual_statistics(candidate, target)
        threshold_met = (
            statistics["relative_residual"]
            <= config.search.candidate_tolerance
        )

        if statistics["relative_residual"] < best_statistics["relative_residual"]:
            best = candidate.copy()
            best_restart = restart
            best_statistics = statistics

        summary: dict[str, Any] = {
            "restart": restart,
            "iterations": int(result.nit),
            "function_evaluations": int(result.nfev),
            "optimizer_success": bool(result.success),
            "optimizer_message": str(result.message),
            "objective": float(result.fun),
            "residual_norm": statistics["residual_norm"],
            "relative_residual": statistics["relative_residual"],
            "max_abs_residual": statistics["max_abs_residual"],
            "candidate_threshold_met": threshold_met,
        }

        decision = RestartDecision()
        if on_restart_complete is not None:
            callback_result = on_restart_complete(
                restart,
                candidate,
                summary.copy(),
            )
            if callback_result is not None:
                decision = callback_result
                summary.update(decision.metadata)

        restart_summaries.append(summary)
        recorder.emit("restart_end", **summary)
        recorder.finish_live_line()

        if decision.stop:
            termination_reason = decision.reason or "callback_requested_stop"
            recorder.emit(
                "search_stop",
                restart=restart,
                reason=termination_reason,
            )
            break

    if best is None:
        raise RuntimeError("search completed without producing a candidate.")

    return SearchOutcome(
        best=best,
        best_restart=best_restart,
        best_statistics=best_statistics,
        restart_summaries=restart_summaries,
        termination_reason=termination_reason,
    )

