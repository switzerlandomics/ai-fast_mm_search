from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from fast_mm.config import ReconstructionConfig
from fast_mm.decomposition import Decomposition, rational_reconstruction_candidates
from fast_mm.verify import verify_exact


@dataclass(frozen=True)
class ReconstructionOutcome:
    status: str
    attempts: int
    best_coefficient_error: float | None
    within_tolerance: bool
    exact_payload: dict[str, Any] | None
    verification_report: dict[str, Any] | None
    selected_max_denominator: int | None
    selected_scale_mode: str | None


def reconstruct_and_verify(
    decomposition: Decomposition,
    config: ReconstructionConfig,
) -> ReconstructionOutcome:
    if not config.enabled:
        return ReconstructionOutcome(
            status="NOT_ATTEMPTED",
            attempts=0,
            best_coefficient_error=None,
            within_tolerance=False,
            exact_payload=None,
            verification_report=None,
            selected_max_denominator=None,
            selected_scale_mode=None,
        )

    if config.method != "rational":
        raise ValueError(f"unsupported reconstruction method: {config.method!r}")

    candidates = rational_reconstruction_candidates(
        decomposition,
        max_denominator=config.max_denominator,
        normalise_scales=config.normalise_term_scales,
        include_unscaled=config.include_unscaled,
    )

    best_error: float | None = None
    for attempt, candidate in enumerate(candidates, start=1):
        if best_error is None or candidate.max_coefficient_error < best_error:
            best_error = candidate.max_coefficient_error

        report = verify_exact(candidate.payload)
        if report["passed"]:
            return ReconstructionOutcome(
                status="VERIFIED",
                attempts=attempt,
                best_coefficient_error=best_error,
                within_tolerance=(
                    candidate.max_coefficient_error <= config.tolerance
                ),
                exact_payload=candidate.payload,
                verification_report=report,
                selected_max_denominator=candidate.max_denominator,
                selected_scale_mode=candidate.scale_mode,
            )

    return ReconstructionOutcome(
        status="NOT_VERIFIED",
        attempts=len(candidates),
        best_coefficient_error=best_error,
        within_tolerance=(
            best_error is not None and best_error <= config.tolerance
        ),
        exact_payload=None,
        verification_report=None,
        selected_max_denominator=None,
        selected_scale_mode=None,
    )

