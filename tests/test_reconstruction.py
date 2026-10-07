from __future__ import annotations

import numpy as np

from fast_mm.config import ReconstructionConfig
from fast_mm.decomposition import rational_reconstruction_candidates
from fast_mm.reconstruction import reconstruct_and_verify


def test_rational_candidates_try_multiple_simple_representatives(strassen_decomposition) -> None:
    candidate = strassen_decomposition.copy()
    candidate.left[0, 0] = 0.83
    candidate.right[0, 0] = 0.71

    candidates = rational_reconstruction_candidates(
        candidate,
        max_denominator=16,
        normalise_scales=True,
        include_unscaled=True,
    )
    denominators = {item.max_denominator for item in candidates}
    assert 1 in denominators
    assert max(denominators) <= 16
    assert len(denominators) > 1
    assert len(candidates) > 2


def test_perturbed_strassen_can_be_recovered_exactly(strassen_decomposition) -> None:
    rng = np.random.default_rng(123)
    candidate = strassen_decomposition.copy()
    candidate.left += rng.normal(0.0, 1e-5, candidate.left.shape)
    candidate.right += rng.normal(0.0, 1e-5, candidate.right.shape)
    candidate.output += rng.normal(0.0, 1e-5, candidate.output.shape)

    outcome = reconstruct_and_verify(
        candidate,
        ReconstructionConfig(
            enabled=True,
            method="rational",
            max_denominator=16,
            tolerance=1e-3,
            normalise_term_scales=True,
            include_unscaled=True,
        ),
    )
    assert outcome.status == "VERIFIED"
    assert outcome.verification_report is not None
    assert outcome.verification_report["passed"] is True
    assert outcome.attempts >= 1


def test_disabled_reconstruction_is_explicit(strassen_decomposition) -> None:
    outcome = reconstruct_and_verify(
        strassen_decomposition,
        ReconstructionConfig(enabled=False),
    )
    assert outcome.status == "NOT_ATTEMPTED"
    assert outcome.attempts == 0

