from __future__ import annotations

import copy

import pytest

from fast_mm.verify import verify_exact


def test_exact_strassen_identity_passes(strassen_payload: dict) -> None:
    report = verify_exact(strassen_payload)
    assert report["passed"] is True
    assert report["mismatch_count"] == 0
    assert report["max_abs_residual"] == "0"
    assert report["rank"] == 7


def test_corrupted_identity_fails(strassen_payload: dict) -> None:
    corrupted = copy.deepcopy(strassen_payload)
    corrupted["factors"]["output"][0][0] = "2"
    report = verify_exact(corrupted)
    assert report["passed"] is False
    assert report["mismatch_count"] > 0
    assert report["first_mismatch"] is not None


def test_exact_verifier_rejects_float_coefficients(strassen_payload: dict) -> None:
    invalid = copy.deepcopy(strassen_payload)
    invalid["factors"]["left"][0][0] = 1.0
    with pytest.raises(TypeError):
        verify_exact(invalid)


def test_exact_verifier_rejects_unimplemented_domain(strassen_payload: dict) -> None:
    invalid = copy.deepcopy(strassen_payload)
    invalid["coefficient_type"] = "complex-rational"
    with pytest.raises(ValueError, match="currently requires"):
        verify_exact(invalid)

