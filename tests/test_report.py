from __future__ import annotations

from pathlib import Path

from fast_mm.report import write_html_summary


def _result(status: str) -> dict:
    return {
        "schema_version": 2,
        "project_version": "test",
        "run_id": "test-run",
        "status": status,
        "claim": "test claim",
        "problem": {"matrix_size": 2, "rank": 7},
        "search": {
            "method": "lbfgsb",
            "configured_restarts": 8,
            "completed_restarts": 8,
            "best_restart": 0,
            "candidate_tolerance": 1e-6,
            "numerical_candidates": 1 if status in {"VERIFIED", "NUMERICAL_CANDIDATE"} else 0,
            "reconstruction_attempts": 5 if status in {"VERIFIED", "NUMERICAL_CANDIDATE"} else 0,
            "verified_restart": 0 if status == "VERIFIED" else None,
            "termination_reason": "verified_identity" if status == "VERIFIED" else "configured_restarts_completed",
            "restart_summaries": [],
        },
        "best": {
            "residual_norm": 1e-8 if status != "ERROR" else None,
            "relative_residual": 1e-8 if status != "ERROR" else None,
            "max_abs_residual": 1e-8 if status != "ERROR" else None,
        },
        "verification": {"status": "VERIFIED" if status == "VERIFIED" else "NOT_VERIFIED"},
        "comparison": {
            "naive_rank": 8,
            "candidate_rank": 7,
            "scalar_multiplications_saved": 1,
            "fraction_of_naive_multiplications": 7 / 8,
            "multiplication_exponent": 2.807354922057604,
        },
        "runtime_seconds": 0.1,
        "environment": {},
        "artifacts": {},
    }


def test_report_renders_all_statuses(tmp_path: Path) -> None:
    for status in ("VERIFIED", "NUMERICAL_CANDIDATE", "NOT_FOUND", "ERROR"):
        directory = tmp_path / status
        directory.mkdir()
        path = write_html_summary(_result(status), directory, [])
        text = path.read_text(encoding="utf-8")
        assert status in text
        assert "Why this search matters" in text
        assert "Reconstruction proposals tested" in text

