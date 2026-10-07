from __future__ import annotations

import json

import pytest

from fast_mm.config import load_config


def _base_config() -> dict:
    return {
        "problem": {"matrix_size": 2, "rank": 7},
        "search": {
            "method": "lbfgsb",
            "restarts": 2,
            "max_iterations": 10,
            "stop_on_verified": True,
        },
        "reconstruction": {
            "enabled": True,
            "method": "rational",
            "max_denominator": 16,
            "tolerance": 1e-8,
        },
        "metrics": {"enabled": False, "live": False, "plots": False},
        "output": {"root": "runs"},
    }


def _write(tmp_path, payload: dict):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_load_config_uses_clean_defaults(tmp_path) -> None:
    config = load_config(_write(tmp_path, _base_config()))
    assert config.problem.matrix_size == 2
    assert config.search.stop_on_verified is True
    assert config.reconstruction.method == "rational"
    assert config.reconstruction.normalise_term_scales is True
    assert config.reconstruction.include_unscaled is True


@pytest.mark.parametrize(
    ("section", "field", "value", "message"),
    [
        ("problem", "matrix_size", 1, "matrix_size"),
        ("problem", "rank", 0, "rank"),
        ("search", "restarts", 0, "restarts"),
        ("search", "candidate_tolerance", 0, "candidate_tolerance"),
        ("reconstruction", "max_denominator", 0, "max_denominator"),
        ("reconstruction", "tolerance", 0, "tolerance"),
    ],
)
def test_invalid_numeric_config_is_rejected(
    tmp_path,
    section: str,
    field: str,
    value,
    message: str,
) -> None:
    payload = _base_config()
    payload[section][field] = value
    with pytest.raises(ValueError, match=message):
        load_config(_write(tmp_path, payload))


def test_unknown_reconstruction_method_is_rejected(tmp_path) -> None:
    payload = _base_config()
    payload["reconstruction"]["method"] = "magic"
    with pytest.raises(ValueError, match="reconstruction method"):
        load_config(_write(tmp_path, payload))

