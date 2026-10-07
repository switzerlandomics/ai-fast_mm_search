from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from fast_mm.decomposition import Decomposition


@pytest.fixture
def strassen_path() -> Path:
    return Path(__file__).resolve().parents[1] / "results" / "verified" / "strassen_2x2.json"


@pytest.fixture
def strassen_payload(strassen_path: Path) -> dict:
    return json.loads(strassen_path.read_text(encoding="utf-8"))


@pytest.fixture
def strassen_decomposition(strassen_payload: dict) -> Decomposition:
    factors = strassen_payload["factors"]
    return Decomposition(
        matrix_size=2,
        left=np.asarray(factors["left"], dtype=float),
        right=np.asarray(factors["right"], dtype=float),
        output=np.asarray(factors["output"], dtype=float),
    )

