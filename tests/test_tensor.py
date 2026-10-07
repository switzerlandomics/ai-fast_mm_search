from __future__ import annotations

import math

import numpy as np

from fast_mm.tensor import (
    compose,
    matrix_multiplication_tensor,
    multiplication_exponent,
    residual_statistics,
)


def test_2x2_target_tensor_has_expected_structure() -> None:
    tensor = matrix_multiplication_tensor(2)
    assert tensor.shape == (4, 4, 4)
    assert int(np.count_nonzero(tensor)) == 8
    assert float(tensor.sum()) == 8.0


def test_exact_strassen_composes_to_target(strassen_decomposition) -> None:
    target = matrix_multiplication_tensor(2)
    actual = compose(strassen_decomposition)
    assert np.array_equal(actual, target)
    statistics = residual_statistics(strassen_decomposition, target)
    assert statistics["relative_residual"] == 0.0
    assert statistics["max_abs_residual"] == 0.0


def test_strassen_exponent() -> None:
    assert math.isclose(
        multiplication_exponent(2, 7),
        math.log(7) / math.log(2),
    )

