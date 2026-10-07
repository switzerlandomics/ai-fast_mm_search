from __future__ import annotations

import math

import numpy as np

from fast_mm.decomposition import Decomposition


def flat_index(row: int, column: int, matrix_size: int) -> int:
    return row * matrix_size + column


def matrix_multiplication_tensor(matrix_size: int, dtype: np.dtype | type = np.float64) -> np.ndarray:
    if matrix_size < 1:
        raise ValueError("matrix_size must be positive.")
    dimension = matrix_size * matrix_size
    tensor = np.zeros((dimension, dimension, dimension), dtype=dtype)
    for row in range(matrix_size):
        for shared in range(matrix_size):
            for column in range(matrix_size):
                left = flat_index(row, shared, matrix_size)
                right = flat_index(shared, column, matrix_size)
                output = flat_index(row, column, matrix_size)
                tensor[left, right, output] = 1
    return tensor


def compose(decomposition: Decomposition) -> np.ndarray:
    decomposition.validate()
    return np.einsum(
        "ra,rb,rc->abc",
        decomposition.left,
        decomposition.right,
        decomposition.output,
        optimize=True,
    )


def residual(decomposition: Decomposition, target: np.ndarray | None = None) -> np.ndarray:
    expected = target
    if expected is None:
        expected = matrix_multiplication_tensor(decomposition.matrix_size)
    return compose(decomposition) - expected


def residual_statistics(
    decomposition: Decomposition,
    target: np.ndarray | None = None,
) -> dict[str, float]:
    expected = target
    if expected is None:
        expected = matrix_multiplication_tensor(decomposition.matrix_size)
    difference = residual(decomposition, expected)
    residual_norm = float(np.linalg.norm(difference))
    target_norm = float(np.linalg.norm(expected))
    return {
        "residual_norm": residual_norm,
        "relative_residual": residual_norm / target_norm if target_norm else math.inf,
        "max_abs_residual": float(np.max(np.abs(difference))),
    }


def multiplication_exponent(matrix_size: int, rank: int) -> float:
    if matrix_size <= 1:
        raise ValueError("matrix_size must be greater than 1.")
    if rank <= 0:
        raise ValueError("rank must be positive.")
    return math.log(rank) / math.log(matrix_size)


