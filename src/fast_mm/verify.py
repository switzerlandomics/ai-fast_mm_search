from __future__ import annotations

from fractions import Fraction
from typing import Any

from fast_mm.decomposition import exact_factor_rows
from fast_mm.tensor import multiplication_exponent


def _target_value(
    left_index: int,
    right_index: int,
    output_index: int,
    matrix_size: int,
) -> int:
    left_row, left_shared = divmod(left_index, matrix_size)
    right_shared, right_column = divmod(right_index, matrix_size)
    output_row, output_column = divmod(output_index, matrix_size)
    return int(
        left_shared == right_shared
        and left_row == output_row
        and right_column == output_column
    )


def verify_exact(payload: dict[str, Any]) -> dict[str, Any]:
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported candidate schema_version.")
    if payload.get("vectorisation") != "row-major":
        raise ValueError("only row-major vectorisation is supported.")
    if payload.get("coefficient_type") != "rational":
        raise ValueError(
            "this verifier currently requires coefficient_type='rational'."
        )

    matrix_size = int(payload["matrix_size"])
    rank = int(payload["rank"])
    dimension = matrix_size * matrix_size
    factors = payload["factors"]
    left = exact_factor_rows(factors["left"])
    right = exact_factor_rows(factors["right"])
    output = exact_factor_rows(factors["output"])

    expected_shape = (rank, dimension)
    for name, rows in (
        ("left", left),
        ("right", right),
        ("output", output),
    ):
        shape = (len(rows), len(rows[0]) if rows else 0)
        if shape != expected_shape or any(len(row) != dimension for row in rows):
            raise ValueError(f"{name} factor has shape {shape}, expected {expected_shape}.")

    mismatch_count = 0
    maximum_residual = Fraction(0)
    first_mismatch: dict[str, Any] | None = None

    for left_index in range(dimension):
        for right_index in range(dimension):
            for output_index in range(dimension):
                actual = sum(
                    left[term][left_index]
                    * right[term][right_index]
                    * output[term][output_index]
                    for term in range(rank)
                )
                expected = Fraction(
                    _target_value(
                        left_index,
                        right_index,
                        output_index,
                        matrix_size,
                    )
                )
                difference = actual - expected
                if difference:
                    mismatch_count += 1
                    maximum_residual = max(maximum_residual, abs(difference))
                    if first_mismatch is None:
                        first_mismatch = {
                            "left_index": left_index,
                            "right_index": right_index,
                            "output_index": output_index,
                            "actual": str(actual),
                            "expected": str(expected),
                            "residual": str(difference),
                        }

    naive_rank = matrix_size**3
    return {
        "schema_version": 1,
        "method": "exact-rational",
        "passed": mismatch_count == 0,
        "matrix_size": matrix_size,
        "rank": rank,
        "naive_rank": naive_rank,
        "scalar_multiplications_saved": naive_rank - rank,
        "multiplication_exponent": multiplication_exponent(matrix_size, rank),
        "mismatch_count": mismatch_count,
        "max_abs_residual": str(maximum_residual),
        "first_mismatch": first_mismatch,
    }

