from __future__ import annotations

import json
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable

import numpy as np


@dataclass
class Decomposition:
    matrix_size: int
    left: np.ndarray
    right: np.ndarray
    output: np.ndarray

    @property
    def rank(self) -> int:
        return int(self.left.shape[0])

    @property
    def vector_dimension(self) -> int:
        return self.matrix_size * self.matrix_size

    def validate(self) -> None:
        expected = (self.rank, self.vector_dimension)
        for name, factor in (
            ("left", self.left),
            ("right", self.right),
            ("output", self.output),
        ):
            if factor.shape != expected:
                raise ValueError(
                    f"{name} factor has shape {factor.shape}, expected {expected}."
                )
            if not np.isfinite(factor).all():
                raise ValueError(f"{name} factor contains non-finite coefficients.")

    def copy(self) -> "Decomposition":
        return Decomposition(
            matrix_size=self.matrix_size,
            left=self.left.copy(),
            right=self.right.copy(),
            output=self.output.copy(),
        )


@dataclass(frozen=True)
class RationalCandidate:
    payload: dict[str, Any]
    max_coefficient_error: float
    max_denominator: int
    scale_mode: str


def unpack_parameters(
    parameters: np.ndarray,
    matrix_size: int,
    rank: int,
) -> Decomposition:
    dimension = matrix_size * matrix_size
    factor_size = rank * dimension
    expected = 3 * factor_size
    if parameters.size != expected:
        raise ValueError(
            f"parameter vector has length {parameters.size}, expected {expected}."
        )

    left = parameters[:factor_size].reshape(rank, dimension)
    right = parameters[factor_size : 2 * factor_size].reshape(rank, dimension)
    output = parameters[2 * factor_size :].reshape(rank, dimension)
    return Decomposition(matrix_size, left, right, output)


def pack_parameters(decomposition: Decomposition) -> np.ndarray:
    decomposition.validate()
    return np.concatenate(
        [
            decomposition.left.ravel(),
            decomposition.right.ravel(),
            decomposition.output.ravel(),
        ]
    )


def normalise_term_scales(decomposition: Decomposition) -> Decomposition:
    """Remove the two scalar gauge freedoms of each rank-one term."""
    result = decomposition.copy()
    for term in range(result.rank):
        left_scale = float(np.max(np.abs(result.left[term])))
        right_scale = float(np.max(np.abs(result.right[term])))
        if left_scale > 0:
            result.left[term] /= left_scale
            result.output[term] *= left_scale
        if right_scale > 0:
            result.right[term] /= right_scale
            result.output[term] *= right_scale
    return result


def _denominator_limits(max_denominator: int) -> list[int]:
    limits = [1]
    value = 2
    while value < max_denominator:
        limits.append(value)
        value *= 2
    if max_denominator not in limits:
        limits.append(max_denominator)
    return sorted(set(limits))


def _fraction_string(value: float, max_denominator: int) -> tuple[str, float]:
    fraction = Fraction(float(value)).limit_denominator(max_denominator)
    error = abs(float(fraction) - float(value))
    if fraction.denominator == 1:
        return str(fraction.numerator), error
    return f"{fraction.numerator}/{fraction.denominator}", error


def _rational_candidate(
    decomposition: Decomposition,
    max_denominator: int,
    scale_mode: str,
) -> RationalCandidate:
    maximum_error = 0.0
    factors: dict[str, list[list[str]]] = {}

    for name, array in (
        ("left", decomposition.left),
        ("right", decomposition.right),
        ("output", decomposition.output),
    ):
        rows: list[list[str]] = []
        for row in array:
            values: list[str] = []
            for value in row:
                text, error = _fraction_string(float(value), max_denominator)
                maximum_error = max(maximum_error, error)
                values.append(text)
            rows.append(values)
        factors[name] = rows

    payload = {
        "schema_version": 1,
        "coefficient_type": "rational",
        "matrix_size": decomposition.matrix_size,
        "rank": decomposition.rank,
        "vectorisation": "row-major",
        "factors": factors,
        "reconstruction": {
            "method": "bounded-rational",
            "max_denominator": max_denominator,
            "scale_mode": scale_mode,
            "max_coefficient_error": maximum_error,
        },
    }
    return RationalCandidate(
        payload=payload,
        max_coefficient_error=maximum_error,
        max_denominator=max_denominator,
        scale_mode=scale_mode,
    )


def rational_reconstruction_candidates(
    decomposition: Decomposition,
    max_denominator: int,
    normalise_scales: bool = True,
    include_unscaled: bool = True,
) -> list[RationalCandidate]:
    """Generate simple rational representatives to be checked exactly.

    This is intentionally a proposal step. Exact verification, not coefficient
    proximity, decides whether a proposed rational identity is valid.
    """
    decomposition.validate()

    sources: list[tuple[str, Decomposition]] = []
    if normalise_scales:
        sources.append(("term-normalised", normalise_term_scales(decomposition)))
    if include_unscaled or not sources:
        sources.append(("raw", decomposition.copy()))

    candidates: list[RationalCandidate] = []
    seen: set[str] = set()

    for scale_mode, source in sources:
        for denominator in _denominator_limits(max_denominator):
            candidate = _rational_candidate(source, denominator, scale_mode)
            key = json.dumps(candidate.payload["factors"], sort_keys=True)
            if key in seen:
                continue
            seen.add(key)
            candidates.append(candidate)

    return candidates


def numerical_candidate_dict(
    decomposition: Decomposition,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    decomposition.validate()
    payload: dict[str, Any] = {
        "schema_version": 1,
        "coefficient_type": "float64",
        "matrix_size": decomposition.matrix_size,
        "rank": decomposition.rank,
        "vectorisation": "row-major",
        "factors": {
            "left": decomposition.left.tolist(),
            "right": decomposition.right.tolist(),
            "output": decomposition.output.tolist(),
        },
    }
    if metadata:
        payload["metadata"] = metadata
    return payload


def numerical_candidate_from_dict(payload: dict[str, Any]) -> Decomposition:
    matrix_size = int(payload["matrix_size"])
    factors = payload["factors"]
    decomposition = Decomposition(
        matrix_size=matrix_size,
        left=np.asarray(factors["left"], dtype=np.float64),
        right=np.asarray(factors["right"], dtype=np.float64),
        output=np.asarray(factors["output"], dtype=np.float64),
    )
    if int(payload["rank"]) != decomposition.rank:
        raise ValueError("candidate rank does not match factor rows.")
    decomposition.validate()
    return decomposition


def load_candidate(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported candidate schema_version.")
    if payload.get("vectorisation") != "row-major":
        raise ValueError("only row-major vectorisation is supported.")
    return payload


def require_exact_scalar(value: Any) -> Fraction:
    if isinstance(value, bool):
        raise TypeError("boolean coefficients are not valid.")
    if isinstance(value, int):
        return Fraction(value)
    if isinstance(value, str):
        return Fraction(value)
    raise TypeError(
        "exact rational verification accepts only integers or rational strings."
    )


def exact_factor_rows(values: Iterable[Iterable[Any]]) -> list[list[Fraction]]:
    return [[require_exact_scalar(value) for value in row] for row in values]

