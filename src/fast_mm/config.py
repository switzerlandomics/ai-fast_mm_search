from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ProblemConfig:
    matrix_size: int
    rank: int


@dataclass(frozen=True)
class SearchConfig:
    method: str = "lbfgsb"
    restarts: int = 8
    max_iterations: int = 3000
    initial_scale: float = 0.5
    seed: int = 1
    candidate_tolerance: float = 1e-6
    ftol: float = 1e-15
    gtol: float = 1e-10
    max_line_search_steps: int = 50
    stop_on_verified: bool = True


@dataclass(frozen=True)
class ReconstructionConfig:
    enabled: bool = True
    method: str = "rational"
    max_denominator: int = 16
    tolerance: float = 1e-8
    normalise_term_scales: bool = True
    include_unscaled: bool = True


@dataclass(frozen=True)
class MetricsConfig:
    enabled: bool = True
    live: bool = True
    interval: int = 10
    plots: bool = True


@dataclass(frozen=True)
class OutputConfig:
    root: str = "runs"


@dataclass(frozen=True)
class RunConfig:
    problem: ProblemConfig
    search: SearchConfig
    reconstruction: ReconstructionConfig
    metrics: MetricsConfig
    output: OutputConfig

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _section(data: dict[str, Any], name: str) -> dict[str, Any]:
    value = data.get(name, {})
    if not isinstance(value, dict):
        raise ValueError(f"Configuration section '{name}' must be an object.")
    return value


def load_config(path: str | Path) -> RunConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    config = RunConfig(
        problem=ProblemConfig(**_section(data, "problem")),
        search=SearchConfig(**_section(data, "search")),
        reconstruction=ReconstructionConfig(**_section(data, "reconstruction")),
        metrics=MetricsConfig(**_section(data, "metrics")),
        output=OutputConfig(**_section(data, "output")),
    )
    validate_config(config)
    return config


def validate_config(config: RunConfig) -> None:
    if config.problem.matrix_size < 2:
        raise ValueError("matrix_size must be at least 2.")
    if config.problem.rank < 1:
        raise ValueError("rank must be positive.")
    if config.search.method != "lbfgsb":
        raise ValueError("search method must be 'lbfgsb' in this version.")
    if config.search.restarts < 1:
        raise ValueError("restarts must be positive.")
    if config.search.max_iterations < 1:
        raise ValueError("max_iterations must be positive.")
    if config.search.initial_scale <= 0:
        raise ValueError("initial_scale must be positive.")
    if config.search.candidate_tolerance <= 0:
        raise ValueError("candidate_tolerance must be positive.")
    if config.search.ftol <= 0 or config.search.gtol <= 0:
        raise ValueError("ftol and gtol must be positive.")
    if config.search.max_line_search_steps < 1:
        raise ValueError("max_line_search_steps must be positive.")
    if config.reconstruction.method != "rational":
        raise ValueError("reconstruction method must be 'rational' in this version.")
    if config.reconstruction.max_denominator < 1:
        raise ValueError("max_denominator must be positive.")
    if config.reconstruction.tolerance <= 0:
        raise ValueError("reconstruction tolerance must be positive.")
    if config.metrics.interval < 1:
        raise ValueError("metrics interval must be positive.")
    if not config.output.root.strip():
        raise ValueError("output root must not be empty.")

