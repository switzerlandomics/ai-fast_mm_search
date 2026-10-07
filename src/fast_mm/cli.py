from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from fast_mm import __version__
from fast_mm.artifacts import (
    atomic_write_json,
    environment_metadata,
    make_run_id,
    utc_now,
)
from fast_mm.campaign import campaign_status, run_campaign
from fast_mm.config import RunConfig, load_config
from fast_mm.decomposition import load_candidate, numerical_candidate_dict
from fast_mm.metrics import MetricsRecorder
from fast_mm.reconstruction import ReconstructionOutcome, reconstruct_and_verify
from fast_mm.report import plot_metrics, write_html_summary
from fast_mm.search import RestartDecision, run_search
from fast_mm.tensor import multiplication_exponent
from fast_mm.verify import verify_exact


def _repository_root() -> Path:
    return Path.cwd()


def _comparison(matrix_size: int, rank: int) -> dict[str, Any]:
    naive_rank = matrix_size**3
    return {
        "naive_rank": naive_rank,
        "candidate_rank": rank,
        "scalar_multiplications_saved": naive_rank - rank,
        "fraction_of_naive_multiplications": rank / naive_rank,
        "multiplication_exponent": multiplication_exponent(matrix_size, rank),
        "interpretation": (
            "The exponent is an arithmetic-complexity comparison for recursive "
            "use of this fixed-size identity. It is not a wall-clock speedup claim."
        ),
    }


def _status_and_claim(
    relative_residual: float,
    tolerance: float,
    verification_status: str,
) -> tuple[str, str]:
    if verification_status == "VERIFIED":
        return (
            "VERIFIED",
            "An exact rational matrix-multiplication identity was independently verified.",
        )
    if relative_residual <= tolerance:
        return (
            "NUMERICAL_CANDIDATE",
            "At least one numerical candidate met the configured residual threshold, "
            "but no exact identity was verified.",
        )
    return (
        "NOT_FOUND",
        "This run did not find a candidate meeting the configured numerical threshold. "
        "This is not evidence of non-existence.",
    )


def _verification_summary(
    reconstruction: ReconstructionOutcome | None,
    numerical_candidate_count: int,
    reconstruction_attempt_count: int,
    reconstruction_enabled: bool,
) -> dict[str, Any]:
    if reconstruction is not None and reconstruction.status == "VERIFIED":
        return {
            "status": "VERIFIED",
            "reconstruction_attempts": reconstruction_attempt_count,
            "best_reconstruction_error": reconstruction.best_coefficient_error,
            "selected_max_denominator": reconstruction.selected_max_denominator,
            "selected_scale_mode": reconstruction.selected_scale_mode,
            "report": "verification.json",
            "candidate": "candidate_rational.json",
        }

    if not reconstruction_enabled:
        return {
            "status": "NOT_ATTEMPTED",
            "reason": "exact reconstruction is disabled",
            "reconstruction_attempts": 0,
        }

    if numerical_candidate_count == 0:
        return {
            "status": "NOT_ATTEMPTED",
            "reason": "no numerical candidate reached the configured threshold",
            "reconstruction_attempts": 0,
        }

    return {
        "status": "NOT_VERIFIED",
        "reason": (
            "bounded rational reconstruction was attempted for every numerical "
            "candidate, but no exact identity passed independent verification"
        ),
        "reconstruction_attempts": reconstruction_attempt_count,
    }


def run_command(config_path: Path, metrics_override: bool | None) -> int:
    config = load_config(config_path)
    if metrics_override is not None:
        config = RunConfig(
            problem=config.problem,
            search=config.search,
            reconstruction=config.reconstruction,
            metrics=type(config.metrics)(
                enabled=metrics_override,
                live=config.metrics.live if metrics_override else False,
                interval=config.metrics.interval,
                plots=config.metrics.plots if metrics_override else False,
            ),
            output=config.output,
        )

    config_dict = config.to_dict()
    run_id = make_run_id(config_dict)
    run_directory = Path(config.output.root) / run_id
    run_directory.mkdir(parents=True, exist_ok=False)
    atomic_write_json(run_directory / "config.json", config_dict)

    recorder = MetricsRecorder(
        run_directory / "metrics.jsonl" if config.metrics.enabled else None,
        enabled=config.metrics.enabled,
        live=config.metrics.live,
        interval=config.metrics.interval,
    )
    started = time.perf_counter()

    restart_directory = run_directory / "restarts"
    restart_directory.mkdir(parents=True, exist_ok=True)

    numerical_candidate_count = 0
    reconstruction_attempt_count = 0
    verified_restart: int | None = None
    verified_outcome: ReconstructionOutcome | None = None

    def evaluate_restart(
        restart: int,
        candidate,
        summary: dict[str, Any],
    ) -> RestartDecision:
        nonlocal numerical_candidate_count
        nonlocal reconstruction_attempt_count
        nonlocal verified_restart
        nonlocal verified_outcome

        numerical_payload = numerical_candidate_dict(
            candidate,
            metadata=summary,
        )
        atomic_write_json(
            restart_directory / f"restart_{restart:03d}.json",
            numerical_payload,
        )

        if not summary["candidate_threshold_met"]:
            return RestartDecision(
                metadata={
                    "reconstruction_status": "NOT_ATTEMPTED",
                    "reconstruction_attempts": 0,
                }
            )

        numerical_candidate_count += 1

        if not config.reconstruction.enabled:
            return RestartDecision(
                metadata={
                    "reconstruction_status": "NOT_ATTEMPTED",
                    "reconstruction_attempts": 0,
                }
            )

        reconstruction = reconstruct_and_verify(
            candidate,
            config.reconstruction,
        )
        reconstruction_attempt_count += reconstruction.attempts

        metadata = {
            "reconstruction_status": reconstruction.status,
            "reconstruction_attempts": reconstruction.attempts,
            "best_reconstruction_error": reconstruction.best_coefficient_error,
            "reconstruction_within_tolerance": reconstruction.within_tolerance,
        }

        if reconstruction.status != "VERIFIED":
            return RestartDecision(metadata=metadata)

        verified_restart = restart
        verified_outcome = reconstruction

        exact_filename = f"restart_{restart:03d}_rational.json"
        verification_filename = f"restart_{restart:03d}_verification.json"
        atomic_write_json(
            restart_directory / exact_filename,
            reconstruction.exact_payload,
        )
        atomic_write_json(
            restart_directory / verification_filename,
            reconstruction.verification_report,
        )

        metadata.update(
            {
                "verified": True,
                "selected_max_denominator": reconstruction.selected_max_denominator,
                "selected_scale_mode": reconstruction.selected_scale_mode,
                "exact_candidate": exact_filename,
                "verification_report": verification_filename,
            }
        )

        return RestartDecision(
            stop=config.search.stop_on_verified,
            reason="verified_identity" if config.search.stop_on_verified else None,
            metadata=metadata,
        )

    try:
        outcome = run_search(
            config,
            recorder,
            on_restart_complete=evaluate_restart,
        )

        best_payload = numerical_candidate_dict(
            outcome.best,
            metadata={
                "best_restart": outcome.best_restart,
                **outcome.best_statistics,
            },
        )
        atomic_write_json(run_directory / "candidate.json", best_payload)

        if verified_outcome is not None:
            atomic_write_json(
                run_directory / "candidate_rational.json",
                verified_outcome.exact_payload,
            )
            atomic_write_json(
                run_directory / "verification.json",
                verified_outcome.verification_report,
            )

        verification = _verification_summary(
            verified_outcome,
            numerical_candidate_count,
            reconstruction_attempt_count,
            config.reconstruction.enabled,
        )

        status, claim = _status_and_claim(
            outcome.best_statistics["relative_residual"],
            config.search.candidate_tolerance,
            verification["status"],
        )

        runtime = time.perf_counter() - started
        figures: list[str] = []
        if config.metrics.enabled and config.metrics.plots:
            figures = plot_metrics(
                run_directory / "metrics.jsonl",
                run_directory,
            )

        artifacts = {
            "config": "config.json",
            "candidate": "candidate.json",
            "candidate_rational": (
                "candidate_rational.json"
                if (run_directory / "candidate_rational.json").exists()
                else None
            ),
            "verification": (
                "verification.json"
                if (run_directory / "verification.json").exists()
                else None
            ),
            "metrics": "metrics.jsonl" if config.metrics.enabled else None,
            "figures": figures,
            "summary": "summary.html",
            "restarts": "restarts",
        }

        result = {
            "schema_version": 2,
            "project_version": __version__,
            "run_id": run_id,
            "created_utc": utc_now(),
            "status": status,
            "claim": claim,
            "problem": {
                "matrix_size": config.problem.matrix_size,
                "rank": config.problem.rank,
            },
            "search": {
                "method": config.search.method,
                "configured_restarts": config.search.restarts,
                "completed_restarts": len(outcome.restart_summaries),
                "best_restart": outcome.best_restart,
                "candidate_tolerance": config.search.candidate_tolerance,
                "numerical_candidates": numerical_candidate_count,
                "reconstruction_attempts": reconstruction_attempt_count,
                "verified_restart": verified_restart,
                "termination_reason": outcome.termination_reason,
                "restart_summaries": outcome.restart_summaries,
            },
            "best": outcome.best_statistics,
            "verification": verification,
            "comparison": _comparison(
                config.problem.matrix_size,
                config.problem.rank,
            ),
            "runtime_seconds": runtime,
            "environment": environment_metadata(_repository_root()),
            "artifacts": artifacts,
        }

        atomic_write_json(run_directory / "result.json", result)
        write_html_summary(result, run_directory, figures)

        print(f"status: {status}")
        print(f"run: {run_directory}")
        print(
            "best relative residual: "
            f"{outcome.best_statistics['relative_residual']:.6e}"
        )
        print(f"numerical candidates: {numerical_candidate_count}")
        print(f"reconstruction attempts: {reconstruction_attempt_count}")
        print(f"exact verification: {verification['status']}")
        print(f"termination: {outcome.termination_reason}")
        print(claim)
        return 0 if status in {"VERIFIED", "NUMERICAL_CANDIDATE"} else 2

    except Exception as error:
        recorder.finish_live_line()
        runtime = time.perf_counter() - started
        result = {
            "schema_version": 2,
            "project_version": __version__,
            "run_id": run_id,
            "created_utc": utc_now(),
            "status": "ERROR",
            "claim": (
                "The run ended with an execution error and produced no scientific conclusion."
            ),
            "problem": {
                "matrix_size": config.problem.matrix_size,
                "rank": config.problem.rank,
            },
            "search": {
                "method": config.search.method,
                "configured_restarts": config.search.restarts,
                "completed_restarts": 0,
                "best_restart": None,
                "candidate_tolerance": config.search.candidate_tolerance,
                "numerical_candidates": 0,
                "reconstruction_attempts": 0,
                "verified_restart": None,
                "termination_reason": "error",
                "restart_summaries": [],
            },
            "best": {
                "residual_norm": None,
                "relative_residual": None,
                "max_abs_residual": None,
            },
            "verification": {"status": "NOT_ATTEMPTED"},
            "comparison": _comparison(
                config.problem.matrix_size,
                config.problem.rank,
            ),
            "runtime_seconds": runtime,
            "error": {
                "type": type(error).__name__,
                "message": str(error),
            },
            "environment": environment_metadata(_repository_root()),
            "artifacts": {
                "config": "config.json",
                "candidate": None,
                "candidate_rational": None,
                "verification": None,
                "metrics": "metrics.jsonl" if config.metrics.enabled else None,
                "figures": [],
                "summary": "summary.html",
                "restarts": "restarts",
            },
        }
        atomic_write_json(run_directory / "result.json", result)
        write_html_summary(result, run_directory, [])
        print(
            f"status: ERROR\nrun: {run_directory}\n"
            f"{type(error).__name__}: {error}"
        )
        return 1


def verify_command(candidate_path: Path, output_path: Path | None) -> int:
    payload = load_candidate(candidate_path)
    report = verify_exact(payload)
    if output_path is not None:
        atomic_write_json(output_path, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 2


def report_command(run_directory: Path) -> int:
    result_path = run_directory / "result.json"
    if not result_path.exists():
        raise FileNotFoundError(f"missing {result_path}")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    figures: list[str] = []
    metrics_path = run_directory / "metrics.jsonl"
    if metrics_path.exists():
        figures = plot_metrics(metrics_path, run_directory)
    path = write_html_summary(result, run_directory, figures)
    print(path)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fast-mm")
    parser.add_argument("--version", action="version", version=__version__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser(
        "run",
        help="run a numerical search with exactification of promising restarts",
    )
    run_parser.add_argument("config", type=Path)
    metrics_group = run_parser.add_mutually_exclusive_group()
    metrics_group.add_argument("--metrics", dest="metrics", action="store_true")
    metrics_group.add_argument("--no-metrics", dest="metrics", action="store_false")
    run_parser.set_defaults(metrics=None)

    verify_parser = subparsers.add_parser(
        "verify",
        help="exactly verify a rational candidate",
    )
    verify_parser.add_argument("candidate", type=Path)
    verify_parser.add_argument("--output", type=Path)

    report_parser = subparsers.add_parser(
        "report",
        help="regenerate a run HTML report",
    )
    report_parser.add_argument("run_directory", type=Path)

    campaign_parser = subparsers.add_parser(
        "campaign",
        help="run or resume a durable multi-seed search campaign",
    )
    campaign_parser.add_argument("config", type=Path)

    campaign_status_parser = subparsers.add_parser(
        "campaign-status",
        help="show durable campaign progress without running a search",
    )
    campaign_status_parser.add_argument("config", type=Path)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "run":
        return run_command(args.config, args.metrics)
    if args.command == "verify":
        return verify_command(args.candidate, args.output)
    if args.command == "report":
        return report_command(args.run_directory)
    if args.command == "campaign":
        return run_campaign(args.config, _repository_root())
    if args.command == "campaign-status":
        return campaign_status(args.config, _repository_root())
    parser.error("unknown command")
    return 2

