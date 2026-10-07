from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator

from fast_mm import __version__
from fast_mm.artifacts import atomic_write_json, utc_now
from fast_mm.config import load_config


TERMINAL_SCIENTIFIC_STATUSES = {
    "VERIFIED",
    "NUMERICAL_CANDIDATE",
    "NOT_FOUND",
}


@dataclass(frozen=True)
class CampaignConfig:
    name: str
    run_config: str
    attempts_per_session: int = 20
    attempt_restarts: int = 1
    start_seed: int = 1
    state_root: str = "runs/campaigns"
    stop_status: str = "VERIFIED"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_campaign_config(path: str | Path) -> CampaignConfig:
    config_path = Path(path)
    payload = _read_json(config_path)
    config = CampaignConfig(**payload)

    if not config.name.strip():
        raise ValueError("campaign name must not be empty.")
    if not config.run_config.strip():
        raise ValueError("campaign run_config must not be empty.")
    if config.attempts_per_session < 1:
        raise ValueError("attempts_per_session must be positive.")
    if config.attempt_restarts < 1:
        raise ValueError("attempt_restarts must be positive.")
    if config.start_seed < 0:
        raise ValueError("start_seed must be non-negative.")
    if not config.state_root.strip():
        raise ValueError("state_root must not be empty.")
    if config.stop_status != "VERIFIED":
        raise ValueError("stop_status must be 'VERIFIED' in this version.")

    return config


def _resolve_run_config(
    campaign_path: Path,
    config: CampaignConfig,
) -> Path:
    path = Path(config.run_config)
    if path.is_absolute():
        return path
    return (campaign_path.parent / path).resolve()


def _campaign_directory(
    project_directory: Path,
    config: CampaignConfig,
) -> Path:
    root = Path(config.state_root)
    if not root.is_absolute():
        root = project_directory / root
    return root / config.name


def _scientific_signature(
    campaign_path: Path,
    config: CampaignConfig,
) -> tuple[str, dict[str, Any]]:
    run_config_path = _resolve_run_config(campaign_path, config)
    run_config = load_config(run_config_path)
    run_payload = run_config.to_dict()

    search = dict(run_payload["search"])
    search["seed"] = 0
    search["restarts"] = config.attempt_restarts

    signature_payload = {
        "project_version": __version__,
        "problem": run_payload["problem"],
        "search": search,
        "reconstruction": run_payload["reconstruction"],
        "attempt_restarts": config.attempt_restarts,
        "stop_status": config.stop_status,
    }
    return _sha256(signature_payload), signature_payload


def _initial_state(
    config: CampaignConfig,
    signature: str,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "campaign_name": config.name,
        "campaign_signature": signature,
        "project_version": __version__,
        "created_utc": utc_now(),
        "updated_utc": utc_now(),
        "attempts_completed": 0,
        "next_seed": config.start_seed,
        "last_completed_seed": None,
        "last_status": None,
        "last_result": None,
        "last_error": None,
        "best_relative_residual": None,
        "best_run": None,
        "verified": False,
        "verified_seed": None,
        "verified_run": None,
    }


def _load_state(
    state_path: Path,
    config: CampaignConfig,
    signature: str,
) -> dict[str, Any]:
    if not state_path.exists():
        return _initial_state(config, signature)

    state = _read_json(state_path)
    if state.get("schema_version") != 1:
        raise ValueError("unsupported campaign state schema_version.")
    if state.get("campaign_name") != config.name:
        raise ValueError("campaign state name does not match the campaign config.")
    if state.get("campaign_signature") != signature:
        raise ValueError(
            "campaign scientific configuration changed. "
            "Start a new campaign name instead of mixing incompatible attempts."
        )
    return state


def _write_state(
    state_path: Path,
    state: dict[str, Any],
) -> None:
    state["updated_utc"] = utc_now()
    atomic_write_json(state_path, state)


@contextmanager
def _campaign_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+", encoding="utf-8")

    try:
        try:
            import fcntl
        except ImportError as error:  # pragma: no cover - project currently targets macOS/Linux
            raise RuntimeError(
                "resumable campaign locking requires fcntl on this platform."
            ) from error

        try:
            fcntl.flock(
                handle.fileno(),
                fcntl.LOCK_EX | fcntl.LOCK_NB,
            )
        except BlockingIOError as error:
            raise RuntimeError(
                "this campaign is already running in another process."
            ) from error

        yield
    finally:
        try:
            if "fcntl" in locals():
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


def _attempt_directory(
    campaign_directory: Path,
    seed: int,
) -> Path:
    return campaign_directory / "attempts" / f"seed_{seed:08d}"


def _attempt_config_payload(
    campaign_path: Path,
    config: CampaignConfig,
    project_directory: Path,
    attempt_directory: Path,
    seed: int,
) -> dict[str, Any]:
    run_config_path = _resolve_run_config(campaign_path, config)
    payload = _read_json(run_config_path)

    payload.setdefault("search", {})
    payload["search"]["seed"] = seed
    payload["search"]["restarts"] = config.attempt_restarts

    try:
        output_root = attempt_directory.relative_to(project_directory)
    except ValueError:
        output_root = attempt_directory

    payload.setdefault("output", {})
    payload["output"]["root"] = str(output_root)
    return payload


def _ensure_attempt_request(
    campaign_path: Path,
    config: CampaignConfig,
    project_directory: Path,
    attempt_directory: Path,
    seed: int,
) -> Path:
    attempt_directory.mkdir(parents=True, exist_ok=True)
    request_path = attempt_directory / "request.json"
    expected = _attempt_config_payload(
        campaign_path,
        config,
        project_directory,
        attempt_directory,
        seed,
    )

    if request_path.exists():
        existing = _read_json(request_path)
        if existing != expected:
            raise ValueError(
                f"existing attempt request differs for seed {seed}; "
                "refusing to overwrite reproducibility data."
            )
    else:
        atomic_write_json(request_path, expected)

    return request_path


def _matching_results(
    attempt_directory: Path,
    seed: int,
    attempt_restarts: int,
) -> list[tuple[Path, dict[str, Any]]]:
    matches: list[tuple[Path, dict[str, Any]]] = []

    for result_path in sorted(attempt_directory.glob("*/result.json")):
        try:
            result = _read_json(result_path)
            run_config = _read_json(result_path.parent / "config.json")
        except (OSError, json.JSONDecodeError, FileNotFoundError):
            continue

        search = run_config.get("search", {})
        if int(search.get("seed", -1)) != seed:
            continue
        if int(search.get("restarts", -1)) != attempt_restarts:
            continue

        matches.append((result_path, result))

    return matches


def _completed_scientific_result(
    attempt_directory: Path,
    seed: int,
    attempt_restarts: int,
) -> tuple[Path, dict[str, Any]] | None:
    completed = [
        item
        for item in _matching_results(
            attempt_directory,
            seed,
            attempt_restarts,
        )
        if item[1].get("status") in TERMINAL_SCIENTIFIC_STATUSES
    ]

    if len(completed) > 1:
        raise RuntimeError(
            f"multiple completed scientific results exist for seed {seed}; "
            "manual review is required before continuing."
        )

    return completed[0] if completed else None


def _latest_error_result(
    attempt_directory: Path,
    seed: int,
    attempt_restarts: int,
) -> tuple[Path, dict[str, Any]] | None:
    errors = [
        item
        for item in _matching_results(
            attempt_directory,
            seed,
            attempt_restarts,
        )
        if item[1].get("status") == "ERROR"
    ]
    return errors[-1] if errors else None


def _relative_to_project(
    path: Path,
    project_directory: Path,
) -> str:
    try:
        return str(path.relative_to(project_directory))
    except ValueError:
        return str(path)


def _commit_completed_attempt(
    state_path: Path,
    state: dict[str, Any],
    project_directory: Path,
    attempt_directory: Path,
    seed: int,
    result_path: Path,
    result: dict[str, Any],
) -> None:
    status = str(result["status"])
    relative_result = _relative_to_project(
        result_path,
        project_directory,
    )
    relative_run = _relative_to_project(
        result_path.parent,
        project_directory,
    )
    residual = result.get("best", {}).get("relative_residual")

    attempt_record = {
        "schema_version": 1,
        "seed": seed,
        "status": status,
        "result": relative_result,
        "run": relative_run,
        "best_relative_residual": residual,
        "completed_utc": result.get("created_utc") or utc_now(),
    }
    atomic_write_json(
        attempt_directory / "attempt.json",
        attempt_record,
    )

    state["attempts_completed"] = int(state["attempts_completed"]) + 1
    state["last_completed_seed"] = seed
    state["next_seed"] = seed + 1
    state["last_status"] = status
    state["last_result"] = relative_result
    state["last_error"] = None

    if residual is not None:
        residual = float(residual)
        previous_best = state.get("best_relative_residual")
        if previous_best is None or residual < float(previous_best):
            state["best_relative_residual"] = residual
            state["best_run"] = relative_run

    if status == "VERIFIED":
        state["verified"] = True
        state["verified_seed"] = seed
        state["verified_run"] = relative_run

    _write_state(state_path, state)


def _record_error(
    state_path: Path,
    state: dict[str, Any],
    project_directory: Path,
    seed: int,
    result_path: Path | None,
    result: dict[str, Any] | None,
    returncode: int | None,
) -> None:
    error_payload = result.get("error", {}) if result else {}
    state["last_error"] = {
        "seed": seed,
        "returncode": returncode,
        "result": (
            _relative_to_project(result_path, project_directory)
            if result_path is not None
            else None
        ),
        "type": error_payload.get("type"),
        "message": error_payload.get("message"),
        "recorded_utc": utc_now(),
    }
    _write_state(state_path, state)


def _execute_attempt(
    project_directory: Path,
    request_path: Path,
) -> int:
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "fast_mm",
            "run",
            str(request_path),
        ],
        cwd=project_directory,
        check=False,
    )
    return int(completed.returncode)


def _write_campaign_snapshot(
    path: Path,
    campaign_path: Path,
    config: CampaignConfig,
    signature: str,
    signature_payload: dict[str, Any],
    project_directory: Path,
) -> None:
    snapshot = {
        "schema_version": 1,
        "campaign": asdict(config),
        "campaign_config": _relative_to_project(
            campaign_path,
            project_directory,
        ),
        "campaign_signature": signature,
        "scientific_signature_payload": signature_payload,
        "project_version": __version__,
    }

    if path.exists():
        existing = _read_json(path)
        if existing.get("campaign_signature") != signature:
            raise ValueError(
                "campaign scientific configuration changed; start a new campaign name "
                "instead of mixing incompatible attempts."
            )

    atomic_write_json(path, snapshot)


def _print_state(
    config: CampaignConfig,
    state: dict[str, Any],
) -> None:
    print()
    print("=" * 72)
    print(f"fast_mm campaign: {config.name}")
    print("=" * 72)
    print(f"Completed attempts: {state['attempts_completed']}")
    print(f"Next seed:          {state['next_seed']}")
    print(f"Best residual:      {state['best_relative_residual']}")
    print(f"Best run:           {state['best_run']}")
    print(f"Last status:        {state['last_status']}")
    print(f"Verified:           {state['verified']}")
    if state["verified"]:
        print(f"Verified seed:      {state['verified_seed']}")
        print(f"Verified run:       {state['verified_run']}")
    if state.get("last_error"):
        print(f"Last error:         {state['last_error']}")
    print("=" * 72)


def campaign_status(
    campaign_path: str | Path,
    project_directory: str | Path | None = None,
) -> int:
    path = Path(campaign_path).resolve()
    project = (
        Path(project_directory).resolve()
        if project_directory is not None
        else Path.cwd().resolve()
    )
    config = load_campaign_config(path)
    signature, _ = _scientific_signature(path, config)
    directory = _campaign_directory(project, config)
    state_path = directory / "state.json"

    if state_path.exists():
        state = _load_state(
            state_path,
            config,
            signature,
        )
    else:
        state = _initial_state(
            config,
            signature,
        )

    _print_state(config, state)
    return 0


def run_campaign(
    campaign_path: str | Path,
    project_directory: str | Path | None = None,
) -> int:
    path = Path(campaign_path).resolve()
    project = (
        Path(project_directory).resolve()
        if project_directory is not None
        else Path.cwd().resolve()
    )
    config = load_campaign_config(path)
    signature, signature_payload = _scientific_signature(
        path,
        config,
    )

    directory = _campaign_directory(project, config)
    directory.mkdir(parents=True, exist_ok=True)
    state_path = directory / "state.json"
    snapshot_path = directory / "campaign.json"
    lock_path = directory / "campaign.lock"

    with _campaign_lock(lock_path):
        _write_campaign_snapshot(
            snapshot_path,
            path,
            config,
            signature,
            signature_payload,
            project,
        )
        state = _load_state(
            state_path,
            config,
            signature,
        )
        if not state_path.exists():
            _write_state(state_path, state)

        if state["verified"]:
            _print_state(config, state)
            print("Campaign already completed successfully.")
            return 0

        print()
        print("=" * 72)
        print(f"fast_mm resumable campaign: {config.name}")
        print("=" * 72)
        print(f"Previous attempts: {state['attempts_completed']}")
        print(f"Next seed:         {state['next_seed']}")
        print(f"Session attempts:  {config.attempts_per_session}")
        print(f"Restarts/attempt:  {config.attempt_restarts}")
        print("=" * 72)

        attempts_run_this_session = 0

        while attempts_run_this_session < config.attempts_per_session:
            seed = int(state["next_seed"])
            attempt_directory = _attempt_directory(
                directory,
                seed,
            )

            completed_result = _completed_scientific_result(
                attempt_directory,
                seed,
                config.attempt_restarts,
            )

            if completed_result is not None:
                result_path, result = completed_result
                print(
                    f"Recovering completed seed {seed} "
                    "that was not yet committed to campaign state."
                )
                _commit_completed_attempt(
                    state_path,
                    state,
                    project,
                    attempt_directory,
                    seed,
                    result_path,
                    result,
                )
                (attempt_directory / "active.json").unlink(
                    missing_ok=True,
                )
                if state["verified"]:
                    _print_state(config, state)
                    return 0
                continue

            request_path = _ensure_attempt_request(
                path,
                config,
                project,
                attempt_directory,
                seed,
            )

            active = {
                "schema_version": 1,
                "seed": seed,
                "started_utc": utc_now(),
                "request": _relative_to_project(
                    request_path,
                    project,
                ),
            }
            atomic_write_json(
                attempt_directory / "active.json",
                active,
            )

            print()
            print(
                f"Campaign attempt "
                f"{attempts_run_this_session + 1}/"
                f"{config.attempts_per_session}  seed={seed}"
            )

            try:
                returncode = _execute_attempt(
                    project,
                    request_path,
                )
            except KeyboardInterrupt:
                print()
                print(
                    f"Campaign interrupted during seed {seed}. "
                    "Completed earlier attempts are preserved. "
                    "Run the same campaign again to resume."
                )
                return 130

            completed_result = _completed_scientific_result(
                attempt_directory,
                seed,
                config.attempt_restarts,
            )

            if completed_result is None:
                error_result = _latest_error_result(
                    attempt_directory,
                    seed,
                    config.attempt_restarts,
                )
                if error_result is None:
                    _record_error(
                        state_path,
                        state,
                        project,
                        seed,
                        None,
                        None,
                        returncode,
                    )
                    print(
                        f"Seed {seed} ended without a completed result. "
                        "The seed was not advanced."
                    )
                else:
                    error_path, error = error_result
                    _record_error(
                        state_path,
                        state,
                        project,
                        seed,
                        error_path,
                        error,
                        returncode,
                    )
                    print(
                        f"Seed {seed} produced ERROR. "
                        "The seed was not advanced; fix the error and rerun "
                        "the campaign to retry the same seed."
                    )
                return 1

            result_path, result = completed_result
            _commit_completed_attempt(
                state_path,
                state,
                project,
                attempt_directory,
                seed,
                result_path,
                result,
            )
            (attempt_directory / "active.json").unlink(
                missing_ok=True,
            )
            attempts_run_this_session += 1

            print(
                f"Committed seed {seed}: {result['status']}  "
                f"next seed={state['next_seed']}"
            )

            if state["verified"]:
                print()
                print("=" * 72)
                print("VERIFIED")
                print("=" * 72)
                print(f"Seed: {state['verified_seed']}")
                print(f"Run:  {state['verified_run']}")
                print("=" * 72)
                return 0

        _print_state(config, state)
        print(
            "No exact identity was verified in this session. "
            "Run the same campaign action again to continue with the next seed."
        )
        return 0
