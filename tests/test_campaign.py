from __future__ import annotations

import json
from pathlib import Path

import pytest

from fast_mm import campaign


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _project(tmp_path: Path, attempts_per_session: int = 2) -> tuple[Path, Path]:
    project = tmp_path / "project"
    configs = project / "configs"
    configs.mkdir(parents=True)

    run_config = {
        "problem": {
            "matrix_size": 2,
            "rank": 7,
        },
        "search": {
            "method": "lbfgsb",
            "restarts": 8,
            "max_iterations": 20,
            "initial_scale": 0.5,
            "seed": 1,
            "candidate_tolerance": 1e-6,
            "ftol": 1e-15,
            "gtol": 1e-10,
            "max_line_search_steps": 50,
            "stop_on_verified": True,
        },
        "reconstruction": {
            "enabled": True,
            "method": "rational",
            "max_denominator": 16,
            "tolerance": 1e-8,
            "normalise_term_scales": True,
            "include_unscaled": True,
        },
        "metrics": {
            "enabled": False,
            "live": False,
            "interval": 10,
            "plots": False,
        },
        "output": {
            "root": "runs",
        },
    }
    _write_json(
        configs / "search.json",
        run_config,
    )

    campaign_config = {
        "name": "test_campaign",
        "run_config": "search.json",
        "attempts_per_session": attempts_per_session,
        "attempt_restarts": 1,
        "start_seed": 1,
        "state_root": "runs/campaigns",
        "stop_status": "VERIFIED",
    }
    campaign_path = configs / "campaign.json"
    _write_json(
        campaign_path,
        campaign_config,
    )
    return project, campaign_path


def _result_for_request(
    project: Path,
    request_path: Path,
    status: str,
    residual: float = 1e-8,
) -> None:
    request = json.loads(
        request_path.read_text(encoding="utf-8")
    )
    seed = int(request["search"]["seed"])
    output_root = Path(request["output"]["root"])
    if not output_root.is_absolute():
        output_root = project / output_root

    run_directory = output_root / f"run_seed_{seed:08d}"
    run_directory.mkdir(parents=True, exist_ok=False)

    _write_json(
        run_directory / "config.json",
        request,
    )
    _write_json(
        run_directory / "result.json",
        {
            "schema_version": 2,
            "project_version": "test",
            "run_id": run_directory.name,
            "created_utc": "2026-10-07T12:00:00+00:00",
            "status": status,
            "problem": {
                "matrix_size": 2,
                "rank": 7,
            },
            "best": {
                "relative_residual": residual,
            },
        },
    )


def _state(project: Path) -> dict:
    path = (
        project
        / "runs"
        / "campaigns"
        / "test_campaign"
        / "state.json"
    )
    return json.loads(
        path.read_text(encoding="utf-8")
    )


def test_campaign_resumes_with_next_seed(
    tmp_path: Path,
    monkeypatch,
) -> None:
    project, campaign_path = _project(
        tmp_path,
        attempts_per_session=2,
    )
    seeds: list[int] = []

    def fake_execute(project_directory: Path, request_path: Path) -> int:
        request = json.loads(
            request_path.read_text(encoding="utf-8")
        )
        seeds.append(int(request["search"]["seed"]))
        _result_for_request(
            project_directory,
            request_path,
            "NUMERICAL_CANDIDATE",
        )
        return 0

    monkeypatch.setattr(
        campaign,
        "_execute_attempt",
        fake_execute,
    )

    assert campaign.run_campaign(
        campaign_path,
        project,
    ) == 0
    assert seeds == [1, 2]
    first_state = _state(project)
    assert first_state["attempts_completed"] == 2
    assert first_state["next_seed"] == 3

    assert campaign.run_campaign(
        campaign_path,
        project,
    ) == 0
    assert seeds == [1, 2, 3, 4]
    second_state = _state(project)
    assert second_state["attempts_completed"] == 4
    assert second_state["next_seed"] == 5


def test_interrupted_attempt_retries_same_seed(
    tmp_path: Path,
    monkeypatch,
) -> None:
    project, campaign_path = _project(
        tmp_path,
        attempts_per_session=1,
    )
    calls: list[int] = []

    def interrupted(project_directory: Path, request_path: Path) -> int:
        request = json.loads(
            request_path.read_text(encoding="utf-8")
        )
        calls.append(int(request["search"]["seed"]))
        raise KeyboardInterrupt

    monkeypatch.setattr(
        campaign,
        "_execute_attempt",
        interrupted,
    )

    assert campaign.run_campaign(
        campaign_path,
        project,
    ) == 130
    assert calls == [1]
    state = _state(project)
    assert state["attempts_completed"] == 0
    assert state["next_seed"] == 1

    def completed(project_directory: Path, request_path: Path) -> int:
        request = json.loads(
            request_path.read_text(encoding="utf-8")
        )
        calls.append(int(request["search"]["seed"]))
        _result_for_request(
            project_directory,
            request_path,
            "NUMERICAL_CANDIDATE",
        )
        return 0

    monkeypatch.setattr(
        campaign,
        "_execute_attempt",
        completed,
    )

    assert campaign.run_campaign(
        campaign_path,
        project,
    ) == 0
    assert calls == [1, 1]
    state = _state(project)
    assert state["attempts_completed"] == 1
    assert state["next_seed"] == 2


def test_completed_result_is_reconciled_after_interruption(
    tmp_path: Path,
    monkeypatch,
) -> None:
    project, campaign_path = _project(
        tmp_path,
        attempts_per_session=1,
    )
    calls: list[int] = []

    def completed_then_interrupted(
        project_directory: Path,
        request_path: Path,
    ) -> int:
        request = json.loads(
            request_path.read_text(encoding="utf-8")
        )
        calls.append(int(request["search"]["seed"]))
        _result_for_request(
            project_directory,
            request_path,
            "NUMERICAL_CANDIDATE",
        )
        raise KeyboardInterrupt

    monkeypatch.setattr(
        campaign,
        "_execute_attempt",
        completed_then_interrupted,
    )

    assert campaign.run_campaign(
        campaign_path,
        project,
    ) == 130
    state = _state(project)
    assert state["attempts_completed"] == 0
    assert state["next_seed"] == 1

    def next_attempt(
        project_directory: Path,
        request_path: Path,
    ) -> int:
        request = json.loads(
            request_path.read_text(encoding="utf-8")
        )
        calls.append(int(request["search"]["seed"]))
        _result_for_request(
            project_directory,
            request_path,
            "NUMERICAL_CANDIDATE",
        )
        return 0

    monkeypatch.setattr(
        campaign,
        "_execute_attempt",
        next_attempt,
    )

    assert campaign.run_campaign(
        campaign_path,
        project,
    ) == 0

    # Seed 1 was recovered from disk rather than recomputed. The one new
    # attempt in this session therefore used seed 2.
    assert calls == [1, 2]
    state = _state(project)
    assert state["attempts_completed"] == 2
    assert state["next_seed"] == 3


def test_verified_result_stops_campaign_immediately(
    tmp_path: Path,
    monkeypatch,
) -> None:
    project, campaign_path = _project(
        tmp_path,
        attempts_per_session=5,
    )
    seeds: list[int] = []

    def fake_execute(project_directory: Path, request_path: Path) -> int:
        request = json.loads(
            request_path.read_text(encoding="utf-8")
        )
        seed = int(request["search"]["seed"])
        seeds.append(seed)
        _result_for_request(
            project_directory,
            request_path,
            "VERIFIED",
            residual=0.0,
        )
        return 0

    monkeypatch.setattr(
        campaign,
        "_execute_attempt",
        fake_execute,
    )

    assert campaign.run_campaign(
        campaign_path,
        project,
    ) == 0
    assert seeds == [1]
    state = _state(project)
    assert state["verified"] is True
    assert state["verified_seed"] == 1
    assert state["attempts_completed"] == 1
    assert state["next_seed"] == 2


def test_error_does_not_advance_seed(
    tmp_path: Path,
    monkeypatch,
) -> None:
    project, campaign_path = _project(
        tmp_path,
        attempts_per_session=1,
    )
    calls: list[int] = []

    def error_execute(project_directory: Path, request_path: Path) -> int:
        request = json.loads(
            request_path.read_text(encoding="utf-8")
        )
        seed = int(request["search"]["seed"])
        calls.append(seed)

        output_root = Path(request["output"]["root"])
        if not output_root.is_absolute():
            output_root = project_directory / output_root
        run_directory = output_root / f"error_seed_{seed:08d}"
        run_directory.mkdir(parents=True, exist_ok=False)
        _write_json(
            run_directory / "config.json",
            request,
        )
        _write_json(
            run_directory / "result.json",
            {
                "status": "ERROR",
                "error": {
                    "type": "RuntimeError",
                    "message": "mock error",
                },
            },
        )
        return 1

    monkeypatch.setattr(
        campaign,
        "_execute_attempt",
        error_execute,
    )

    assert campaign.run_campaign(
        campaign_path,
        project,
    ) == 1
    state = _state(project)
    assert state["next_seed"] == 1
    assert state["attempts_completed"] == 0
    assert state["last_error"]["seed"] == 1

    def success_execute(project_directory: Path, request_path: Path) -> int:
        request = json.loads(
            request_path.read_text(encoding="utf-8")
        )
        calls.append(int(request["search"]["seed"]))
        _result_for_request(
            project_directory,
            request_path,
            "NOT_FOUND",
            residual=1e-3,
        )
        return 2

    monkeypatch.setattr(
        campaign,
        "_execute_attempt",
        success_execute,
    )

    assert campaign.run_campaign(
        campaign_path,
        project,
    ) == 0
    assert calls == [1, 1]
    state = _state(project)
    assert state["next_seed"] == 2
    assert state["attempts_completed"] == 1


def test_changed_scientific_config_requires_new_campaign(
    tmp_path: Path,
    monkeypatch,
) -> None:
    project, campaign_path = _project(
        tmp_path,
        attempts_per_session=1,
    )

    def fake_execute(project_directory: Path, request_path: Path) -> int:
        _result_for_request(
            project_directory,
            request_path,
            "NUMERICAL_CANDIDATE",
        )
        return 0

    monkeypatch.setattr(
        campaign,
        "_execute_attempt",
        fake_execute,
    )
    assert campaign.run_campaign(
        campaign_path,
        project,
    ) == 0

    run_config_path = project / "configs" / "search.json"
    run_config = json.loads(
        run_config_path.read_text(encoding="utf-8")
    )
    run_config["search"]["max_iterations"] += 1
    _write_json(
        run_config_path,
        run_config,
    )

    with pytest.raises(
        ValueError,
        match="scientific configuration changed",
    ):
        campaign.run_campaign(
            campaign_path,
            project,
        )


def test_operational_session_size_can_change(
    tmp_path: Path,
    monkeypatch,
) -> None:
    project, campaign_path = _project(
        tmp_path,
        attempts_per_session=1,
    )

    def fake_execute(project_directory: Path, request_path: Path) -> int:
        _result_for_request(
            project_directory,
            request_path,
            "NUMERICAL_CANDIDATE",
        )
        return 0

    monkeypatch.setattr(
        campaign,
        "_execute_attempt",
        fake_execute,
    )
    assert campaign.run_campaign(
        campaign_path,
        project,
    ) == 0

    payload = json.loads(
        campaign_path.read_text(encoding="utf-8")
    )
    payload["attempts_per_session"] = 3
    _write_json(
        campaign_path,
        payload,
    )

    assert campaign.run_campaign(
        campaign_path,
        project,
    ) == 0
    state = _state(project)
    assert state["attempts_completed"] == 4
    assert state["next_seed"] == 5
