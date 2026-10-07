from __future__ import annotations

from fast_mm.cli import _status_and_claim, build_parser


def test_verified_status_requires_exact_verification() -> None:
    status, _ = _status_and_claim(1e-2, 1e-6, "VERIFIED")
    assert status == "VERIFIED"


def test_small_residual_without_proof_is_numerical_candidate() -> None:
    status, _ = _status_and_claim(1e-8, 1e-6, "NOT_VERIFIED")
    assert status == "NUMERICAL_CANDIDATE"


def test_failed_numerical_search_is_not_found() -> None:
    status, _ = _status_and_claim(1e-3, 1e-6, "NOT_ATTEMPTED")
    assert status == "NOT_FOUND"



def test_cli_exposes_campaign_commands() -> None:
    parser = build_parser()
    campaign_args = parser.parse_args(
        ["campaign", "configs/strassen_campaign.json"]
    )
    status_args = parser.parse_args(
        ["campaign-status", "configs/strassen_campaign.json"]
    )
    assert campaign_args.command == "campaign"
    assert status_args.command == "campaign-status"
