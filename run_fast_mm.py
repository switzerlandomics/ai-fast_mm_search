# -*- coding: utf-8 -*-

# ============================================================================
# PROJECT PURPOSE — KEEP THIS BLOCK
# ============================================================================
#
# Reference:
# https://github.com/openai/math/blob/main/preprints/Matrix-Multiplication-Nine-Fourths-October-2-2026/build/paper.tex
#
# fast_mm is being built to automatically discover and exactly verify finite
# matrix-multiplication rules that use unusually few scalar multiplications.
#
# The key idea is that one efficient small rule can be reused recursively
# inside larger matrices. Strassen's known 2 × 2 rule is our validation
# benchmark: it uses 7 scalar multiplications instead of 8. Recursion turns
# this into 49 instead of 64 multiplications for 4 × 4 matrices, and 343
# instead of 512 for 8 × 8 matrices.
#
# The 2 × 2, rank-7 Strassen rule is NOT the final research target. It is used
# to prove that the complete discovery -> exactification -> verification
# pipeline works without being given the known solution during discovery.
#
# The research goal is to discover and exactly verify a new finite (u, r)
# rule with a better asymptotic exponent log_u(r). The OpenAI 9/4 result
# proves that sufficiently efficient finite exact decompositions exist
# asymptotically, but does not specify a competitive finite matrix size u
# or provide the explicit coefficients that fast_mm is intended to search for.
#
# Once a new rule is found, the same small recipe can be applied recursively
# to larger matrix blocks, so a saving at the base level compounds at every
# recursive level. A later engineering stage will test whether that theoretical
# advantage also produces faster practical matrix multiplication on real
# hardware.
#
# Long-running searches must be resumable. Completed attempts, seeds,
# candidates and verification outcomes must be preserved so a campaign can be
# stopped and later continued without repeating completed work.
#
# IMPORTANT:
# The current campaign is deliberately a Strassen VALIDATION campaign. It is
# not yet the final unknown-(u, r) research campaign. A research campaign
# should only be added after this validation pipeline reaches VERIFIED.
# ============================================================================

"""Spyder entry point for fast_mm validation, benchmarks and campaigns."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys


PROJECT_DIR = Path(__file__).resolve().parent

# Work through these actions in order. Keep exactly ONE ACTION uncommented.
# ACTION = "tests"                 # 1. Prove the current code passes all tests.
# ACTION = "verify_baseline"     # 2. Verify the stored exact Strassen identity.
# ACTION = "benchmark_quick"     # 3. Short end-to-end numerical benchmark.
# ACTION = "benchmark_search"    # 4. Longer one-off benchmark.
# ACTION = "campaign_status"     # 5. Inspect resumable campaign progress.
ACTION = "campaign"            # 6. Run/resume the durable validation campaign.

# ============================================================================
# CAMPAIGN RUNTIME GUIDE — DO NOT REMOVE
# ============================================================================
#
# The campaign is resumable. Each invocation runs up to the configured number
# of NEW attempts, starting from the next unfinished seed.
#
# Rough guide at ~1 attempt/second:
#     20       = quick check
#     3600     = ~1 hour
#     21600    = ~6 hours
#     43200    = ~12 hours
#
# Set in configs/strassen_campaign.json:
#     "attempts_per_session": <value above>
#     "attempt_restarts": 1
#
# Completed attempts are preserved. The campaign stops immediately if VERIFIED.
# ============================================================================

CAMPAIGN_CONFIG = "configs/strassen_campaign.json"


def run_fast_mm(
    arguments: list[str],
    accepted_returncodes: set[int],
) -> None:
    command = [
        sys.executable,
        "-m",
        "fast_mm",
        *arguments,
    ]

    print()
    print("=" * 72)
    print("fast_mm")
    print("=" * 72)
    print(f"Project: {PROJECT_DIR}")
    print(f"Python:  {sys.executable}")
    print(f"Action:  {ACTION}")
    print("=" * 72)
    print()

    try:
        completed = subprocess.run(
            command,
            cwd=PROJECT_DIR,
            check=False,
        )
    except KeyboardInterrupt:
        print()
        print(
            "Interrupted. Durable campaign state and completed attempts are "
            "preserved. Run the same action again to resume."
        )
        return

    if completed.returncode not in accepted_returncodes:
        raise SystemExit(completed.returncode)


def run_tests() -> None:
    print()
    print("=" * 72)
    print("fast_mm tests")
    print("=" * 72)
    print(f"Project: {PROJECT_DIR}")
    print(f"Python:  {sys.executable}")
    print("=" * 72)
    print()

    subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
        ],
        cwd=PROJECT_DIR,
        check=True,
    )


def main() -> None:
    actions = {
        "verify_baseline": (
            [
                "verify",
                "results/verified/strassen_2x2.json",
            ],
            {0},
        ),
        "benchmark_quick": (
            [
                "run",
                "configs/strassen_quick.json",
            ],
            {0, 2},
        ),
        "benchmark_search": (
            [
                "run",
                "configs/strassen_search.json",
            ],
            {0, 2},
        ),
        "campaign_status": (
            [
                "campaign-status",
                CAMPAIGN_CONFIG,
            ],
            {0},
        ),
        "campaign": (
            [
                "campaign",
                CAMPAIGN_CONFIG,
            ],
            {0, 130},
        ),
    }

    if ACTION == "tests":
        run_tests()
        return

    if ACTION not in actions:
        valid = [
            "tests",
            *actions,
        ]
        raise ValueError(
            f"Unknown ACTION: {ACTION!r}. "
            f"Choose one of: {', '.join(valid)}"
        )

    arguments, accepted_returncodes = actions[ACTION]
    run_fast_mm(
        arguments,
        accepted_returncodes,
    )


if __name__ == "__main__":
    main()
