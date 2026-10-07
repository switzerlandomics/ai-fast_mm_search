from __future__ import annotations

import html
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from fast_mm.decomposition import numerical_candidate_from_dict


def read_metrics(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def plot_metrics(metrics_path: Path, output_directory: Path) -> list[str]:
    records = read_metrics(metrics_path)
    iteration_records = [
        record for record in records if record.get("event") == "iteration"
    ]
    restart_records = [
        record for record in records if record.get("event") == "restart_end"
    ]
    created: list[str] = []

    if iteration_records:
        figure, axis = plt.subplots(figsize=(8, 4.5))
        restart_ids = sorted({int(record["restart"]) for record in iteration_records})
        for restart in restart_ids:
            selected = [
                record
                for record in iteration_records
                if int(record["restart"]) == restart
            ]
            axis.plot(
                [int(record["iteration"]) for record in selected],
                [
                    max(float(record["relative_residual"]), 1e-300)
                    for record in selected
                ],
                label=f"restart {restart + 1}",
            )
        axis.set_yscale("log")
        axis.set_xlabel("Iteration")
        axis.set_ylabel("Relative residual")
        axis.set_title("Search convergence")
        if len(restart_ids) <= 12:
            axis.legend()
        axis.grid(True, alpha=0.25)
        figure.tight_layout()
        path = output_directory / "convergence.png"
        figure.savefig(path, dpi=160)
        plt.close(figure)
        created.append(path.name)

    if restart_records:
        figure, axis = plt.subplots(figsize=(8, 4.5))
        restarts = [int(record["restart"]) + 1 for record in restart_records]
        residuals = [
            max(float(record["relative_residual"]), 1e-300)
            for record in restart_records
        ]
        axis.bar(restarts, residuals)
        axis.set_yscale("log")
        axis.set_xlabel("Restart")
        axis.set_ylabel("Final relative residual")
        axis.set_title("Restart outcomes")
        axis.grid(True, axis="y", alpha=0.25)
        figure.tight_layout()
        path = output_directory / "restarts.png"
        figure.savefig(path, dpi=160)
        plt.close(figure)
        created.append(path.name)

    return created


def _format_number(value: Any) -> str:
    if value is None:
        return "not available"
    if isinstance(value, float):
        if math.isfinite(value):
            return f"{value:.6g}"
        return str(value)
    return str(value)


def _format_scientific(value: Any) -> str:
    if value is None:
        return "not available"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(number):
        return str(number)
    if number == 0:
        return "0"
    return f"{number:.6e}"


def _format_percentage(value: float) -> str:
    return f"{100.0 * value:.1f}%"


def _table_html(rows: list[tuple[Any, Any]]) -> str:
    return "\n".join(
        "<tr>"
        f"<th>{html.escape(str(key))}</th>"
        f"<td>{html.escape(str(value))}</td>"
        "</tr>"
        for key, value in rows
    )


def _matrix_html(matrix: np.ndarray, precision: int = 8) -> str:
    rows = []
    for row in matrix:
        values = []
        for value in row:
            number = float(value)
            if abs(number - round(number)) < 1e-12:
                text = str(int(round(number)))
            else:
                text = f"{number:.{precision}g}"
            values.append(f"<td>{html.escape(text)}</td>")
        rows.append("<tr>" + "".join(values) + "</tr>")
    return '<table class="matrix">' + "".join(rows) + "</table>"


def _symbolic_matrix_html(matrix: list[list[str]]) -> str:
    rows = []
    for row in matrix:
        values = [f"<td>{html.escape(value)}</td>" for value in row]
        rows.append("<tr>" + "".join(values) + "</tr>")
    return '<table class="matrix symbolic-matrix">' + "".join(rows) + "</table>"


def _candidate_matrix_product(
    candidate_path: Path,
    matrix_size: int,
) -> dict[str, Any] | None:
    if not candidate_path.exists():
        return None
    try:
        payload = json.loads(candidate_path.read_text(encoding="utf-8"))
        decomposition = numerical_candidate_from_dict(payload)
        if decomposition.matrix_size != matrix_size:
            return None

        value_count = matrix_size * matrix_size
        left_matrix = np.arange(
            1,
            value_count + 1,
            dtype=np.float64,
        ).reshape(matrix_size, matrix_size)
        right_matrix = np.arange(
            value_count + 1,
            2 * value_count + 1,
            dtype=np.float64,
        ).reshape(matrix_size, matrix_size)

        products = (
            decomposition.left @ left_matrix.reshape(-1)
        ) * (
            decomposition.right @ right_matrix.reshape(-1)
        )
        candidate_matrix = (
            decomposition.output.T @ products
        ).reshape(matrix_size, matrix_size)
        expected_matrix = left_matrix @ right_matrix
        difference = candidate_matrix - expected_matrix
        expected_norm = float(np.linalg.norm(expected_matrix))

        return {
            "left": left_matrix,
            "right": right_matrix,
            "expected": expected_matrix,
            "candidate": candidate_matrix,
            "difference": difference,
            "relative_error": (
                float(np.linalg.norm(difference)) / expected_norm
                if expected_norm
                else math.inf
            ),
            "max_abs_error": float(np.max(np.abs(difference))),
        }
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None


def _worked_example_html(example: dict[str, Any] | None, rank: int) -> str:
    if example is None:
        return "<p>No numerical candidate was available for a worked example.</p>"
    return f"""
<div class="matrix-row">
    <div><h3>A</h3>{_matrix_html(example['left'])}</div>
    <div class="matrix-operator">×</div>
    <div><h3>B</h3>{_matrix_html(example['right'])}</div>
</div>

<h3>Standard matrix multiplication</h3>
{_matrix_html(example['expected'])}

<h3>Candidate result</h3>
{_matrix_html(example['candidate'])}

<h3>Difference</h3>
{_matrix_html(example['difference'], precision=5)}

<table>
{_table_html([
    ('Candidate bilinear products', rank),
    ('Maximum output error', _format_scientific(example['max_abs_error'])),
    ('Relative output error', _format_scientific(example['relative_error'])),
])}
</table>

<p class="secondary">
This example shows one pair of matrices. It is not the proof. Exact
verification checks the complete identity independently.
</p>
"""


def _introduction_html() -> str:
    left_matrix = _symbolic_matrix_html([["a", "b"], ["c", "d"]])
    right_matrix = _symbolic_matrix_html([["e", "f"], ["g", "h"]])
    products = _table_html(
        [
            ("M₁", "(a + d)(e + h)"),
            ("M₂", "(c + d)e"),
            ("M₃", "a(f − h)"),
            ("M₄", "d(g − e)"),
            ("M₅", "(a + b)h"),
            ("M₆", "(c − a)(e + f)"),
            ("M₇", "(b − d)(g + h)"),
        ]
    )
    outputs = _table_html(
        [
            ("C₁₁", "M₁ + M₄ − M₅ + M₇"),
            ("C₁₂", "M₃ + M₅"),
            ("C₂₁", "M₂ + M₄"),
            ("C₂₂", "M₁ − M₂ + M₃ + M₆"),
        ]
    )
    return f"""
<section class="introduction">
<h2>Why this search matters</h2>
<p>
Matrix multiplication is a basic operation used throughout computing,
including statistics, genomics, graphics and machine learning.
</p>
<p>For two 2 × 2 matrices,</p>
<div class="matrix-row equation-row">
    <div>{left_matrix}</div>
    <div class="matrix-operator">×</div>
    <div>{right_matrix}</div>
</div>
<p>the ordinary method computes four output values:</p>
<p class="equation">ae + bg,&nbsp;&nbsp; af + bh,&nbsp;&nbsp; ce + dg,&nbsp;&nbsp; cf + dh.</p>
<p>
Each output requires two multiplications, for a total of
<strong>8 scalar multiplications</strong>. A rule that uses fewer products can
change the asymptotic arithmetic cost when applied recursively.
</p>

<h3>Strassen's exact seven-product construction</h3>
<p>
Strassen's result was constructive: he wrote down an exact recipe showing how
two 2 × 2 matrices can be multiplied using only
<strong>7 scalar multiplications</strong> instead of the usual 8.
</p>
<table class="equation-table">{products}</table>
<p>The four entries are reconstructed using additions and subtractions:</p>
<table class="equation-table">{outputs}</table>
<p>
So Strassen supplied the exact coefficients needed to perform and verify the
algorithm, not merely an existence argument.
</p>

<h3>From an existence proof to an explicit algorithm</h3>
<p>
The recent OpenAI result proves the asymptotic upper bound
</p>
<p class="equation">ω ≤ 9/4 = 2.25,</p>
<p>
which implies that sufficiently efficient finite matrix-multiplication
decompositions exist. Roughly, for every small <strong>ε &gt; 0</strong>, there
must be some fixed matrix size <strong>u</strong> admitting an exact
decomposition with fewer than about
</p>
<p class="equation">u<sup>2.25 + ε</sup></p>
<p>
products. The proof does not provide the explicit finite coefficients in the
way Strassen did. That distinction motivates <code>fast_mm</code>.
</p>
<p>
The 2 × 2, rank-7 case is a benchmark. The programme starts from numerical
coefficients, searches for a low-residual decomposition, proposes exact
representatives for every promising restart, and checks each proposal with an
independent exact verifier. A small residual is never treated as a proof.
</p>
</section>
"""


def _artifact_links(result: dict[str, Any], output_directory: Path) -> str:
    artifacts = result.get("artifacts", {})
    ordered = [
        ("Configuration", artifacts.get("config")),
        ("Best numerical candidate", artifacts.get("candidate")),
        ("Verified exact candidate", artifacts.get("candidate_rational")),
        ("Exact verification", artifacts.get("verification")),
        ("Metrics", artifacts.get("metrics")),
    ]
    links = []
    for label, filename in ordered:
        if not filename:
            continue
        if not (output_directory / filename).exists():
            continue
        links.append(
            f'<li><a href="{html.escape(filename)}">{html.escape(label)}</a></li>'
        )
    return "<ul>" + "".join(links) + "</ul>" if links else "<p>No additional artefacts are available.</p>"


def _termination_text(search: dict[str, Any]) -> str:
    reason = search.get("termination_reason")
    if reason == "verified_identity":
        return "An exact identity was verified, so the search stopped early."
    if reason == "configured_restarts_completed":
        return "All configured restarts completed."
    if reason == "error":
        return "The run terminated because of an execution error."
    return str(reason or "not available")


def write_html_summary(
    result: dict[str, Any],
    output_directory: Path,
    figures: list[str],
) -> Path:
    status = str(result["status"])
    claim = str(result["claim"])
    problem = result["problem"]
    best = result["best"]
    verification = result["verification"]
    comparison = result["comparison"]
    search = result["search"]

    matrix_size = int(problem["matrix_size"])
    rank = int(problem["rank"])
    naive_rank = int(comparison["naive_rank"])
    saved = int(comparison["scalar_multiplications_saved"])
    fraction_of_naive = float(comparison["fraction_of_naive_multiplications"])
    candidate_exponent = float(comparison["multiplication_exponent"])
    reduction_fraction = 1.0 - fraction_of_naive
    tensor_coefficients = matrix_size**6
    nonzero_coefficients = matrix_size**3
    zero_coefficients = tensor_coefficients - nonzero_coefficients

    if status == "VERIFIED":
        headline = "Exact matrix multiplication identity verified"
    elif status == "NUMERICAL_CANDIDATE":
        headline = "Numerical candidate found, but not proved"
    elif status == "NOT_FOUND":
        headline = "No candidate reached the configured threshold"
    else:
        headline = "The run ended with an execution error"

    example = _candidate_matrix_product(
        output_directory / "candidate.json",
        matrix_size,
    )

    overview = _table_html(
        [
            ("Problem", f"{matrix_size} × {matrix_size} matrix multiplication"),
            ("Naive bilinear products", naive_rank),
            ("Candidate bilinear products", rank),
            ("Products saved", f"{saved} ({_format_percentage(reduction_fraction)})"),
            ("Scientific status", status),
        ]
    )

    proof_status = verification.get("status", "NOT_ATTEMPTED")
    if proof_status == "VERIFIED":
        reconstruction_stage = "PASSED"
        exact_stage = "PASSED"
    elif int(search.get("numerical_candidates", 0)) > 0:
        reconstruction_stage = "ATTEMPTED"
        exact_stage = "NOT VERIFIED"
    else:
        reconstruction_stage = "NOT ATTEMPTED"
        exact_stage = "NOT ATTEMPTED"

    proof_rows = [
        ("Numerical candidates", search.get("numerical_candidates", 0)),
        ("Exact reconstruction", reconstruction_stage),
        ("Reconstruction proposals tested", search.get("reconstruction_attempts", 0)),
        ("Exact identity verification", exact_stage),
        ("Verified restart", (
            int(search["verified_restart"]) + 1
            if search.get("verified_restart") is not None
            else "none"
        )),
        ("Scientific conclusion", status),
    ]
    if verification.get("best_reconstruction_error") is not None:
        proof_rows.append(
            (
                "Best reconstruction coefficient error",
                _format_scientific(verification["best_reconstruction_error"]),
            )
        )
    proof_table = _table_html(proof_rows)

    identity_table = _table_html(
        [
            ("Tensor coefficients tested numerically", tensor_coefficients),
            ("Expected non-zero coefficients", nonzero_coefficients),
            ("Expected zero coefficients", zero_coefficients),
            ("Residual norm", _format_scientific(best.get("residual_norm"))),
            ("Relative residual", _format_scientific(best.get("relative_residual"))),
            ("Maximum coefficient error", _format_scientific(best.get("max_abs_residual"))),
            ("Candidate threshold", _format_scientific(search.get("candidate_tolerance"))),
        ]
    )

    search_table = _table_html(
        [
            ("Method", search.get("method", "not available")),
            ("Configured restarts", search.get("configured_restarts", "not available")),
            ("Completed restarts", search.get("completed_restarts", "not available")),
            ("Best numerical restart", (
                int(search["best_restart"]) + 1
                if search.get("best_restart") is not None
                else "not available"
            )),
            ("Numerical candidates", search.get("numerical_candidates", 0)),
            ("Reconstruction proposals tested", search.get("reconstruction_attempts", 0)),
            ("Termination", _termination_text(search)),
            ("Runtime", f"{_format_number(result.get('runtime_seconds'))} s"),
        ]
    )

    environment = result.get("environment", {})
    reproducibility_table = _table_html(
        [
            ("Run ID", result.get("run_id", "not available")),
            ("Project version", result.get("project_version", "not available")),
            ("Git commit", environment.get("git_commit") or "not available"),
            ("Python", environment.get("python", "not available")),
            ("NumPy", environment.get("numpy", "not available")),
            ("SciPy", environment.get("scipy", "not available")),
            ("Platform", environment.get("platform", "not available")),
            ("Machine", environment.get("machine", "not available")),
        ]
    )

    recursive_rows = []
    for level in range(1, 6):
        size = matrix_size**level
        recursive_rows.append(
            "<tr>"
            f"<td>{size:,}</td>"
            f"<td>{size**3:,}</td>"
            f"<td>{rank**level:,}</td>"
            "</tr>"
        )
    recursive_table = "".join(recursive_rows)

    images = "\n".join(
        "<figure>"
        f'<img src="{html.escape(name)}" alt="{html.escape(name)}">'
        "</figure>"
        for name in figures
    )

    verification_text = {
        "VERIFIED": (
            "A rational representative reconstructed from a numerical restart passed "
            "independent exact coefficient-by-coefficient verification."
        ),
        "NUMERICAL_CANDIDATE": (
            "At least one restart reached the numerical threshold, but none of the "
            "configured rational reconstruction proposals passed exact verification. "
            "This is not a proof."
        ),
        "NOT_FOUND": (
            "No restart reached the numerical candidate threshold. This describes "
            "only this finite search and does not establish non-existence."
        ),
        "ERROR": (
            "The run ended before a scientific conclusion could be produced."
        ),
    }.get(status, claim)

    document = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>fast_mm result</title>
<style>
html {{ font-size: 16px; }}
body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
    max-width: 900px;
    margin: 48px auto;
    padding: 0 24px 80px;
    line-height: 1.55;
    color: #1d1d1f;
    background: #fff;
}}
h1 {{ margin: 0 0 0.4rem; font-size: 2rem; line-height: 1.15; font-weight: 600; }}
h2 {{ margin: 3rem 0 1rem; padding-top: 1.25rem; border-top: 1px solid #d2d2d7; font-size: 1.45rem; }}
h3 {{ margin: 1.8rem 0 0.6rem; font-size: 1.08rem; }}
p {{ margin: 0.8rem 0; }}
a {{ color: #06c; }}
.subtitle, .secondary {{ color: #6e6e73; }}
.subtitle {{ margin: 0; font-size: 1.05rem; }}
.result-heading {{ margin: 3rem 0 0.5rem; padding-top: 1.25rem; border-top: 1px solid #d2d2d7; font-size: 1.5rem; font-weight: 600; }}
.status {{ font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; color: #6e6e73; overflow-wrap: anywhere; }}
.equation {{ font-family: ui-serif, Georgia, "Times New Roman", serif; font-size: 1.08rem; }}
table {{ width: 100%; margin: 1.25rem 0; border-collapse: collapse; }}
th, td {{ padding: 0.65rem 0.4rem; border-bottom: 1px solid #e5e5e7; text-align: left; vertical-align: top; overflow-wrap: anywhere; }}
th {{ width: 42%; padding-left: 0; font-weight: 500; color: #515154; }}
.equation-table {{ max-width: 620px; }}
.equation-table th {{ width: 16%; color: #1d1d1f; }}
.matrix-row {{ display: flex; align-items: center; gap: 1.5rem; flex-wrap: wrap; }}
.matrix {{ display: inline-table; width: auto; margin: 0.25rem 0 1rem; border-left: 1px solid #86868b; border-right: 1px solid #86868b; font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }}
.symbolic-matrix {{ font-family: ui-serif, Georgia, "Times New Roman", serif; font-style: italic; }}
.matrix td {{ width: auto; padding: 0.3rem 0.65rem; border: 0; text-align: right; color: #1d1d1f; }}
.matrix-operator {{ margin-top: 1.4rem; font-size: 1.25rem; }}
figure {{ margin: 2rem 0; }}
img {{ display: block; max-width: 100%; height: auto; }}
code {{ font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; background: #f5f5f7; padding: 0.1rem 0.25rem; }}
</style>
</head>
<body>
<h1>fast_mm</h1>
<p class="subtitle">Search for finite matrix multiplication identities</p>
{_introduction_html()}

<p class="result-heading">Result for this run</p>
<p><strong>{html.escape(headline)}</strong></p>
<p class="status">{html.escape(status)}</p>
<p>{html.escape(claim)}</p>
<table>{overview}</table>

<h2>1. What was tested?</h2>
<p>
Ordinary {matrix_size} × {matrix_size} matrix multiplication uses
<strong>{naive_rank}</strong> bilinear products in the direct method. This run
searched for a representation using <strong>{rank}</strong>.
</p>
<p class="secondary">
The product count is an arithmetic-complexity quantity. It is not by itself a
wall-clock speed claim.
</p>

<h2>2. Worked example using the best numerical candidate</h2>
{_worked_example_html(example, rank)}

<h2>3. Complete multiplication identity</h2>
<p>
The numerical objective compares the candidate with the complete multiplication
tensor. Exact verification, when reached, checks every required coefficient with
exact rational arithmetic.
</p>
<table>{identity_table}</table>

<h2>4. Exactification and verification</h2>
<table>{proof_table}</table>
<p>{html.escape(verification_text)}</p>

<h2>5. Recursive interpretation</h2>
<p>
A verified {matrix_size} × {matrix_size} identity using {rank} products has
recursive arithmetic exponent <strong>{candidate_exponent:.8f}</strong>, given by
<code>log(rank) / log(matrix_size)</code>. The direct method has exponent 3.
</p>
<table>
<thead><tr><th>Matrix size</th><th>Naive recursive products</th><th>{rank}-product recursion</th></tr></thead>
<tbody>{recursive_table}</tbody>
</table>

<h2>6. Search behaviour</h2>
<table>{search_table}</table>
{images}

<h2>7. Reproducibility</h2>
<table>{reproducibility_table}</table>
<h3>Run artefacts</h3>
{_artifact_links(result, output_directory)}
<p class="secondary">
Every numerical restart is retained under <code>restarts/</code>. Exact files are
written only for identities that pass the independent verifier.
</p>

<h2>8. Technical interpretation</h2>
<p>{html.escape(claim)}</p>
<p>
The candidate arithmetic exponent describes recursive arithmetic complexity. It
does not imply the same improvement in practical CPU or GPU execution time.
</p>
</body>
</html>
"""

    path = output_directory / "summary.html"
    path.write_text(document, encoding="utf-8")
    return path

