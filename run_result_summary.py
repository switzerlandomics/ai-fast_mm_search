# -*- coding: utf-8 -*-

"""
Create a human-readable HTML summary for a verified fast_mm discovery.

Selection order:
1. RUN_DIR, when set manually below.
2. CAMPAIGN_STATE["verified_run"], when available.
3. Best VERIFIED result found under runs/, ranked by arithmetic exponent.

The exact candidate is independently verified again before any report is written.
"""

from __future__ import annotations

import html
import json
import math
from fractions import Fraction
from pathlib import Path
from typing import Any

from fast_mm.verify import verify_exact


PROJECT_DIR = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# User settings
# ---------------------------------------------------------------------------

# Manual run directory. Leave as None to select automatically.
#
# Example:
# RUN_DIR = (
#     "runs/campaigns/strassen_validation/attempts/seed_00000581/"
#     "20261007T131948890708Z_u2_r7_06557cb6"
# )
RUN_DIR: str | None = None

# Preferred campaign state for automatic selection.
CAMPAIGN_STATE = "runs/campaigns/strassen_validation/state.json"

# None writes discovery_summary.html inside the selected run directory.
OUTPUT_FILE: str | None = None

# Number of recursive levels shown in the arithmetic comparison table.
RECURSIVE_LEVELS = 5


# ---------------------------------------------------------------------------
# File handling
# ---------------------------------------------------------------------------

def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_DIR / path


def validate_run_directory(path: Path) -> Path:
    path = path.resolve()
    required = (
        "result.json",
        "candidate_rational.json",
        "verification.json",
        "config.json",
    )
    missing = [name for name in required if not (path / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"{path} is not a complete verified run directory. "
            f"Missing: {', '.join(missing)}"
        )

    result = read_json(path / "result.json")
    if result.get("status") != "VERIFIED":
        raise ValueError(
            f"{path} has status {result.get('status')!r}, not 'VERIFIED'."
        )
    return path


def automatic_run_directory() -> Path:
    state_path = project_path(CAMPAIGN_STATE)

    if state_path.is_file():
        state = read_json(state_path)
        verified_run = state.get("verified_run")
        if state.get("verified") and verified_run:
            return validate_run_directory(project_path(verified_run))

    candidates: list[tuple[float, float, Path]] = []

    for result_path in PROJECT_DIR.glob("runs/**/result.json"):
        try:
            result = read_json(result_path)
        except (OSError, json.JSONDecodeError):
            continue

        if result.get("status") != "VERIFIED":
            continue

        exponent = result.get("comparison", {}).get(
            "multiplication_exponent",
            math.inf,
        )
        try:
            exponent = float(exponent)
        except (TypeError, ValueError):
            exponent = math.inf

        # Lower exponent is better. For equal exponents, prefer the newest run.
        candidates.append(
            (
                exponent,
                -result_path.stat().st_mtime,
                result_path.parent,
            )
        )

    if not candidates:
        raise FileNotFoundError(
            "No VERIFIED fast_mm run was found under runs/."
        )

    candidates.sort(key=lambda item: (item[0], item[1]))
    return validate_run_directory(candidates[0][2])


def select_run_directory() -> Path:
    if RUN_DIR:
        return validate_run_directory(project_path(RUN_DIR))
    return automatic_run_directory()


# ---------------------------------------------------------------------------
# Exact decomposition handling
# ---------------------------------------------------------------------------

def exact_rows(
    values: list[list[Any]],
) -> list[list[Fraction]]:
    return [
        [Fraction(value) for value in row]
        for row in values
    ]


def first_nonzero_sign(values: list[Fraction]) -> int:
    for value in values:
        if value > 0:
            return 1
        if value < 0:
            return -1
    return 1


def canonical_display_factors(
    payload: dict[str, Any],
) -> tuple[
    list[list[Fraction]],
    list[list[Fraction]],
    list[list[Fraction]],
]:
    """
    Apply sign-only gauge transformations for cleaner display.

    Each tensor term L * R * O is unchanged because any sign moved into
    L or R is compensated in O.
    """
    factors = payload["factors"]
    left = exact_rows(factors["left"])
    right = exact_rows(factors["right"])
    output = exact_rows(factors["output"])

    for term in range(len(left)):
        left_sign = first_nonzero_sign(left[term])
        right_sign = first_nonzero_sign(right[term])

        if left_sign < 0:
            left[term] = [-value for value in left[term]]

        if right_sign < 0:
            right[term] = [-value for value in right[term]]

        output_scale = left_sign * right_sign
        if output_scale < 0:
            output[term] = [-value for value in output[term]]

    return left, right, output


def variable_labels(
    matrix_size: int,
) -> tuple[list[str], list[str], list[str]]:
    """Plain labels used in code and text output."""
    if matrix_size == 2:
        return (
            ["a", "b", "c", "d"],
            ["e", "f", "g", "h"],
            ["C11", "C12", "C21", "C22"],
        )

    left = []
    right = []
    output = []

    for row in range(1, matrix_size + 1):
        for column in range(1, matrix_size + 1):
            left.append(f"A{row}{column}")
            right.append(f"B{row}{column}")
            output.append(f"C{row}{column}")

    return left, right, output


def latex_variable_labels(
    matrix_size: int,
) -> tuple[list[str], list[str], list[str]]:
    """LaTeX labels used in the article-style algorithm display."""
    if matrix_size == 2:
        return (
            ["a", "b", "c", "d"],
            ["e", "f", "g", "h"],
            [r"C_{11}", r"C_{12}", r"C_{21}", r"C_{22}"],
        )

    left = []
    right = []
    output = []

    for row in range(1, matrix_size + 1):
        for column in range(1, matrix_size + 1):
            left.append(rf"A_{{{row}{column}}}")
            right.append(rf"B_{{{row}{column}}}")
            output.append(rf"C_{{{row}{column}}}")

    return left, right, output


def fraction_text(value: Fraction) -> str:
    if value.denominator == 1:
        return str(value.numerator)
    return f"{value.numerator}/{value.denominator}"


def linear_form_text(
    coefficients: list[Fraction],
    labels: list[str],
) -> str:
    terms: list[str] = []

    for coefficient, label in zip(coefficients, labels):
        if coefficient == 0:
            continue

        sign = -1 if coefficient < 0 else 1
        magnitude = abs(coefficient)

        if magnitude == 1:
            body = label
        else:
            body = f"{fraction_text(magnitude)}*{label}"

        if not terms:
            terms.append(f"-{body}" if sign < 0 else body)
        else:
            terms.append(
                f" - {body}" if sign < 0 else f" + {body}"
            )

    return "".join(terms) if terms else "0"



def fraction_latex(value: Fraction) -> str:
    value = Fraction(value)
    if value.denominator == 1:
        return str(value.numerator)
    return rf"\frac{{{value.numerator}}}{{{value.denominator}}}"


def linear_form_latex(
    coefficients: list[Fraction],
    labels: list[str],
) -> str:
    """Render an exact linear form as readable LaTeX."""
    terms: list[str] = []

    for coefficient, label in zip(coefficients, labels):
        if coefficient == 0:
            continue

        sign = -1 if coefficient < 0 else 1
        magnitude = abs(coefficient)

        if magnitude == 1:
            body = label
        else:
            body = rf"{fraction_latex(magnitude)}\,{label}"

        if not terms:
            terms.append(f"-{body}" if sign < 0 else body)
        else:
            terms.append(
                f" - {body}" if sign < 0 else f" + {body}"
            )

    return "".join(terms) if terms else "0"


def matrix_latex(
    entries: list[str],
    matrix_size: int,
) -> str:
    rows = []
    for row in range(matrix_size):
        start = row * matrix_size
        rows.append(" & ".join(entries[start:start + matrix_size]))
    return r"\begin{pmatrix}" + r" \\ ".join(rows) + r"\end{pmatrix}"


def product_formulas_latex(
    left: list[list[Fraction]],
    right: list[list[Fraction]],
    left_labels: list[str],
    right_labels: list[str],
) -> list[str]:
    formulas = []

    for index, (left_row, right_row) in enumerate(
        zip(left, right),
        start=1,
    ):
        left_text = linear_form_latex(left_row, left_labels)
        right_text = linear_form_latex(right_row, right_labels)
        formulas.append(
            rf"P_{{{index}}} = \left({left_text}\right)\left({right_text}\right)"
        )

    return formulas


def output_expressions_latex(
    output: list[list[Fraction]],
) -> list[str]:
    rank = len(output)
    output_count = len(output[0]) if output else 0
    expressions = []

    for output_index in range(output_count):
        coefficients = [
            output[term][output_index]
            for term in range(rank)
        ]
        products = [
            rf"P_{{{term + 1}}}"
            for term in range(rank)
        ]
        expressions.append(
            linear_form_latex(coefficients, products)
        )

    return expressions


def standard_output_expressions_latex(
    matrix_size: int,
    left_labels: list[str],
    right_labels: list[str],
) -> list[str]:
    expressions = []

    for row in range(matrix_size):
        for column in range(matrix_size):
            terms = []
            for shared in range(matrix_size):
                left_label = left_labels[row * matrix_size + shared]
                right_label = right_labels[shared * matrix_size + column]
                terms.append(rf"{left_label}{right_label}")
            expressions.append(" + ".join(terms))

    return expressions


def display_math(expression: str) -> str:
    return f'<div class="math-block">\\[{expression}\\]</div>'


def aligned_math(formulas: list[str]) -> str:
    body = r" \\ ".join(formulas)
    return display_math(r"\begin{aligned}" + body + r"\end{aligned}")


def product_formulas(
    left: list[list[Fraction]],
    right: list[list[Fraction]],
    left_labels: list[str],
    right_labels: list[str],
) -> list[str]:
    formulas = []

    for index, (left_row, right_row) in enumerate(
        zip(left, right),
        start=1,
    ):
        left_text = linear_form_text(left_row, left_labels)
        right_text = linear_form_text(right_row, right_labels)
        formulas.append(
            f"P{index} = ({left_text})({right_text})"
        )

    return formulas


def output_formulas(
    output: list[list[Fraction]],
    output_labels: list[str],
) -> list[str]:
    rank = len(output)
    formulas = []

    for output_index, label in enumerate(output_labels):
        coefficients = [
            output[term][output_index]
            for term in range(rank)
        ]
        products = [
            f"P{term + 1}"
            for term in range(rank)
        ]
        formulas.append(
            f"{label} = {linear_form_text(coefficients, products)}"
        )

    return formulas


# ---------------------------------------------------------------------------
# Exact worked example
# ---------------------------------------------------------------------------

def exact_dot(
    coefficients: list[Fraction],
    values: list[Fraction],
) -> Fraction:
    return sum(
        (coefficient * value for coefficient, value in zip(coefficients, values)),
        Fraction(0),
    )


def exact_example(
    matrix_size: int,
    left: list[list[Fraction]],
    right: list[list[Fraction]],
    output: list[list[Fraction]],
) -> dict[str, Any]:
    dimension = matrix_size * matrix_size

    left_values = [
        Fraction(value)
        for value in range(1, dimension + 1)
    ]
    right_values = [
        Fraction(value)
        for value in range(dimension + 1, 2 * dimension + 1)
    ]

    products = [
        exact_dot(left_row, left_values)
        * exact_dot(right_row, right_values)
        for left_row, right_row in zip(left, right)
    ]

    discovered = [
        sum(
            (
                output[term][index] * products[term]
                for term in range(len(products))
            ),
            Fraction(0),
        )
        for index in range(dimension)
    ]

    expected: list[Fraction] = []

    for row in range(matrix_size):
        for column in range(matrix_size):
            value = sum(
                (
                    left_values[row * matrix_size + shared]
                    * right_values[shared * matrix_size + column]
                    for shared in range(matrix_size)
                ),
                Fraction(0),
            )
            expected.append(value)

    if discovered != expected:
        raise RuntimeError(
            "The displayed exact decomposition failed the worked example."
        )

    return {
        "left": left_values,
        "right": right_values,
        "products": products,
        "discovered": discovered,
        "expected": expected,
    }


# ---------------------------------------------------------------------------
# Python validation snippet
# ---------------------------------------------------------------------------

def python_fraction(value: Fraction) -> str:
    if value.denominator == 1:
        return str(value.numerator)
    return f"F({value.numerator}, {value.denominator})"


def python_matrix(values: list[list[Fraction]]) -> str:
    rows = []
    for row in values:
        rows.append(
            "    ["
            + ", ".join(python_fraction(value) for value in row)
            + "]"
        )
    return "[\n" + ",\n".join(rows) + "\n]"


def validation_code(
    matrix_size: int,
    left: list[list[Fraction]],
    right: list[list[Fraction]],
    output: list[list[Fraction]],
) -> str:
    dimension = matrix_size * matrix_size
    a_values = list(range(1, dimension + 1))
    b_values = list(range(dimension + 1, 2 * dimension + 1))

    return f"""from fractions import Fraction as F

u = {matrix_size}
L = {python_matrix(left)}
R = {python_matrix(right)}
O = {python_matrix(output)}


def standard_mm(A, B):
    return [
        [
            sum(F(A[i][k]) * F(B[k][j]) for k in range(u))
            for j in range(u)
        ]
        for i in range(u)
    ]


def discovered_mm(A, B):
    a = [F(x) for row in A for x in row]
    b = [F(x) for row in B for x in row]

    products = [
        sum(L[k][i] * a[i] for i in range(u * u))
        * sum(R[k][i] * b[i] for i in range(u * u))
        for k in range(len(L))
    ]

    c = [
        sum(O[k][j] * products[k] for k in range(len(L)))
        for j in range(u * u)
    ]

    return [c[i * u:(i + 1) * u] for i in range(u)]


def show_matrix(name, matrix):
    print(name)
    for row in matrix:
        print("   ", [str(value) for value in row])


A = {[a_values[i:i + matrix_size] for i in range(0, dimension, matrix_size)]}
B = {[b_values[i:i + matrix_size] for i in range(0, dimension, matrix_size)]}

standard_result = standard_mm(A, B)
discovered_result = discovered_mm(A, B)

show_matrix("A", A)
show_matrix("B", B)
show_matrix("Standard multiplication A @ B", standard_result)
show_matrix("Discovered algorithm", discovered_result)

print("Match:", standard_result == discovered_result)
assert discovered_result == standard_result
"""


# ---------------------------------------------------------------------------
# HTML helpers
# ---------------------------------------------------------------------------

def table(rows: list[tuple[str, Any]]) -> str:
    return (
        "<table>"
        + "".join(
            "<tr>"
            f"<th>{html.escape(str(label))}</th>"
            f"<td>{html.escape(str(value))}</td>"
            "</tr>"
            for label, value in rows
        )
        + "</table>"
    )


def matrix_html(
    values: list[Fraction],
    matrix_size: int,
) -> str:
    rows = []

    for row in range(matrix_size):
        cells = []
        for column in range(matrix_size):
            value = values[row * matrix_size + column]
            cells.append(
                f"<td>{html.escape(fraction_text(value))}</td>"
            )
        rows.append("<tr>" + "".join(cells) + "</tr>")

    return '<table class="matrix">' + "".join(rows) + "</table>"


def formula_table(
    formulas: list[str],
) -> str:
    rows = []

    for formula in formulas:
        name, expression = formula.split(" = ", 1)
        rows.append(
            "<tr>"
            f"<th>{html.escape(name)}</th>"
            f"<td><code>{html.escape(expression)}</code></td>"
            "</tr>"
        )

    return '<table class="formula">' + "".join(rows) + "</table>"


def recursive_table(
    matrix_size: int,
    rank: int,
) -> str:
    rows = []

    for level in range(1, RECURSIVE_LEVELS + 1):
        size = matrix_size**level
        naive = size**3
        discovered = rank**level
        rows.append(
            "<tr>"
            f"<td>{size:,}</td>"
            f"<td>{naive:,}</td>"
            f"<td>{discovered:,}</td>"
            f"<td>{100 * (1 - discovered / naive):.1f}%</td>"
            "</tr>"
        )

    return "".join(rows)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def build_report(run_directory: Path) -> str:
    result = read_json(run_directory / "result.json")
    config = read_json(run_directory / "config.json")
    exact_candidate = read_json(
        run_directory / "candidate_rational.json"
    )
    stored_verification = read_json(
        run_directory / "verification.json"
    )

    fresh_verification = verify_exact(exact_candidate)

    if not fresh_verification["passed"]:
        raise RuntimeError(
            "Independent exact verification failed. "
            "No discovery report was written."
        )

    if stored_verification.get("passed") is not True:
        raise RuntimeError(
            "Stored verification.json is not marked as passed."
        )

    matrix_size = int(exact_candidate["matrix_size"])
    rank = int(exact_candidate["rank"])
    naive_rank = matrix_size**3
    exponent = math.log(rank) / math.log(matrix_size)
    strassen_exponent = math.log(7) / math.log(2)

    left, right, output = canonical_display_factors(
        exact_candidate
    )

    left_labels, right_labels, output_labels = variable_labels(
        matrix_size
    )

    products = product_formulas(
        left,
        right,
        left_labels,
        right_labels,
    )
    outputs = output_formulas(
        output,
        output_labels,
    )

    latex_left_labels, latex_right_labels, latex_output_labels = (
        latex_variable_labels(matrix_size)
    )
    latex_products = product_formulas_latex(
        left,
        right,
        latex_left_labels,
        latex_right_labels,
    )
    latex_output_expressions = output_expressions_latex(output)
    latex_standard_expressions = standard_output_expressions_latex(
        matrix_size,
        latex_left_labels,
        latex_right_labels,
    )
    latex_left_matrix = matrix_latex(
        latex_left_labels,
        matrix_size,
    )
    latex_right_matrix = matrix_latex(
        latex_right_labels,
        matrix_size,
    )
    latex_output_matrix = matrix_latex(
        latex_output_expressions,
        matrix_size,
    )
    latex_standard_matrix = matrix_latex(
        latex_standard_expressions,
        matrix_size,
    )

    example = exact_example(
        matrix_size,
        left,
        right,
        output,
    )

    seed = config.get("search", {}).get("seed")
    residual = result.get("best", {}).get("relative_residual")
    reconstruction_attempts = result.get(
        "search",
        {},
    ).get("reconstruction_attempts")

    if exponent < strassen_exponent - 1e-12:
        comparison = (
            "This verified rule has a lower recursive arithmetic exponent "
            "than Strassen's 2 × 2, rank-7 rule."
        )
    elif abs(exponent - strassen_exponent) <= 1e-12:
        comparison = (
            "This verified rule matches the recursive arithmetic exponent "
            "of Strassen's 2 × 2, rank-7 construction."
        )
    else:
        comparison = (
            "This verified rule does not improve on Strassen's recursive "
            "arithmetic exponent."
        )

    source_path = str(run_directory)
    try:
        source_path = str(
            run_directory.relative_to(PROJECT_DIR)
        )
    except ValueError:
        pass

    validation = validation_code(
        matrix_size,
        left,
        right,
        output,
    )

    # Precompute MathJax fragments before the large HTML f-string.
    # Python 3.11 does not allow backslashes inside f-string expressions.
    input_matrices_math = display_math(
        "A=" + latex_left_matrix + r",\qquad B=" + latex_right_matrix
    )
    standard_product_math = display_math(
        "AB=" + latex_standard_matrix
    )
    discovered_products_math = aligned_math(
        latex_products
    )
    discovered_output_math = display_math(
        "C=AB=" + latex_output_matrix
    )

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>fast_mm verified discovery</title>
<script>
window.MathJax = {{
    tex: {{
        inlineMath: [['\\(', '\\)']],
        displayMath: [['\\[', '\\]']]
    }}
}};
</script>
<script defer src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
<style>
html {{ font-size: 16px; }}
body {{
    max-width: 900px;
    margin: 48px auto;
    padding: 0 24px 80px;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
    color: #1d1d1f;
    line-height: 1.55;
}}
h1 {{ font-size: 2rem; margin-bottom: .4rem; }}
h2 {{ font-size: 1.4rem; margin-top: 2.8rem; padding-top: 1rem; border-top: 1px solid #ddd; }}
p {{ margin: .8rem 0; }}
.status {{ font-weight: 600; }}
.secondary {{ color: #6e6e73; }}
table {{ border-collapse: collapse; width: 100%; margin: 1rem 0 1.5rem; }}
th, td {{ padding: .55rem .4rem; border-bottom: 1px solid #e5e5e7; text-align: left; vertical-align: top; }}
th {{ width: 36%; font-weight: 500; }}
.formula th {{ width: 12%; }}
code, pre {{ font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }}
pre {{ overflow-x: auto; background: #f5f5f7; padding: 1rem; }}
.matrix {{ width: auto; display: inline-table; border-left: 1px solid #888; border-right: 1px solid #888; }}
.matrix td {{ border: 0; text-align: right; min-width: 2.2rem; }}
.matrix-row {{ display: flex; gap: 1.25rem; align-items: center; flex-wrap: wrap; }}
.math-block {{ overflow-x: auto; margin: 1.1rem 0; }}
.algorithm-step {{ margin: 1.8rem 0; }}
.algorithm-step h3 {{ margin-bottom: .5rem; }}
.result-pair {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 1.5rem; }}
.result-card {{ padding: 0 1rem 1rem; border: 1px solid #e5e5e7; border-radius: 8px; }}
</style>
</head>
<body>

<h1>fast_mm verified discovery</h1>
<p class="status">VERIFIED</p>
<p>
An exact {matrix_size} × {matrix_size} matrix-multiplication identity using
<strong>{rank}</strong> scalar products was independently verified with exact
rational arithmetic.
</p>

{table([
    ("Matrix size", f"{matrix_size} × {matrix_size}"),
    ("Naive scalar products", naive_rank),
    ("Discovered scalar products", rank),
    ("Products saved at the base size", naive_rank - rank),
    ("Recursive arithmetic exponent", f"{exponent:.9f}"),
    ("Search seed", seed),
    ("Numerical residual before exactification", f"{float(residual):.6e}" if residual is not None else "not available"),
    ("Exact reconstruction attempts", reconstruction_attempts),
    ("Exact verifier mismatches", fresh_verification["mismatch_count"]),
])}

<p>{html.escape(comparison)}</p>

<h2>The discovered algorithm</h2>
<p>
Write the two input matrices as
</p>

{input_matrices_math}

<p>
Ordinary matrix multiplication would produce
</p>

{standard_product_math}

<p>
The discovered rule reaches the same result using
<strong>{rank} bilinear products</strong> instead of {naive_rank}.
</p>

<div class="algorithm-step">
<h3>Step 1. Compute the {rank} products</h3>
{discovered_products_math}
</div>

<div class="algorithm-step">
<h3>Step 2. Assemble the output matrix</h3>
{discovered_output_math}
</div>

<p>
Those two steps are the complete discovered multiplication algorithm.
The equations are generated directly from the exact verified coefficient
matrices, so the same presentation works for any future verified
<em>u</em> × <em>u</em>, rank-<em>r</em> discovery.
</p>

<h2>A concrete check</h2>
<p>For the example below, ordinary multiplication gives:</p>

<div class="matrix-row">
<div>{matrix_html(example["left"], matrix_size)}</div>
<div>×</div>
<div>{matrix_html(example["right"], matrix_size)}</div>
<div>=</div>
<div>{matrix_html(example["expected"], matrix_size)}</div>
</div>

<p>The discovered algorithm gives:</p>

<div class="matrix-row">
<div>{matrix_html(example["discovered"], matrix_size)}</div>
</div>

<p><strong>The two results are exactly identical.</strong></p>

<p class="secondary">
This numerical example is only illustrative. The proof is the exact
coefficient-by-coefficient verification recorded above.
</p>

<h2>Why the small rule matters</h2>
<p>
The same fixed rule can be applied recursively to blocks of larger matrices.
The arithmetic multiplication counts therefore compound.
</p>

<table>
<thead>
<tr>
<th>Matrix size</th>
<th>Naive products</th>
<th>Discovered-rule products</th>
<th>Reduction</th>
</tr>
</thead>
<tbody>
{recursive_table(matrix_size, rank)}
</tbody>
</table>

<h2>Standard multiplication versus the discovered algorithm</h2>
<p>
The code below implements the ordinary matrix product and the discovered rule
separately, evaluates both on the same matrices, prints both results, and
checks that they are exactly equal. <code>Fraction</code> keeps rational
coefficients exact.
</p>

<pre>{html.escape(validation)}</pre>

<h2>Verification and provenance</h2>
{table([
    ("Status", result.get("status")),
    ("Verification method", fresh_verification.get("method")),
    ("Mismatch count", fresh_verification.get("mismatch_count")),
    ("Maximum exact residual", fresh_verification.get("max_abs_residual")),
    ("Run ID", result.get("run_id")),
    ("Source run", source_path),
    ("Project version", result.get("project_version")),
])}

<p class="secondary">
Arithmetic-product counts and recursive exponents are theoretical complexity
measures. They do not by themselves establish a wall-clock speed improvement
on a CPU or GPU.
</p>

</body>
</html>
"""


def main() -> None:
    run_directory = select_run_directory()

    output_path = (
        project_path(OUTPUT_FILE)
        if OUTPUT_FILE
        else run_directory / "discovery_summary.html"
    )

    document = build_report(run_directory)
    output_path.write_text(document, encoding="utf-8")

    print("fast_mm discovery summary")
    print(f"Run:    {run_directory}")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()

