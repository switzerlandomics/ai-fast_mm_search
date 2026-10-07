# fast_mm

`fast_mm` searches for explicit finite matrix-multiplication algorithms, reconstructs promising numerical solutions as exact coefficients, and verifies the resulting identities independently.

> **TLDR:** run `run_fast_mm.py` to execute or resume a search campaign. A campaign continues through independent random starts until an exact identity is verified or the configured search budget is exhausted. After a verified discovery, run `run_result_summary.py` to generate a human-readable HTML description of the recovered algorithm.

## Status

The first validation phase is complete.

Starting from random coefficients and without being given Strassen's formula, `fast_mm` independently recovered and exactly verified a rank-7 algorithm for $$2\times2$$ matrix multiplication. The recovered equations differ from the familiar textbook form of Strassen's algorithm while representing the same optimal seven-product result.

This establishes the complete discovery pipeline:

```text
random coefficients
        ↓
numerical optimisation
        ↓
candidate threshold
        ↓
exact reconstruction
        ↓
independent exact verification
        ↓
VERIFIED
```

A low floating-point residual is never accepted as proof. `VERIFIED` requires an exact matrix-multiplication identity checked coefficient by coefficient.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
```

## Main workflow

The simplest entry point is:

```bash
python run_fast_mm.py
```

`run_fast_mm.py` is the project runner for tests, baseline verification, numerical benchmarks and resumable campaigns. The selected action and configuration are set near the top of the file.

The principal validation campaign searches independent random initialisations until an exact identity is found. Completed attempts are retained, so the campaign can be stopped and resumed without repeating finished seeds.

After a verified result, generate the readable algorithm summary with:

```bash
python run_result_summary.py
```

By default, the summary script selects the recorded verified run. A specific run directory can also be supplied manually near the top of the script.

The generated report includes the recovered multiplication rule, exact verification result, recursive arithmetic interpretation, a standard-versus-discovered worked example and a small Python implementation of the algorithm.

## Basic checks

Verify the tracked exact Strassen baseline:

```bash
fast-mm verify results/verified/strassen_2x2.json
```

Run the test suite:

```bash
pytest
```

Run a short numerical benchmark:

```bash
fast-mm run configs/strassen_quick.json
```

Run the longer rank-7 benchmark:

```bash
fast-mm run configs/strassen_search.json
```

The search evaluates every restart that reaches the numerical candidate threshold. A failed reconstruction attempt does not terminate the run. Early termination occurs only after exact verification when `stop_on_verified` is enabled.

## Validation result

Ordinary $$2\times2$$ matrix multiplication uses eight scalar products. Strassen showed that seven are sufficient.

The validation search was deliberately not supplied with Strassen's known coefficients. Independent random starts were optimised against the complete matrix-multiplication tensor until a candidate could be reconstructed exactly.

The successful campaign reached `VERIFIED` at seed 581. The numerical precursor had relative residual

$$
3.09\times10^{-8},
$$

after which exact reconstruction produced an integer-coefficient identity. Independent verification reported zero mismatches and zero exact residual.

This result validates the numerical search, exact reconstruction and verification pipeline. It does not constitute a new $$2\times2$$ multiplication bound; the recovered algorithm belongs to the known optimal rank-7 family.

## Result states

`VERIFIED` means an exact identity was reconstructed and independently verified.

`NUMERICAL_CANDIDATE` means at least one restart reached the configured numerical threshold, but no exact identity passed verification.

`NOT_FOUND` means no restart reached the numerical threshold within the configured search. It is not evidence that an identity does not exist.

`ERROR` means execution failed before a scientific result was produced.

## Run artefacts

Each search run records its numerical candidate, configuration, metrics and reproducibility information. Successful exact verification adds the exact candidate and verification record.

```text
config.json
candidate.json
candidate_rational.json      # VERIFIED runs only
verification.json            # VERIFIED runs only
result.json
metrics.jsonl
convergence.png
restarts.png
summary.html
discovery_summary.html       # generated after a verified discovery
restarts/
```

Every numerical restart is retained under `restarts/`. Exact root-level candidate files are written only after independent verification succeeds.

## Why exact reconstruction is separate

Matrix-multiplication decompositions possess continuous scaling and change-of-representation symmetries. A highly accurate numerical solution therefore need not lie close, coefficient by coefficient, to a simple rational representation.

Numerical optimisation answers whether a low-residual decomposition has been found. Exact reconstruction searches for a simple exact representative. Independent verification then determines whether that representative is a true matrix-multiplication identity.

These are deliberately separate stages.

## Recursive interpretation

For an exact fixed-size algorithm multiplying $$u\times u$$ matrices with $$r$$ bilinear products, recursive application gives arithmetic exponent

$$
\log_u r.
$$

For the validated $$u=2,\ r=7$$ case,

$$
\log_2 7 \approx 2.807354922.
$$

This measures arithmetic multiplication complexity. It does not by itself imply an equivalent wall-clock improvement on CPUs or GPUs.

## Scientific scope

The project is motivated by the search for explicit finite matrix-multiplication algorithms with better recursive complexity.

The recent OpenAI result [*An Upper Bound of 9/4 for the Matrix Multiplication Exponent*](https://github.com/openai/math/blob/main/preprints/Matrix-Multiplication-Nine-Fourths-October-2-2026/paper.pdf) proves

$$
\omega \leq \frac{9}{4}
$$

over $$\mathbb{C}$$. The theorem establishes an asymptotic upper bound but does not provide a small competitive finite $$(u,r)$$ rule with explicit coefficients.

The completed $$2\times2$$, rank-7 campaign was therefore a validation target rather than the final research problem.

The next phase is to apply the same search, reconstruction and verification machinery to finite $$(u,r)$$ targets where an improved explicit construction is not already known.

The current exact verifier supports rational coefficients. Future searches involving algebraic or complex exact coefficients should extend the coefficient domain explicitly rather than assuming every exact decomposition has a small rational representation.
