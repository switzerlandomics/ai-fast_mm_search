# Validation

The corrected snapshot was validated on 7 October 2026.

## Automated tests

```text
25 passed
```

The suite covers tensor construction, the analytic optimisation gradient, configuration validation, exact Strassen verification, deliberate corruption, rational reconstruction proposals, restart continuation after a numerical candidate, callback-controlled early stopping, result status classification, and HTML report rendering.

## Execution checks

The stored exact Strassen identity passes with zero mismatches.

The short benchmark completes both configured restarts when no exact identity is verified.

The longer benchmark completes all eight configured restarts when no exact identity is verified. In the validation run, seven restarts crossed the numerical candidate threshold and 70 bounded-rational reconstruction proposals were checked exactly. None passed exact verification, so the scientifically correct status was `NUMERICAL_CANDIDATE`.

This confirms the corrected control flow: crossing the numerical threshold triggers exactification but does not terminate the search. Early termination is reserved for an independently verified exact identity when `stop_on_verified` is enabled.
