
# ============================================================================
# PROJECT PURPOSE — KEEP THIS BLOCK
# ============================================================================

Reference:
https://github.com/openai/math/blob/main/preprints/Matrix-Multiplication-Nine-Fourths-October-2-2026/build/paper.tex

fast_mm is being built to automatically discover and exactly verify finite
matrix-multiplication rules that use unusually few scalar multiplications.

The key idea is that one efficient small rule can be reused recursively
inside larger matrices. Strassen's known 2 × 2 rule is our validation
benchmark: it uses 7 scalar multiplications instead of 8. Recursion turns
this into 49 instead of 64 multiplications for 4 × 4 matrices, and 343
instead of 512 for 8 × 8 matrices.

The 2 × 2, rank-7 Strassen rule is NOT the final research target. It is used
to prove that the complete discovery -> exactification -> verification
pipeline works without being given the known solution during discovery.

The research goal is to discover and exactly verify a new finite (u, r)
rule with a better asymptotic exponent log_u(r). The OpenAI 9/4 result
proves that sufficiently efficient finite exact decompositions exist
asymptotically, but does not specify a competitive finite matrix size u
or provide the explicit coefficients that fast_mm is intended to search for.

Once a new rule is found, the same small recipe can be applied recursively
to larger matrix blocks, so a saving at the base level compounds at every
recursive level. A later engineering stage will test whether that theoretical
advantage also produces faster practical matrix multiplication on real
hardware.

Long-running research searches must be resumable: completed results, seeds,
candidates and verification outcomes must be preserved so a campaign can be
stopped and later continued without losing or repeating completed work.

============================================================================

