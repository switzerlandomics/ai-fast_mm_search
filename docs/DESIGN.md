# Design

The project keeps four concerns separate:

1. `search.py` performs floating-point numerical optimisation.
2. `reconstruction.py` proposes exact representatives for promising numerical candidates.
3. `verify.py` independently checks exact identities.
4. `report.py` reports results without influencing computation.

A numerical threshold is a gate into exactification, not a success condition. Search termination may be requested by the controller after exact verification succeeds.

The current exactification method is bounded rational reconstruction with several denominator limits and optional term-scale normalisation. It is intentionally labelled as one strategy, not as a complete solution to exact recovery. Continuous equivalences between matrix-multiplication decompositions can move an exact algorithm away from small rational coordinates.

The next research stage should only begin after the benchmark, verifier, search control and artefact tests pass. Larger experiments must specify a concrete matrix size and target rank. Search failure is never interpreted as non-existence.
