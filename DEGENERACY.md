# Self-interaction table degeneracy

For identical molecules, swapping A↔B gives the same physical configuration but
`inverse_orient` maps it to two different points in (ω, dir_a, dir_b) space.
The table stores correct energies at all grid points, but interpolation between
grid points breaks the exchange symmetry.

## Current workaround (in faunus)

Average both lookups: `E = 0.5 * [lookup(sep, q_a, q_b) + lookup(-sep, q_b, q_a)]`.
This restores physical symmetry at the cost of two lookups per pair evaluation.

Tracking `|exp(-βU_fwd) - exp(-βU_rev)|` confirms the asymmetry decreases
with higher angular resolution, as expected for an interpolation artifact.

## Rejected approach: pre-symmetrize table in Duello

Averaging each grid point with its interpolated swap partner during table
generation (`E_sym = 0.5 * (E + E_partner)`). This fails because:

- Arithmetic mean in energy space corrupts the repulsive boundary: an accessible
  orientation averaged with a repulsive partner (~∞) becomes artificially blocked.
- Boltzmann-space averaging (`-ln(0.5*(exp(-βE₁)+exp(-βE₂)))/β`) would fix the
  repulsive problem but produces a free-energy-like quantity that doesn't compose
  correctly with the runtime Boltzmann-weighted interpolation (double nonlinear
  transform).
- Verified on trp-cage (N=10): symmetrized table shifts the RDF contact peak
  ~3 Å outward and reduces its height by ~20% compared to explicit nonbonded,
  while the non-symmetrized table matches almost exactly.

## Proposed fix: canonical coordinates

Make `inverse_orient` return a unique representative for each degenerate pair.
For self-interaction tables this would:

1. Eliminate the averaging (single lookup)
2. Enable half-storage tables — only the canonical half-space needs to be stored
3. Give exact exchange symmetry by construction

### Implementation

1. Compute both decompositions: (ω₁, dir_a₁, dir_b₁) and (ω₂, dir_a₂, dir_b₂)
2. Pick one via deterministic tie-breaker (e.g., lexicographic order)
3. In Duello: generate only canonical grid points, halving table size

Step 1-2 is straightforward in icotable. Step 3 requires Duello changes to
identify and skip redundant grid points near the symmetry boundary.
