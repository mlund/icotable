# icotable

Icosphere-based 6D angular lookup tables for rigid molecule-molecule interactions.

## Overview

`icotable` stores pre-computed pairwise energies over six degrees of freedom
(R, &omega;, &theta;&#x2081;&phi;&#x2081;, &theta;&#x2082;&phi;&#x2082;) using icosphere tessellation for the
angular dimensions. Barycentric interpolation on the spherical mesh gives smooth energy
surfaces from a discrete set of vertex samples.

The crate provides:

- **`Table6DAdaptive<T>`** &mdash; adaptive 6D table with per-slab angular resolution.
  Slabs are classified into four tiers based on angular gradients and Boltzmann weights:
  *repulsive* (exp(&minus;&beta;U) &approx; 0, zero storage), *scalar* (single mean value),
  *nearest-vertex* (no interpolation), or *interpolated* (full barycentric).
  Includes Voronoi quadrature weights for correct angular integration.
  The repulsive classification is temperature-dependent.
- **`AdaptiveBuilder`** &mdash; drives the table generation protocol, taking
  &beta; = 1/kT to classify repulsive slabs and angular gradients to decide
  resolution and interpolation tier.
- **`Table6DFlat<T>`** &mdash; flat, bincode-serializable 6D representation with uniform
  angular resolution. Generic over `f32` or `f16` (half-precision via
  [`half`](https://crates.io/crates/half)). BFS vertex reordering improves cache locality.
- **`Table6D`** &mdash; nested periodic table over (R, &omega;) with `IcoTable4D` angular layers,
  used during table construction.
- **`Table3DFlat<T>`** &mdash; flat, bincode-serializable 3D table (R, &theta;, &phi;) for
  rigid body + single atom interactions.
- **`orient` / `inverse_orient`** &mdash; forward and inverse coordinate transforms between
  6D table indices and quaternion + separation-vector representation.
- **`IcoTable2D<T>`** &mdash; single-icosphere angular table with barycentric interpolation,
  useful for mapping scalar fields on S&sup2;.
- Icosphere mesh generation and vertex/face utilities built on
  [`hexasphere`](https://crates.io/crates/hexasphere).

## Usage

This crate is a shared dependency between:

- [**Duello**](https://github.com/mlund/duello) &mdash; computes 6D energy tables by scanning
  all orientations of two rigid molecules using `AdaptiveBuilder`,
  and 3D tables for rigid body + single atom interactions (`Table3DFlat`).
- [**Faunus**](https://github.com/mlund/faunus-rs) &mdash; loads pre-computed tables for
  O(1) energy lookups during Monte Carlo simulations. Supports both adaptive
  and flat table formats with automatic format detection.

## License

Apache-2.0
