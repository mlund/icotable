# icotable

Icosphere-based 6D angular lookup tables for rigid molecule-molecule interactions.

## Overview

`icotable` stores pre-computed pairwise energies over six degrees of freedom
(R, &omega;, &theta;&#x2081;&phi;&#x2081;, &theta;&#x2082;&phi;&#x2082;) using icosphere tessellation for the
angular dimensions. Barycentric interpolation on the spherical mesh gives smooth energy
surfaces from a discrete set of vertex samples.

The crate provides:

- **`Table6D`** &mdash; nested periodic table over (R, &omega;) with `IcoTable4D` angular layers,
  used during table construction.
- **`Table6DFlat<T>`** &mdash; flat, bincode-serializable representation for fast runtime lookup.
  Supports optional gzip compression (`.gz` suffix).
- **`orient` / `inverse_orient`** &mdash; forward and inverse coordinate transforms between
  6D table indices and quaternion + separation-vector representation.
- **`IcoTable2D<T>`** &mdash; single-icosphere angular table with barycentric interpolation,
  useful for mapping scalar fields on S&sup2;.
- Icosphere mesh generation and vertex/face utilities built on
  [`hexasphere`](https://crates.io/crates/hexasphere).

## Usage

This crate is a shared dependency between:

- [**Duello**](https://github.com/mlund/duello) &mdash; computes 6D energy tables by scanning
  all orientations of two rigid molecules (`Table6D` &rarr; `Table6DFlat::save()`).
- [**Faunus**](https://github.com/mlund/faunus-rs) &mdash; loads pre-computed tables for
  O(1) energy lookups during rigid-body Monte Carlo simulations (`Table6DFlat::load()`).

## License

Apache-2.0
