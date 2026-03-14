#![warn(missing_docs)]
//! Icosphere-based 6D angular lookup tables for rigid molecule-molecule interactions.
//!
//! Stores pre-computed pairwise energies over six degrees of freedom
//! (R, ω, θ₁φ₁, θ₂φ₂) using icosphere tessellation for the angular dimensions.
//!
//! # Table types
//!
//! - [`Table6DAdaptive`] — adaptive resolution per (R, ω) slab. Slabs are
//!   classified as repulsive, scalar, nearest-vertex, or fully interpolated
//!   based on angular gradients. Built via [`AdaptiveBuilder`].
//! - [`Table6DFlat`] — uniform angular resolution across all separations (legacy).
//! - [`Table3DFlat`] — 3D table (R, θ, φ) for rigid body + single atom interactions.
//!
//! Both 6D formats support optional [`TableMetadata`] with tail correction
//! terms for extrapolation beyond the table cutoff.

/// Adaptive 6D tables with per-R-slice resolution for fast generation and compact storage.
pub mod adaptive;
/// Portable, flat representations of angular tables for file I/O and fast lookup.
pub mod flat;
/// Core icosphere tables and barycentric interpolation.
pub mod ico;
mod icosphere;
/// Forward and inverse coordinate transforms.
pub mod orient;
mod spherical;
/// Periodic, equidistant padded tables.
pub mod table;
mod vertex;

// Public API
pub use adaptive::{AdaptiveBuilder, SlabResolution, Table6DAdaptive};
pub use flat::{Table3DFlat, Table6DFlat, TableMetadata, TailCorrectionTerm};
/// Half-precision float for compact table storage.
pub use half::f16;
pub use ico::{Face, IcoTable2D, IcoTable4D, Table6D};
pub use icosphere::make_icosphere_vertices;
pub use orient::{inverse_orient, orient};
pub use spherical::SphericalCoord;
pub use table::PaddedTable;

// Crate-internal re-exports
pub(crate) use icosphere::{make_icosphere, make_icosphere_by_ndiv, make_weights};
pub(crate) use vertex::{make_vertices, Vertex};

/// 3×3 matrix of `f64`.
pub type Matrix3 = nalgebra::Matrix3<f64>;
/// 3D column vector of `f64`.
pub type Vector3 = nalgebra::Vector3<f64>;
/// Unit quaternion for `f64` rotations.
pub type UnitQuaternion = nalgebra::UnitQuaternion<f64>;

/// Subdivided icosphere mesh.
pub(crate) type IcoSphere = hexasphere::Subdivided<(), hexasphere::shapes::IcoSphereBase>;
