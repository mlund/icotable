#![warn(missing_docs)]
//! Icosphere-based 6D angular lookup tables for rigid molecule-molecule interactions.
//!
//! Stores pre-computed energies over (R, ω, θ₁φ₁, θ₂φ₂) using icosphere
//! tessellation with barycentric interpolation for the angular dimensions.

/// Portable, flat representations of angular tables for file I/O and fast lookup.
pub mod flat;
mod icosphere;
/// Icosphere-based angular lookup tables and interpolation.
pub mod icotable;
/// Forward and inverse coordinate transforms.
pub mod orient;
mod spherical;
/// Periodic, equidistant padded tables.
pub mod table;
mod vertex;

// Public API
pub use flat::{Table3DFlat, Table6DFlat};
pub use icosphere::make_icosphere_vertices;
pub use icotable::{Face, IcoTable2D, IcoTable4D, Table6D};
pub use orient::{inverse_orient, orient};
pub use spherical::SphericalCoord;
pub use table::PaddedTable;

// Crate-internal re-exports
pub(crate) use icosphere::make_icosphere;
pub(crate) use vertex::{make_vertices, Vertex};

/// 3×3 matrix of `f64`.
pub type Matrix3 = nalgebra::Matrix3<f64>;
/// 3D column vector of `f64`.
pub type Vector3 = nalgebra::Vector3<f64>;
/// Unit quaternion for `f64` rotations.
pub type UnitQuaternion = nalgebra::UnitQuaternion<f64>;

/// Subdivided icosphere mesh.
pub(crate) type IcoSphere = hexasphere::Subdivided<(), hexasphere::shapes::IcoSphereBase>;
