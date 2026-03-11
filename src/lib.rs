//! Icosphere-based 6D angular lookup tables for rigid molecule-molecule interactions.
//!
//! Stores pre-computed energies over (R, ω, θ₁φ₁, θ₂φ₂) using icosphere
//! tessellation with barycentric interpolation for the angular dimensions.

pub mod flat;
mod icosphere;
pub mod icotable;
pub mod orient;
mod spherical;
pub mod table;
mod vertex;

pub use flat::{Table3DFlat, Table6DFlat};
pub use icosphere::{extract_vertices, make_icosphere, make_icosphere_vertices, make_weights};
pub use icotable::{Face, IcoTable2D, IcoTable4D, Table6D};
pub use orient::{inverse_orient, orient};
pub use spherical::SphericalCoord;
pub use vertex::{make_vertices, Vertex};

pub type IcoSphere = hexasphere::Subdivided<(), hexasphere::shapes::IcoSphereBase>;
pub type Matrix3 = nalgebra::Matrix3<f64>;
pub type Vector3 = nalgebra::Vector3<f64>;
pub type UnitQuaternion = nalgebra::UnitQuaternion<f64>;
