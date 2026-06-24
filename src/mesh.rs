//! Scheme-independent view of a subdivided sphere as a weighted graph of
//! angular sample points.
//!
//! Analysis that integrates or diffuses over the directions (Duello's spectral
//! and PMF computations) needs the sample points, their quadrature weights, and
//! the connectivity — but not the concrete icosphere representation. Consuming a
//! mesh through this trait keeps such analysis independent of the subdivision
//! scheme that produced it.

use crate::Vector3;

/// A subdivided sphere as a weighted graph of angular sample points.
///
/// Indices `0..len()` enumerate the points; each carries a unit [`direction`],
/// a quadrature [`weight`] (solid-angle fraction), and graph [`neighbors`].
///
/// [`direction`]: AngularMesh::direction
/// [`weight`]: AngularMesh::weight
/// [`neighbors`]: AngularMesh::neighbors
pub trait AngularMesh {
    /// Number of sample points.
    fn len(&self) -> usize;

    /// Whether the mesh has no points.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Unit direction of point `i`.
    fn direction(&self, i: usize) -> Vector3;

    /// Quadrature weight (solid-angle fraction) of point `i`, normalized so that
    /// uniform weighting is `1.0`.
    fn weight(&self, i: usize) -> f64;

    /// Graph neighbors of point `i`.
    fn neighbors(&self, i: usize) -> &[u16];
}
