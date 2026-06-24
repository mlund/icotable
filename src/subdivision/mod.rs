//! Pluggable subdivision schemes for a single icosphere.
//!
//! A subdivision scheme determines a sphere's sample directions, their
//! quadrature weights, the neighbor graph, and the locator that maps a query
//! direction to its containing cell. Everything above the single sphere (the
//! pair-of-spheres 4D structure, R/ω, interpolation) is scheme-agnostic.
//!
//! `hexasphere` usage is confined to [`geodesic`]; no other module may touch it.

pub mod geodesic;

use crate::flat::FaceGrid;
use crate::ico::Face;
use crate::Vector3;
use serde::{Deserialize, Serialize};

/// The mesh data a subdivision produces for one sphere: unit-sphere vertex
/// positions, per-vertex quadrature weights (normalized so uniform = 1.0), and
/// per-vertex neighbor indices. All three share one index order and are derived
/// from a single adjacency pass.
pub(crate) struct MeshData {
    pub vertices: Vec<[f64; 3]>,
    pub weights: Vec<f64>,
    pub neighbors: Vec<Vec<u16>>,
}

/// Which subdivision produced a mesh — stored in the table (serde) and used to
/// rebuild the locator on load. A new scheme adds a variant here plus a module.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum Subdivision {
    /// Conventional geodesic icosphere (hexasphere), located by spatial search.
    #[default]
    Geodesic,
    // Lattice — Jeremi's regular planar lattice with a closed-form locator — lands next.
}

impl Subdivision {
    /// Generate the sphere mesh (vertices, weights, neighbors) at level `n_div`.
    pub(crate) fn build_mesh(self, n_div: usize) -> MeshData {
        match self {
            Self::Geodesic => geodesic::build_mesh(n_div),
        }
    }

    /// Build this scheme's locator over an already-built mesh.
    pub(crate) fn build_locator(self, vertices: &[[f64; 3]], neighbors: &[Vec<u16>]) -> Locator {
        match self {
            Self::Geodesic => Locator::Geodesic(FaceGrid::new(vertices, neighbors)),
        }
    }
}

/// A scheme's direction→cell locator. Rebuilt on load (never serialized); one
/// `match` per query — the only place the scheme is dispatched at lookup time.
#[derive(Clone, Debug)]
pub(crate) enum Locator {
    Geodesic(FaceGrid),
}

impl Locator {
    /// Containing cell (3 vertex indices) + barycentric weights for `dir`.
    pub(crate) fn locate(&self, dir: &Vector3) -> (Face, [f64; 3]) {
        match self {
            Self::Geodesic(grid) => grid.locate(dir),
        }
    }

    /// Nearest vertex index to `dir` (no triangle search).
    pub(crate) fn find_nearest_vertex(&self, dir: &Vector3) -> usize {
        match self {
            Self::Geodesic(grid) => grid.find_nearest_vertex(dir),
        }
    }
}
