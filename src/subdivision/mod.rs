//! Pluggable subdivision schemes for a single icosphere.
//!
//! A subdivision scheme determines a sphere's sample directions, their
//! quadrature weights, the neighbor graph, and the locator that maps a query
//! direction to its containing cell. Everything above the single sphere (the
//! pair-of-spheres 4D structure, R/ω, interpolation) is scheme-agnostic.
//!
//! `hexasphere` usage is confined to [`geodesic`]; no other module may touch it.

pub mod geodesic;
pub mod lattice;

use crate::flat::FaceGrid;
use crate::ico::Face;
use crate::Vector3;
use lattice::AnalyticLattice;
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
    /// Jeremi's regular planar lattice on each icosahedron face, with a
    /// closed-form locator (no search).
    Lattice,
}

impl Subdivision {
    /// Generate the sphere mesh (vertices, weights, neighbors) for builder
    /// `level` (0-based, coarsest first). Each scheme maps the level to its own
    /// native parameter so that level 0 is the 12-vertex base for **both**:
    /// geodesic uses subdivision `n_div = level`; the lattice uses frequency
    /// `level + 1` (frequency 0 is degenerate).
    pub(crate) fn build_mesh(self, level: usize) -> MeshData {
        match self {
            Self::Geodesic => geodesic::build_mesh(level),
            Self::Lattice => lattice::build_mesh(level + 1),
        }
    }

    /// Build this scheme's locator for builder `level` over an already-built
    /// mesh. The geodesic search locator needs the mesh; the lattice locator
    /// needs only its frequency (`level + 1`), so this stays consistent with a
    /// deserialized table (whose `MeshLevel` carries the level as `n_div`).
    pub(crate) fn build_locator(
        self,
        level: usize,
        vertices: &[[f64; 3]],
        neighbors: &[Vec<u16>],
    ) -> Locator {
        match self {
            Self::Geodesic => Locator::Geodesic(FaceGrid::new(vertices, neighbors)),
            Self::Lattice => Locator::Lattice(AnalyticLattice::new(level + 1)),
        }
    }

    /// Vertex count at builder `level`. Both schemes share the resolution ladder
    /// — level ℓ maps to effective frequency ℓ+1, giving `10·(ℓ+1)²+2` vertices,
    /// so level 0 is the 12-vertex base either way. The schemes differ in their
    /// locator and weight uniformity, not in which resolutions they can reach.
    pub fn n_vertices(self, level: usize) -> usize {
        let f = level + 1;
        10 * f * f + 2
    }

    /// Approximate angular spacing between adjacent vertices (radians) at builder
    /// `level` — the mean nearest-neighbor arc, `√(4π / N)`.
    pub fn angular_resolution(self, level: usize) -> f64 {
        (4.0 * std::f64::consts::PI / self.n_vertices(level) as f64).sqrt()
    }
}

/// A scheme's direction→cell locator. Rebuilt on load (never serialized); one
/// `match` per query — the only place the scheme is dispatched at lookup time.
#[derive(Clone, Debug)]
pub(crate) enum Locator {
    Geodesic(FaceGrid),
    Lattice(AnalyticLattice),
}

impl Locator {
    /// Containing cell (3 vertex indices) + barycentric weights for `dir`.
    pub(crate) fn locate(&self, dir: &Vector3) -> (Face, [f64; 3]) {
        match self {
            Self::Geodesic(grid) => grid.locate(dir),
            Self::Lattice(loc) => loc.locate(dir),
        }
    }

    /// Nearest vertex index to `dir` (no triangle search).
    pub(crate) fn find_nearest_vertex(&self, dir: &Vector3) -> usize {
        match self {
            Self::Geodesic(grid) => grid.find_nearest_vertex(dir),
            Self::Lattice(loc) => loc.find_nearest_vertex(dir),
        }
    }
}

/// Per-vertex quadrature weights (normalized so uniform = 1.0) for an arbitrary
/// vertex graph — the Voronoi solid angle, summed as a fan of spherical
/// triangles over each vertex's neighbors in angular order. Unlike
/// `geodesic`'s variant it sorts the fan itself, so it works for schemes whose
/// neighbor lists are not pre-ordered (the lattice).
pub(crate) fn solid_angle_weights(vertices: &[[f64; 3]], neighbors: &[Vec<u16>]) -> Vec<f64> {
    let dir = |i: usize| Vector3::from(vertices[i]).normalize();
    let mut weights = Vec::with_capacity(vertices.len());
    for (i, nbrs) in neighbors.iter().enumerate() {
        let vi = dir(i);
        // Tangent basis at `vi` to sort the neighbors by azimuth.
        let seed = dir(nbrs[0] as usize);
        let e1 = (seed - vi * vi.dot(&seed)).normalize();
        let e2 = vi.cross(&e1);
        let mut ring: Vec<(f64, Vector3)> = nbrs
            .iter()
            .map(|&n| {
                let d = dir(n as usize);
                let t = d - vi * vi.dot(&d);
                (t.dot(&e2).atan2(t.dot(&e1)), d)
            })
            .collect();
        ring.sort_by(|a, b| a.0.total_cmp(&b.0));
        // Neighbor-polygon area; each face's area is shared by its 3 vertices.
        let mut area = 0.0;
        for j in 0..ring.len() {
            area += spherical_triangle_area(&vi, &ring[j].1, &ring[(j + 1) % ring.len()].1);
        }
        weights.push(area / 3.0);
    }
    let ideal = 4.0 * std::f64::consts::PI / vertices.len() as f64;
    weights.iter_mut().for_each(|w| *w /= ideal);
    weights
}

/// Area of the spherical triangle `(a, b, c)` via spherical excess.
#[allow(non_snake_case)]
fn spherical_triangle_area(a: &Vector3, b: &Vector3, c: &Vector3) -> f64 {
    let angle = |u: &Vector3, v: &Vector3, w: &Vector3| {
        let vu = (u - v * v.dot(u)).normalize();
        let vw = (w - v * v.dot(w)).normalize();
        vu.dot(&vw).clamp(-1.0, 1.0).acos()
    };
    angle(b, a, c) + angle(c, b, a) + angle(a, c, b) - std::f64::consts::PI
}
