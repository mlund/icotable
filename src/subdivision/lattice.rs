//! The **Lattice** subdivision: Jeremi's regular planar lattice on each
//! icosahedron face, with a closed-form locator (~6 ns, no search).
//!
//! Generated directly by barycentric enumeration (no half-edge mesh). Reuses
//! Jeremi's geometric core — the 12-vertex icosahedron, the 20 master faces, and
//! the sign-bit master-face classifier — but locates via standard triangular-
//! lattice point location (containing sub-triangle + barycentric weights), and
//! stores a deduplicated vertex graph for interpolation.

use super::MeshData;
use crate::ico::Face;
use crate::Vector3;
use std::collections::{BTreeSet, HashMap};

const EPS: f64 = 1e-12;
/// Dedup quantization: distinct lattice vertices are ≫ 1e-6 apart for usable `n`.
const DEDUP_SCALE: f64 = 1e7;

/// A master icosahedron face: plane geometry for projection + classification,
/// and the edge Gram matrix for barycentric coordinates of a projected point.
#[derive(Clone, Debug)]
struct MasterFace {
    index: usize,
    p1: Vector3,
    p2: Vector3,
    p3: Vector3,
    /// Outward face-plane normal.
    normal: Vector3,
    /// Plane offset `normal · p1`, for the radial projection.
    plane_offset: f64,
    /// Normals of the three `(Pi, O, Pj)` side planes, for face classification.
    edge_planes: [Vector3; 3],
    v0: Vector3, // p2 - p1
    v1: Vector3, // p3 - p1
    d00: f64,
    d01: f64,
    d11: f64,
    inv_denom: f64,
}

impl MasterFace {
    fn new(p1: Vector3, p2: Vector3, p3: Vector3, index: usize) -> Self {
        let v0 = p2 - p1;
        let v1 = p3 - p1;
        let d00 = v0.dot(&v0);
        let d01 = v0.dot(&v1);
        let d11 = v1.dot(&v1);
        let normal = v0.cross(&v1).normalize();
        MasterFace {
            index,
            plane_offset: normal.dot(&p1),
            normal,
            edge_planes: [p1.cross(&p2), p2.cross(&p3), p3.cross(&p1)],
            v0,
            v1,
            d00,
            d01,
            d11,
            inv_denom: 1.0 / (d00 * d11 - d01 * d01),
            p1,
            p2,
            p3,
        }
    }

    /// Radially project `p` onto the face plane.
    #[inline]
    fn projection(&self, p: &Vector3) -> Vector3 {
        p * (self.plane_offset / p.dot(&self.normal))
    }

    /// Barycentric `(α, β, γ)` of a projected point `pp` w.r.t. `(p1, p2, p3)`.
    #[inline]
    fn barycentric(&self, pp: &Vector3) -> (f64, f64, f64) {
        let v2 = pp - self.p1;
        let d20 = v2.dot(&self.v0);
        let d21 = v2.dot(&self.v1);
        let beta = (self.d11 * d20 - self.d01 * d21) * self.inv_denom;
        let gamma = (self.d00 * d21 - self.d01 * d20) * self.inv_denom;
        (1.0 - beta - gamma, beta, gamma)
    }
}

/// The 12 icosahedron vertices (pole, two rings at ±atan(2), pole).
fn icosahedron_points() -> Vec<Vector3> {
    let a = 2.0f64.atan();
    let mut pts = Vec::with_capacity(12);
    pts.push(Vector3::new(0.0, 0.0, 1.0));
    for k in 0..5 {
        let phi = 2.0 * k as f64 * std::f64::consts::PI / 5.0;
        pts.push(Vector3::new(a.sin() * phi.cos(), a.sin() * phi.sin(), a.cos()));
    }
    for k in 0..5 {
        let phi = (2.0 * k as f64 + 1.0) * std::f64::consts::PI / 5.0;
        let b = std::f64::consts::PI - a;
        pts.push(Vector3::new(b.sin() * phi.cos(), b.sin() * phi.sin(), b.cos()));
    }
    pts.push(Vector3::new(0.0, 0.0, -1.0));
    pts
}

/// The 12 icosahedron vertex indices of each of the 20 master faces, in index
/// order (top 0..5, middle 5..15, bottom 15..20). Constant topology.
const FACE_VERTICES: [[usize; 3]; 20] = [
    [0, 1, 2], [0, 2, 3], [0, 3, 4], [0, 4, 5], [0, 5, 1], // top
    [6, 2, 1], [2, 6, 7], [7, 3, 2], [3, 7, 8], [8, 4, 3], // middle
    [4, 8, 9], [9, 5, 4], [5, 9, 10], [10, 1, 5], [1, 10, 6],
    [11, 7, 6], [11, 8, 7], [11, 9, 8], [11, 10, 9], [11, 6, 10], // bottom
];

/// The 20 master faces and the 3 symmetry planes — Jeremi's classifier geometry.
/// (Not `const`: the vertex positions need `sin`/`cos`/`atan`.)
fn master_faces() -> (Vec<MasterFace>, [Vector3; 3]) {
    let pts = icosahedron_points();
    let faces = FACE_VERTICES
        .iter()
        .enumerate()
        .map(|(idx, &[i, j, k])| MasterFace::new(pts[i], pts[j], pts[k], idx))
        .collect();
    let symmetry_planes = [pts[2] + pts[7], pts[3] + pts[4], pts[0] + pts[1]];
    (faces, symmetry_planes)
}

/// `master_face` candidate table: 3 symmetry-plane sign bits → 4 candidate face
/// indices (then one side-plane test picks among them). Constant geometry, so
/// it lives here rather than in every locator instance.
type FaceMapping = [[[[usize; 4]; 2]; 2]; 2];
const FACE_MAPPING: FaceMapping = {
    let mut m = [[[[0usize; 4]; 2]; 2]; 2];
    m[1][1][1] = [18, 19, 12, 17];
    m[1][1][0] = [13, 14, 4, 12];
    m[1][0][1] = [10, 9, 17, 11];
    m[1][0][0] = [3, 2, 11, 4];
    m[0][1][1] = [15, 16, 6, 19];
    m[0][1][0] = [5, 6, 0, 14];
    m[0][0][1] = [8, 7, 16, 9];
    m[0][0][0] = [1, 0, 7, 2];
    m
};

/// Index of barycentric corner `(a, b)` (with `c = n − a − b`) within a face's
/// flat vertex-id array: rows `a = 0..=n`, each of length `n − a + 1`. The row
/// offset is the triangular sum `Σ_{r<a}(n−r+1)`.
#[inline]
const fn tri_index(n: usize, a: usize, b: usize) -> usize {
    a * (n + 1) - a * a.saturating_sub(1) / 2 + b
}

/// Deduplicated lattice geometry at frequency `n`: a pure function of `n`, so the
/// locator can reproduce it on load.
struct LatticeGeometry {
    vertices: Vec<[f64; 3]>,
    /// Per master face, `(a,b) → global vertex id`, indexed by [`tri_index`].
    face_ids: Vec<Vec<u32>>,
    neighbors: Vec<Vec<u16>>,
}

fn lattice_geometry(n: usize) -> LatticeGeometry {
    assert!(n >= 1, "lattice frequency must be ≥ 1 (got {n}); level 0 is degenerate");
    let n_verts = 10 * n * n + 2;
    // Vertex ids are u16 (mesh neighbor lists are `Vec<u16>`); fail loudly rather
    // than silently wrapping `vertices.len() as u16` past 65535 (frequency ≥ 81).
    assert!(
        n_verts <= 1 << 16,
        "lattice frequency {n} yields {n_verts} vertices, exceeding the u16 id limit (65536)"
    );
    let (faces, _) = master_faces();
    let nf = n as f64;
    let per_face = (n + 1) * (n + 2) / 2;

    let mut vertices: Vec<[f64; 3]> = Vec::with_capacity(n_verts);
    let mut key_to_id: HashMap<(i64, i64, i64), u16> = HashMap::with_capacity(n_verts);
    let mut face_ids: Vec<Vec<u32>> = Vec::with_capacity(20);

    for f in &faces {
        let mut ids = vec![0u32; per_face];
        for a in 0..=n {
            for b in 0..=(n - a) {
                let c = n - a - b;
                let p = ((a as f64) * f.p1 + (b as f64) * f.p2 + (c as f64) * f.p3) / nf;
                let p = p.normalize();
                let key = (
                    (p.x * DEDUP_SCALE).round() as i64,
                    (p.y * DEDUP_SCALE).round() as i64,
                    (p.z * DEDUP_SCALE).round() as i64,
                );
                let id = *key_to_id.entry(key).or_insert_with(|| {
                    let id = vertices.len() as u16;
                    vertices.push([p.x, p.y, p.z]);
                    id
                });
                ids[tri_index(n, a, b)] = u32::from(id);
            }
        }
        face_ids.push(ids);
    }
    assert_eq!(
        vertices.len(),
        n_verts,
        "lattice dedup produced {} vertices, expected {n_verts}",
        vertices.len()
    );

    // Triangular-lattice adjacency: the 6 barycentric steps (keeping a+b+c=n),
    // unioned across every face a vertex appears in (so shared-edge vertices get
    // neighbors from both faces).
    let steps: [(i64, i64); 6] = [(1, -1), (-1, 1), (1, 0), (-1, 0), (0, 1), (0, -1)];
    let mut sets: Vec<BTreeSet<u16>> = vec![BTreeSet::new(); vertices.len()];
    for ids in &face_ids {
        for a in 0..=n {
            for b in 0..=(n - a) {
                let id = ids[tri_index(n, a, b)] as u16;
                for (da, db) in steps {
                    let (na, nb) = (a as i64 + da, b as i64 + db);
                    let nc = n as i64 - na - nb;
                    if na >= 0 && nb >= 0 && nc >= 0 && na <= n as i64 && nb <= n as i64 {
                        let nid = ids[tri_index(n, na as usize, nb as usize)] as u16;
                        sets[id as usize].insert(nid);
                    }
                }
            }
        }
    }
    let neighbors = sets.into_iter().map(|s| s.into_iter().collect()).collect();

    LatticeGeometry {
        vertices,
        face_ids,
        neighbors,
    }
}

/// Closed-form lattice locator (Jeremi's classifier + standard triangular-lattice
/// point location). Rebuilt on load from the vertex count.
#[derive(Clone, Debug)]
pub(crate) struct AnalyticLattice {
    faces: Vec<MasterFace>,
    symmetry_planes: [Vector3; 3],
    n: usize,
    face_ids: Vec<Vec<u32>>,
}

impl AnalyticLattice {
    /// Build the locator for frequency `n` (matching `lattice_geometry(n)`).
    pub(crate) fn new(n: usize) -> Self {
        let (faces, symmetry_planes) = master_faces();
        let face_ids = lattice_geometry(n).face_ids;
        AnalyticLattice {
            faces,
            symmetry_planes,
            n,
            face_ids,
        }
    }

    /// Classify the master face containing `p` (constant time, no search).
    #[inline]
    fn master_face(&self, p: &Vector3) -> &MasterFace {
        let i = (p.dot(&self.symmetry_planes[0]) < 0.0) as usize;
        let j = (p.dot(&self.symmetry_planes[1]) < 0.0) as usize;
        let k = (p.dot(&self.symmetry_planes[2]) < 0.0) as usize;
        let cand = &FACE_MAPPING[i][j][k];
        let f0 = &self.faces[cand[0]];
        let mut side = (p.dot(&f0.edge_planes[0]) < -EPS) as usize;
        side += 2 * ((p.dot(&f0.edge_planes[1]) < -EPS) as usize);
        side += 3 * ((p.dot(&f0.edge_planes[2]) < -EPS) as usize);
        &self.faces[cand[side]]
    }

    /// Containing sub-triangle (3 vertex ids) + barycentric weights for `dir`.
    pub(crate) fn locate(&self, dir: &Vector3) -> (Face, [f64; 3]) {
        let face = self.master_face(dir);
        let pp = face.projection(dir);
        let (alpha, beta, gamma) = face.barycentric(&pp);

        let nf = self.n as f64;
        let max = self.n as i64;
        let (u, v, w) = (alpha * nf, beta * nf, gamma * nf);
        let i = (u.floor() as i64).clamp(0, max);
        let j = (v.floor() as i64).clamp(0, max);
        let k = (w.floor() as i64).clamp(0, max);
        let (fu, fv, fw) = (u - i as f64, v - j as f64, w - k as f64);

        // u+v+w = n exactly, so fu+fv+fw is an integer ≈1 (up sub-triangle) or
        // ≈2 (down); split at 1.5 to be robust to FP noise around the integer.
        // Up: corners (i+1,j,k),(i,j+1,k),(i,j,k+1). Down: (i,j+1,k+1),(i+1,j,k+1),(i+1,j+1,k).
        let (corners, weights) = if fu + fv + fw < 1.5 {
            ([(i + 1, j), (i, j + 1), (i, j)], [fu, fv, fw])
        } else {
            (
                [(i, j + 1), (i + 1, j), (i + 1, j + 1)],
                [1.0 - fu, 1.0 - fv, 1.0 - fw],
            )
        };

        let ids = &self.face_ids[face.index];
        let lookup = |(a, b): (i64, i64)| {
            let a = a.clamp(0, max) as usize;
            let b = b.clamp(0, max - a as i64) as usize;
            ids[tri_index(self.n, a, b)] as usize
        };
        (
            [lookup(corners[0]), lookup(corners[1]), lookup(corners[2])],
            weights,
        )
    }

    /// Nearest vertex id to `dir` (the locate corner with the largest weight).
    pub(crate) fn find_nearest_vertex(&self, dir: &Vector3) -> usize {
        let (face, weights) = self.locate(dir);
        let best = (0..3).max_by(|&a, &b| weights[a].total_cmp(&weights[b])).unwrap();
        face[best]
    }
}

/// Build the lattice mesh (vertices, weights, neighbors) at frequency `n`.
pub(crate) fn build_mesh(n: usize) -> MeshData {
    let geom = lattice_geometry(n);
    let weights = super::solid_angle_weights(&geom.vertices, &geom.neighbors);
    MeshData {
        vertices: geom.vertices,
        weights,
        neighbors: geom.neighbors,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{Rng, SeedableRng};

    fn rand_dir(rng: &mut impl Rng) -> Vector3 {
        loop {
            let x: f64 = rng.gen_range(-1.0..1.0);
            let y: f64 = rng.gen_range(-1.0..1.0);
            let z: f64 = rng.gen_range(-1.0..1.0);
            let r2 = x * x + y * y + z * z;
            if r2 > 0.01 && r2 < 1.0 {
                break Vector3::new(x, y, z).normalize();
            }
        }
    }

    #[test]
    fn vertex_count_and_dedup() {
        for &n in &[2usize, 3, 4, 8, 16, 32] {
            let g = lattice_geometry(n);
            assert_eq!(g.vertices.len(), 10 * n * n + 2);
            // neighbor graph symmetric
            for (i, nbrs) in g.neighbors.iter().enumerate() {
                for &nb in nbrs {
                    assert!(g.neighbors[nb as usize].contains(&(i as u16)));
                }
            }
        }
    }

    /// Ground-truth cross-check: our generated vertex **positions** must coincide
    /// with Jeremi's C++ `HedgeTess` mesh (same icosahedron), not merely match in
    /// count. Reference files in `tests/data/` were dumped from his half-edge mesh
    /// (`subdivide_regular(n); push_to_sphere`).
    #[test]
    fn matches_cpp_reference_positions() {
        for &n in &[2usize, 4, 8] {
            let path = format!(
                "{}/tests/data/lattice_cpp_n{n}.txt",
                env!("CARGO_MANIFEST_DIR")
            );
            let cpp: Vec<[f64; 3]> = std::fs::read_to_string(&path)
                .unwrap()
                .lines()
                .map(|line| {
                    let mut it = line.split_whitespace().map(|s| s.parse::<f64>().unwrap());
                    [it.next().unwrap(), it.next().unwrap(), it.next().unwrap()]
                })
                .collect();

            let ours = lattice_geometry(n).vertices;
            assert_eq!(ours.len(), cpp.len(), "n={n}: vertex count");

            // Every C++ vertex must coincide with one of ours (positions, not count).
            // Combined with equal counts + our dedup uniqueness, this is a bijection.
            for c in &cpp {
                let nearest_sq = ours
                    .iter()
                    .map(|o| (o[0] - c[0]).powi(2) + (o[1] - c[1]).powi(2) + (o[2] - c[2]).powi(2))
                    .fold(f64::INFINITY, f64::min);
                assert!(
                    nearest_sq < 1e-18,
                    "n={n}: C++ vertex {c:?} has no Rust match (nearest dist²={nearest_sq:.2e})"
                );
            }
        }
    }

    /// Boundary stress-test mirroring Jeremi's C++ `edgeCase`: directions exactly
    /// on master-face edges and at the 12 icosahedron corners — where the sign-bit
    /// classifier is most likely to misfire. Each must land in a containing cell.
    #[test]
    fn locate_handles_edges_and_corners() {
        let loc = AnalyticLattice::new(8);
        let pts = icosahedron_points();
        for &[i, j, k] in &FACE_VERTICES {
            let (p1, p2, p3) = (pts[i], pts[j], pts[k]);
            for p in [
                0.3 * p1 + 0.7 * p2,
                0.3 * p2 + 0.7 * p3,
                0.3 * p3 + 0.7 * p1,
                p1,
                p2,
                p3,
            ] {
                let dir = p.normalize();
                let (_face, w) = loc.locate(&dir);
                let min = w[0].min(w[1]).min(w[2]);
                assert!(min >= -1e-6, "edge/corner dir {dir:?}: min weight {min}");
            }
        }
    }

    #[test]
    fn weights_normalized_and_positive() {
        for &n in &[2usize, 4, 8, 16] {
            let mesh = build_mesh(n);
            let sum: f64 = mesh.weights.iter().sum();
            // Mean weight = 1.0 (Voronoi solid angles normalized by 4π/N), so Σ = N.
            assert!(
                (sum - mesh.vertices.len() as f64).abs() < 1e-6,
                "n={n}: weights sum {sum} vs N={}",
                mesh.vertices.len()
            );
            assert!(mesh.weights.iter().all(|&w| w > 0.0));
        }
    }

    #[test]
    fn locate_returns_containing_cell() {
        let loc = AnalyticLattice::new(8);
        let mut rng = rand::rngs::StdRng::seed_from_u64(0x1A77);
        for _ in 0..50_000 {
            let dir = rand_dir(&mut rng);
            let (_face, w) = loc.locate(&dir);
            let sum: f64 = w.iter().sum();
            let min = w[0].min(w[1]).min(w[2]);
            assert!((sum - 1.0).abs() < 1e-9, "weights sum {sum}");
            assert!(min >= -1e-6, "extrapolation: min weight {min}");
        }
    }
}
