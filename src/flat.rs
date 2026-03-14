//! Portable, flat representations of angular tables for file I/O and fast lookup.
//!
//! [`Table6DFlat`] and [`Table3DFlat`] store all data in contiguous vectors
//! (no nested Arc/OnceLock) for efficient serialization with bincode and
//! simple interpolation at runtime. Both are generic over `f32` and
//! [`half::f16`] for half-precision storage. Vertices are BFS-reordered
//! during construction to improve cache locality.

use crate::ico::{Face, IcoTable2D, Table6D};
use crate::table::PaddedTable;
use crate::Vector3;
use anyhow::Result;
use flate2::{read::GzDecoder, write::GzEncoder, Compression};
use get_size::GetSize;
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use std::collections::VecDeque;
use std::path::Path;
use std::sync::OnceLock;

/// Save a bincode-serializable value. A `.gz` suffix enables gzip compression.
pub(crate) fn save_bincode(value: &impl Serialize, path: &Path) -> Result<()> {
    let file = std::io::BufWriter::new(std::fs::File::create(path)?);
    if path.extension().is_some_and(|ext| ext == "gz") {
        bincode::serialize_into(GzEncoder::new(file, Compression::default()), value)?;
    } else {
        bincode::serialize_into(file, value)?;
    }
    Ok(())
}

/// Load a bincode-deserializable value. A `.gz` suffix enables gzip decompression.
pub(crate) fn load_bincode<T: DeserializeOwned>(path: &Path) -> Result<T> {
    let file = std::io::BufReader::new(std::fs::File::open(path)?);
    Ok(if path.extension().is_some_and(|ext| ext == "gz") {
        bincode::deserialize_from(GzDecoder::new(file))?
    } else {
        bincode::deserialize_from(file)?
    })
}

/// Project a 3D point onto one of 6 cube faces, returning `(face_id, u, w)`.
///
/// The cube face is determined by the dominant axis component. `u` and `w`
/// are the projected coordinates in [-1, 1] on that face.
pub(crate) fn cube_project(v: &[f64; 3]) -> (usize, f64, f64) {
    let (ax, ay, az) = (v[0].abs(), v[1].abs(), v[2].abs());
    let (face_id, u, w, d) = if ax >= ay && ax >= az {
        if v[0] > 0.0 {
            (0, v[1], v[2], v[0])
        } else {
            (1, -v[1], v[2], -v[0])
        }
    } else if ay >= az {
        if v[1] > 0.0 {
            (2, v[0], v[2], v[1])
        } else {
            (3, -v[0], v[2], -v[1])
        }
    } else if v[2] > 0.0 {
        (4, v[0], v[1], v[2])
    } else {
        (5, -v[0], v[1], -v[2])
    };
    (face_id, u / d, w / d)
}

/// Search mesh triangles adjacent to `nearest` for the best-fitting face.
///
/// Enumerates all neighbor pairs sharing an edge (actual mesh triangles)
/// rather than just the two closest neighbors, which would often select
/// non-adjacent vertices and yield a spurious triangle.
pub(crate) fn search_triangles(
    nearest: usize,
    dir: &Vector3,
    vertices: &[[f64; 3]],
    neighbors: &[Vec<u16>],
) -> (Face, [f64; 3]) {
    let nbrs = &neighbors[nearest];
    let mut best_face = [0usize; 3];
    let mut best_min_bary = f64::NEG_INFINITY;

    for (idx_i, &ni) in nbrs.iter().enumerate() {
        let ni = ni as usize;
        for &nj in &nbrs[idx_i + 1..] {
            let nj = nj as usize;
            if !neighbors[ni].contains(&(nj as u16)) {
                continue;
            }
            let face = [nearest, ni, nj];
            let bary = projected_barycentric(
                dir,
                &vertex_vec3(vertices, face[0]),
                &vertex_vec3(vertices, face[1]),
                &vertex_vec3(vertices, face[2]),
            );
            let min_b = bary[0].min(bary[1]).min(bary[2]);
            if min_b > best_min_bary {
                best_min_bary = min_b;
                best_face = face;
            }
        }
    }

    best_face.sort_unstable();
    let best_bary = projected_barycentric(
        dir,
        &vertex_vec3(vertices, best_face[0]),
        &vertex_vec3(vertices, best_face[1]),
        &vertex_vec3(vertices, best_face[2]),
    );
    (best_face, best_bary)
}

/// O(1) nearest-vertex lookup via cube-face projected grids.
///
/// Each vertex is projected onto the six faces of a cube and binned into a
/// G×G grid, with 1-cell margin overlap to handle boundary cases. A direction
/// lookup maps to a single cube face and grid cell, scanning only the few
/// candidate vertices stored there instead of all V vertices.
#[derive(Clone, Debug)]
pub(crate) struct VertexLocator {
    grid_size: usize,
    /// Flat layout: index = `face_id * g*g + ci*g + cj`.
    cells: Vec<Vec<u16>>,
}

impl VertexLocator {
    pub(crate) fn new(vertices: &[[f64; 3]]) -> Self {
        let n = vertices.len();
        let grid_size = ((n as f64 / 6.0).sqrt() * 1.2).ceil() as usize;
        let grid_size = grid_size.max(2);
        let g = grid_size;
        let mut cells = vec![Vec::new(); 6 * g * g];

        for (vi, v) in vertices.iter().enumerate() {
            for face_id in 0..6 {
                let (u, w, d) = match face_id {
                    0 => (v[1], v[2], v[0]),
                    1 => (-v[1], v[2], -v[0]),
                    2 => (v[0], v[2], v[1]),
                    3 => (-v[0], v[2], -v[1]),
                    4 => (v[0], v[1], v[2]),
                    5 => (-v[0], v[1], -v[2]),
                    _ => unreachable!(),
                };
                if d <= 0.0 {
                    continue;
                }
                let (u, w) = (u / d, w / d);
                let ci = ((u * 0.5 + 0.5) * g as f64) as isize;
                let cj = ((w * 0.5 + 0.5) * g as f64) as isize;
                for di in -1..=1isize {
                    for dj in -1..=1isize {
                        let ci2 = ci + di;
                        let cj2 = cj + dj;
                        if ci2 >= 0 && ci2 < g as isize && cj2 >= 0 && cj2 < g as isize {
                            let idx = face_id * g * g + ci2 as usize * g + cj2 as usize;
                            let cell = &mut cells[idx];
                            if !cell.contains(&(vi as u16)) {
                                cell.push(vi as u16);
                            }
                        }
                    }
                }
            }
        }

        Self { grid_size, cells }
    }

    pub(crate) fn find_face_bary(
        &self,
        dir: &Vector3,
        vertices: &[[f64; 3]],
        neighbors: &[Vec<u16>],
    ) -> (Face, [f64; 3]) {
        let dir = dir.normalize();
        let (face_id, u, w) = cube_project(&[dir.x, dir.y, dir.z]);
        let g = self.grid_size;
        let ci = ((u * 0.5 + 0.5) * g as f64).clamp(0.0, g as f64 - 1.0) as usize;
        let cj = ((w * 0.5 + 0.5) * g as f64).clamp(0.0, g as f64 - 1.0) as usize;
        let candidates = &self.cells[face_id * g * g + ci * g + cj];

        let nearest = candidates
            .iter()
            .map(|&vi| {
                let vi = vi as usize;
                let d = (dir - Vector3::from(vertices[vi])).norm_squared();
                (vi, d)
            })
            .min_by(|a, b| a.1.total_cmp(&b.1))
            .expect("grid cell must not be empty")
            .0;

        search_triangles(nearest, &dir, vertices, neighbors)
    }
}

/// Find the mesh face containing `dir` and return barycentric coordinates.
///
/// O(V) linear scan reference implementation, kept for test verification.
#[cfg(test)]
fn find_face_bary(
    dir: &Vector3,
    vertices: &[[f64; 3]],
    neighbors: &[Vec<u16>],
) -> (Face, [f64; 3]) {
    let dir = dir.normalize();

    let nearest = vertices
        .iter()
        .enumerate()
        .map(|(i, v)| (i, (dir - Vector3::from(*v)).norm_squared()))
        .min_by(|a, b| a.1.total_cmp(&b.1))
        .expect("vertices must not be empty")
        .0;

    search_triangles(nearest, &dir, vertices, neighbors)
}

fn vertex_vec3(vertices: &[[f64; 3]], i: usize) -> Vector3 {
    Vector3::from(vertices[i])
}

/// Extract normalized vertex positions and neighbor lists from an `IcoTable2D`.
fn extract_mesh<T: Clone + GetSize>(ico: &IcoTable2D<T>) -> (Vec<[f64; 3]>, Vec<Vec<u16>>) {
    let vertices = ico
        .iter_positions()
        .map(|p| <[f64; 3]>::from(p.normalize()))
        .collect();
    let neighbors = ico.iter_vertices().map(|v| v.neighbors.clone()).collect();
    (vertices, neighbors)
}

/// Single term in a tail correction: `coefficient * exp(-kappa * r) / r^power`.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct TailCorrectionTerm {
    /// Pre-exponential coefficient.
    pub coefficient: f64,
    /// Screening parameter (inverse length).
    pub kappa: f64,
    /// Power of r in the denominator.
    pub power: u32,
}

/// Optional metadata attached to a 6D table for tail corrections beyond the table cutoff.
#[derive(Serialize, Deserialize, Clone, Debug, Default)]
pub struct TableMetadata {
    /// Tail correction terms evaluated as sum of `C * exp(-κr) / r^p`.
    pub tail_terms: Vec<TailCorrectionTerm>,
    /// Net charges of the two molecules `[q_a, q_b]` in elementary charges.
    pub charges: Option<[f64; 2]>,
    /// Dipole moment magnitudes `[μ_a, μ_b]` in e·Å.
    pub dipole_moments: Option<[f64; 2]>,
    /// Temperature in Kelvin used during table generation.
    pub temperature: Option<f64>,
    /// Coulomb prefactor e²/(4πε₀εᵣ) in kJ·Å/mol/e², multiplied into all tail terms.
    pub electric_prefactor: Option<f64>,
}

impl TableMetadata {
    /// Evaluate tail correction energy at distance `r`.
    pub fn tail_energy(&self, r: f64) -> f64 {
        if self.tail_terms.is_empty() {
            return 0.0;
        }
        let prefactor = self.electric_prefactor.unwrap_or(0.0);
        prefactor
            * self
                .tail_terms
                .iter()
                .map(|t| t.coefficient * (-t.kappa * r).exp() / r.powi(t.power as i32))
                .sum::<f64>()
    }

    /// Validate that tail correction metadata is self-consistent.
    pub fn validate(&self) -> Result<()> {
        if !self.tail_terms.is_empty() && self.electric_prefactor.is_none() {
            anyhow::bail!("tail correction terms present but electric_prefactor is missing");
        }
        Ok(())
    }
}

/// Flat, serializable 6D lookup table.
///
/// Layout: `data[r_idx * (n_omega * n_v * n_v) + omega_idx * (n_v * n_v) + vi * n_v + vj]`
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Table6DFlat<T: num_traits::Float> {
    /// Minimum radial distance.
    pub rmin: f64,
    /// Maximum radial distance.
    pub rmax: f64,
    /// Radial bin width.
    pub dr: f64,
    /// Number of radial bins.
    pub n_r: usize,
    /// Dihedral angle bin width (radians).
    pub omega_step: f64,
    /// Number of dihedral angle bins.
    pub n_omega: usize,
    /// Number of icosphere vertices.
    pub n_vertices: usize,
    /// Normalized unit-sphere vertex positions `[x, y, z]`.
    pub vertices: Vec<[f64; 3]>,
    /// Neighbor indices per vertex.
    pub neighbors: Vec<Vec<u16>>,
    /// Flat data array.
    pub data: Vec<T>,
    /// Optional metadata for tail corrections beyond the table cutoff.
    pub metadata: Option<TableMetadata>,
    /// Cached O(1) vertex locator, lazy-initialized on first lookup.
    #[serde(skip, default)]
    locator: OnceLock<VertexLocator>,
}

/// Legacy layout without metadata — bincode is not self-describing, so adding
/// `metadata` changed the wire format and old files need a separate struct.
#[derive(Serialize, Deserialize)]
struct Table6DFlatLegacy<T: num_traits::Float> {
    rmin: f64,
    rmax: f64,
    dr: f64,
    n_r: usize,
    omega_step: f64,
    n_omega: usize,
    n_vertices: usize,
    vertices: Vec<[f64; 3]>,
    neighbors: Vec<Vec<u16>>,
    data: Vec<T>,
}

impl<T: num_traits::Float> From<Table6DFlatLegacy<T>> for Table6DFlat<T> {
    fn from(legacy: Table6DFlatLegacy<T>) -> Self {
        Self {
            rmin: legacy.rmin,
            rmax: legacy.rmax,
            dr: legacy.dr,
            n_r: legacy.n_r,
            omega_step: legacy.omega_step,
            n_omega: legacy.n_omega,
            n_vertices: legacy.n_vertices,
            vertices: legacy.vertices,
            neighbors: legacy.neighbors,
            data: legacy.data,
            metadata: None,
            locator: OnceLock::new(),
        }
    }
}

impl<T: num_traits::Float + Serialize + DeserializeOwned> Table6DFlat<T> {
    /// Save to a bincode file (gzip-compressed if path ends in `.gz`).
    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        save_bincode(self, path.as_ref())
    }

    /// Load from a bincode file, falling back to legacy format without metadata.
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        match load_bincode::<Self>(path) {
            Ok(table) => Ok(table),
            Err(primary_err) => {
                log::debug!("New format failed ({}), trying legacy format", primary_err);
                Ok(load_bincode::<Table6DFlatLegacy<T>>(path)?.into())
            }
        }
    }

    /// Evaluate tail correction energy beyond the table cutoff.
    ///
    /// Returns 0 if no tail terms or no electric prefactor is present.
    pub fn tail_energy(&self, r: f64) -> f64 {
        self.metadata.as_ref().map_or(0.0, |m| m.tail_energy(r))
    }

    /// Validate that tail correction metadata is self-consistent.
    pub fn validate_metadata(&self) -> Result<()> {
        if let Some(m) = &self.metadata {
            m.validate()?;
        }
        Ok(())
    }
}

/// Build a flat 6D table with element type `T`, converting each f64 value via `convert`.
fn build_flat_6d<T: num_traits::Float + Copy>(
    table: &Table6D,
    convert: impl Fn(f64) -> T,
) -> Result<Table6DFlat<T>> {
    let r_min = table.min_key();
    let r_max = table.max_key();
    let dr = table.key_step();
    let n_r = table.len() - 2;

    let first_omega_table = table.get(r_min)?;
    let omega_step = first_omega_table.key_step();
    let omega_min = first_omega_table.min_key();
    let n_omega = first_omega_table.len() - 2;

    let first_ico = first_omega_table.get(omega_min)?;
    let n_vertices = first_ico.len();
    let (mut vertices, mut neighbors) = extract_mesh(first_ico);

    let stride = n_omega * n_vertices * n_vertices;
    let zero = convert(0.0);
    let mut data = vec![zero; n_r * stride];

    for ri in 0..n_r {
        let r = (ri as f64).mul_add(dr, r_min);
        for oi in 0..n_omega {
            let omega = (oi as f64).mul_add(omega_step, omega_min);
            let ico4d = table.get_icospheres(r, omega)?;
            let base = ri * stride + oi * (n_vertices * n_vertices);
            for vi in 0..n_vertices {
                let inner = ico4d.get_data(vi).expect("missing 4D table entry");
                for vj in 0..n_vertices {
                    if let Some(val) = inner.get_data(vj) {
                        data[base + vi * n_vertices + vj] = convert(*val);
                    }
                }
            }
        }
    }

    let perm = bfs_vertex_permutation(&neighbors);
    apply_vertex_permutation(
        &perm,
        &mut vertices,
        &mut neighbors,
        &mut data,
        n_vertices,
        n_vertices,
    );

    Ok(Table6DFlat {
        rmin: r_min,
        rmax: r_max,
        dr,
        n_r,
        omega_step,
        n_omega,
        n_vertices,
        vertices,
        neighbors,
        data,
        metadata: None,
        locator: OnceLock::new(),
    })
}

impl TryFrom<&Table6D> for Table6DFlat<f32> {
    type Error = anyhow::Error;
    fn try_from(table: &Table6D) -> Result<Self> {
        build_flat_6d(table, |v| v as f32)
    }
}

impl TryFrom<&Table6D> for Table6DFlat<half::f16> {
    type Error = anyhow::Error;
    fn try_from(table: &Table6D) -> Result<Self> {
        build_flat_6d(table, half::f16::from_f64)
    }
}

impl<T: num_traits::Float + Into<f64>> Table6DFlat<T> {
    /// Resolve R/ω bins and icosphere faces for a given query point.
    fn resolve_bins(
        &self,
        r: f64,
        omega: f64,
        dir_a: &Vector3,
        dir_b: &Vector3,
    ) -> Option<(usize, Face, [f64; 3], Face, [f64; 3])> {
        if r < self.rmin || r > self.rmax {
            return None;
        }
        let ri = ((r - self.rmin) / self.dr + 0.5) as usize;
        let ri = ri.min(self.n_r.saturating_sub(1));
        let omega = omega.rem_euclid(std::f64::consts::TAU);
        let oi = (omega / self.omega_step + 0.5) as usize % self.n_omega;
        let base = ri * self.n_omega * self.n_vertices * self.n_vertices
            + oi * self.n_vertices * self.n_vertices;
        let loc = self
            .locator
            .get_or_init(|| VertexLocator::new(&self.vertices));
        let (face_a, bary_a) = loc.find_face_bary(dir_a, &self.vertices, &self.neighbors);
        let (face_b, bary_b) = loc.find_face_bary(dir_b, &self.vertices, &self.neighbors);
        Some((base, face_a, bary_a, face_b, bary_b))
    }

    /// Lookup value by nearest R/ω bin and barycentric interpolation on icospheres.
    pub fn lookup(&self, r: f64, omega: f64, dir_a: &Vector3, dir_b: &Vector3) -> f64 {
        let (base, face_a, bary_a, face_b, bary_b) = match self.resolve_bins(r, omega, dir_a, dir_b)
        {
            Some(v) => v,
            None => return 0.0,
        };
        let mut result = 0.0;
        for i in 0..3 {
            for j in 0..3 {
                let idx = base + face_a[i] * self.n_vertices + face_b[j];
                let val: f64 = self.data[idx].into();
                result += bary_a[i] * val * bary_b[j];
            }
        }
        result
    }

    /// Boltzmann-weighted interpolation: interpolate exp(-beta*u) then invert.
    ///
    /// Linear interpolation of a convex energy surface overestimates (Jensen's
    /// inequality). By interpolating the Boltzmann factor instead, we avoid
    /// this systematic bias at repulsive contacts. `beta` must be positive.
    pub fn lookup_boltzmann(
        &self,
        r: f64,
        omega: f64,
        dir_a: &Vector3,
        dir_b: &Vector3,
        beta: f64,
    ) -> f64 {
        debug_assert!(beta > 0.0, "beta must be positive, got {beta}");
        let (base, face_a, bary_a, face_b, bary_b) = match self.resolve_bins(r, omega, dir_a, dir_b)
        {
            Some(v) => v,
            None => return 0.0,
        };

        // Shift by u_min before exp() to prevent fp overflow (log-sum-exp trick);
        // the final `+ u_min` restores the correct result.
        let mut vals = [0.0f64; 9];
        let mut u_min = f64::INFINITY;
        for i in 0..3 {
            for j in 0..3 {
                let idx = base + face_a[i] * self.n_vertices + face_b[j];
                let val: f64 = self.data[idx].into();
                vals[i * 3 + j] = val;
                if val < u_min {
                    u_min = val;
                }
            }
        }
        if u_min.is_infinite() {
            return f64::INFINITY;
        }

        let mut sum = 0.0;
        for i in 0..3 {
            for j in 0..3 {
                sum += bary_a[i] * bary_b[j] * (-beta * (vals[i * 3 + j] - u_min)).exp();
            }
        }

        if sum <= 0.0 {
            return f64::INFINITY;
        }
        u_min - sum.ln() / beta
    }
}

/// BFS traversal from vertex 0, returning a permutation `old_index → new_index`.
pub(crate) fn bfs_vertex_permutation(neighbors: &[Vec<u16>]) -> Vec<usize> {
    let n = neighbors.len();
    let mut new_of_old = vec![usize::MAX; n];
    let mut queue = VecDeque::with_capacity(n);
    let mut next_id = 0;

    new_of_old[0] = 0;
    next_id += 1;
    queue.push_back(0usize);

    while let Some(old) = queue.pop_front() {
        for &nb in &neighbors[old] {
            let nb = nb as usize;
            if new_of_old[nb] == usize::MAX {
                new_of_old[nb] = next_id;
                next_id += 1;
                queue.push_back(nb);
            }
        }
    }
    debug_assert!(
        new_of_old.iter().all(|&p| p != usize::MAX),
        "disconnected vertex graph: not all vertices reachable from vertex 0"
    );
    new_of_old
}

/// Reorder vertices, neighbors, and data in-place according to a vertex permutation.
///
/// `entries_per_vertex` is 1 for 3D tables or `n_v` for 6D tables (where each
/// slab has `n_v * n_v` entries indexed by two vertex indices).
pub(crate) fn apply_vertex_permutation<T: Copy>(
    perm: &[usize],
    vertices: &mut [[f64; 3]],
    neighbors: &mut [Vec<u16>],
    data: &mut [T],
    n_v: usize,
    entries_per_vertex: usize,
) {
    debug_assert_eq!(perm.len(), n_v);

    // Inverse permutation: old_of_new[new] = old
    let mut old_of_new = vec![0usize; n_v];
    for (old, &new) in perm.iter().enumerate() {
        old_of_new[new] = old;
    }

    // In-place cycle-following for vertices
    let mut done = vec![false; n_v];
    for start in 0..n_v {
        if done[start] || old_of_new[start] == start {
            done[start] = true;
            continue;
        }
        let mut cur = start;
        let tmp_v = vertices[start];
        let tmp_n = std::mem::take(&mut neighbors[start]);
        loop {
            let src = old_of_new[cur];
            if src == start {
                vertices[cur] = tmp_v;
                neighbors[cur] = tmp_n;
                done[cur] = true;
                break;
            }
            vertices[cur] = vertices[src];
            neighbors[cur] = std::mem::take(&mut neighbors[src]);
            done[cur] = true;
            cur = src;
        }
    }

    // Remap neighbor indices to new IDs
    for nbrs in neighbors.iter_mut() {
        for nb in nbrs.iter_mut() {
            *nb = perm[*nb as usize] as u16;
        }
    }

    // Reorder data slabs — requires a clone since 6D rows are interleaved
    let slab_size = entries_per_vertex * n_v;
    let n_slabs = data.len() / slab_size;
    let old_data = data.to_vec();
    if entries_per_vertex == n_v {
        for s in 0..n_slabs {
            let base = s * slab_size;
            for ni in 0..n_v {
                let oi = old_of_new[ni];
                for nj in 0..n_v {
                    data[base + ni * n_v + nj] = old_data[base + oi * n_v + old_of_new[nj]];
                }
            }
        }
    } else {
        for s in 0..n_slabs {
            let base = s * n_v;
            for ni in 0..n_v {
                data[base + ni] = old_data[base + old_of_new[ni]];
            }
        }
    }
}

/// Projected barycentric coordinates (Ericson, "Real-Time Collision Detection", p141-142).
///
/// Single free function shared by both `Table6DFlat::find_face_bary` and
/// `IcoTable2D::barycentric` to avoid duplicating this algorithm.
pub(crate) fn projected_barycentric(
    p: &Vector3,
    a: &Vector3,
    b: &Vector3,
    c: &Vector3,
) -> [f64; 3] {
    let ab = b - a;
    let ac = c - a;
    let ap = p - a;
    let d1 = ab.dot(&ap);
    let d2 = ac.dot(&ap);
    if d1 <= 0.0 && d2 <= 0.0 {
        return [1.0, 0.0, 0.0];
    }
    let bp = p - b;
    let d3 = ab.dot(&bp);
    let d4 = ac.dot(&bp);
    if d3 >= 0.0 && d4 <= d3 {
        return [0.0, 1.0, 0.0];
    }
    let vc = d1.mul_add(d4, -(d3 * d2));
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        let v = d1 / (d1 - d3);
        return [1.0 - v, v, 0.0];
    }
    let cp = p - c;
    let d5 = ab.dot(&cp);
    let d6 = ac.dot(&cp);
    if d6 >= 0.0 && d5 <= d6 {
        return [0.0, 0.0, 1.0];
    }
    let vb = d5.mul_add(d2, -(d1 * d6));
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        let w = d2 / (d2 - d6);
        return [1.0 - w, 0.0, w];
    }
    let va = d3.mul_add(d6, -(d5 * d4));
    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
        let w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return [0.0, 1.0 - w, w];
    }
    let denom = 1.0 / (va + vb + vc);
    let v = vb * denom;
    let w = vc * denom;
    [1.0 - v - w, v, w]
}

/// Flat, serializable 3D lookup table for rigid body + single atom interactions.
///
/// Layout: `data[r_idx * n_vertices + vi]`
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Table3DFlat<T: num_traits::Float> {
    /// Minimum radial distance.
    pub rmin: f64,
    /// Maximum radial distance.
    pub rmax: f64,
    /// Radial bin width.
    pub dr: f64,
    /// Number of radial bins.
    pub n_r: usize,
    /// Number of icosphere vertices.
    pub n_vertices: usize,
    /// Normalized unit-sphere vertex positions `[x, y, z]`.
    pub vertices: Vec<[f64; 3]>,
    /// Neighbor indices per vertex.
    pub neighbors: Vec<Vec<u16>>,
    /// Flat data array.
    pub data: Vec<T>,
    /// Cached O(1) vertex locator, lazy-initialized on first lookup.
    #[serde(skip, default)]
    locator: OnceLock<VertexLocator>,
}

impl<T: num_traits::Float + Serialize + DeserializeOwned> Table3DFlat<T> {
    /// Save to a bincode file (gzip-compressed if path ends in `.gz`).
    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        save_bincode(self, path.as_ref())
    }
    /// Load from a bincode file (gzip-decompressed if path ends in `.gz`).
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        load_bincode(path.as_ref())
    }
}

/// Build a flat 3D table with element type `T`, converting each f64 value via `convert`.
fn build_flat_3d<T: num_traits::Float + Copy>(
    table: &PaddedTable<IcoTable2D<f64>>,
    convert: impl Fn(f64) -> T,
) -> Result<Table3DFlat<T>> {
    let r_min = table.min_key();
    let r_max = table.max_key();
    let dr = table.key_step();
    let n_r = table.len() - 2;

    let first_ico = table.get(r_min)?;
    let n_vertices = first_ico.len();
    let (mut vertices, mut neighbors) = extract_mesh(first_ico);

    let zero = convert(0.0);
    let mut data = vec![zero; n_r * n_vertices];

    for ri in 0..n_r {
        let r = (ri as f64).mul_add(dr, r_min);
        let ico = table.get(r)?;
        for (vi, val) in ico.vertex_data().enumerate() {
            data[ri * n_vertices + vi] = convert(*val);
        }
    }

    let perm = bfs_vertex_permutation(&neighbors);
    apply_vertex_permutation(
        &perm,
        &mut vertices,
        &mut neighbors,
        &mut data,
        n_vertices,
        1,
    );

    Ok(Table3DFlat {
        rmin: r_min,
        rmax: r_max,
        dr,
        n_r,
        n_vertices,
        vertices,
        neighbors,
        data,
        locator: OnceLock::new(),
    })
}

impl TryFrom<&PaddedTable<IcoTable2D<f64>>> for Table3DFlat<f32> {
    type Error = anyhow::Error;
    fn try_from(table: &PaddedTable<IcoTable2D<f64>>) -> Result<Self> {
        build_flat_3d(table, |v| v as f32)
    }
}

impl TryFrom<&PaddedTable<IcoTable2D<f64>>> for Table3DFlat<half::f16> {
    type Error = anyhow::Error;
    fn try_from(table: &PaddedTable<IcoTable2D<f64>>) -> Result<Self> {
        build_flat_3d(table, half::f16::from_f64)
    }
}

impl<T: num_traits::Float + Into<f64>> Table3DFlat<T> {
    /// Lookup value by nearest R bin and barycentric interpolation on icosphere.
    pub fn lookup(&self, r: f64, direction: &Vector3) -> f64 {
        if r < self.rmin || r > self.rmax {
            return 0.0;
        }
        let ri = ((r - self.rmin) / self.dr + 0.5) as usize;
        let ri = ri.min(self.n_r.saturating_sub(1));

        let loc = self
            .locator
            .get_or_init(|| VertexLocator::new(&self.vertices));
        let (face, bary) = loc.find_face_bary(direction, &self.vertices, &self.neighbors);
        let base = ri * self.n_vertices;

        let mut result = 0.0;
        for i in 0..3 {
            let val: f64 = self.data[base + face[i]].into();
            result += bary[i] * val;
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_mesh(min_points: usize) -> (usize, Vec<[f64; 3]>, Vec<Vec<u16>>) {
        let ico = IcoTable2D::<f64>::from_min_points(min_points).unwrap();
        let (vertices, neighbors) = extract_mesh(&ico);
        (ico.len(), vertices, neighbors)
    }

    /// Build a small table from a real icosphere for lookup testing.
    fn make_test_table(min_points: usize, n_r: usize, n_omega: usize) -> Table6DFlat<f32> {
        let (n_v, vertices, neighbors) = make_test_mesh(min_points);
        let rmin = 5.0;
        let dr = 1.0;
        let rmax = rmin + (n_r as f64) * dr;
        let omega_step = std::f64::consts::TAU / n_omega as f64;

        // Fill with constant 42.0 so lookup should return ~42 for any query
        let data = vec![42.0f32; n_r * n_omega * n_v * n_v];
        Table6DFlat {
            rmin,
            rmax,
            dr,
            n_r,
            omega_step,
            n_omega,
            n_vertices: n_v,
            vertices,
            neighbors,
            data,
            metadata: None,
            locator: OnceLock::new(),
        }
    }

    #[test]
    fn lookup_constant_table() {
        let table = make_test_table(1, 5, 8);
        // Any direction inside the R range should return ~42
        let dir_a = Vector3::new(1.0, 0.0, 0.0);
        let dir_b = Vector3::new(0.0, 1.0, 0.0);
        let e = table.lookup(7.0, 1.0, &dir_a, &dir_b);
        assert!((e - 42.0).abs() < 1e-4, "Expected ~42, got {}", e);
        // Different directions should also give 42
        let dir_c = Vector3::new(1.0, 1.0, 1.0).normalize();
        let e2 = table.lookup(6.5, 3.0, &dir_c, &dir_a);
        assert!((e2 - 42.0).abs() < 1e-4, "Expected ~42, got {}", e2);
    }

    #[test]
    fn boltzmann_lookup_constant_table() {
        let table = make_test_table(1, 5, 8);
        let dir_a = Vector3::new(1.0, 0.0, 0.0);
        let dir_b = Vector3::new(0.0, 1.0, 0.0);
        // Constant table: Boltzmann lookup should also return ~42 for any beta
        let e = table.lookup_boltzmann(7.0, 1.0, &dir_a, &dir_b, 1.0);
        assert!((e - 42.0).abs() < 1e-4, "Expected ~42, got {}", e);
        let e2 = table.lookup_boltzmann(7.0, 1.0, &dir_a, &dir_b, 0.5);
        assert!((e2 - 42.0).abs() < 1e-4, "Expected ~42, got {}", e2);
    }

    #[test]
    fn boltzmann_vs_linear_convex() {
        // For a convex function, Boltzmann interpolation should give
        // lower (less biased) values than linear interpolation.
        let table = make_table_with_fn(1, |va, vb| 10.0 * (va - vb).norm());

        // Off-vertex direction: Boltzmann should be <= linear for convex data
        let dir_a = Vector3::new(1.0, 1.0, 1.0).normalize();
        let dir_b = Vector3::new(-1.0, 0.5, 0.3).normalize();
        let linear = table.lookup(6.0, 0.0, &dir_a, &dir_b);
        let boltz = table.lookup_boltzmann(6.0, 0.0, &dir_a, &dir_b, 1.0);
        assert!(
            boltz <= linear + 1e-10,
            "Boltzmann ({boltz:.4}) should be <= linear ({linear:.4}) for convex function"
        );
    }

    #[test]
    fn lookup_out_of_range_returns_zero() {
        let table = make_test_table(1, 5, 8);
        let dir = Vector3::new(1.0, 0.0, 0.0);
        assert_eq!(table.lookup(3.0, 0.0, &dir, &dir), 0.0); // below rmin
        assert_eq!(table.lookup(20.0, 0.0, &dir, &dir), 0.0); // above rmax
    }

    #[test]
    fn lookup_at_vertex_exact() {
        let (n_v, vertices, neighbors) = make_test_mesh(1);
        let n_r = 3;
        let n_omega = 4;

        // Set data[vi, vj] = (vi + vj) as f32 for r_idx=1, omega_idx=0
        let mut data = vec![0.0f32; n_r * n_omega * n_v * n_v];
        let stride = n_omega * n_v * n_v;
        let ri = 1; // r = rmin + 1*dr = 6.0
        for vi in 0..n_v {
            for vj in 0..n_v {
                data[ri * stride + 0 * n_v * n_v + vi * n_v + vj] = (vi + vj) as f32;
            }
        }

        let table = Table6DFlat {
            rmin: 5.0,
            rmax: 8.0,
            dr: 1.0,
            n_r,
            omega_step: std::f64::consts::TAU / n_omega as f64,
            n_omega,
            n_vertices: n_v,
            vertices,
            neighbors,
            data,
            metadata: None,
            locator: OnceLock::new(),
        };

        // Query exactly at vertex 0 direction for both → should get 0+0=0
        let dir0 = Vector3::from(table.vertices[0]);
        let e = table.lookup(6.0, 0.0, &dir0, &dir0);
        assert!(e.abs() < 1.0, "Expected ~0 at vertex (0,0), got {}", e);
    }

    /// Analytic test function: f(dir_a, dir_b) = dot(dir_a, ref)^2 + dot(dir_b, ref)^2
    /// Smooth on S², well-suited for barycentric interpolation accuracy checks.
    fn analytic_fn(dir_a: &Vector3, dir_b: &Vector3) -> f64 {
        let reference = Vector3::new(1.0, 1.0, 1.0).normalize();
        let da = dir_a.normalize().dot(&reference);
        let db = dir_b.normalize().dot(&reference);
        da * da + db * db
    }

    /// Build a table where each vertex pair is filled by `f(dir_a, dir_b)`.
    fn make_table_with_fn(
        min_points: usize,
        f: impl Fn(&Vector3, &Vector3) -> f64,
    ) -> Table6DFlat<f32> {
        let (n_v, vertices, neighbors) = make_test_mesh(min_points);
        let n_r = 3;
        let n_omega = 4;
        let rmin = 5.0;
        let dr = 1.0;
        let rmax = rmin + n_r as f64 * dr;
        let omega_step = std::f64::consts::TAU / n_omega as f64;
        let stride = n_omega * n_v * n_v;

        let mut data = vec![0.0f32; n_r * stride];
        for ri in 0..n_r {
            for oi in 0..n_omega {
                for vi in 0..n_v {
                    let va = Vector3::from(vertices[vi]);
                    for vj in 0..n_v {
                        let vb = Vector3::from(vertices[vj]);
                        let idx = ri * stride + oi * n_v * n_v + vi * n_v + vj;
                        data[idx] = f(&va, &vb) as f32;
                    }
                }
            }
        }

        Table6DFlat {
            rmin,
            rmax,
            dr,
            n_r,
            omega_step,
            n_omega,
            n_vertices: n_v,
            vertices,
            neighbors,
            data,
            metadata: None,
            locator: OnceLock::new(),
        }
    }

    fn make_analytic_table(min_points: usize) -> Table6DFlat<f32> {
        make_table_with_fn(min_points, analytic_fn)
    }

    /// Test that orient → inverse_orient → lookup recovers the correct table value
    /// for exact vertex orientations with arbitrary initial quaternions.
    #[test]
    fn lookup_via_orient_inverse_roundtrip() {
        use crate::orient::{inverse_orient, orient};
        use rand::Rng;

        let table = make_analytic_table(42); // match resolution used in production
        let mut rng = rand::thread_rng();
        let r = 6.0; // within table range

        for vi in 0..table.n_vertices.min(10) {
            for vj in 0..table.n_vertices.min(10) {
                let va = Vector3::from(table.vertices[vi]);
                let vb = Vector3::from(table.vertices[vj]);

                for _ in 0..5 {
                    let omega = rng.gen_range(0.0..std::f64::consts::TAU);

                    // Forward: get quaternions and separation for this pose
                    let (q_a, q_b, sep) = orient(r, omega, &va, &vb);

                    // Now simulate what Faunus does: apply an arbitrary rotation
                    // to both molecules (as if they were rotated during placement)
                    let random_q = nalgebra::UnitQuaternion::from_euler_angles(
                        rng.gen_range(0.0..std::f64::consts::TAU),
                        rng.gen_range(0.0..std::f64::consts::PI),
                        rng.gen_range(0.0..std::f64::consts::TAU),
                    );
                    let q_a_rot = random_q * q_a;
                    let q_b_rot = random_q * q_b;
                    let sep_rot = random_q.transform_vector(&sep);

                    // Inverse: recover 6D coordinates
                    let (r2, omega2, dir_a2, dir_b2) = inverse_orient(&sep_rot, &q_a_rot, &q_b_rot);

                    // Lookup should give the same value as the exact vertex value
                    let looked_up = table.lookup(r2, omega2, &dir_a2, &dir_b2);
                    let exact = analytic_fn(&va, &vb);

                    // On-vertex queries should be near-exact (only f32 rounding)
                    assert!(
                        (looked_up - exact).abs() < 0.05,
                        "vi={vi}, vj={vj}: looked_up={looked_up:.4}, exact={exact:.4}, \
                         dir_a={dir_a2:?}, dir_b={dir_b2:?}, omega={omega2:.4}"
                    );
                }
            }
        }
    }

    #[test]
    fn lookup_off_vertex_interpolation() {
        use rand::Rng;

        // 162 vertices (subdivision 2) for reasonable angular resolution
        let table = make_analytic_table(162);

        let mut rng = rand::thread_rng();
        let mut max_err = 0.0f64;

        for _ in 0..500 {
            let random_dir = |rng: &mut rand::rngs::ThreadRng| loop {
                let x: f64 = rng.gen_range(-1.0..1.0);
                let y: f64 = rng.gen_range(-1.0..1.0);
                let z: f64 = rng.gen_range(-1.0..1.0);
                let r2 = x * x + y * y + z * z;
                if r2 > 0.01 && r2 < 1.0 {
                    return Vector3::new(x, y, z).normalize();
                }
            };

            let dir_a = random_dir(&mut rng);
            let dir_b = random_dir(&mut rng);

            let interpolated = table.lookup(6.0, 0.0, &dir_a, &dir_b);
            let exact = analytic_fn(&dir_a, &dir_b);
            max_err = max_err.max((interpolated - exact).abs());
        }

        assert!(
            max_err < 0.3,
            "Max interpolation error {:.4} exceeds tolerance for smooth quadratic",
            max_err
        );
    }

    #[test]
    fn round_trip_save_load() {
        let table = Table6DFlat::<f32> {
            rmin: 5.0,
            rmax: 10.0,
            dr: 1.0,
            n_r: 5,
            omega_step: 0.5,
            n_omega: 4,
            n_vertices: 3,
            vertices: vec![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            neighbors: vec![vec![1, 2], vec![0, 2], vec![0, 1]],
            data: vec![1.0; 5 * 4 * 3 * 3],
            metadata: None,
            locator: OnceLock::new(),
        };
        let dir = tempfile::tempdir().unwrap();

        // Uncompressed round-trip
        let path = dir.path().join("test.bin");
        table.save(&path).unwrap();
        let loaded = Table6DFlat::<f32>::load(&path).unwrap();
        assert_eq!(loaded.n_vertices, 3);
        assert_eq!(loaded.data.len(), table.data.len());

        // Compressed round-trip
        let gz_path = dir.path().join("test.bin.gz");
        table.save(&gz_path).unwrap();
        let loaded_gz = Table6DFlat::<f32>::load(&gz_path).unwrap();
        assert_eq!(loaded_gz.n_vertices, 3);
        assert_eq!(loaded_gz.data, table.data);

        // Compressed file should be smaller
        let raw_size = std::fs::metadata(&path).unwrap().len();
        let gz_size = std::fs::metadata(&gz_path).unwrap().len();
        assert!(gz_size < raw_size);
    }

    fn make_test_table_3d(min_points: usize, n_r: usize) -> Table3DFlat<f32> {
        let (n_v, vertices, neighbors) = make_test_mesh(min_points);
        let rmin = 5.0;
        let dr = 1.0;
        let rmax = rmin + (n_r as f64) * dr;
        let data = vec![42.0f32; n_r * n_v];
        Table3DFlat {
            rmin,
            rmax,
            dr,
            n_r,
            n_vertices: n_v,
            vertices,
            neighbors,
            data,
            locator: OnceLock::new(),
        }
    }

    #[test]
    fn table3d_lookup_constant() {
        let table = make_test_table_3d(1, 5);
        let dir = Vector3::new(1.0, 0.0, 0.0);
        let e = table.lookup(7.0, &dir);
        assert!((e - 42.0).abs() < 1e-4, "Expected ~42, got {}", e);
        let dir2 = Vector3::new(1.0, 1.0, 1.0).normalize();
        let e2 = table.lookup(6.5, &dir2);
        assert!((e2 - 42.0).abs() < 1e-4, "Expected ~42, got {}", e2);
    }

    #[test]
    fn table3d_out_of_range_returns_zero() {
        let table = make_test_table_3d(1, 5);
        let dir = Vector3::new(1.0, 0.0, 0.0);
        assert_eq!(table.lookup(3.0, &dir), 0.0);
        assert_eq!(table.lookup(20.0, &dir), 0.0);
    }

    #[test]
    fn table3d_round_trip_save_load() {
        let table = make_test_table_3d(1, 5);
        let dir = tempfile::tempdir().unwrap();

        let gz_path = dir.path().join("test3d.bin.gz");
        table.save(&gz_path).unwrap();
        let loaded = Table3DFlat::<f32>::load(&gz_path).unwrap();
        assert_eq!(loaded.n_vertices, table.n_vertices);
        assert_eq!(loaded.data, table.data);
    }

    #[test]
    fn f16_round_trip_save_load() {
        use half::f16;
        let table = Table6DFlat::<f16> {
            rmin: 5.0,
            rmax: 10.0,
            dr: 1.0,
            n_r: 5,
            omega_step: 0.5,
            n_omega: 4,
            n_vertices: 3,
            vertices: vec![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            neighbors: vec![vec![1, 2], vec![0, 2], vec![0, 1]],
            data: vec![f16::from_f64(1.0); 5 * 4 * 3 * 3],
            metadata: None,
            locator: OnceLock::new(),
        };
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test_f16.bin.gz");
        table.save(&path).unwrap();
        let loaded = Table6DFlat::<f16>::load(&path).unwrap();
        assert_eq!(loaded.n_vertices, 3);
        assert_eq!(loaded.data, table.data);
    }

    #[test]
    fn f16_lookup_constant_table() {
        use half::f16;
        let (n_v, vertices, neighbors) = make_test_mesh(1);
        let n_r = 5;
        let n_omega = 8;
        let rmin = 5.0;
        let dr = 1.0;
        let rmax = rmin + (n_r as f64) * dr;
        let omega_step = std::f64::consts::TAU / n_omega as f64;
        let data = vec![f16::from_f64(42.0); n_r * n_omega * n_v * n_v];
        let table = Table6DFlat {
            rmin,
            rmax,
            dr,
            n_r,
            omega_step,
            n_omega,
            n_vertices: n_v,
            vertices,
            neighbors,
            data,
            metadata: None,
            locator: OnceLock::new(),
        };
        let dir_a = Vector3::new(1.0, 0.0, 0.0);
        let dir_b = Vector3::new(0.0, 1.0, 0.0);
        let e = table.lookup(7.0, 1.0, &dir_a, &dir_b);
        assert!((e - 42.0).abs() < 0.1, "Expected ~42, got {e}");
    }

    #[test]
    fn bfs_permutation_is_bijection() {
        let (_, _, neighbors) = make_test_mesh(42);
        let perm = bfs_vertex_permutation(&neighbors);
        let n = neighbors.len();
        assert_eq!(perm.len(), n);
        let mut seen = vec![false; n];
        for &new_id in &perm {
            assert!(new_id < n, "out of range: {new_id}");
            assert!(!seen[new_id], "duplicate: {new_id}");
            seen[new_id] = true;
        }
    }

    #[test]
    fn reordered_lookup_matches_original() {
        // Build a table with non-trivial per-vertex data, then reorder and compare lookups
        let (n_v, vertices, neighbors) = make_test_mesh(42);
        let n_r = 3;
        let n_omega = 4;
        let rmin = 5.0;
        let dr = 1.0;
        let rmax = rmin + n_r as f64 * dr;
        let omega_step = std::f64::consts::TAU / n_omega as f64;
        let stride = n_omega * n_v * n_v;

        // Fill with f(vi,vj) = vi*n_v + vj
        let mut data = vec![0.0f32; n_r * stride];
        for s in 0..n_r * n_omega {
            for vi in 0..n_v {
                for vj in 0..n_v {
                    data[s * n_v * n_v + vi * n_v + vj] = (vi * n_v + vj) as f32;
                }
            }
        }

        let orig = Table6DFlat {
            rmin,
            rmax,
            dr,
            n_r,
            omega_step,
            n_omega,
            n_vertices: n_v,
            vertices: vertices.clone(),
            neighbors: neighbors.clone(),
            data: data.clone(),
            metadata: None,
            locator: OnceLock::new(),
        };

        // Apply BFS reordering
        let perm = bfs_vertex_permutation(&neighbors);
        let mut rv = vertices.clone();
        let mut rn = neighbors.clone();
        let mut rd = data.clone();
        apply_vertex_permutation(&perm, &mut rv, &mut rn, &mut rd, n_v, n_v);

        let reordered = Table6DFlat {
            rmin,
            rmax,
            dr,
            n_r,
            omega_step,
            n_omega,
            n_vertices: n_v,
            vertices: rv,
            neighbors: rn,
            data: rd,
            metadata: None,
            locator: OnceLock::new(),
        };

        // Lookups at random directions should match
        use rand::Rng;
        let mut rng = rand::thread_rng();
        for _ in 0..200 {
            let random_dir = |rng: &mut rand::rngs::ThreadRng| loop {
                let x: f64 = rng.gen_range(-1.0..1.0);
                let y: f64 = rng.gen_range(-1.0..1.0);
                let z: f64 = rng.gen_range(-1.0..1.0);
                let r2 = x * x + y * y + z * z;
                if r2 > 0.01 && r2 < 1.0 {
                    return Vector3::new(x, y, z).normalize();
                }
            };
            let da = random_dir(&mut rng);
            let db = random_dir(&mut rng);
            let r = rng.gen_range(rmin..rmax);
            let omega = rng.gen_range(0.0..std::f64::consts::TAU);
            let e_orig = orig.lookup(r, omega, &da, &db);
            let e_reord = reordered.lookup(r, omega, &da, &db);
            assert!(
                (e_orig - e_reord).abs() < 1e-4,
                "Mismatch: orig={e_orig}, reordered={e_reord}"
            );
        }
    }

    /// Verify that `VertexLocator` produces identical results to the O(V) reference.
    #[test]
    fn locator_matches_linear_scan() {
        use rand::Rng;

        let (_, vertices, neighbors) = make_test_mesh(162);
        let locator = VertexLocator::new(&vertices);
        let mut rng = rand::thread_rng();

        for _ in 0..1000 {
            let dir = loop {
                let x: f64 = rng.gen_range(-1.0..1.0);
                let y: f64 = rng.gen_range(-1.0..1.0);
                let z: f64 = rng.gen_range(-1.0..1.0);
                let r2 = x * x + y * y + z * z;
                if r2 > 0.01 && r2 < 1.0 {
                    break Vector3::new(x, y, z);
                }
            };
            let (face_ref, bary_ref) = find_face_bary(&dir, &vertices, &neighbors);
            let (face_loc, bary_loc) = locator.find_face_bary(&dir, &vertices, &neighbors);
            assert_eq!(face_ref, face_loc, "Face mismatch for dir={dir:?}");
            for k in 0..3 {
                assert!(
                    (bary_ref[k] - bary_loc[k]).abs() < 1e-12,
                    "Bary mismatch for dir={dir:?}: ref={bary_ref:?}, loc={bary_loc:?}"
                );
            }
        }
    }

    #[test]
    fn round_trip_with_metadata() {
        let table = Table6DFlat::<f32> {
            rmin: 5.0,
            rmax: 10.0,
            dr: 1.0,
            n_r: 5,
            omega_step: 0.5,
            n_omega: 4,
            n_vertices: 3,
            vertices: vec![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            neighbors: vec![vec![1, 2], vec![0, 2], vec![0, 1]],
            data: vec![1.0; 5 * 4 * 3 * 3],
            metadata: Some(TableMetadata {
                tail_terms: vec![
                    TailCorrectionTerm {
                        coefficient: 100.0,
                        kappa: 0.3,
                        power: 1,
                    },
                    TailCorrectionTerm {
                        coefficient: -5.0,
                        kappa: 0.1,
                        power: 4,
                    },
                ],
                charges: Some([1.0, -1.0]),
                dipole_moments: Some([2.5, 0.0]),
                temperature: Some(298.15),
                electric_prefactor: Some(17.83),
            }),
            locator: OnceLock::new(),
        };
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("meta.bin.gz");
        table.save(&path).unwrap();
        let loaded = Table6DFlat::<f32>::load(&path).unwrap();
        let meta = loaded.metadata.as_ref().expect("metadata missing");
        assert_eq!(meta.tail_terms.len(), 2);
        assert_eq!(meta.charges, Some([1.0, -1.0]));
        assert_eq!(meta.temperature, Some(298.15));
    }

    #[test]
    fn load_legacy_without_metadata() {
        // Serialize a legacy struct (no metadata field), then load via Table6DFlat
        let legacy = Table6DFlatLegacy::<f32> {
            rmin: 5.0,
            rmax: 10.0,
            dr: 1.0,
            n_r: 5,
            omega_step: 0.5,
            n_omega: 4,
            n_vertices: 3,
            vertices: vec![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            neighbors: vec![vec![1, 2], vec![0, 2], vec![0, 1]],
            data: vec![1.0; 5 * 4 * 3 * 3],
        };
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("legacy.bin");
        save_bincode(&legacy, &path).unwrap();
        let loaded = Table6DFlat::<f32>::load(&path).unwrap();
        assert!(loaded.metadata.is_none());
        assert_eq!(loaded.n_vertices, 3);
        assert_eq!(loaded.data.len(), 5 * 4 * 3 * 3);
    }

    #[test]
    fn tail_energy_evaluation() {
        let mut table = make_test_table(1, 3, 4);
        // No metadata → tail_energy returns 0
        assert_eq!(table.tail_energy(20.0), 0.0);

        table.metadata = Some(TableMetadata {
            tail_terms: vec![TailCorrectionTerm {
                coefficient: 100.0,
                kappa: 0.1,
                power: 1,
            }],
            electric_prefactor: Some(2.5),
            ..Default::default()
        });
        // 2.5 * 100 * exp(-0.1 * 20) / 20
        let expected = 2.5 * 100.0 * (-2.0_f64).exp() / 20.0;
        let got = table.tail_energy(20.0);
        assert!(
            (got - expected).abs() < 1e-10,
            "expected {expected}, got {got}"
        );
    }
}
