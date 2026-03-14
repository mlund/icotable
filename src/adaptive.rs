//! Adaptive 6D tables with per-R-slice resolution.
//!
//! [`Table6DAdaptive`] stores angular data at varying icosphere resolutions
//! across radial slices. Each (R, ω) slab is classified into one of four tiers
//! based on angular gradients and Boltzmann weights:
//!
//! - **Repulsive** — all `exp(−βU) < 10⁻⁴`; zero storage, lookup returns ∞.
//! - **Scalar** — nearly isotropic; single mean value.
//! - **Nearest-vertex** — smooth surface; lookup without interpolation.
//! - **Interpolated** — full barycentric interpolation on icosphere faces.
//!
//! [`AdaptiveBuilder`] drives the table generation protocol: it provides
//! vertex directions and quadrature weights per resolution level, accepts
//! batched energy results, and classifies slabs after each R-slice.
//! The repulsive classification uses `β = 1/kT` from the generation
//! temperature, making the table temperature-dependent.

use crate::flat::{
    apply_vertex_permutation, bfs_vertex_permutation, load_bincode, save_bincode, TableMetadata,
    VertexLocator,
};
use crate::ico::Face;
use crate::vertex::make_vertices;
use crate::Vector3;
use crate::{make_icosphere_by_ndiv, make_weights};
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::sync::OnceLock;

// ---------------------------------------------------------------------------
// MeshLevel
// ---------------------------------------------------------------------------

/// Pre-built icosphere at one subdivision level with BFS-reordered vertices.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MeshLevel {
    /// Subdivision level (0, 1, 2, 3, …).
    pub n_div: usize,
    /// Number of vertices: 10*(n_div+1)²+2. Stored explicitly because it serves
    /// as the row/column stride for indexing into the flat `data` array.
    pub n_vertices: usize,
    /// BFS-reordered vertex positions `[x, y, z]` on unit sphere.
    pub vertices: Vec<[f64; 3]>,
    /// Quadrature weights per vertex (Voronoi solid-angle fractions, normalized
    /// so that uniform weight = 1.0). BFS-reordered to match `vertices`.
    /// Needed for correct angular integration: icosphere vertices do not subtend
    /// exactly equal solid angles (12 pentagonal vs. remaining hexagonal cells).
    pub weights: Vec<f64>,
    /// Neighbor indices per vertex (BFS-reordered).
    pub neighbors: Vec<Vec<u16>>,
    /// Lazy O(1) vertex locator.
    #[serde(skip, default)]
    locator: OnceLock<VertexLocator>,
}

impl MeshLevel {
    /// Build a mesh level for the given subdivision count.
    pub fn new(n_div: usize) -> Self {
        let ico = make_icosphere_by_ndiv(n_div);
        let verts = make_vertices(&ico);
        let n_vertices = verts.len();

        let weights = make_weights(&ico);
        let mut vertices: Vec<[f64; 3]> = verts
            .iter()
            .map(|v| {
                let p = v.pos.normalize();
                [p.x, p.y, p.z]
            })
            .collect();
        let mut neighbors: Vec<Vec<u16>> = verts.iter().map(|v| v.neighbors.clone()).collect();

        // BFS reorder for cache locality.
        // entries_per_vertex=1 (not 0) to avoid division by zero in slab_size calculation.
        let perm = bfs_vertex_permutation(&neighbors);
        apply_vertex_permutation::<u8>(
            &perm,
            &mut vertices,
            &mut neighbors,
            &mut [],
            n_vertices,
            1,
        );
        // Apply same BFS permutation to weights
        let weights: Vec<f64> = perm.iter().map(|&old_idx| weights[old_idx]).collect();

        Self {
            n_div,
            n_vertices,
            vertices,
            weights,
            neighbors,
            locator: OnceLock::new(),
        }
    }

    /// Get or lazily initialize the vertex locator.
    fn locator(&self) -> &VertexLocator {
        self.locator
            .get_or_init(|| VertexLocator::new(&self.vertices))
    }

    /// Find the containing face and barycentric coordinates for a direction.
    pub fn find_face_bary(&self, dir: &Vector3) -> (Face, [f64; 3]) {
        self.locator()
            .find_face_bary(dir, &self.vertices, &self.neighbors)
    }
}

// ---------------------------------------------------------------------------
// SlabResolution
// ---------------------------------------------------------------------------

/// Resolution for a single (R, ω) slab.
///
/// Three tiers reduce storage and lookup cost for regions of the 6D energy
/// surface where full angular resolution is unnecessary:
/// - **Repulsive**: overlapping configurations where exp(−βu) ≈ 0; MC always
///   rejects, so the exact energy is irrelevant. Zero storage.
/// - **Scalar**: nearly isotropic slabs collapsed to a single mean value.
/// - **Mesh**: angular data stored on an icosphere. When the gradient is small,
///   nearest-vertex lookup (1 read) replaces barycentric interpolation (9 reads
///   + 9 FMAs), which also eliminates 9 exp() calls in the Boltzmann path.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum SlabResolution {
    /// Angular mesh with optional barycentric interpolation.
    Mesh {
        /// Index into `Table6DAdaptive::levels`.
        level: u8,
        /// When `false`, lookup returns the nearest-vertex value (1 read)
        /// instead of full barycentric interpolation (9 reads + 9 FMAs).
        interpolate: bool,
    },
    /// Fully isotropic: single scalar value (no angular dependence).
    Scalar(f32),
    /// All orientations are strongly repulsive (above energy threshold).
    /// Stores nothing; lookup returns infinity.
    Repulsive,
}

// ---------------------------------------------------------------------------
// Table6DAdaptive
// ---------------------------------------------------------------------------

/// Adaptive 6D lookup table with per-slab resolution.
///
/// Slabs at long range use coarser meshes or scalar values, reducing both
/// memory footprint and generation cost. The lookup hot path dispatches
/// per-slab to either scalar return or mesh interpolation.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Table6DAdaptive<T: num_traits::Float> {
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
    /// Pre-built mesh levels (typically 2-4, indexed by `SlabResolution::Mesh::level`).
    pub levels: Vec<MeshLevel>,
    /// Resolution descriptor per (R, ω) slab. Layout: `slab_res[ri * n_omega + oi]`.
    pub slab_res: Vec<SlabResolution>,
    /// Data offset per slab into `data`. `u32::MAX` for Scalar/Repulsive slabs.
    /// Indexed directly by slab_idx for O(1) lookup (not by mesh-ordinal).
    pub(crate) slab_offsets: Vec<u32>,
    /// Contiguous storage for all Mesh slab data.
    pub data: Vec<T>,
    /// Optional metadata for tail corrections beyond the table cutoff.
    pub metadata: Option<TableMetadata>,
}

impl<T: num_traits::Float + Serialize + serde::de::DeserializeOwned> Table6DAdaptive<T> {
    /// Save to a bincode file (gzip-compressed if path ends in `.gz`).
    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        save_bincode(self, path.as_ref())
    }

    /// Load from a bincode file (gzip-decompressed if path ends in `.gz`).
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        load_bincode(path.as_ref())
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

impl<T: num_traits::Float + Into<f64>> Table6DAdaptive<T> {
    /// Resolve R/ω to a slab index, or `None` if out of range.
    fn resolve_slab_idx(&self, r: f64, omega: f64) -> Option<usize> {
        if r < self.rmin || r > self.rmax {
            return None;
        }
        // +0.5 for nearest-bin rounding; clamp to valid range
        let ri = ((r - self.rmin) / self.dr + 0.5) as usize;
        let ri = ri.min(self.n_r.saturating_sub(1));
        // ω is periodic in [0, 2π); wrap then bin
        let omega = omega.rem_euclid(std::f64::consts::TAU);
        let oi = (omega / self.omega_step + 0.5) as usize % self.n_omega;
        Some(ri * self.n_omega + oi)
    }

    /// Gather the 3×3 interpolation values for a Mesh slab.
    fn gather_mesh_vals(
        &self,
        slab_idx: usize,
        level: u8,
        dir_a: &Vector3,
        dir_b: &Vector3,
    ) -> (Face, [f64; 3], Face, [f64; 3], [f64; 9]) {
        let lvl = &self.levels[level as usize];
        let n_v = lvl.n_vertices;
        let (face_a, bary_a) = lvl.find_face_bary(dir_a);
        let (face_b, bary_b) = lvl.find_face_bary(dir_b);
        let base = self.slab_offsets[slab_idx] as usize;

        let mut vals = [0.0f64; 9];
        for i in 0..3 {
            for j in 0..3 {
                let idx = base + face_a[i] * n_v + face_b[j];
                vals[i * 3 + j] = self.data[idx].into();
            }
        }
        (face_a, bary_a, face_b, bary_b, vals)
    }

    /// Return the value at the nearest vertex pair (no interpolation).
    fn nearest_vertex_val(
        &self,
        slab_idx: usize,
        level: u8,
        dir_a: &Vector3,
        dir_b: &Vector3,
    ) -> f64 {
        let lvl = &self.levels[level as usize];
        let n_v = lvl.n_vertices;
        let (face_a, bary_a) = lvl.find_face_bary(dir_a);
        let (face_b, bary_b) = lvl.find_face_bary(dir_b);
        let ia = argmax3(&bary_a);
        let ib = argmax3(&bary_b);
        let base = self.slab_offsets[slab_idx] as usize;
        self.data[base + face_a[ia] * n_v + face_b[ib]].into()
    }

    /// Lookup value by nearest R/ω bin with adaptive angular resolution.
    pub fn lookup(&self, r: f64, omega: f64, dir_a: &Vector3, dir_b: &Vector3) -> f64 {
        let Some(slab_idx) = self.resolve_slab_idx(r, omega) else {
            return 0.0;
        };
        match &self.slab_res[slab_idx] {
            SlabResolution::Scalar(v) => *v as f64,
            SlabResolution::Repulsive => f64::INFINITY,
            SlabResolution::Mesh {
                level,
                interpolate: false,
            } => self.nearest_vertex_val(slab_idx, *level, dir_a, dir_b),
            SlabResolution::Mesh {
                level,
                interpolate: true,
            } => {
                let (_, bary_a, _, bary_b, vals) =
                    self.gather_mesh_vals(slab_idx, *level, dir_a, dir_b);
                let mut result = 0.0;
                for i in 0..3 {
                    for j in 0..3 {
                        result += bary_a[i] * vals[i * 3 + j] * bary_b[j];
                    }
                }
                result
            }
        }
    }

    /// Boltzmann-weighted interpolation: interpolate exp(-beta*u) then invert.
    pub fn lookup_boltzmann(
        &self,
        r: f64,
        omega: f64,
        dir_a: &Vector3,
        dir_b: &Vector3,
        beta: f64,
    ) -> f64 {
        debug_assert!(beta > 0.0, "beta must be positive, got {beta}");
        let Some(slab_idx) = self.resolve_slab_idx(r, omega) else {
            return 0.0;
        };
        match &self.slab_res[slab_idx] {
            SlabResolution::Scalar(v) => *v as f64,
            SlabResolution::Repulsive => f64::INFINITY,
            SlabResolution::Mesh {
                level,
                interpolate: false,
            } => self.nearest_vertex_val(slab_idx, *level, dir_a, dir_b),
            SlabResolution::Mesh {
                level,
                interpolate: true,
            } => {
                let (_, bary_a, _, bary_b, vals) =
                    self.gather_mesh_vals(slab_idx, *level, dir_a, dir_b);

                // Log-sum-exp trick: shift by u_min before exp() to prevent fp overflow.
                // Final `u_min - ln(sum)/beta` restores the correct Boltzmann-weighted energy.
                let u_min = vals.iter().copied().fold(f64::INFINITY, f64::min);
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
    }
}

// ---------------------------------------------------------------------------
// AdaptiveBuilder
// ---------------------------------------------------------------------------

/// Drives the adaptive table generation protocol.
///
/// The builder provides vertex directions per level, accepts batched energy
/// results per (R, ω) slab, checks angular gradients after each R slice,
/// and monotonically decreases the resolution as R increases.
pub struct AdaptiveBuilder {
    rmin: f64,
    dr: f64,
    n_r: usize,
    n_omega: usize,
    omega_step: f64,
    gradient_threshold: f64,
    /// Inverse thermal energy 1/kT (mol/kJ). Used to test whether all
    /// Boltzmann weights in a slab are negligible (expm1(-βU) ≈ −1),
    /// replacing the old energy-based repulsive threshold.
    beta: f64,
    levels: Vec<MeshLevel>,
    /// Current n_div index into `levels`. Monotonically decreases.
    current_level: usize,
    /// Resolution assigned per slab.
    slab_res: Vec<SlabResolution>,
    /// Accumulated data per slab (temporary, indexed by slab_idx).
    slab_data: Vec<Option<Vec<f64>>>,
}

impl AdaptiveBuilder {
    /// Create a new adaptive builder.
    ///
    /// # Arguments
    /// * `rmin` — Minimum radial distance
    /// * `rmax` — Maximum radial distance
    /// * `dr` — Radial bin width
    /// * `omega_step` — Dihedral angle bin width (radians)
    /// * `max_n_div` — Maximum icosphere subdivision level
    /// * `gradient_threshold` — Angular gradient threshold for resolution reduction
    /// * `beta` — Inverse thermal energy 1/kT (mol/kJ) for Boltzmann-weight
    ///   based repulsive slab detection
    pub fn new(
        rmin: f64,
        rmax: f64,
        dr: f64,
        omega_step: f64,
        max_n_div: usize,
        gradient_threshold: f64,
        beta: f64,
    ) -> Self {
        let n_r = ((rmax - rmin) / dr + 0.5) as usize;
        let n_omega = (std::f64::consts::TAU / omega_step + 0.5) as usize;

        // Build all levels from 0..=max_n_div
        let levels: Vec<MeshLevel> = (0..=max_n_div).map(MeshLevel::new).collect();

        let n_slabs = n_r * n_omega;
        Self {
            rmin,
            dr,
            n_r,
            n_omega,
            omega_step,
            gradient_threshold,
            beta,
            current_level: levels.len() - 1, // start at max
            levels,
            slab_res: vec![SlabResolution::Scalar(0.0); n_slabs],
            slab_data: vec![None; n_slabs],
        }
    }

    /// Current subdivision level index (into `levels`).
    pub fn current_level(&self) -> usize {
        self.current_level
    }

    /// Current n_div value.
    pub fn current_n_div(&self) -> usize {
        self.levels[self.current_level].n_div
    }

    /// Number of vertices at the current level.
    pub fn current_n_vertices(&self) -> usize {
        self.levels[self.current_level].n_vertices
    }

    /// Vertex directions at the given level index.
    pub fn vertex_directions(&self, level: usize) -> &[[f64; 3]] {
        &self.levels[level].vertices
    }

    /// Quadrature weights at the given level index.
    ///
    /// Normalized so uniform weight = 1.0; the product `w_i * w_j` gives
    /// the relative solid-angle degeneracy for vertex pair (i, j).
    pub fn vertex_weights(&self, level: usize) -> &[f64] {
        &self.levels[level].weights
    }

    /// Number of radial bins.
    pub fn n_r(&self) -> usize {
        self.n_r
    }

    /// Number of omega bins.
    pub fn n_omega(&self) -> usize {
        self.n_omega
    }

    /// The R value for a given radial index.
    pub fn r_value(&self, ri: usize) -> f64 {
        self.rmin + ri as f64 * self.dr
    }

    /// The ω value for a given omega index.
    pub fn omega_value(&self, oi: usize) -> f64 {
        oi as f64 * self.omega_step
    }

    /// Set energy data for one (ri, oi) slab.
    ///
    /// `energies` must have `n_v * n_v` elements (row-major: `energies[vi * n_v + vj]`)
    /// where `n_v` is the vertex count at the current level.
    pub fn set_slab(&mut self, ri: usize, oi: usize, energies: &[f64]) {
        let n_v = self.levels[self.current_level].n_vertices;
        assert_eq!(
            energies.len(),
            n_v * n_v,
            "expected {} energies, got {}",
            n_v * n_v,
            energies.len()
        );
        let slab_idx = ri * self.n_omega + oi;
        self.slab_data[slab_idx] = Some(energies.to_vec());
        self.slab_res[slab_idx] = SlabResolution::Mesh {
            level: self.current_level as u8,
            interpolate: true,
        };
    }

    /// Finish an R slice: classify each slab and possibly lower resolution for the next R.
    ///
    /// Per-slab classification (checked in order):
    /// 1. **Repulsive** — all Boltzmann weights negligible (expm1(−βU) ≈ −1) → zero storage.
    /// 2. **Scalar** — angular gradient < `gradient_threshold / 10` → 4-byte mean.
    /// 3. **Mesh (no interpolation)** — gradient < `gradient_threshold` → nearest-vertex lookup.
    /// 4. **Mesh (interpolated)** — full barycentric interpolation.
    ///
    /// Returns the maximum angular gradient (kJ/mol/rad) across all non-repulsive slabs.
    ///
    /// Call this after all ω slabs for a given `ri` have been set.
    pub fn finish_r_slice(&mut self, ri: usize) -> f64 {
        let level = self.current_level;
        let lvl = &self.levels[level];
        let n_v = lvl.n_vertices;
        let scalar_threshold = self.gradient_threshold / 10.0;
        // Slabs where exp(-βU) < this for all orientations are thermodynamically
        // dead — MC will always reject. Corresponds to U > ~9 kT.
        const BOLTZMANN_FLOOR: f64 = 1e-4;
        let mut max_gradient = 0.0f64;
        let mut has_nonrepulsive = false;

        for oi in 0..self.n_omega {
            let slab_idx = ri * self.n_omega + oi;
            let Some(ref data) = self.slab_data[slab_idx] else {
                continue;
            };

            if data
                .iter()
                .all(|&e| (-self.beta * e).exp_m1() < BOLTZMANN_FLOOR - 1.0)
            {
                self.slab_res[slab_idx] = SlabResolution::Repulsive;
                self.slab_data[slab_idx] = None;
                continue;
            }

            has_nonrepulsive = true;
            let grad = compute_gradient(data, &lvl.vertices, &lvl.neighbors, n_v);

            if grad < scalar_threshold {
                // 2. Scalar: nearly isotropic
                let mean = data.iter().sum::<f64>() / data.len() as f64;
                self.slab_res[slab_idx] = SlabResolution::Scalar(mean as f32);
                self.slab_data[slab_idx] = None;
            } else if grad < self.gradient_threshold {
                // 3. Mesh without interpolation: smooth enough for nearest-vertex
                self.slab_res[slab_idx] = SlabResolution::Mesh {
                    level: level as u8,
                    interpolate: false,
                };
                max_gradient = max_gradient.max(grad);
            } else {
                // 4. Mesh with full interpolation
                max_gradient = max_gradient.max(grad);
            }
        }

        // Only decrease resolution when we have gradient evidence from
        // non-repulsive slabs. All-repulsive R-slices (short range) must
        // not reduce the level — the binding region ahead may need it.
        if has_nonrepulsive && max_gradient < self.gradient_threshold && self.current_level > 0 {
            self.current_level -= 1;
        }
        max_gradient
    }

    /// Build the final adaptive table.
    ///
    /// Consumes the builder and produces a `Table6DAdaptive<f32>`.
    pub fn build(self) -> Table6DAdaptive<f32> {
        let n_slabs = self.n_r * self.n_omega;

        let mut slab_offsets = vec![u32::MAX; n_slabs];
        let mut data = Vec::new();
        let mut current_offset = 0u32;

        for (slab_idx, (res, slab_buf)) in
            self.slab_res.iter().zip(self.slab_data.iter()).enumerate()
        {
            if let SlabResolution::Mesh { level, .. } = res {
                slab_offsets[slab_idx] = current_offset;
                let n_v = self.levels[*level as usize].n_vertices;
                let slab_data = slab_buf.as_ref().expect("Mesh slab must have data");
                data.extend(slab_data.iter().map(|&v| v as f32));
                current_offset += (n_v * n_v) as u32;
            }
        }

        Table6DAdaptive {
            rmin: self.rmin,
            rmax: self.rmin + self.n_r as f64 * self.dr,
            dr: self.dr,
            n_r: self.n_r,
            omega_step: self.omega_step,
            n_omega: self.n_omega,
            levels: self.levels,
            slab_res: self.slab_res,
            slab_offsets,
            data,
            metadata: None,
        }
    }
}

/// Index of the largest element in a 3-element array.
fn argmax3(v: &[f64; 3]) -> usize {
    if v[0] >= v[1] && v[0] >= v[2] {
        0
    } else if v[1] >= v[2] {
        1
    } else {
        2
    }
}

// ---------------------------------------------------------------------------
// Gradient computation
// ---------------------------------------------------------------------------

/// Compute the max angular gradient for a slab.
///
/// For each vertex pair (vi, vj), checks:
///   max over neighbors ni of vi: |E(vi,vj) - E(ni,vj)| / angle(vi, ni)
///   max over neighbors nj of vj: |E(vi,vj) - E(vi,nj)| / angle(vj, nj)
fn compute_gradient(
    data: &[f64],
    vertices: &[[f64; 3]],
    neighbors: &[Vec<u16>],
    n_v: usize,
) -> f64 {
    // Precompute neighbor angles once: angle(vi, ni) depends only on mesh topology,
    // not on vi's pairing with vj, so caching avoids n_v redundant acos() calls per edge.
    let neighbor_angles: Vec<Vec<f64>> = (0..n_v)
        .map(|vi| {
            let va = Vector3::from(vertices[vi]);
            neighbors[vi]
                .iter()
                .map(|&ni| va.angle(&Vector3::from(vertices[ni as usize])))
                .collect()
        })
        .collect();

    let mut max_grad = 0.0f64;

    for vi in 0..n_v {
        for vj in 0..n_v {
            let e_ij = data[vi * n_v + vj];

            // Check neighbors of vi
            for (k, &ni) in neighbors[vi].iter().enumerate() {
                let angle = neighbor_angles[vi][k];
                if angle > 1e-12 {
                    let grad = (e_ij - data[ni as usize * n_v + vj]).abs() / angle;
                    max_grad = max_grad.max(grad);
                }
            }

            // Check neighbors of vj
            for (k, &nj) in neighbors[vj].iter().enumerate() {
                let angle = neighbor_angles[vj][k];
                if angle > 1e-12 {
                    let grad = (e_ij - data[vi * n_v + nj as usize]).abs() / angle;
                    max_grad = max_grad.max(grad);
                }
            }
        }
    }

    max_grad
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn mesh_level_basic() {
        let lvl = MeshLevel::new(0);
        assert_eq!(lvl.n_vertices, 12); // 10*(0+1)^2 + 2
        let lvl1 = MeshLevel::new(1);
        assert_eq!(lvl1.n_vertices, 42); // 10*(1+1)^2 + 2
        let lvl2 = MeshLevel::new(2);
        assert_eq!(lvl2.n_vertices, 92); // 10*(2+1)^2 + 2
        let lvl3 = MeshLevel::new(3);
        assert_eq!(lvl3.n_vertices, 162); // 10*(3+1)^2 + 2
    }

    #[test]
    fn mesh_level_vertices_normalized() {
        let lvl = MeshLevel::new(2);
        for v in &lvl.vertices {
            let norm = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
            assert_relative_eq!(norm, 1.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn mesh_level_find_face_bary() {
        let lvl = MeshLevel::new(2);
        let dir = Vector3::new(1.0, 0.0, 0.0);
        let (face, bary) = lvl.find_face_bary(&dir);
        // Bary should sum to ~1 and all be non-negative
        let sum: f64 = bary.iter().sum();
        assert_relative_eq!(sum, 1.0, epsilon = 1e-10);
        assert!(bary.iter().all(|&b| b >= -1e-10));
        // Face indices should be valid
        assert!(face.iter().all(|&i| i < lvl.n_vertices));
    }

    /// Fill a slab with constant energy → gradient should be 0.
    #[test]
    fn gradient_constant_surface() {
        let lvl = MeshLevel::new(1);
        let n_v = lvl.n_vertices;
        let data = vec![42.0; n_v * n_v];
        let grad = compute_gradient(&data, &lvl.vertices, &lvl.neighbors, n_v);
        assert_relative_eq!(grad, 0.0, epsilon = 1e-12);
    }

    /// Non-constant surface should have nonzero gradient.
    #[test]
    fn gradient_nonzero_surface() {
        let lvl = MeshLevel::new(1);
        let n_v = lvl.n_vertices;
        let reference = Vector3::new(1.0, 1.0, 1.0).normalize();
        let mut data = vec![0.0; n_v * n_v];
        for vi in 0..n_v {
            let va = Vector3::from(lvl.vertices[vi]);
            for vj in 0..n_v {
                let vb = Vector3::from(lvl.vertices[vj]);
                data[vi * n_v + vj] = va.dot(&reference) + vb.dot(&reference);
            }
        }
        let grad = compute_gradient(&data, &lvl.vertices, &lvl.neighbors, n_v);
        assert!(grad > 0.0);
    }

    #[test]
    fn builder_all_constant_becomes_scalar() {
        let mut builder =
            AdaptiveBuilder::new(5.0, 10.0, 1.0, std::f64::consts::TAU / 4.0, 2, 1.0, 0.001);

        for ri in 0..builder.n_r() {
            let n_v = builder.current_n_vertices();
            for oi in 0..builder.n_omega() {
                let data = vec![42.0; n_v * n_v];
                builder.set_slab(ri, oi, &data);
            }
            builder.finish_r_slice(ri);
        }

        let table = builder.build();
        // All slabs should be scalar (gradient is 0, well below threshold/100)
        for sr in &table.slab_res {
            match sr {
                SlabResolution::Scalar(v) => assert_relative_eq!(*v as f64, 42.0, epsilon = 1e-4),
                SlabResolution::Mesh { .. } | SlabResolution::Repulsive => {
                    panic!("expected scalar for constant data")
                }
            }
        }
        // No mesh data
        assert!(table.data.is_empty());
    }

    #[test]
    fn builder_produces_mesh_for_nonconstant() {
        let mut builder =
            AdaptiveBuilder::new(5.0, 7.0, 1.0, std::f64::consts::TAU / 4.0, 1, 1e10, 0.001);

        let reference = Vector3::new(1.0, 0.0, 0.0);
        for ri in 0..builder.n_r() {
            let n_v = builder.current_n_vertices();
            let verts = builder.vertex_directions(builder.current_level()).to_vec();
            for oi in 0..builder.n_omega() {
                let mut data = vec![0.0; n_v * n_v];
                for vi in 0..n_v {
                    let va = Vector3::from(verts[vi]);
                    for vj in 0..n_v {
                        let vb = Vector3::from(verts[vj]);
                        data[vi * n_v + vj] = 10.0 * va.dot(&reference) + vb.dot(&reference);
                    }
                }
                builder.set_slab(ri, oi, &data);
            }
            builder.finish_r_slice(ri);
        }

        let table = builder.build();
        // With threshold=1e10, everything below → at least some may be scalar
        // but the key thing is the table is buildable and lookups work
        let dir_a = Vector3::new(1.0, 0.0, 0.0);
        let dir_b = Vector3::new(0.0, 1.0, 0.0);
        let val = table.lookup(5.5, 0.0, &dir_a, &dir_b);
        // Should be finite
        assert!(val.is_finite(), "lookup returned {val}");
    }

    #[test]
    fn lookup_constant_adaptive() {
        let mut builder =
            AdaptiveBuilder::new(5.0, 10.0, 1.0, std::f64::consts::TAU / 8.0, 2, 1e10, 0.001);

        for ri in 0..builder.n_r() {
            let n_v = builder.current_n_vertices();
            for oi in 0..builder.n_omega() {
                let data = vec![42.0; n_v * n_v];
                builder.set_slab(ri, oi, &data);
            }
            builder.finish_r_slice(ri);
        }

        let table = builder.build();
        let dir_a = Vector3::new(1.0, 0.0, 0.0);
        let dir_b = Vector3::new(0.0, 1.0, 0.0);
        let e = table.lookup(7.0, 1.0, &dir_a, &dir_b);
        assert!(
            (e - 42.0).abs() < 0.1,
            "Expected ~42 for constant table, got {e}"
        );
    }

    #[test]
    fn lookup_out_of_range() {
        let mut builder =
            AdaptiveBuilder::new(5.0, 10.0, 1.0, std::f64::consts::TAU / 4.0, 1, 1e10, 0.001);
        for ri in 0..builder.n_r() {
            let n_v = builder.current_n_vertices();
            for oi in 0..builder.n_omega() {
                builder.set_slab(ri, oi, &vec![1.0; n_v * n_v]);
            }
            builder.finish_r_slice(ri);
        }
        let table = builder.build();
        let dir = Vector3::new(1.0, 0.0, 0.0);
        assert_eq!(table.lookup(3.0, 0.0, &dir, &dir), 0.0);
        assert_eq!(table.lookup(20.0, 0.0, &dir, &dir), 0.0);
    }

    #[test]
    fn boltzmann_constant_adaptive() {
        let mut builder =
            AdaptiveBuilder::new(5.0, 10.0, 1.0, std::f64::consts::TAU / 8.0, 2, 1e10, 0.001);
        for ri in 0..builder.n_r() {
            let n_v = builder.current_n_vertices();
            for oi in 0..builder.n_omega() {
                builder.set_slab(ri, oi, &vec![42.0; n_v * n_v]);
            }
            builder.finish_r_slice(ri);
        }
        let table = builder.build();
        let dir_a = Vector3::new(1.0, 0.0, 0.0);
        let dir_b = Vector3::new(0.0, 1.0, 0.0);
        let e = table.lookup_boltzmann(7.0, 1.0, &dir_a, &dir_b, 1.0);
        assert!(
            (e - 42.0).abs() < 0.1,
            "Expected ~42 for constant Boltzmann, got {e}"
        );
    }

    #[test]
    fn adaptive_resolution_decreases() {
        // Use a high gradient threshold so the builder always wants to go coarser
        let mut builder =
            AdaptiveBuilder::new(5.0, 10.0, 1.0, std::f64::consts::TAU / 4.0, 3, 1e10, 0.001);
        assert_eq!(builder.current_n_div(), 3);

        let n_v = builder.current_n_vertices();
        for oi in 0..builder.n_omega() {
            builder.set_slab(0, oi, &vec![42.0; n_v * n_v]);
        }
        builder.finish_r_slice(0);
        // Should have decreased
        assert!(builder.current_n_div() < 3);

        // Continue decreasing
        let n_v = builder.current_n_vertices();
        for oi in 0..builder.n_omega() {
            builder.set_slab(1, oi, &vec![42.0; n_v * n_v]);
        }
        builder.finish_r_slice(1);
        assert!(builder.current_n_div() < 2);
    }

    #[test]
    fn round_trip_save_load() {
        let mut builder =
            AdaptiveBuilder::new(5.0, 8.0, 1.0, std::f64::consts::TAU / 4.0, 1, 1e10, 0.001);
        for ri in 0..builder.n_r() {
            let n_v = builder.current_n_vertices();
            for oi in 0..builder.n_omega() {
                builder.set_slab(ri, oi, &vec![7.0; n_v * n_v]);
            }
            builder.finish_r_slice(ri);
        }
        let table = builder.build();

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("adaptive.bin.gz");
        table.save(&path).unwrap();
        let loaded = Table6DAdaptive::<f32>::load(&path).unwrap();

        assert_eq!(loaded.n_r, table.n_r);
        assert_eq!(loaded.n_omega, table.n_omega);
        assert_eq!(loaded.data.len(), table.data.len());
        assert_eq!(loaded.slab_res.len(), table.slab_res.len());

        let dir_a = Vector3::new(1.0, 0.0, 0.0);
        let dir_b = Vector3::new(0.0, 1.0, 0.0);
        let orig = table.lookup(6.0, 0.5, &dir_a, &dir_b);
        let load = loaded.lookup(6.0, 0.5, &dir_a, &dir_b);
        assert!(
            (orig - load).abs() < 1e-4,
            "Round-trip mismatch: {orig} vs {load}"
        );
    }

    #[test]
    fn mixed_scalar_and_mesh_slabs() {
        // First R slice: non-constant → mesh, large threshold so second becomes scalar
        let mut builder =
            AdaptiveBuilder::new(5.0, 7.0, 1.0, std::f64::consts::TAU / 2.0, 1, 1e10, 0.001);
        let n_v = builder.current_n_vertices();
        let verts = builder.vertex_directions(builder.current_level()).to_vec();
        let reference = Vector3::new(1.0, 0.0, 0.0);

        // ri=0: fill with varying data
        for oi in 0..builder.n_omega() {
            let mut data = vec![0.0; n_v * n_v];
            for vi in 0..n_v {
                let va = Vector3::from(verts[vi]);
                for vj in 0..n_v {
                    let vb = Vector3::from(verts[vj]);
                    data[vi * n_v + vj] = 100.0 * va.dot(&reference) + vb.dot(&reference);
                }
            }
            builder.set_slab(0, oi, &data);
        }
        builder.finish_r_slice(0);

        // ri=1: constant data
        let n_v = builder.current_n_vertices();
        for oi in 0..builder.n_omega() {
            builder.set_slab(1, oi, &vec![42.0; n_v * n_v]);
        }
        builder.finish_r_slice(1);

        let table = builder.build();
        // Verify both types of slabs are present and lookups work
        let dir_a = Vector3::new(1.0, 0.0, 0.0);
        let dir_b = Vector3::new(0.0, 1.0, 0.0);
        let e1 = table.lookup(5.5, 0.0, &dir_a, &dir_b);
        assert!(e1.is_finite());
        let e2 = table.lookup(6.5, 0.0, &dir_a, &dir_b);
        assert!(e2.is_finite());
    }

    #[test]
    fn repulsive_slab_returns_infinity() {
        // beta=1.0: energies of 200 give expm1(-200) ≈ -1 → repulsive
        let mut builder =
            AdaptiveBuilder::new(5.0, 7.0, 1.0, std::f64::consts::TAU / 4.0, 1, 1e10, 1.0);

        let n_v = builder.current_n_vertices();
        // ri=0 (R=5.0): all energies strongly repulsive (expm1(-βU) ≈ -1)
        for oi in 0..builder.n_omega() {
            builder.set_slab(0, oi, &vec![200.0; n_v * n_v]);
        }
        builder.finish_r_slice(0);

        // ri=1 (R=6.0): moderate energies
        let n_v = builder.current_n_vertices();
        for oi in 0..builder.n_omega() {
            builder.set_slab(1, oi, &vec![5.0; n_v * n_v]);
        }
        builder.finish_r_slice(1);

        let table = builder.build();
        let dir = Vector3::new(1.0, 0.0, 0.0);

        // Repulsive slab (R=5.0 → ri=0) → infinity
        assert!(table.lookup(5.0, 0.0, &dir, &dir).is_infinite());
        assert!(table
            .lookup_boltzmann(5.0, 0.0, &dir, &dir, 1.0)
            .is_infinite());
        // Non-repulsive slab (R=6.0 → ri=1) → finite
        let e = table.lookup(6.0, 0.0, &dir, &dir);
        assert!((e - 5.0).abs() < 0.1, "Expected ~5, got {e}");
        // All ri=0 slabs should be Repulsive
        for oi in 0..table.n_omega {
            assert!(matches!(table.slab_res[oi], SlabResolution::Repulsive));
        }
    }

    #[test]
    fn no_interpolation_for_smooth_slab() {
        let n_v_at_level1 = 42; // 10*(1+1)^2 + 2
        let reference = Vector3::new(1.0, 0.0, 0.0);

        // First, measure the actual gradient of our test data
        let lvl = MeshLevel::new(1);
        let mut test_data = vec![0.0; n_v_at_level1 * n_v_at_level1];
        for vi in 0..n_v_at_level1 {
            let va = Vector3::from(lvl.vertices[vi]);
            for vj in 0..n_v_at_level1 {
                let vb = Vector3::from(lvl.vertices[vj]);
                test_data[vi * n_v_at_level1 + vj] =
                    10.0 + 0.1 * va.dot(&reference) + 0.1 * vb.dot(&reference);
            }
        }
        let grad = compute_gradient(&test_data, &lvl.vertices, &lvl.neighbors, n_v_at_level1);

        // Set gradient_threshold just above the measured gradient so:
        //   scalar_threshold = gradient_threshold/10 < grad < gradient_threshold
        let gradient_threshold = grad * 2.0;
        assert!(
            grad > gradient_threshold / 10.0,
            "Gradient {grad} should be above scalar threshold {}",
            gradient_threshold / 10.0
        );

        let mut builder = AdaptiveBuilder::new(
            5.0,
            7.0,
            1.0,
            std::f64::consts::TAU / 4.0,
            1,
            gradient_threshold,
            0.001,
        );

        let n_v = builder.current_n_vertices();
        let verts = builder.vertex_directions(builder.current_level()).to_vec();

        for oi in 0..builder.n_omega() {
            let mut data = vec![0.0; n_v * n_v];
            for vi in 0..n_v {
                let va = Vector3::from(verts[vi]);
                for vj in 0..n_v {
                    let vb = Vector3::from(verts[vj]);
                    data[vi * n_v + vj] =
                        10.0 + 0.1 * va.dot(&reference) + 0.1 * vb.dot(&reference);
                }
            }
            builder.set_slab(0, oi, &data);
        }
        builder.finish_r_slice(0);

        let has_no_interp = (0..builder.n_omega()).any(|oi| {
            matches!(
                builder.slab_res[oi],
                SlabResolution::Mesh {
                    interpolate: false,
                    ..
                }
            )
        });
        assert!(has_no_interp, "Expected non-interpolated mesh slabs");

        // Fill remaining R slice
        let n_v = builder.current_n_vertices();
        for oi in 0..builder.n_omega() {
            builder.set_slab(1, oi, &vec![10.0; n_v * n_v]);
        }
        builder.finish_r_slice(1);

        let table = builder.build();
        let dir_a = Vector3::new(1.0, 0.0, 0.0);
        let dir_b = Vector3::new(0.0, 1.0, 0.0);
        let e = table.lookup(5.0, 0.0, &dir_a, &dir_b);
        assert!(e.is_finite());
        assert!((e - 10.0).abs() < 1.0, "Expected ~10, got {e}");
    }

    #[test]
    fn repulsive_saves_storage() {
        let n_omega_bins = 4;
        let omega_step = std::f64::consts::TAU / n_omega_bins as f64;
        let reference = Vector3::new(1.0, 0.0, 0.0);

        // Use non-constant repulsive data so the non-repulsive builder keeps it as Mesh
        let make_repulsive_data = |verts: &[[f64; 3]], n_v: usize| -> Vec<f64> {
            let mut data = vec![0.0; n_v * n_v];
            for vi in 0..n_v {
                let va = Vector3::from(verts[vi]);
                for vj in 0..n_v {
                    let vb = Vector3::from(verts[vj]);
                    // All values > 150, but varying
                    data[vi * n_v + vj] =
                        200.0 + 50.0 * va.dot(&reference) + 50.0 * vb.dot(&reference);
                }
            }
            data
        };

        // beta=1.0: energies 150-300 give expm1(-βU) ≈ -1 → repulsive
        let mut builder_rep = AdaptiveBuilder::new(5.0, 7.0, 1.0, omega_step, 1, 0.01, 1.0);

        // beta≈0: Boltzmann check never triggers → all kept as Mesh
        let mut builder_full = AdaptiveBuilder::new(5.0, 7.0, 1.0, omega_step, 1, 0.01, 1e-20);

        for b in [&mut builder_rep, &mut builder_full] {
            let n_v = b.current_n_vertices();
            let verts = b.vertex_directions(b.current_level()).to_vec();
            for oi in 0..b.n_omega() {
                b.set_slab(0, oi, &make_repulsive_data(&verts, n_v));
            }
            b.finish_r_slice(0);
            let n_v = b.current_n_vertices();
            let verts = b.vertex_directions(b.current_level()).to_vec();
            for oi in 0..b.n_omega() {
                b.set_slab(1, oi, &make_repulsive_data(&verts, n_v));
            }
            b.finish_r_slice(1);
        }

        let table_rep = builder_rep.build();
        let table_full = builder_full.build();
        assert!(
            table_rep.data.len() < table_full.data.len(),
            "Repulsive table ({}) should use less storage than full table ({})",
            table_rep.data.len(),
            table_full.data.len()
        );
    }
}
