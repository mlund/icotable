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

use crate::flat::{load_bincode, save_bincode, TableMetadata};
use crate::ico::Face;
use crate::subdivision::{Locator, MeshData, Subdivision};
use crate::Vector3;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::sync::OnceLock;

/// Boltzmann weight floor: slabs where `exp(−βU) < BOLTZMANN_FLOOR` for all
/// orientations are thermodynamically dead (MC always rejects). ~9 kT.
const BOLTZMANN_FLOOR: f64 = 1e-4;

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
    /// Which subdivision produced this mesh; drives locator rebuild on load.
    /// `#[serde(default)]` only helps self-describing formats — bincode (the
    /// table loader) is not one, so adaptive tables saved before this field was
    /// added must be regenerated rather than loaded.
    #[serde(default)]
    subdivision: Subdivision,
    /// Lazy direction→cell locator, rebuilt per `subdivision`.
    #[serde(skip, default)]
    locator: OnceLock<Locator>,
}

impl MeshLevel {
    /// Build a mesh level for the given subdivision count.
    pub fn new(n_div: usize) -> Self {
        Self::with_subdivision(Subdivision::default(), n_div)
    }

    /// Build a mesh level using a specific subdivision scheme. The scheme's
    /// `build_mesh` returns vertices in its own final order (geodesic BFS-reorders
    /// for cache locality; lattice keeps its `(face,i,j)` order).
    pub fn with_subdivision(subdivision: Subdivision, n_div: usize) -> Self {
        let MeshData {
            vertices,
            weights,
            neighbors,
        } = subdivision.build_mesh(n_div);
        Self {
            n_div,
            n_vertices: vertices.len(),
            vertices,
            weights,
            neighbors,
            subdivision,
            locator: OnceLock::new(),
        }
    }

    /// Get or lazily build the scheme's locator.
    fn grid(&self) -> &Locator {
        self.locator.get_or_init(|| {
            self.subdivision
                .build_locator(self.n_div, &self.vertices, &self.neighbors)
        })
    }

    /// Find the containing face and barycentric coordinates for a direction.
    pub fn locate(&self, dir: &Vector3) -> (Face, [f64; 3]) {
        self.grid().locate(dir)
    }

    /// Find the nearest vertex index for a direction (no triangle search).
    fn find_nearest_vertex(&self, dir: &Vector3) -> usize {
        self.grid().find_nearest_vertex(dir)
    }
}

impl crate::mesh::AngularMesh for MeshLevel {
    fn len(&self) -> usize {
        self.n_vertices
    }
    fn direction(&self, i: usize) -> Vector3 {
        Vector3::from(self.vertices[i])
    }
    fn weight(&self, i: usize) -> f64 {
        self.weights[i]
    }
    fn neighbors(&self, i: usize) -> &[u16] {
        &self.neighbors[i]
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
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
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
    /// Dihedral angle bin width (radians). Internal to the lookup machinery.
    omega_step: f64,
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

    /// Validate that tail correction metadata is self-consistent.
    pub fn validate_metadata(&self) -> Result<()> {
        if let Some(m) = &self.metadata {
            m.validate()?;
        }
        Ok(())
    }
}

impl<T: num_traits::Float + Into<f64>> Table6DAdaptive<T> {
    /// Get the mesh level used at radial index `ri` (from first Mesh slab found).
    ///
    /// Returns `None` if all slabs at this R are Scalar or Repulsive.
    pub fn mesh_level_at_r(&self, ri: usize) -> Option<&MeshLevel> {
        let base = ri * self.n_omega;
        (0..self.n_omega).find_map(|oi| match &self.slab_res[base + oi] {
            SlabResolution::Mesh { level, .. } => Some(&self.levels[*level as usize]),
            _ => None,
        })
    }

    /// Extract all energies at radial index `ri` as a flat array.
    ///
    /// Layout: `energies[vi * n_v * n_omega + vj * n_omega + oi]`
    /// where `n_v` is the vertex count of the mesh level at this R.
    ///
    /// - Mesh slabs: per-vertex-pair energies from the data array.
    /// - Scalar slabs: the scalar value replicated for all vertex pairs.
    /// - Repulsive slabs: `f64::INFINITY` for all vertex pairs.
    ///
    /// Returns `None` if all slabs are Repulsive, or if `ri` is out of range.
    pub fn energies_at_r(&self, ri: usize) -> Option<(Vec<f64>, &MeshLevel)> {
        if ri >= self.n_r {
            return None;
        }
        let level = self.mesh_level_at_r(ri)?;
        let n_v = level.n_vertices;
        let n_omega = self.n_omega;
        let mut energies = vec![f64::INFINITY; n_v * n_v * n_omega];

        for oi in 0..n_omega {
            let slab_idx = ri * n_omega + oi;
            match &self.slab_res[slab_idx] {
                SlabResolution::Repulsive => {} // already INFINITY
                SlabResolution::Scalar(v) => {
                    let val = *v as f64;
                    for vi in 0..n_v {
                        for vj in 0..n_v {
                            energies[vi * n_v * n_omega + vj * n_omega + oi] = val;
                        }
                    }
                }
                SlabResolution::Mesh {
                    level: lvl_idx, ..
                } => {
                    let slab_n_v = self.levels[*lvl_idx as usize].n_vertices;
                    let base = self.slab_offsets[slab_idx] as usize;
                    for vi in 0..slab_n_v.min(n_v) {
                        for vj in 0..slab_n_v.min(n_v) {
                            energies[vi * n_v * n_omega + vj * n_omega + oi] =
                                self.data[base + vi * slab_n_v + vj].into();
                        }
                    }
                }
            }
        }
        Some((energies, level))
    }

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
        let (face_a, bary_a) = lvl.locate(dir_a);
        let (face_b, bary_b) = lvl.locate(dir_b);
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
    ///
    /// Uses `find_nearest_vertex` which only does the cube-face grid lookup
    /// and candidate scan, skipping the triangle search and barycentric
    /// computation that `locate` would perform.
    fn nearest_vertex_val(
        &self,
        slab_idx: usize,
        level: u8,
        dir_a: &Vector3,
        dir_b: &Vector3,
    ) -> f64 {
        let lvl = &self.levels[level as usize];
        let n_v = lvl.n_vertices;
        let va = lvl.find_nearest_vertex(dir_a);
        let vb = lvl.find_nearest_vertex(dir_b);
        let base = self.slab_offsets[slab_idx] as usize;
        self.data[base + va * n_v + vb].into()
    }

    /// Lookup value by nearest R/ω bin with adaptive angular resolution.
    pub(crate) fn lookup(&self, r: f64, omega: f64, dir_a: &Vector3, dir_b: &Vector3) -> f64 {
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
    pub(crate) fn lookup_boltzmann(
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

impl<T: num_traits::Float> crate::lookup::TabulatedInteraction for Table6DAdaptive<T> {
    fn r_range(&self) -> (f64, f64) {
        (self.rmin, self.rmax)
    }
    fn metadata(&self) -> Option<&TableMetadata> {
        self.metadata.as_ref()
    }
}

impl<T: num_traits::Float + Into<f64>> crate::lookup::Lookup6D for Table6DAdaptive<T> {
    fn lookup(&self, r: f64, omega: f64, dir_a: &Vector3, dir_b: &Vector3, beta: Option<f64>) -> f64 {
        match beta {
            Some(beta) => self.lookup_boltzmann(r, omega, dir_a, dir_b, beta),
            None => self.lookup(r, omega, dir_a, dir_b),
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
    /// * `gradient_threshold` — Boltzmann-weight gradient threshold (1/rad) for
    ///   resolution reduction. Measures max |Δexp(-βU)| / Δangle across neighbors.
    /// * `beta` — Inverse thermal energy 1/kT (mol/kJ)
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        rmin: f64,
        rmax: f64,
        dr: f64,
        omega_step: f64,
        max_n_div: usize,
        gradient_threshold: f64,
        beta: f64,
    ) -> Self {
        Self::with_subdivision(
            Subdivision::default(),
            rmin,
            rmax,
            dr,
            omega_step,
            max_n_div,
            gradient_threshold,
            beta,
        )
    }

    /// Like [`new`](Self::new) but with an explicit subdivision scheme for the
    /// angular meshes (the future `--locator` CLI choice).
    #[allow(clippy::too_many_arguments)]
    pub fn with_subdivision(
        subdivision: Subdivision,
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

        // Build all levels from 0..=max_n_div with the chosen scheme.
        let levels: Vec<MeshLevel> = (0..=max_n_div)
            .map(|n| MeshLevel::with_subdivision(subdivision, n))
            .collect();

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
    pub const fn current_level(&self) -> usize {
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
    pub const fn n_r(&self) -> usize {
        self.n_r
    }

    /// Number of omega bins.
    pub const fn n_omega(&self) -> usize {
        self.n_omega
    }

    /// The R value for a given radial index.
    pub fn r_value(&self, ri: usize) -> f64 {
        (ri as f64).mul_add(self.dr, self.rmin)
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
    /// Returns the maximum Boltzmann-weight gradient (1/rad) across all
    /// non-repulsive slabs.
    ///
    /// Call this after all ω slabs for a given `ri` have been set.
    pub fn finish_r_slice(&mut self, ri: usize) -> f64 {
        let level = self.current_level;
        let lvl = &self.levels[level];
        let n_v = lvl.n_vertices;
        let scalar_threshold = self.gradient_threshold / 10.0;
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
            let grad = compute_gradient(data, &lvl.vertices, &lvl.neighbors, n_v, self.beta);

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

        // Only decrease resolution once fully past the repulsive wall:
        // mixed slices (some ω repulsive, some not) sit at the binding
        // region boundary where the next R-slice needs fine resolution.
        // `has_nonrepulsive` also guards against empty-data slices whose
        // default Scalar(0) would otherwise pass the Repulsive check.
        let all_nonrepulsive = has_nonrepulsive
            && (0..self.n_omega).all(|oi| {
                !matches!(
                    self.slab_res[ri * self.n_omega + oi],
                    SlabResolution::Repulsive
                )
            });
        if all_nonrepulsive && max_gradient < self.gradient_threshold && self.current_level > 0 {
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
            rmax: (self.n_r as f64).mul_add(self.dr, self.rmin),
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

// ---------------------------------------------------------------------------
// Table3DAdaptive
// ---------------------------------------------------------------------------

/// Adaptive 3D lookup table with per-R-slice resolution.
///
/// Each R-bin has a single slab (no dihedral angle dimension). Slab data
/// is `n_v` values indexed by vertex. Supports the same tier classification
/// as [`Table6DAdaptive`]: repulsive, scalar, nearest-vertex, and interpolated.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Table3DAdaptive<T: num_traits::Float> {
    /// Minimum radial distance.
    pub rmin: f64,
    /// Maximum radial distance.
    pub rmax: f64,
    /// Radial bin width.
    pub dr: f64,
    /// Number of radial bins.
    pub n_r: usize,
    /// Pre-built mesh levels (indexed by `SlabResolution::Mesh::level`).
    pub levels: Vec<MeshLevel>,
    /// Resolution descriptor per R-bin.
    pub slab_res: Vec<SlabResolution>,
    /// Data offset per slab into `data`. `u32::MAX` for Scalar/Repulsive slabs.
    pub(crate) slab_offsets: Vec<u32>,
    /// Contiguous storage for all Mesh slab data.
    pub data: Vec<T>,
    /// Optional metadata for tail corrections beyond the table cutoff.
    pub metadata: Option<TableMetadata>,
}

impl<T: num_traits::Float + Serialize + serde::de::DeserializeOwned> Table3DAdaptive<T> {
    /// Save to a bincode file (gzip-compressed if path ends in `.gz`).
    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        save_bincode(self, path.as_ref())
    }

    /// Load from a bincode file (gzip-decompressed if path ends in `.gz`).
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        load_bincode(path.as_ref())
    }

    /// Validate that tail correction metadata is self-consistent.
    pub fn validate_metadata(&self) -> Result<()> {
        if let Some(m) = &self.metadata {
            m.validate()?;
        }
        Ok(())
    }
}

impl<T: num_traits::Float + Into<f64>> Table3DAdaptive<T> {
    /// Resolve R to a slab index, or `None` if out of range.
    fn resolve_slab_idx(&self, r: f64) -> Option<usize> {
        if r < self.rmin || r > self.rmax {
            return None;
        }
        let ri = ((r - self.rmin) / self.dr + 0.5) as usize;
        Some(ri.min(self.n_r.saturating_sub(1)))
    }

    /// Return the value at the nearest vertex (no interpolation).
    fn nearest_vertex_val(&self, slab_idx: usize, level: u8, dir: &Vector3) -> f64 {
        let lvl = &self.levels[level as usize];
        let va = lvl.find_nearest_vertex(dir);
        let base = self.slab_offsets[slab_idx] as usize;
        self.data[base + va].into()
    }

    /// Gather the 3 interpolation values for a Mesh slab.
    fn gather_mesh_vals(
        &self,
        slab_idx: usize,
        level: u8,
        dir: &Vector3,
    ) -> (Face, [f64; 3], [f64; 3]) {
        let lvl = &self.levels[level as usize];
        let (face, bary) = lvl.locate(dir);
        let base = self.slab_offsets[slab_idx] as usize;
        let vals = [
            self.data[base + face[0]].into(),
            self.data[base + face[1]].into(),
            self.data[base + face[2]].into(),
        ];
        (face, bary, vals)
    }

    /// Lookup value by nearest R bin with adaptive angular resolution.
    pub(crate) fn lookup(&self, r: f64, dir: &Vector3) -> f64 {
        let Some(slab_idx) = self.resolve_slab_idx(r) else {
            return 0.0;
        };
        match &self.slab_res[slab_idx] {
            SlabResolution::Scalar(v) => *v as f64,
            SlabResolution::Repulsive => f64::INFINITY,
            SlabResolution::Mesh {
                level,
                interpolate: false,
            } => self.nearest_vertex_val(slab_idx, *level, dir),
            SlabResolution::Mesh {
                level,
                interpolate: true,
            } => {
                let (_, bary, vals) = self.gather_mesh_vals(slab_idx, *level, dir);
                bary[0] * vals[0] + bary[1] * vals[1] + bary[2] * vals[2]
            }
        }
    }

    /// Boltzmann-weighted interpolation: interpolate exp(-beta*u) then invert.
    pub(crate) fn lookup_boltzmann(&self, r: f64, dir: &Vector3, beta: f64) -> f64 {
        debug_assert!(beta > 0.0, "beta must be positive, got {beta}");
        let Some(slab_idx) = self.resolve_slab_idx(r) else {
            return 0.0;
        };
        match &self.slab_res[slab_idx] {
            SlabResolution::Scalar(v) => *v as f64,
            SlabResolution::Repulsive => f64::INFINITY,
            SlabResolution::Mesh {
                level,
                interpolate: false,
            } => self.nearest_vertex_val(slab_idx, *level, dir),
            SlabResolution::Mesh {
                level,
                interpolate: true,
            } => {
                let (_, bary, vals) = self.gather_mesh_vals(slab_idx, *level, dir);
                let u_min = vals[0].min(vals[1]).min(vals[2]);
                if u_min.is_infinite() {
                    return f64::INFINITY;
                }
                let sum = bary[0] * (-beta * (vals[0] - u_min)).exp()
                    + bary[1] * (-beta * (vals[1] - u_min)).exp()
                    + bary[2] * (-beta * (vals[2] - u_min)).exp();
                if sum <= 0.0 {
                    return f64::INFINITY;
                }
                u_min - sum.ln() / beta
            }
        }
    }
}

impl<T: num_traits::Float> crate::lookup::TabulatedInteraction for Table3DAdaptive<T> {
    fn r_range(&self) -> (f64, f64) {
        (self.rmin, self.rmax)
    }
    fn metadata(&self) -> Option<&TableMetadata> {
        self.metadata.as_ref()
    }
}

impl<T: num_traits::Float + Into<f64>> crate::lookup::Lookup3D for Table3DAdaptive<T> {
    fn lookup(&self, r: f64, dir: &Vector3, beta: Option<f64>) -> f64 {
        match beta {
            Some(beta) => self.lookup_boltzmann(r, dir, beta),
            None => self.lookup(r, dir),
        }
    }
}

// ---------------------------------------------------------------------------
// Adaptive3DBuilder
// ---------------------------------------------------------------------------

/// Drives the adaptive 3D table generation protocol.
///
/// Mirrors [`AdaptiveBuilder`] without the dihedral angle dimension: each
/// R-slice has a single slab of `n_v` energies (not `n_v × n_v`).
pub struct Adaptive3DBuilder {
    rmin: f64,
    dr: f64,
    n_r: usize,
    gradient_threshold: f64,
    beta: f64,
    levels: Vec<MeshLevel>,
    current_level: usize,
    slab_res: Vec<SlabResolution>,
    slab_data: Vec<Option<Vec<f64>>>,
}

impl Adaptive3DBuilder {
    /// Create a new adaptive 3D builder.
    ///
    /// # Arguments
    /// * `rmin` — Minimum radial distance
    /// * `rmax` — Maximum radial distance
    /// * `dr` — Radial bin width
    /// * `max_n_div` — Maximum icosphere subdivision level
    /// * `gradient_threshold` — Boltzmann-weight gradient threshold (1/rad) for
    ///   resolution reduction. Measures max |Δexp(-βU)| / Δangle across neighbors.
    /// * `beta` — Inverse thermal energy 1/kT (mol/kJ)
    pub fn new(
        rmin: f64,
        rmax: f64,
        dr: f64,
        max_n_div: usize,
        gradient_threshold: f64,
        beta: f64,
    ) -> Self {
        Self::with_subdivision(
            Subdivision::default(),
            rmin,
            rmax,
            dr,
            max_n_div,
            gradient_threshold,
            beta,
        )
    }

    /// Like [`new`](Self::new) but with an explicit subdivision scheme.
    #[allow(clippy::too_many_arguments)]
    pub fn with_subdivision(
        subdivision: Subdivision,
        rmin: f64,
        rmax: f64,
        dr: f64,
        max_n_div: usize,
        gradient_threshold: f64,
        beta: f64,
    ) -> Self {
        let n_r = ((rmax - rmin) / dr + 0.5) as usize;
        let levels: Vec<MeshLevel> = (0..=max_n_div)
            .map(|n| MeshLevel::with_subdivision(subdivision, n))
            .collect();

        Self {
            rmin,
            dr,
            n_r,
            gradient_threshold,
            beta,
            current_level: levels.len() - 1,
            levels,
            slab_res: vec![SlabResolution::Scalar(0.0); n_r],
            slab_data: vec![None; n_r],
        }
    }

    /// Current subdivision level index (into `levels`).
    pub const fn current_level(&self) -> usize {
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
    pub fn vertex_weights(&self, level: usize) -> &[f64] {
        &self.levels[level].weights
    }

    /// Number of radial bins.
    pub const fn n_r(&self) -> usize {
        self.n_r
    }

    /// The R value for a given radial index.
    pub fn r_value(&self, ri: usize) -> f64 {
        (ri as f64).mul_add(self.dr, self.rmin)
    }

    /// Set energy data for one R-bin.
    ///
    /// `energies` must have `n_v` elements where `n_v` is the vertex count
    /// at the current level.
    pub fn set_slab(&mut self, ri: usize, energies: &[f64]) {
        let n_v = self.levels[self.current_level].n_vertices;
        assert_eq!(
            energies.len(),
            n_v,
            "expected {} energies, got {}",
            n_v,
            energies.len()
        );
        self.slab_data[ri] = Some(energies.to_vec());
        self.slab_res[ri] = SlabResolution::Mesh {
            level: self.current_level as u8,
            interpolate: true,
        };
    }

    /// Finish an R slice: classify the slab and possibly lower resolution.
    ///
    /// Returns the Boltzmann-weight gradient (1/rad) for the slab.
    pub fn finish_r_slice(&mut self, ri: usize) -> f64 {
        let level = self.current_level;
        let lvl = &self.levels[level];
        let scalar_threshold = self.gradient_threshold / 10.0;

        let Some(ref data) = self.slab_data[ri] else {
            return 0.0;
        };

        if data
            .iter()
            .all(|&e| (-self.beta * e).exp_m1() < BOLTZMANN_FLOOR - 1.0)
        {
            self.slab_res[ri] = SlabResolution::Repulsive;
            self.slab_data[ri] = None;
            return 0.0;
        }

        let grad = compute_gradient_1d(data, &lvl.vertices, &lvl.neighbors, self.beta);

        if grad < scalar_threshold {
            let mean = data.iter().sum::<f64>() / data.len() as f64;
            self.slab_res[ri] = SlabResolution::Scalar(mean as f32);
            self.slab_data[ri] = None;
        } else if grad < self.gradient_threshold {
            self.slab_res[ri] = SlabResolution::Mesh {
                level: level as u8,
                interpolate: false,
            };
        }

        if grad < self.gradient_threshold && self.current_level > 0 {
            self.current_level -= 1;
        }
        grad
    }

    /// Build the final adaptive 3D table.
    ///
    /// Consumes the builder and produces a `Table3DAdaptive<f32>`.
    pub fn build(self) -> Table3DAdaptive<f32> {
        let mut slab_offsets = vec![u32::MAX; self.n_r];
        let mut data = Vec::new();
        let mut current_offset = 0u32;

        for (ri, (res, slab_buf)) in self.slab_res.iter().zip(self.slab_data.iter()).enumerate() {
            if let SlabResolution::Mesh { level, .. } = res {
                slab_offsets[ri] = current_offset;
                let n_v = self.levels[*level as usize].n_vertices;
                let slab_data = slab_buf.as_ref().expect("Mesh slab must have data");
                data.extend(slab_data.iter().map(|&v| v as f32));
                current_offset += n_v as u32;
            }
        }

        Table3DAdaptive {
            rmin: self.rmin,
            rmax: (self.n_r as f64).mul_add(self.dr, self.rmin),
            dr: self.dr,
            n_r: self.n_r,
            levels: self.levels,
            slab_res: self.slab_res,
            slab_offsets,
            data,
            metadata: None,
        }
    }
}

// ---------------------------------------------------------------------------
// Gradient computation
// ---------------------------------------------------------------------------

/// Compute the max angular gradient for a 1D (single-direction) slab.
///
/// For each vertex `vi`, checks max |exp(-βE_vi) - exp(-βE_ni)| / angle(vi, ni).
///
/// Using Boltzmann weights instead of raw energies makes the gradient
/// dimensionless/rad and scale-independent: repulsive configurations
/// (exp≈0) and bulk-like configurations (exp≈1) both have small gradients,
/// so resolution reduces earlier in the transition zone.
fn compute_gradient_1d(
    data: &[f64],
    vertices: &[[f64; 3]],
    neighbors: &[Vec<u16>],
    beta: f64,
) -> f64 {
    let boltzmann: Vec<f64> = data.iter().map(|&e| (-beta * e).exp()).collect();
    let mut max_grad = 0.0f64;
    for (vi, &bf_vi) in boltzmann.iter().enumerate() {
        let va = Vector3::from(vertices[vi]);
        for &ni in &neighbors[vi] {
            let ni = ni as usize;
            let angle = va.angle(&Vector3::from(vertices[ni]));
            if angle > 1e-12 {
                let grad = (bf_vi - boltzmann[ni]).abs() / angle;
                max_grad = max_grad.max(grad);
            }
        }
    }
    max_grad
}

/// Compute the max angular Boltzmann-weight gradient for a 6D slab.
///
/// For each vertex pair (vi, vj), checks:
///   max over neighbors ni of vi: |bf(vi,vj) - bf(ni,vj)| / angle(vi, ni)
///   max over neighbors nj of vj: |bf(vi,vj) - bf(vi,nj)| / angle(vj, nj)
/// where bf = exp(-βE). Units: 1/rad (dimensionless).
fn compute_gradient(
    data: &[f64],
    vertices: &[[f64; 3]],
    neighbors: &[Vec<u16>],
    n_v: usize,
    beta: f64,
) -> f64 {
    // Precompute all Boltzmann weights upfront: each entry is accessed ~12 times
    // as a neighbor in the stencil, avoiding redundant exp() calls.
    let boltzmann: Vec<f64> = data.iter().map(|&e| (-beta * e).exp()).collect();

    // Neighbor angles depend only on mesh topology, not on the (vi, vj) pairing
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
            let bf_ij = boltzmann[vi * n_v + vj];

            for (k, &ni) in neighbors[vi].iter().enumerate() {
                let angle = neighbor_angles[vi][k];
                if angle > 1e-12 {
                    let grad = (bf_ij - boltzmann[ni as usize * n_v + vj]).abs() / angle;
                    max_grad = max_grad.max(grad);
                }
            }

            for (k, &nj) in neighbors[vj].iter().enumerate() {
                let angle = neighbor_angles[vj][k];
                if angle > 1e-12 {
                    let grad = (bf_ij - boltzmann[vi * n_v + nj as usize]).abs() / angle;
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
    fn mesh_level_locate() {
        let lvl = MeshLevel::new(2);
        let dir = Vector3::new(1.0, 0.0, 0.0);
        let (face, bary) = lvl.locate(&dir);
        // Bary should sum to ~1 and all be non-negative
        let sum: f64 = bary.iter().sum();
        assert_relative_eq!(sum, 1.0, epsilon = 1e-10);
        assert!(bary.iter().all(|&b| b >= -1e-10));
        // Face indices should be valid
        assert!(face.iter().all(|&i| i < lvl.n_vertices));
    }

    #[test]
    fn lattice_meshlevel_locator_rebuilds_on_load() {
        use rand::{Rng, SeedableRng};
        let lvl = MeshLevel::with_subdivision(Subdivision::Lattice, 4);
        // Serialize (locator is #[serde(skip)]) and reload: the restored level must
        // rebuild its lattice locator from the stored subdivision tag + vertex count.
        let bytes = bincode::serialize(&lvl).unwrap();
        let restored: MeshLevel = bincode::deserialize(&bytes).unwrap();

        let mut rng = rand::rngs::StdRng::seed_from_u64(7);
        for _ in 0..2000 {
            let dir = Vector3::new(
                rng.gen_range(-1.0..1.0),
                rng.gen_range(-1.0..1.0),
                rng.gen_range(-1.0..1.0),
            )
            .normalize();
            assert_eq!(lvl.locate(&dir), restored.locate(&dir));
        }
    }

    #[test]
    fn lattice_builder_level_zero_is_valid_base() {
        // The builder constructs levels 0..=max; for the lattice, level 0 must map
        // to the 12-vertex base (frequency 1), not the degenerate frequency 0.
        let base = MeshLevel::with_subdivision(Subdivision::Lattice, 0);
        assert_eq!(base.n_vertices, 12);
        let (_face, w) = base.locate(&Vector3::new(0.3, 0.5, 0.8).normalize());
        assert!((w.iter().sum::<f64>() - 1.0).abs() < 1e-9);

        // A full builder over levels 0..=2 must construct without panicking.
        let builder = AdaptiveBuilder::with_subdivision(
            Subdivision::Lattice,
            1.0,
            2.0,
            0.5,
            0.5,
            2,
            0.1,
            1.0,
        );
        assert_eq!(builder.current_n_div(), 2);
    }

    /// Fill a slab with constant energy → gradient should be 0.
    #[test]
    fn gradient_constant_surface() {
        let lvl = MeshLevel::new(1);
        let n_v = lvl.n_vertices;
        let data = vec![42.0; n_v * n_v];
        let grad = compute_gradient(&data, &lvl.vertices, &lvl.neighbors, n_v, 1.0);
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
        let grad = compute_gradient(&data, &lvl.vertices, &lvl.neighbors, n_v, 1.0);
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

    /// Analytic test function: f(dir_a, dir_b) = dot(dir_a, ref)^2 + dot(dir_b, ref)^2.
    /// Smooth on S² and bounded to [0, 2], suiting both interpolation-accuracy
    /// checks and non-repulsive classification at β = 1.
    fn analytic_fn(dir_a: &Vector3, dir_b: &Vector3) -> f64 {
        let reference = Vector3::new(1.0, 1.0, 1.0).normalize();
        let da = dir_a.normalize().dot(&reference);
        let db = dir_b.normalize().dot(&reference);
        da * da + db * db
    }

    /// Build an adaptive table whose every vertex pair is filled by `f(dir_a, dir_b)`.
    ///
    /// A tiny `gradient_threshold` (1e-9) forces every varying slab to
    /// `Mesh { interpolate: true }` and suppresses resolution coarsening, so the
    /// mesh level stays at `max_n_div` for all R. The same spatial function is
    /// used for every (R, ω) slab, making lookups robust to R/ω bin snapping.
    fn make_adaptive_table_with_fn(
        max_n_div: usize,
        f: impl Fn(&Vector3, &Vector3) -> f64,
    ) -> Table6DAdaptive<f32> {
        let mut builder = AdaptiveBuilder::new(
            5.0,
            8.0,
            1.0,
            std::f64::consts::TAU / 4.0,
            max_n_div,
            1e-9,
            1.0,
        );
        for ri in 0..builder.n_r() {
            let n_v = builder.current_n_vertices();
            let verts = builder.vertex_directions(builder.current_level()).to_vec();
            for oi in 0..builder.n_omega() {
                let mut data = vec![0.0; n_v * n_v];
                for vi in 0..n_v {
                    let va = Vector3::from(verts[vi]);
                    for vj in 0..n_v {
                        let vb = Vector3::from(verts[vj]);
                        data[vi * n_v + vj] = f(&va, &vb);
                    }
                }
                builder.set_slab(ri, oi, &data);
            }
            builder.finish_r_slice(ri);
        }
        builder.build()
    }

    /// Quaternion-driven lookups that land on exact icosphere vertices should
    /// return the stored value. Drives the full `orient → inverse_orient →
    /// lookup` path (as Faunus does) with an arbitrary lab-frame rotation, and
    /// checks the O(1) nearest-vertex / barycentric lookup recovers the exact
    /// sampled energy. Mirrors the `Table6DFlat` test of the same name.
    #[test]
    fn lookup_via_orient_inverse_roundtrip() {
        use crate::orient::{inverse_orient, orient};
        use nalgebra::UnitQuaternion;
        use rand::Rng;

        let table = make_adaptive_table_with_fn(3, analytic_fn); // 162 vertices
        let mut rng = rand::thread_rng();

        // Skip ri = 0: its bin center sits exactly on rmin, where fp noise in
        // the recovered r2 can dip below the table's lower bound. Interior
        // slabs hold identical data, so this loses no coverage.
        let mut checked = 0usize;
        for ri in 1..table.n_r {
            let Some(level) = table.mesh_level_at_r(ri) else {
                continue; // Scalar/Repulsive slab — no per-vertex storage
            };
            let verts = &level.vertices;
            let r = table.rmin + ri as f64 * table.dr;

            // Cover the 12 pentagonal (degree-5) vertices — the icosphere's
            // special points — plus a spread of ordinary vertices. Vertices are
            // BFS-reordered, so pentagonal ones are identified by neighbor
            // count, not by index.
            let mut idxs: Vec<usize> = (0..verts.len())
                .filter(|&i| level.neighbors[i].len() == 5)
                .collect();
            idxs.extend((0..verts.len()).step_by(15));
            idxs.sort_unstable();
            idxs.dedup();

            for &vi in &idxs {
                for &vj in &idxs {
                    let va = Vector3::from(verts[vi]);
                    let vb = Vector3::from(verts[vj]);
                    let exact = analytic_fn(&va, &vb);

                    for _ in 0..3 {
                        let omega = rng.gen_range(0.0..std::f64::consts::TAU);
                        let (q_a, q_b, sep) = orient(r, omega, &va, &vb);

                        // Apply an arbitrary rotation to both molecules, as if
                        // they were placed/rotated during a simulation step.
                        let random_q = UnitQuaternion::from_euler_angles(
                            rng.gen_range(0.0..std::f64::consts::TAU),
                            rng.gen_range(0.0..std::f64::consts::PI),
                            rng.gen_range(0.0..std::f64::consts::TAU),
                        );
                        let q_a_rot = random_q * q_a;
                        let q_b_rot = random_q * q_b;
                        let sep_rot = random_q.transform_vector(&sep);

                        let (r2, omega2, dir_a2, dir_b2) =
                            inverse_orient(&sep_rot, &q_a_rot, &q_b_rot);
                        let looked_up = table.lookup(r2, omega2, &dir_a2, &dir_b2);

                        assert!(
                            (looked_up - exact).abs() < 0.05,
                            "ri={ri}, vi={vi}, vj={vj}: looked_up={looked_up:.4}, \
                             exact={exact:.4}, omega={omega2:.4}"
                        );
                        checked += 1;
                    }
                }
            }
        }
        assert!(checked > 0, "no mesh slabs were exercised");
    }

    /// Off-vertex queries should interpolate close to the underlying smooth
    /// function. Confirms at least one slab actually uses barycentric
    /// interpolation, then bounds the interpolation error over random
    /// directions. Mirrors the `Table6DFlat` test of the same name.
    #[test]
    fn lookup_off_vertex_interpolation() {
        use rand::Rng;

        let table = make_adaptive_table_with_fn(3, analytic_fn); // 162 vertices

        // The test is only meaningful if interpolation is actually exercised.
        assert!(
            table
                .slab_res
                .iter()
                .any(|sr| matches!(sr, SlabResolution::Mesh { interpolate: true, .. })),
            "expected at least one interpolated mesh slab"
        );

        let mut rng = rand::thread_rng();
        let mut max_err = 0.0f64;

        let random_dir = |rng: &mut rand::rngs::ThreadRng| loop {
            let x: f64 = rng.gen_range(-1.0..1.0);
            let y: f64 = rng.gen_range(-1.0..1.0);
            let z: f64 = rng.gen_range(-1.0..1.0);
            let r2 = x * x + y * y + z * z;
            if r2 > 0.01 && r2 < 1.0 {
                return Vector3::new(x, y, z).normalize();
            }
        };

        for _ in 0..500 {
            let dir_a = random_dir(&mut rng);
            let dir_b = random_dir(&mut rng);

            // r = rmin, omega = 0 selects a mesh slab (ri = 0, oi = 0).
            let interpolated = table.lookup(table.rmin, 0.0, &dir_a, &dir_b);
            let exact = analytic_fn(&dir_a, &dir_b);
            max_err = max_err.max((interpolated - exact).abs());
        }

        // Axis-aligned (cube-face centre) and body-diagonal (cube-corner)
        // directions fall in the projected grid's degenerate cells; verify
        // they interpolate within the same tolerance as generic directions.
        let inv_sqrt3 = 1.0 / 3.0_f64.sqrt();
        let degenerate = [
            Vector3::new(1.0, 0.0, 0.0),
            Vector3::new(-1.0, 0.0, 0.0),
            Vector3::new(0.0, 1.0, 0.0),
            Vector3::new(0.0, -1.0, 0.0),
            Vector3::new(0.0, 0.0, 1.0),
            Vector3::new(0.0, 0.0, -1.0),
            Vector3::new(inv_sqrt3, inv_sqrt3, inv_sqrt3),
            Vector3::new(-inv_sqrt3, -inv_sqrt3, inv_sqrt3),
        ];
        for &dir_a in &degenerate {
            for &dir_b in &degenerate {
                let interpolated = table.lookup(table.rmin, 0.0, &dir_a, &dir_b);
                let exact = analytic_fn(&dir_a, &dir_b);
                max_err = max_err.max((interpolated - exact).abs());
            }
        }

        assert!(
            max_err < 0.3,
            "Max interpolation error {max_err:.4} exceeds tolerance for smooth quadratic"
        );
    }

    /// Exercises the O(1) nearest-vertex lookup path (`Mesh { interpolate:
    /// false }`), which the barycentric tests never reach. At an exact vertex it
    /// must return the stored value; a small perturbation toward a neighbour —
    /// where the same vertex is still nearest — must return the *same* value,
    /// confirming the lookup is piecewise-constant (no interpolation leaks in).
    #[test]
    fn lookup_nearest_vertex_no_interpolation() {
        let n_div = 2;
        let lvl = MeshLevel::new(n_div);
        let n_v = lvl.n_vertices;
        let beta = 1.0;

        // Fill with a smooth function and measure its Boltzmann-weight gradient,
        // then set the threshold so the slab lands in the nearest-vertex band:
        //   scalar_threshold = threshold/10 < grad < threshold.
        let mut data = vec![0.0; n_v * n_v];
        for vi in 0..n_v {
            let va = Vector3::from(lvl.vertices[vi]);
            for vj in 0..n_v {
                let vb = Vector3::from(lvl.vertices[vj]);
                data[vi * n_v + vj] = analytic_fn(&va, &vb);
            }
        }
        let grad = compute_gradient(&data, &lvl.vertices, &lvl.neighbors, n_v, beta);
        let gradient_threshold = grad * 2.0;

        let mut builder = AdaptiveBuilder::new(
            5.0,
            8.0,
            1.0,
            std::f64::consts::TAU / 4.0,
            n_div,
            gradient_threshold,
            beta,
        );
        // Fill an interior R slab (ri = 1, r = 6.0) to dodge the rmin boundary.
        let ri = 1;
        for oi in 0..builder.n_omega() {
            builder.set_slab(ri, oi, &data);
        }
        builder.finish_r_slice(ri);
        let table = builder.build();

        let r = table.rmin + ri as f64 * table.dr;
        assert!(
            matches!(
                table.slab_res[ri * table.n_omega],
                SlabResolution::Mesh {
                    interpolate: false,
                    ..
                }
            ),
            "expected a nearest-vertex (non-interpolated) mesh slab, got {:?}",
            table.slab_res[ri * table.n_omega]
        );

        let level = table.mesh_level_at_r(ri).unwrap();
        for vi in (0..n_v).step_by(7) {
            for vj in (0..n_v).step_by(11) {
                let va = Vector3::from(level.vertices[vi]);
                let vb = Vector3::from(level.vertices[vj]);
                let exact = analytic_fn(&va, &vb);

                // Exact vertex → stored value, no interpolation contamination.
                let at_vertex = table.lookup(r, 0.0, &va, &vb);
                assert!(
                    (at_vertex - exact).abs() < 1e-3,
                    "vi={vi}, vj={vj}: nearest-vertex lookup {at_vertex:.5} != stored {exact:.5}"
                );

                // Nudge dir_a 10% toward a neighbour: va is still the nearest
                // vertex, so a nearest-vertex lookup must be unchanged.
                let nb = level.neighbors[vi][0] as usize;
                let neighbor = Vector3::from(level.vertices[nb]);
                let perturbed = (va + 0.1 * (neighbor - va)).normalize();
                let near_vertex = table.lookup(r, 0.0, &perturbed, &vb);
                assert!(
                    (near_vertex - exact).abs() < 1e-3,
                    "vi={vi}, vj={vj}: perturbed lookup {near_vertex:.5} should equal vertex \
                     value {exact:.5} (piecewise-constant); interpolation leaked in"
                );
            }
        }
    }

    /// Verifies ω-bin selection through `lookup`, including the periodic wrap at
    /// ω = 0 / TAU and nearest-bin rounding. Every other lookup test fills
    /// identical data across ω, so this is the only check that the ω index is
    /// actually computed correctly.
    #[test]
    fn lookup_selects_correct_omega_bin() {
        let omega_step = std::f64::consts::TAU / 8.0;
        let mut builder = AdaptiveBuilder::new(5.0, 8.0, 1.0, omega_step, 1, 1.0, 0.1);
        let n_omega = builder.n_omega();

        // Each ω slab is a distinct constant → classified Scalar, so a lookup
        // returns exactly that slab's value, isolating ω-bin selection from
        // angular interpolation.
        let value = |oi: usize| 10.0 + oi as f64;
        for ri in 0..builder.n_r() {
            // n_v is read per R-slice: all-Scalar slices make the builder coarsen.
            let n_v = builder.current_n_vertices();
            for oi in 0..n_omega {
                builder.set_slab(ri, oi, &vec![value(oi); n_v * n_v]);
            }
            builder.finish_r_slice(ri);
        }
        let table = builder.build();

        let dir = Vector3::new(1.0, 0.0, 0.0);
        let r = table.rmin + table.dr; // interior R bin

        // Each bin centre maps to its own value.
        for oi in 0..n_omega {
            let e = table.lookup(r, oi as f64 * omega_step, &dir, &dir);
            assert!(
                (e - value(oi)).abs() < 1e-3,
                "ω bin {oi}: lookup {e:.3} != {:.3}",
                value(oi)
            );
        }

        // Periodic wrap via rem_euclid: ω = TAU → bin 0; ω = −step → last bin;
        // a tiny negative ω wraps forward to bin 0.
        let last = n_omega - 1;
        let e_tau = table.lookup(r, std::f64::consts::TAU, &dir, &dir);
        assert!(
            (e_tau - value(0)).abs() < 1e-3,
            "ω = TAU should wrap to bin 0, got {e_tau:.3}"
        );
        let e_neg = table.lookup(r, -omega_step, &dir, &dir);
        assert!(
            (e_neg - value(last)).abs() < 1e-3,
            "ω = −step should wrap to bin {last}, got {e_neg:.3}"
        );
        let e_tiny_neg = table.lookup(r, -1e-9, &dir, &dir);
        assert!(
            (e_tiny_neg - value(0)).abs() < 1e-3,
            "ω = −ε should wrap to bin 0, got {e_tiny_neg:.3}"
        );

        // The seam between the last bin and bin 0 sits at (last + 0.5)·step:
        // a positive ω just past it wraps *forward* to bin 0 (it is nearer the
        // bin-0 centre at TAU), rather than rounding back to the last bin.
        let e_seam_lo = table.lookup(r, (last as f64 + 0.4) * omega_step, &dir, &dir);
        assert!(
            (e_seam_lo - value(last)).abs() < 1e-3,
            "ω = (last+0.4)·step should stay in bin {last}, got {e_seam_lo:.3}"
        );
        let e_seam_hi = table.lookup(r, (last as f64 + 0.6) * omega_step, &dir, &dir);
        assert!(
            (e_seam_hi - value(0)).abs() < 1e-3,
            "ω = (last+0.6)·step should wrap to bin 0, got {e_seam_hi:.3}"
        );

        // Nearest-bin rounding around a bin boundary (between bins 2 and 3 at
        // 2.5·step): just below stays in 2, just above rounds up to 3.
        let e_below = table.lookup(r, 2.4 * omega_step, &dir, &dir);
        assert!(
            (e_below - value(2)).abs() < 1e-3,
            "ω = 2.4·step should bin to 2, got {e_below:.3}"
        );
        let e_above = table.lookup(r, 2.6 * omega_step, &dir, &dir);
        assert!(
            (e_above - value(3)).abs() < 1e-3,
            "ω = 2.6·step should bin to 3, got {e_above:.3}"
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
        let beta = 1.0;

        // First, measure the actual Boltzmann-weight gradient of our test data
        let lvl = MeshLevel::new(1);
        let mut test_data = vec![0.0; n_v_at_level1 * n_v_at_level1];
        for vi in 0..n_v_at_level1 {
            let va = Vector3::from(lvl.vertices[vi]);
            for vj in 0..n_v_at_level1 {
                let vb = Vector3::from(lvl.vertices[vj]);
                test_data[vi * n_v_at_level1 + vj] =
                    0.1 * va.dot(&reference) + 0.1 * vb.dot(&reference);
            }
        }
        let grad = compute_gradient(
            &test_data,
            &lvl.vertices,
            &lvl.neighbors,
            n_v_at_level1,
            beta,
        );

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
            beta,
        );

        let n_v = builder.current_n_vertices();
        let verts = builder.vertex_directions(builder.current_level()).to_vec();

        for oi in 0..builder.n_omega() {
            let mut data = vec![0.0; n_v * n_v];
            for vi in 0..n_v {
                let va = Vector3::from(verts[vi]);
                for vj in 0..n_v {
                    let vb = Vector3::from(verts[vj]);
                    data[vi * n_v + vj] = 0.1 * va.dot(&reference) + 0.1 * vb.dot(&reference);
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
        assert!(e.abs() < 1.0, "Expected ~0, got {e}");
    }

    #[test]
    fn repulsive_saves_storage() {
        let n_omega_bins = 4;
        let omega_step = std::f64::consts::TAU / n_omega_bins as f64;
        let beta = 1.0;
        let reference = Vector3::new(1.0, 0.0, 0.0);

        // Repulsive data: all energies >> kT, so expm1(-βU) ≈ -1
        let make_repulsive_data = |verts: &[[f64; 3]], n_v: usize| -> Vec<f64> {
            let mut data = vec![0.0; n_v * n_v];
            for vi in 0..n_v {
                let va = Vector3::from(verts[vi]);
                for vj in 0..n_v {
                    let vb = Vector3::from(verts[vj]);
                    data[vi * n_v + vj] =
                        200.0 + 50.0 * va.dot(&reference) + 50.0 * vb.dot(&reference);
                }
            }
            data
        };

        // Non-repulsive data with Boltzmann-weight variation above threshold
        let make_accessible_data = |verts: &[[f64; 3]], n_v: usize| -> Vec<f64> {
            let mut data = vec![0.0; n_v * n_v];
            for vi in 0..n_v {
                let va = Vector3::from(verts[vi]);
                for vj in 0..n_v {
                    let vb = Vector3::from(verts[vj]);
                    data[vi * n_v + vj] = 2.0 * va.dot(&reference) + 2.0 * vb.dot(&reference);
                }
            }
            data
        };

        // Repulsive builder: high energies → classified as Repulsive → zero storage
        let mut builder_rep = AdaptiveBuilder::new(5.0, 7.0, 1.0, omega_step, 1, 100.0, beta);
        // Full builder: moderate energies with angular variation → kept as Mesh
        let mut builder_full = AdaptiveBuilder::new(5.0, 7.0, 1.0, omega_step, 1, 100.0, beta);

        for ri in 0..2 {
            let n_v = builder_rep.current_n_vertices();
            let verts = builder_rep
                .vertex_directions(builder_rep.current_level())
                .to_vec();
            for oi in 0..builder_rep.n_omega() {
                builder_rep.set_slab(ri, oi, &make_repulsive_data(&verts, n_v));
            }
            builder_rep.finish_r_slice(ri);

            let n_v = builder_full.current_n_vertices();
            let verts = builder_full
                .vertex_directions(builder_full.current_level())
                .to_vec();
            for oi in 0..builder_full.n_omega() {
                builder_full.set_slab(ri, oi, &make_accessible_data(&verts, n_v));
            }
            builder_full.finish_r_slice(ri);
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

    // -----------------------------------------------------------------------
    // Table3DAdaptive / Adaptive3DBuilder tests
    // -----------------------------------------------------------------------

    #[test]
    fn gradient_1d_constant_surface() {
        let lvl = MeshLevel::new(1);
        let data = vec![42.0; lvl.n_vertices];
        let grad = compute_gradient_1d(&data, &lvl.vertices, &lvl.neighbors, 1.0);
        assert_relative_eq!(grad, 0.0, epsilon = 1e-12);
    }

    #[test]
    fn gradient_1d_nonzero_surface() {
        let lvl = MeshLevel::new(1);
        let reference = Vector3::new(1.0, 1.0, 1.0).normalize();
        let data: Vec<f64> = lvl
            .vertices
            .iter()
            .map(|v| Vector3::from(*v).dot(&reference))
            .collect();
        let grad = compute_gradient_1d(&data, &lvl.vertices, &lvl.neighbors, 1.0);
        assert!(grad > 0.0);
    }

    #[test]
    fn builder_3d_constant_becomes_scalar() {
        let mut builder = Adaptive3DBuilder::new(5.0, 10.0, 1.0, 2, 1.0, 0.001);
        for ri in 0..builder.n_r() {
            let n_v = builder.current_n_vertices();
            builder.set_slab(ri, &vec![42.0; n_v]);
            builder.finish_r_slice(ri);
        }
        let table = builder.build();
        for sr in &table.slab_res {
            match sr {
                SlabResolution::Scalar(v) => assert_relative_eq!(*v as f64, 42.0, epsilon = 1e-4),
                _ => panic!("expected scalar for constant data"),
            }
        }
        assert!(table.data.is_empty());
    }

    #[test]
    fn builder_3d_varying_becomes_mesh() {
        let mut builder = Adaptive3DBuilder::new(5.0, 7.0, 1.0, 1, 1e10, 0.001);
        let reference = Vector3::new(1.0, 0.0, 0.0);
        for ri in 0..builder.n_r() {
            let verts = builder.vertex_directions(builder.current_level()).to_vec();
            let data: Vec<f64> = verts
                .iter()
                .map(|v| 10.0 * Vector3::from(*v).dot(&reference))
                .collect();
            builder.set_slab(ri, &data);
            builder.finish_r_slice(ri);
        }
        let table = builder.build();
        let dir = Vector3::new(1.0, 0.0, 0.0);
        let val = table.lookup(5.5, &dir);
        assert!(val.is_finite(), "lookup returned {val}");
    }

    #[test]
    fn builder_3d_repulsive() {
        let mut builder = Adaptive3DBuilder::new(5.0, 7.0, 1.0, 1, 1e10, 1.0);
        let n_v = builder.current_n_vertices();
        // ri=0: strongly repulsive
        builder.set_slab(0, &vec![200.0; n_v]);
        builder.finish_r_slice(0);
        // ri=1: moderate
        let n_v = builder.current_n_vertices();
        builder.set_slab(1, &vec![5.0; n_v]);
        builder.finish_r_slice(1);

        let table = builder.build();
        let dir = Vector3::new(1.0, 0.0, 0.0);
        assert!(table.lookup(5.0, &dir).is_infinite());
        let e = table.lookup(6.0, &dir);
        assert!((e - 5.0).abs() < 0.1, "Expected ~5, got {e}");
    }

    #[test]
    fn builder_3d_no_interpolation_for_smooth() {
        let lvl = MeshLevel::new(1);
        let reference = Vector3::new(1.0, 0.0, 0.0);
        let beta = 1.0;
        let test_data: Vec<f64> = lvl
            .vertices
            .iter()
            .map(|v| 0.1 * Vector3::from(*v).dot(&reference))
            .collect();
        let grad = compute_gradient_1d(&test_data, &lvl.vertices, &lvl.neighbors, beta);
        let gradient_threshold = grad * 2.0;
        assert!(grad > gradient_threshold / 10.0);

        let mut builder = Adaptive3DBuilder::new(5.0, 7.0, 1.0, 1, gradient_threshold, beta);
        let verts = builder.vertex_directions(builder.current_level()).to_vec();
        let data: Vec<f64> = verts
            .iter()
            .map(|v| 0.1 * Vector3::from(*v).dot(&reference))
            .collect();
        builder.set_slab(0, &data);
        builder.finish_r_slice(0);

        assert!(matches!(
            builder.slab_res[0],
            SlabResolution::Mesh {
                interpolate: false,
                ..
            }
        ));

        let n_v = builder.current_n_vertices();
        builder.set_slab(1, &vec![10.0; n_v]);
        builder.finish_r_slice(1);
        let table = builder.build();
        let dir = Vector3::new(1.0, 0.0, 0.0);
        let e = table.lookup(5.0, &dir);
        assert!(e.is_finite());
    }

    #[test]
    fn lookup_3d_constant() {
        let mut builder = Adaptive3DBuilder::new(5.0, 10.0, 1.0, 2, 1e10, 0.001);
        for ri in 0..builder.n_r() {
            let n_v = builder.current_n_vertices();
            builder.set_slab(ri, &vec![42.0; n_v]);
            builder.finish_r_slice(ri);
        }
        let table = builder.build();
        let dir = Vector3::new(1.0, 0.0, 0.0);
        let e = table.lookup(7.0, &dir);
        assert!(
            (e - 42.0).abs() < 0.1,
            "Expected ~42 for constant table, got {e}"
        );
    }

    #[test]
    fn lookup_3d_out_of_range() {
        let mut builder = Adaptive3DBuilder::new(5.0, 10.0, 1.0, 1, 1e10, 0.001);
        for ri in 0..builder.n_r() {
            let n_v = builder.current_n_vertices();
            builder.set_slab(ri, &vec![1.0; n_v]);
            builder.finish_r_slice(ri);
        }
        let table = builder.build();
        let dir = Vector3::new(1.0, 0.0, 0.0);
        assert_eq!(table.lookup(3.0, &dir), 0.0);
        assert_eq!(table.lookup(20.0, &dir), 0.0);
    }

    #[test]
    fn boltzmann_3d_constant() {
        let mut builder = Adaptive3DBuilder::new(5.0, 10.0, 1.0, 2, 1e10, 0.001);
        for ri in 0..builder.n_r() {
            let n_v = builder.current_n_vertices();
            builder.set_slab(ri, &vec![42.0; n_v]);
            builder.finish_r_slice(ri);
        }
        let table = builder.build();
        let dir = Vector3::new(1.0, 0.0, 0.0);
        let e = table.lookup_boltzmann(7.0, &dir, 1.0);
        assert!(
            (e - 42.0).abs() < 0.1,
            "Expected ~42 for constant Boltzmann, got {e}"
        );
    }

    #[test]
    fn adaptive_3d_resolution_decreases() {
        let mut builder = Adaptive3DBuilder::new(5.0, 10.0, 1.0, 3, 1e10, 0.001);
        assert_eq!(builder.current_n_div(), 3);

        let n_v = builder.current_n_vertices();
        builder.set_slab(0, &vec![42.0; n_v]);
        builder.finish_r_slice(0);
        assert!(builder.current_n_div() < 3);

        let n_v = builder.current_n_vertices();
        builder.set_slab(1, &vec![42.0; n_v]);
        builder.finish_r_slice(1);
        assert!(builder.current_n_div() < 2);
    }

    #[test]
    fn round_trip_3d_save_load() {
        let mut builder = Adaptive3DBuilder::new(5.0, 8.0, 1.0, 1, 1e10, 0.001);
        for ri in 0..builder.n_r() {
            let n_v = builder.current_n_vertices();
            builder.set_slab(ri, &vec![7.0; n_v]);
            builder.finish_r_slice(ri);
        }
        let table = builder.build();

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("adaptive3d.bin.gz");
        table.save(&path).unwrap();
        let loaded = Table3DAdaptive::<f32>::load(&path).unwrap();

        assert_eq!(loaded.n_r, table.n_r);
        assert_eq!(loaded.data.len(), table.data.len());
        assert_eq!(loaded.slab_res.len(), table.slab_res.len());

        let d = Vector3::new(1.0, 0.0, 0.0);
        let orig = table.lookup(6.0, &d);
        let load = loaded.lookup(6.0, &d);
        assert!(
            (orig - load).abs() < 1e-4,
            "Round-trip mismatch: {orig} vs {load}"
        );
    }
}
