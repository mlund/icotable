//! Portable, flat representation of a 6D table for file I/O and fast lookup.
//!
//! [`Table6DFlat`] stores all data in contiguous vectors (no nested Arc/OnceLock)
//! for efficient serialization with bincode and simple interpolation at runtime.

use crate::icotable::{Face, IcoTable4D, Table6D};
use crate::Vector3;
use anyhow::Result;
use flate2::{read::GzDecoder, write::GzEncoder, Compression};
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Flat, serializable 6D lookup table.
///
/// Layout: `data[r_idx * (n_omega * n_v * n_v) + omega_idx * (n_v * n_v) + vi * n_v + vj]`
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Table6DFlat<T: num_traits::Float> {
    pub rmin: f64,
    pub rmax: f64,
    pub dr: f64,
    pub n_r: usize,
    pub omega_step: f64,
    pub n_omega: usize,
    pub n_vertices: usize,
    /// Normalized unit-sphere vertex positions `[x, y, z]`.
    pub vertices: Vec<[f64; 3]>,
    /// Neighbor indices per vertex.
    pub neighbors: Vec<Vec<u16>>,
    /// Flat data array.
    pub data: Vec<T>,
}

impl<T: num_traits::Float + Serialize + serde::de::DeserializeOwned> Table6DFlat<T> {
    /// Save to a bincode file. A `.gz` suffix enables gzip compression.
    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        let path = path.as_ref();
        let file = std::io::BufWriter::new(std::fs::File::create(path)?);
        if path.extension().is_some_and(|ext| ext == "gz") {
            let encoder = GzEncoder::new(file, Compression::default());
            bincode::serialize_into(encoder, self)?;
        } else {
            bincode::serialize_into(file, self)?;
        }
        Ok(())
    }

    /// Load from a bincode file. A `.gz` suffix enables gzip decompression.
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let file = std::io::BufReader::new(std::fs::File::open(path)?);
        let table: Self = if path.extension().is_some_and(|ext| ext == "gz") {
            bincode::deserialize_from(GzDecoder::new(file))?
        } else {
            bincode::deserialize_from(file)?
        };
        Ok(table)
    }
}

impl TryFrom<&Table6D> for Table6DFlat<f32> {
    type Error = anyhow::Error;

    fn try_from(table: &Table6D) -> Result<Self> {
        let r_min = table.min_key();
        let r_max = table.max_key();
        let dr = table.key_step();
        // Interior bins (excluding PaddedTable's boundary padding)
        let n_r = table.len() - 2;

        // Extract vertex/omega info from first (r, omega=0) entry
        let first_omega_table = table.get(r_min)?;
        let omega_step = first_omega_table.key_step();
        let omega_min = first_omega_table.min_key();
        let n_omega = first_omega_table.len() - 2;

        let first_ico = first_omega_table.get(omega_min)?;
        let n_vertices = first_ico.len();

        // Collect normalized vertex positions and neighbors
        let vertices: Vec<[f64; 3]> = first_ico
            .iter_positions()
            .map(|p| {
                let n = p.normalize();
                [n.x, n.y, n.z]
            })
            .collect();

        let neighbors: Vec<Vec<u16>> = first_ico
            .iter_vertices()
            .map(|v| v.neighbors.clone())
            .collect();

        let stride = n_omega * n_vertices * n_vertices;
        let mut data = vec![0.0f32; n_r * stride];

        for ri in 0..n_r {
            let r = r_min + ri as f64 * dr;
            for oi in 0..n_omega {
                let omega = omega_min + oi as f64 * omega_step;
                let ico4d = table.get_icospheres(r, omega)?;
                for (vi, vj, energy) in flat_iter_indexed(ico4d, n_vertices) {
                    let idx = ri * stride + oi * (n_vertices * n_vertices) + vi * n_vertices + vj;
                    data[idx] = *energy as f32;
                }
            }
        }

        Ok(Self {
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
        })
    }
}

impl<T: num_traits::Float + Into<f64>> Table6DFlat<T> {
    /// Lookup energy by nearest R/ω bin and barycentric interpolation on icospheres.
    pub fn lookup(&self, r: f64, omega: f64, dir_a: &Vector3, dir_b: &Vector3) -> f64 {
        if r < self.rmin || r > self.rmax {
            return 0.0;
        }

        let ri = ((r - self.rmin) / self.dr + 0.5) as usize;
        let ri = ri.min(self.n_r.saturating_sub(1));

        // Wrap omega into [0, 2π)
        let omega = omega.rem_euclid(std::f64::consts::TAU);
        let oi = (omega / self.omega_step + 0.5) as usize % self.n_omega;

        // Find nearest faces and barycentric coords for both directions
        let (face_a, bary_a) = self.find_face_bary(dir_a);
        let (face_b, bary_b) = self.find_face_bary(dir_b);

        // Bilinear barycentric: bary_a^T * M * bary_b
        let base = ri * self.n_omega * self.n_vertices * self.n_vertices
            + oi * self.n_vertices * self.n_vertices;

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

    /// Find the mesh face containing `dir` and return barycentric coordinates.
    ///
    /// Searches all triangles incident on the nearest vertex (formed by pairs of
    /// mutual neighbors) for one where all barycentric coordinates are non-negative.
    fn find_face_bary(&self, dir: &Vector3) -> (Face, [f64; 3]) {
        let dir = dir.normalize();

        // Brute-force nearest vertex
        let nearest = self
            .vertices
            .iter()
            .enumerate()
            .map(|(i, v)| {
                let d = (dir.x - v[0]).powi(2) + (dir.y - v[1]).powi(2) + (dir.z - v[2]).powi(2);
                (i, d)
            })
            .min_by(|a, b| a.1.total_cmp(&b.1))
            .expect("vertices must not be empty")
            .0;

        let nbrs = &self.neighbors[nearest];

        // Search all neighbor pairs that share an edge (= actual mesh triangle).
        // Picking the two closest neighbors would often select non-adjacent vertices,
        // yielding a spurious triangle and poor interpolation.
        let mut best_face = [0usize; 3];
        let mut best_min_bary = f64::NEG_INFINITY;

        for (idx_i, &ni) in nbrs.iter().enumerate() {
            let ni = ni as usize;
            for &nj in &nbrs[idx_i + 1..] {
                let nj = nj as usize;
                // Only consider neighbor pairs that share an edge (valid mesh triangle)
                if !self.neighbors[ni].contains(&(nj as u16)) {
                    continue;
                }
                let face = [nearest, ni, nj];
                let a = self.vertex_vec3(face[0]);
                let b = self.vertex_vec3(face[1]);
                let c = self.vertex_vec3(face[2]);
                let bary = projected_barycentric(&dir, &a, &b, &c);
                let min_b = bary[0].min(bary[1]).min(bary[2]);
                if min_b > best_min_bary {
                    best_min_bary = min_b;
                    best_face = face;
                }
            }
        }

        // Recompute barycentric coords for the sorted face (index order matters for lookup)
        best_face.sort_unstable();
        let a = self.vertex_vec3(best_face[0]);
        let b = self.vertex_vec3(best_face[1]);
        let c = self.vertex_vec3(best_face[2]);
        let best_bary = projected_barycentric(&dir, &a, &b, &c);

        (best_face, best_bary)
    }

    fn vertex_vec3(&self, i: usize) -> Vector3 {
        let v = &self.vertices[i];
        Vector3::new(v[0], v[1], v[2])
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

/// Indexed flat iterator yielding `(vi, vj, &f64)` for ordered vertex pairs.
fn flat_iter_indexed(ico4d: &IcoTable4D, n_vertices: usize) -> Vec<(usize, usize, &f64)> {
    let mut result = Vec::with_capacity(n_vertices * n_vertices);
    for vi in 0..n_vertices {
        let inner = ico4d.get_data(vi).expect("missing 4D table entry");
        for vj in 0..n_vertices {
            if let Some(val) = inner.get_data(vj) {
                result.push((vi, vj, val));
            }
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a small table from a real icosphere for lookup testing.
    fn make_test_table(min_points: usize, n_r: usize, n_omega: usize) -> Table6DFlat<f32> {
        use crate::{make_icosphere, make_vertices};
        let ico = make_icosphere(min_points).unwrap();
        let verts = make_vertices(&ico);
        let n_v = verts.len();
        let vertices: Vec<[f64; 3]> = verts
            .iter()
            .map(|v| {
                let p = v.pos.normalize();
                [p.x, p.y, p.z]
            })
            .collect();
        let neighbors: Vec<Vec<u16>> = verts.iter().map(|v| v.neighbors.clone()).collect();
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
    fn lookup_out_of_range_returns_zero() {
        let table = make_test_table(1, 5, 8);
        let dir = Vector3::new(1.0, 0.0, 0.0);
        assert_eq!(table.lookup(3.0, 0.0, &dir, &dir), 0.0); // below rmin
        assert_eq!(table.lookup(20.0, 0.0, &dir, &dir), 0.0); // above rmax
    }

    #[test]
    fn lookup_at_vertex_exact() {
        use crate::{make_icosphere, make_vertices};
        let ico = make_icosphere(1).unwrap();
        let verts = make_vertices(&ico);
        let n_v = verts.len();
        let n_r = 3;
        let n_omega = 4;
        let vertices: Vec<[f64; 3]> = verts
            .iter()
            .map(|v| {
                let p = v.pos.normalize();
                [p.x, p.y, p.z]
            })
            .collect();
        let neighbors: Vec<Vec<u16>> = verts.iter().map(|v| v.neighbors.clone()).collect();

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
        };

        // Query exactly at vertex 0 direction for both → should get 0+0=0
        let dir0 = Vector3::new(
            table.vertices[0][0],
            table.vertices[0][1],
            table.vertices[0][2],
        );
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

    /// Build a table filled with `analytic_fn` at each vertex pair.
    fn make_analytic_table(min_points: usize) -> Table6DFlat<f32> {
        use crate::{make_icosphere, make_vertices};
        let ico = make_icosphere(min_points).unwrap();
        let verts = make_vertices(&ico);
        let n_v = verts.len();
        let vertices: Vec<[f64; 3]> = verts
            .iter()
            .map(|v| {
                let p = v.pos.normalize();
                [p.x, p.y, p.z]
            })
            .collect();
        let neighbors: Vec<Vec<u16>> = verts.iter().map(|v| v.neighbors.clone()).collect();

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
                    let va = Vector3::new(vertices[vi][0], vertices[vi][1], vertices[vi][2]);
                    for vj in 0..n_v {
                        let vb = Vector3::new(vertices[vj][0], vertices[vj][1], vertices[vj][2]);
                        let idx = ri * stride + oi * n_v * n_v + vi * n_v + vj;
                        data[idx] = analytic_fn(&va, &vb) as f32;
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
}
