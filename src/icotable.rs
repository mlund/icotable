use crate::icosphere::make_weights;
use crate::{
    make_icosphere, make_vertices, table::PaddedTable, IcoSphere, SphericalCoord, Vector3, Vertex,
};
use anyhow::{ensure, Result};
use core::f64::consts::PI;
use get_size::GetSize;
use itertools::Itertools;
use nalgebra::Matrix3;
use std::fmt::Display;
use std::io::Write;
use std::sync::{Arc, OnceLock};

/// A 4D icotable where each vertex holds an icotable of floats.
pub type IcoTable4D = IcoTable2D<IcoTable2D<f64>>;

/// 6D table: R → ω → (θφ) → (θφ) → f64.
pub type Table6D = PaddedTable<PaddedTable<IcoTable4D>>;

/// Three vertex indices defining a face.
pub type Face = [usize; 3];

/// Stores data on icosphere vertices with barycentric interpolation.
///
/// Vertex positions and neighbors are shared via `Arc` to avoid duplication
/// across nested tables.
#[derive(Clone, GetSize)]
pub struct IcoTable2D<T: Clone + GetSize> {
    #[get_size(size = 8)]
    vertices: Arc<Vec<Vertex>>,
    #[get_size(size_fn = oncelock_size_helper)]
    data: Vec<OnceLock<T>>,
}

fn oncelock_size_helper<T: GetSize>(value: &Vec<OnceLock<T>>) -> usize {
    value.get_size() + std::mem::size_of::<T>() * value.len()
}

impl<T: Clone + GetSize> IcoTable2D<T> {
    /// Iterate over all vertices.
    pub(crate) fn iter_vertices(&self) -> impl Iterator<Item = &Vertex> {
        self.vertices.iter()
    }
    /// Normalized position of vertex at `index`.
    pub fn get_normalized_pos(&self, index: usize) -> Vector3 {
        self.vertices[index].pos.normalize()
    }
    /// Data stored at vertex `index`, if set.
    pub fn get_data(&self, index: usize) -> Option<&T> {
        self.data[index].get()
    }
    /// Neighbor indices for vertex at `index`.
    pub fn get_neighbors(&self, index: usize) -> &[u16] {
        &self.vertices[index].neighbors
    }
    fn get_with_lock(&self, index: usize) -> (&Vector3, &[u16], &OnceLock<T>) {
        (
            &self.vertices[index].pos,
            &self.vertices[index].neighbors,
            &self.data[index],
        )
    }
    /// Iterate over vertex positions.
    pub fn iter_positions(&self) -> impl Iterator<Item = &Vector3> {
        self.iter_vertices().map(|v| &v.pos)
    }
    /// Iterate over `(position, data)` pairs.
    pub fn iter(&self) -> impl Iterator<Item = (&Vector3, Option<&T>)> {
        self.vertices
            .iter()
            .zip(self.data.iter())
            .map(|(v, d)| (&v.pos, d.get()))
    }

    fn iter_with_lock(&self) -> impl Iterator<Item = (&Vector3, &[u16], &OnceLock<T>)> {
        (0..self.data.len()).map(move |i| self.get_with_lock(i))
    }

    /// Create from shared vertices, optionally pre-filling all cells with `data`.
    pub(crate) fn from_vertices(vertices: Arc<Vec<Vertex>>, data: Option<T>) -> Self {
        let num_vertices = vertices.len();
        let data = data.map(OnceLock::from);
        Self {
            vertices,
            data: vec![data.unwrap_or_default(); num_vertices],
        }
    }

    fn from_icosphere_without_data(icosphere: &IcoSphere) -> Self {
        Self {
            vertices: Arc::new(make_vertices(icosphere)),
            data: vec![OnceLock::default(); icosphere.raw_points().len()],
        }
    }

    /// Create from an icosphere, filling all vertices with `default_data`.
    #[cfg(test)]
    pub(crate) fn from_icosphere(icosphere: &IcoSphere, default_data: T) -> Self {
        let table = Self::from_icosphere_without_data(icosphere);
        table.set_vertex_data(|_, _| default_data.clone()).unwrap();
        table
    }

    /// Average angular spacing between vertices.
    pub fn angle_resolution(&self) -> f64 {
        (4.0 * PI / self.data.len() as f64).sqrt()
    }

    /// Number of vertices.
    pub fn len(&self) -> usize {
        self.vertices.len()
    }

    /// Returns `true` if there are no vertices.
    pub fn is_empty(&self) -> bool {
        self.vertices.is_empty()
    }

    /// Set data via generator; can only be called once per vertex due to `OnceLock`.
    pub fn set_vertex_data(&self, f: impl Fn(usize, &Vector3) -> T) -> Result<()> {
        ensure!(
            self.data.iter().any(|v| v.get().is_none()),
            "Data already set for some vertices"
        );
        self.iter_with_lock()
            .enumerate()
            .try_for_each(|(i, (pos, _, data))| {
                let value = f(i, pos);
                ensure!(data.set(value).is_ok(), "Data already set for vertex {}", i);
                Ok(())
            })
    }

    /// Reset all vertex data to unset.
    pub fn clear_vertex_data(&mut self) {
        for data in self.data.iter_mut() {
            *data = OnceLock::default();
        }
    }

    /// Iterate over set vertex data values (panics if any vertex is unset).
    pub fn vertex_data(&self) -> impl Iterator<Item = &T> {
        self.data.iter().map(|v| v.get().unwrap())
    }

    /// Apply a transform to all vertex positions.
    pub fn transform_vertex_positions(&mut self, f: impl Fn(&Vector3) -> Vector3) {
        let new_vertices = self
            .iter_vertices()
            .map(|v| Vertex {
                pos: f(&v.pos),
                neighbors: v.neighbors.clone(),
            })
            .collect_vec();
        self.vertices = Arc::new(new_vertices);
    }

    /// Projected barycentric coordinates (Ericson, "Real-Time Collision Detection", p141-142).
    pub fn barycentric(&self, p: &Vector3, face: &Face) -> Vector3 {
        let (a, b, c) = self.face_positions(face);
        let b = crate::flat::projected_barycentric(p, &a, &b, &c);
        Vector3::new(b[0], b[1], b[2])
    }

    /// Naive barycentric (assumes point is on the face plane).
    #[allow(clippy::suspicious_operation_groupings)]
    pub fn naive_barycentric(&self, p: &Vector3, face: &Face) -> Vector3 {
        let (a, b, c) = self.face_positions(face);
        let ab = b - a;
        let ac = c - a;
        let ap = p - a;
        let d00 = ab.dot(&ab);
        let d01 = ab.dot(&ac);
        let d11 = ac.dot(&ac);
        let d20 = ap.dot(&ab);
        let d21 = ap.dot(&ac);
        let denom = d00.mul_add(d11, -(d01 * d01));
        let v = d11.mul_add(d20, -(d01 * d21)) / denom;
        let w = d00.mul_add(d21, -(d01 * d20)) / denom;
        Vector3::new(1.0 - v - w, v, w)
    }

    /// Normalized positions of the three vertices of a face.
    pub fn face_positions(&self, face: &Face) -> (Vector3, Vector3, Vector3) {
        (
            self.get_normalized_pos(face[0]),
            self.get_normalized_pos(face[1]),
            self.get_normalized_pos(face[2]),
        )
    }

    /// O(n) brute-force nearest vertex.
    pub fn nearest_vertex(&self, point: &Vector3) -> usize {
        self.iter_vertices()
            .map(|v| (v.pos.normalize() - point).norm_squared())
            .enumerate()
            .min_by(|a, b| a.1.total_cmp(&b.1))
            .expect("vertices must not be empty")
            .0
    }

    /// Find the face (triangle) enclosing a given direction.
    pub fn nearest_face(&self, point: &Vector3) -> Face {
        let point = point.normalize();
        let nearest = self.nearest_vertex(&point);
        let (a, b) = self
            .get_neighbors(nearest)
            .iter()
            .copied()
            .map(|i| {
                (
                    i as usize,
                    (self.get_normalized_pos(i as usize) - point).norm_squared(),
                )
            })
            .sorted_by(|a, b| a.1.total_cmp(&b.1))
            .map(|(i, _)| i)
            .take(2)
            .collect_tuple()
            .expect("Face requires at least two neighbors");

        let mut face = [a, b, nearest];
        face.sort_unstable();
        debug_assert_eq!(face.iter().unique().count(), 3);
        face
    }
}

impl Display for IcoTable2D<f64> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "# x y z θ φ data")?;
        for (pos, data) in self.iter() {
            let spherical = SphericalCoord::from_cartesian(*pos);
            writeln!(
                f,
                "{} {} {} {} {} {:?}",
                pos.x,
                pos.y,
                pos.z,
                spherical.theta(),
                spherical.phi(),
                data
            )?;
        }
        Ok(())
    }
}

impl IcoTable4D {
    /// Flat iterator over all (vertex_a_pos, vertex_b_pos, data) triples.
    pub fn flat_iter(&self) -> impl Iterator<Item = (&Vector3, &Vector3, &OnceLock<f64>)> {
        self.iter_with_lock().flat_map(|(pos_a, _, data_a)| {
            data_a
                .get()
                .unwrap()
                .iter_with_lock()
                .map(move |(pos_b, _, data_b)| (pos_a, pos_b, data_b))
        })
    }

    /// Create a 4D table with at least `min_points` vertices, filled with `default_data`.
    #[cfg(test)]
    pub(crate) fn from_min_points(min_points: usize, default_data: IcoTable2D<f64>) -> Result<Self> {
        let icosphere = make_icosphere(min_points)?;
        let vertices = Arc::new(make_vertices(&icosphere));
        Ok(Self::from_vertices(vertices, Some(default_data)))
    }

    /// 4D barycentric interpolation via bilinear matrix form.
    pub fn interpolate(
        &self,
        face_a: &Face,
        face_b: &Face,
        bary_a: &Vector3,
        bary_b: &Vector3,
    ) -> f64 {
        let data_ab = Matrix3::from_fn(|i, j| {
            *self
                .get_data(face_a[i])
                .unwrap()
                .get_data(face_b[j])
                .unwrap()
        });
        (bary_a.transpose() * data_ab * bary_b).to_scalar()
    }
}

impl Table6D {
    /// Create a 6D table spanning `[r_min, r_max]` with given radial and angular resolution.
    pub fn from_resolution(r_min: f64, r_max: f64, dr: f64, angle_resolution: f64) -> Result<Self> {
        let n_points = (4.0 * PI / angle_resolution.powi(2)).round() as usize;
        let icosphere = make_icosphere(n_points)?;
        let weights = make_weights(&icosphere);

        // Encode weights into vertex magnitudes for integration
        let mut vertices = make_vertices(&icosphere);
        vertices
            .iter_mut()
            .zip(weights)
            .for_each(|(v, w)| v.pos *= w);

        let vertices = Arc::new(vertices);
        let table_b = IcoTable2D::from_vertices(vertices.clone(), None);
        let angle_resolution = table_b.angle_resolution();
        log::info!("Actual angle resolution = {angle_resolution:.2} radians");

        let table_a = IcoTable4D::from_vertices(vertices, Some(table_b));
        let table_omega = PaddedTable::<IcoTable4D>::new(0.0, 2.0 * PI, angle_resolution, table_a);
        Ok(Self::new(r_min, r_max, dr, table_omega))
    }

    /// Get the 4D angular table for a given radial distance and dihedral angle.
    pub fn get_icospheres(&self, r: f64, omega: f64) -> Result<&IcoTable4D> {
        self.get(r)?.get(omega)
    }

    /// Write 5D angular space data to a stream for a single radial distance.
    pub fn stream_angular_space(&self, r: f64, stream: &mut impl Write) -> Result<()> {
        writeln!(stream, "# r ω θ1 φ1 θ2 φ2 data")?;
        for (omega, angles) in self.get(r)?.iter() {
            for (vertex1, vertex2, data) in angles.flat_iter() {
                let Some(data) = data.get() else {
                    continue;
                };
                let (s1, s2) = (
                    SphericalCoord::from_cartesian(vertex1.normalize()),
                    SphericalCoord::from_cartesian(vertex2.normalize()),
                );
                writeln!(
                    stream,
                    "{:.2} {:.3} {:.3} {:.3} {:.3} {:.3} {:.4e}",
                    r,
                    omega,
                    s1.theta(),
                    s1.phi(),
                    s2.theta(),
                    s2.phi(),
                    data
                )?;
            }
        }
        Ok(())
    }
}

impl IcoTable2D<f64> {
    /// Barycentric interpolation on the icosphere surface.
    pub fn interpolate(&self, point: &Vector3) -> f64 {
        let face = self.nearest_face(point);
        let bary = self.barycentric(point, &face);
        bary[0] * self.data[face[0]].get().unwrap()
            + bary[1] * self.data[face[1]].get().unwrap()
            + bary[2] * self.data[face[2]].get().unwrap()
    }

    /// Create an empty 2D table with at least `min_points` vertices.
    pub fn from_min_points(min_points: usize) -> Result<Self> {
        let icosphere = make_icosphere(min_points)?;
        Ok(Self::from_icosphere_without_data(&icosphere))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_icosphere_table() {
        let icosphere = make_icosphere(3).unwrap();
        let icotable = IcoTable2D::from_icosphere(&icosphere, 0.0);
        assert_eq!(icotable.data.len(), 12);

        let point = icotable.get_normalized_pos(0);
        let face = icotable.nearest_face(&point);
        let bary = icotable.barycentric(&point, &face);
        assert_eq!(face, [0, 2, 5]);
        assert_relative_eq!(bary[0], 1.0);
        assert_relative_eq!(bary[1], 0.0);
        assert_relative_eq!(bary[2], 0.0);
    }

    #[test]
    fn test_icosphere_interpolate() {
        let icosphere = make_icosphere(3).unwrap();
        let icotable = IcoTable2D::from_icosphere_without_data(&icosphere);
        icotable.set_vertex_data(|i, _| i as f64 + 1.0).unwrap();

        let point = Vector3::new(0.5, 0.5, 0.5).normalize();
        let data = icotable.interpolate(&point);
        assert_relative_eq!(data, 2.59977558757542, epsilon = 1e-6);
    }

    #[test]
    fn test_4d_interpolation() {
        let n_points = 12;
        let icosphere = make_icosphere(n_points).unwrap();
        let icotable = IcoTable2D::from_icosphere_without_data(&icosphere);
        icotable.set_vertex_data(|i, _| i as f64).unwrap();
        let icotable_of_spheres = IcoTable4D::from_min_points(n_points, icotable).unwrap();

        let face = [0, 1, 2];
        let bary_corner = Vector3::new(0.0, 1.0, 0.0);
        let data = icotable_of_spheres.interpolate(&face, &face, &bary_corner, &bary_corner);
        assert_relative_eq!(data, 1.0);

        let bary_center = Vector3::new(1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0);
        let data = icotable_of_spheres.interpolate(&face, &face, &bary_center, &bary_center);
        assert_relative_eq!(data, 1.0); // (0+1+2)/3 = 1
    }
}
