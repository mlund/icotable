use std::f64::consts::PI;

use crate::{IcoSphere, Vector3};
use anyhow::{Context, Result};
use glam::f32::Vec3A;
use hexasphere::AdjacencyBuilder;

const UNIT_SPHERE_AREA: f64 = 4.0 * PI;

/// Create icosphere with at least `min_points` surface vertices.
///
/// Vertex count: N = 10 × (n_divisions + 1)² + 2.
pub fn make_icosphere(min_points: usize) -> Result<IcoSphere> {
    let points_per_division = |n_div: usize| 10 * (n_div + 1) * (n_div + 1) + 2;
    let n_points = (0..200).map(points_per_division);

    let n_divisions = n_points
        .enumerate()
        .find(|(_, n)| *n >= min_points)
        .map(|(n_div, _)| n_div)
        .context("too many vertices")?;

    log::debug!(
        "Creating icosphere with {} divisions, {} vertices",
        n_divisions,
        points_per_division(n_divisions)
    );

    Ok(IcoSphere::new(n_divisions, |_| ()))
}

/// Get icosphere vertices as 3D vectors.
pub fn make_icosphere_vertices(min_points: usize) -> Result<Vec<Vector3>> {
    let icosphere = make_icosphere(min_points)?;
    Ok(extract_vertices(&icosphere))
}

/// Get icosphere vertices as nalgebra vectors.
pub fn extract_vertices(icosphere: &IcoSphere) -> Vec<Vector3> {
    icosphere
        .raw_points()
        .iter()
        .map(|p| Vector3::new(p.x as f64, p.y as f64, p.z as f64))
        .collect()
}

/// Per-vertex area weights normalized to fluctuate around 1.
pub fn make_weights(icosphere: &IcoSphere) -> Vec<f64> {
    let indices = icosphere.get_all_indices();
    let vertices = icosphere.raw_points();
    let mut weights = Vec::with_capacity(vertices.len());
    let mut adjacency = AdjacencyBuilder::new(vertices.len());
    adjacency.add_indices(&indices);

    for (i, neighbors) in adjacency.finish().iter().enumerate() {
        let mut area = 0.0;
        area += spherical_face_area(
            &vertices[i],
            &vertices[*neighbors.first().unwrap()],
            &vertices[*neighbors.last().unwrap()],
        );
        for j in 0..neighbors.len() - 1 {
            area += spherical_face_area(
                &vertices[i],
                &vertices[neighbors[j]],
                &vertices[neighbors[j + 1]],
            );
        }
        weights.push(area / 3.0);
    }
    debug_assert_eq!(weights.len(), vertices.len());

    let total_area = weights.iter().sum::<f64>();
    debug_assert!((total_area - UNIT_SPHERE_AREA).abs() < 1e-4);

    let ideal_vertex_area = UNIT_SPHERE_AREA / vertices.len() as f64;
    weights.iter_mut().for_each(|w| *w /= ideal_vertex_area);
    weights
}

/// Spherical face area via spherical excess.
///
/// See <https://en.wikipedia.org/wiki/Spherical_trigonometry>
#[allow(non_snake_case)]
fn spherical_face_area(a: &Vec3A, b: &Vec3A, c: &Vec3A) -> f64 {
    debug_assert!(a.is_normalized());
    debug_assert!(b.is_normalized());
    debug_assert!(c.is_normalized());

    let angle = |u: &Vec3A, v: &Vec3A, w: &Vec3A| {
        let vu = u - v * v.dot(*u);
        let vw = w - v * v.dot(*w);
        vu.angle_between(vw)
    };
    let A = angle(b, a, c);
    let B = angle(c, b, a);
    let C = angle(a, c, b);
    (A + B + C) as f64 - PI
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_icosphere_vertex_counts() {
        assert_eq!(make_icosphere_vertices(1).unwrap().len(), 12);
        assert_eq!(make_icosphere_vertices(13).unwrap().len(), 42);
        assert_eq!(make_icosphere_vertices(43).unwrap().len(), 92);
    }

    #[test]
    fn test_weights_sum_to_one() {
        let icosphere = make_icosphere(1).unwrap();
        let weights = make_weights(&icosphere);
        let min = weights.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = weights.iter().cloned().fold(0.0, f64::max);
        assert_relative_eq!(min, 1.0, epsilon = 1e-6);
        assert_relative_eq!(max, 1.0, epsilon = 1e-6);
    }

    #[test]
    fn test_spherical_face_area_octant() {
        let [a, b, c] = [
            Vec3A::new(1.0, 0.0, 0.0),
            Vec3A::new(0.0, 1.0, 0.0),
            Vec3A::new(0.0, 0.0, 1.0),
        ];
        let area = spherical_face_area(&a, &b, &c);
        assert_relative_eq!(area, UNIT_SPHERE_AREA / 8.0, epsilon = 1e-6);
    }
}
