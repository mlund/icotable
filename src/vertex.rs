use crate::{IcoSphere, Vector3};
use get_size::GetSize;
use hexasphere::AdjacencyBuilder;
use itertools::Itertools;

/// Vertex on an icosphere with position and neighbor connectivity.
#[derive(Clone, GetSize, Debug)]
pub struct Vertex {
    /// 3D coordinates on a unit sphere
    #[get_size(size = 24)]
    pub pos: Vector3,
    /// Indices of neighboring vertices
    pub neighbors: Vec<u16>,
}

/// Extract vertices and neighbour lists from an icosphere.
pub fn make_vertices(icosphere: &IcoSphere) -> Vec<Vertex> {
    let vertex_positions = icosphere
        .raw_points()
        .iter()
        .map(|p| Vector3::new(p.x as f64, p.y as f64, p.z as f64));

    let indices = icosphere.get_all_indices();
    let mut builder = AdjacencyBuilder::new(icosphere.raw_points().len());
    builder.add_indices(indices.as_slice());
    let neighbors = builder.finish().iter().map(|i| i.to_vec()).collect_vec();

    assert_eq!(vertex_positions.len(), neighbors.len());

    vertex_positions
        .zip(neighbors)
        .map(|(pos, neighbors)| Vertex {
            pos,
            neighbors: neighbors
                .into_iter()
                .map(|i| u16::try_from(i).unwrap())
                .collect(),
        })
        .collect()
}
