use crate::Vector3;
use get_size::GetSize;

/// Vertex on an icosphere with position and neighbor connectivity.
#[derive(Clone, GetSize, Debug)]
pub struct Vertex {
    /// 3D coordinates on a unit sphere
    #[get_size(size = 24)]
    pub pos: Vector3,
    /// Indices of neighboring vertices
    pub neighbors: Vec<u16>,
}
