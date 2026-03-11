//! Forward and inverse coordinate transforms between 6D table coordinates
//! and quaternion + separation representations used in simulations.

use crate::{UnitQuaternion, Vector3};
use nalgebra::{Quaternion, UnitVector3};

const Z: Vector3 = Vector3::new(0.0, 0.0, 1.0);
const NEG_Z: Vector3 = Vector3::new(0.0, 0.0, -1.0);

/// `rotation_between` that handles anti-parallel vectors (where nalgebra returns `None`).
fn robust_rotation_between(from: &Vector3, to: &Vector3) -> UnitQuaternion {
    UnitQuaternion::rotation_between(from, to).unwrap_or_else(|| {
        // Anti-parallel: 180° rotation around any axis perpendicular to `from`
        let perp = if from.x.abs() < 0.9 {
            Vector3::x()
        } else {
            Vector3::y()
        };
        let axis = UnitVector3::new_normalize(from.cross(&perp));
        UnitQuaternion::from_axis_angle(&axis, std::f64::consts::PI)
    })
}

/// Forward: 6D point → (quaternion_a, quaternion_b, separation vector).
///
/// Molecule A sits at origin with identity orientation.
/// Molecule B is placed at distance `r` along the direction `vertex_i`,
/// rotated so that its local axis aligns with `vertex_j` and the
/// dihedral angle between them is `omega`.
pub fn orient(
    r: f64,
    omega: f64,
    vertex_i: &Vector3,
    vertex_j: &Vector3,
) -> (UnitQuaternion, UnitQuaternion, Vector3) {
    let vertex_i = vertex_i.normalize();
    let vertex_j = vertex_j.normalize();
    let r_vec = Vector3::new(0.0, 0.0, r);
    let z_unit = UnitVector3::new_unchecked(Z);

    let q1 = robust_rotation_between(&vertex_j, &NEG_Z);
    let q2 = UnitQuaternion::from_axis_angle(&z_unit, omega);
    let q3 = robust_rotation_between(&Z, &vertex_i);

    let q_b = q3 * q1 * q2;
    let separation = q3.transform_vector(&r_vec);

    (UnitQuaternion::identity(), q_b, separation)
}

/// Inverse: separation vector + quaternions → (R, ω, dir_a, dir_b).
///
/// Given the mass-center separation and orientation quaternions of two
/// rigid molecules, recover the 6D table coordinates for lookup.
pub fn inverse_orient(
    separation: &Vector3,
    q_a: &UnitQuaternion,
    q_b: &UnitQuaternion,
) -> (f64, f64, Vector3, Vector3) {
    let r = separation.norm();
    if r < 1e-15 {
        return (0.0, 0.0, Z, Z);
    }

    let q_a_inv = q_a.inverse();

    // dir_a: separation direction in molecule A's body frame
    let dir_a = q_a_inv.transform_vector(separation).normalize();

    // Reconstruct q3 = rotation_between(z, dir_a)
    let q3 = robust_rotation_between(&Z, &dir_a);

    // q_rel = q3⁻¹ * q_a⁻¹ * q_b = q1 * q2
    let q_rel = q3.inverse() * q_a_inv * q_b;

    // Swing-twist decomposition around z-axis extracts the dihedral angle ω
    // as the twist component, and the face direction dir_b as the swing.
    let q = q_rel.quaternion();
    let twist_quat = Quaternion::new(q.w, 0.0, 0.0, q.k);
    let twist_norm = twist_quat.norm();

    let (twist, omega) = if twist_norm > 1e-15 {
        let t = UnitQuaternion::new_normalize(twist_quat);
        let omega = 2.0 * q.k.atan2(q.w);
        (t, omega)
    } else {
        (UnitQuaternion::identity(), 0.0)
    };
    let omega = omega.rem_euclid(std::f64::consts::TAU);

    // q1 = q_rel * twist⁻¹; vertex_j = q1⁻¹(-z)
    let q1 = q_rel * twist.inverse();
    let dir_b = q1.inverse().transform_vector(&NEG_Z).normalize();

    (r, omega, dir_a, dir_b)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use rand::Rng;

    #[test]
    fn round_trip_orient_inverse() {
        let mut rng = rand::thread_rng();
        for _ in 0..1000 {
            let r = rng.gen_range(3.0..20.0);
            let omega = rng.gen_range(0.0..std::f64::consts::TAU);
            let vertex_i = Vector3::new(
                rng.gen_range(-1.0..1.0),
                rng.gen_range(-1.0..1.0),
                rng.gen_range(-1.0..1.0),
            )
            .normalize();
            let vertex_j = Vector3::new(
                rng.gen_range(-1.0..1.0),
                rng.gen_range(-1.0..1.0),
                rng.gen_range(-1.0..1.0),
            )
            .normalize();

            let (q_a, q_b, sep) = orient(r, omega, &vertex_i, &vertex_j);
            let (r2, omega2, dir_a2, dir_b2) = inverse_orient(&sep, &q_a, &q_b);

            assert_relative_eq!(r, r2, epsilon = 1e-10);
            assert_relative_eq!(dir_a2.dot(&vertex_i), 1.0, epsilon = 1e-10);
            assert_relative_eq!(dir_b2.dot(&vertex_j), 1.0, epsilon = 1e-10);
            assert_relative_eq!(omega, omega2, epsilon = 1e-10);

            // Verify the reconstructed pose matches the original
            let (_, q_b_recon, sep2) = orient(r2, omega2, &dir_a2, &dir_b2);
            assert_relative_eq!(sep.normalize().dot(&sep2.normalize()), 1.0, epsilon = 1e-10);
            let test_vec = Vector3::new(1.0, 2.0, 3.0);
            let orig = q_b.transform_vector(&test_vec);
            let recon = q_b_recon.transform_vector(&test_vec);
            assert_relative_eq!(orig, recon, epsilon = 1e-10);
        }
    }

    #[test]
    fn anti_parallel_cases() {
        // vertex_i ≈ -z (triggers anti-parallel in q3)
        let (q_a, q_b, sep) = orient(10.0, 1.0, &NEG_Z, &Vector3::x());
        let (r, omega, dir_a, dir_b) = inverse_orient(&sep, &q_a, &q_b);
        assert_relative_eq!(r, 10.0, epsilon = 1e-10);
        assert_relative_eq!(omega, 1.0, epsilon = 1e-10);
        assert_relative_eq!(dir_a.dot(&(-Z).normalize()), 1.0, epsilon = 1e-10);
        assert_relative_eq!(dir_b.dot(&Vector3::x()), 1.0, epsilon = 1e-10);

        // vertex_j ≈ -z (triggers anti-parallel in q1: rotation_between(-z, -z))
        let (q_a, q_b, sep) = orient(10.0, 1.0, &Vector3::x(), &NEG_Z);
        let (r, omega, _, _) = inverse_orient(&sep, &q_a, &q_b);
        assert_relative_eq!(r, 10.0, epsilon = 1e-10);
        assert_relative_eq!(omega, 1.0, epsilon = 1e-10);
    }
}
