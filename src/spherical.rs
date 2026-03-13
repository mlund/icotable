use crate::Vector3;
use std::f64::consts::PI;

/// Spherical coordinates (r, θ, φ): radius, polar angle, azimuthal angle.
#[derive(Debug, Clone)]
pub struct SphericalCoord {
    r: f64,
    theta: f64,
    phi: f64,
}

impl SphericalCoord {
    /// Radial distance.
    pub const fn radius(&self) -> f64 {
        self.r
    }
    /// Polar angle (0..π)
    pub const fn theta(&self) -> f64 {
        self.theta
    }
    /// Azimuthal angle (0..2π)
    pub const fn phi(&self) -> f64 {
        self.phi
    }
    /// Create from `(r, θ, φ)`.
    pub const fn new(r: f64, theta: f64, phi: f64) -> Self {
        let phi = (phi + 2.0 * PI) % (2.0 * PI);
        let theta = (theta + PI) % PI;
        Self { r, theta, phi }
    }
    /// Convert from Cartesian coordinates.
    pub fn from_cartesian(cartesian: Vector3) -> Self {
        let r = cartesian.norm();
        let theta = (cartesian.z / r).acos();
        let phi = cartesian.y.atan2(cartesian.x);
        let phi = 2.0f64.mul_add(PI, phi) % (2.0 * PI);
        Self::new(r, theta, phi)
    }
    /// Convert to Cartesian coordinates.
    pub fn to_cartesian(&self) -> Vector3 {
        let (theta_sin, theta_cos) = self.theta.sin_cos();
        let (phi_sin, phi_cos) = self.phi.sin_cos();
        Vector3::new(theta_sin * phi_cos, theta_sin * phi_sin, theta_cos).scale(self.r)
    }
}

impl From<SphericalCoord> for (f64, f64, f64) {
    fn from(spherical: SphericalCoord) -> Self {
        (spherical.r, spherical.theta, spherical.phi)
    }
}

impl From<SphericalCoord> for Vector3 {
    fn from(spherical: SphericalCoord) -> Self {
        spherical.to_cartesian()
    }
}

impl From<Vector3> for SphericalCoord {
    fn from(cartesian: Vector3) -> Self {
        Self::from_cartesian(cartesian)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn round_trip_spherical_cartesian() {
        const TOL: f64 = 1e-6;
        for theta in [0.1, 0.5, 1.0, 1.5, 2.5, 3.0] {
            for phi in [0.0, 0.5, 1.0, 2.0, 4.0, 5.5] {
                let s1 = SphericalCoord::new(1.0, theta, phi);
                let cartesian = Vector3::from(s1.clone()).scale(2.0);
                let s2 = SphericalCoord::from(cartesian);
                assert_relative_eq!(s1.theta(), s2.theta(), epsilon = TOL);
                assert_relative_eq!(s1.phi(), s2.phi(), epsilon = TOL);
                assert_relative_eq!(s1.radius() * 2.0, s2.radius(), epsilon = TOL);
            }
        }
    }
}
