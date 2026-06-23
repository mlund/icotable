//! Scheme-independent lookup traits for tabulated interaction tables.
//!
//! A caller queries an energy table through these traits without knowing how it
//! is stored or which icosphere subdivision it uses. This keeps the subdivision
//! and interpolation scheme an implementation detail, so a table backed by a
//! different scheme can be substituted behind the same interface.

use crate::{TableMetadata, Vector3};

/// Properties shared by every tabulated interaction table, independent of
/// dimensionality and subdivision scheme.
pub trait TabulatedInteraction {
    /// The inclusive radial range `[rmin, rmax]` the table covers.
    fn r_range(&self) -> (f64, f64);

    /// Metadata (charges, dipole moments, tail corrections, …) if the table
    /// carries any. Defaults to `None` for tables that store none.
    fn metadata(&self) -> Option<&TableMetadata> {
        None
    }

    /// Long-range tail-correction energy at separation `r`, derived from the
    /// table's metadata (zero when there is none).
    fn tail_energy(&self, r: f64) -> f64 {
        self.metadata().map_or(0.0, |meta| meta.tail_energy(r))
    }
}

/// Lookup over a 6D molecule–molecule table: separation, dihedral, and the two
/// molecules' body-frame orientations.
pub trait Lookup6D: TabulatedInteraction {
    /// Interpolated energy at separation `r`, dihedral angle `omega`, and
    /// body-frame directions `dir_a`, `dir_b`.
    ///
    /// `beta = Some(β)` applies Boltzmann weighting across the interpolation
    /// stencil; `None` is plain linear interpolation. The two are the same
    /// operation: Boltzmann weighting reduces to linear interpolation as β→0.
    fn lookup(
        &self,
        r: f64,
        omega: f64,
        dir_a: &Vector3,
        dir_b: &Vector3,
        beta: Option<f64>,
    ) -> f64;
}

/// Lookup over a 3D molecule–atom table: separation and one body-frame direction.
pub trait Lookup3D: TabulatedInteraction {
    /// Interpolated energy at separation `r` and body-frame direction `dir`.
    /// `beta` behaves as in [`Lookup6D::lookup`].
    fn lookup(&self, r: f64, dir: &Vector3, beta: Option<f64>) -> f64;
}

#[cfg(test)]
mod tests {
    use super::{Lookup3D, Lookup6D};
    use crate::{Table3DAdaptive, Table3DFlat, Table6DAdaptive, Table6DFlat};

    /// Compile-time proof that every concrete table implements the lookup traits,
    /// so callers can be generic or `dyn` over them regardless of scheme.
    #[allow(dead_code)]
    fn all_tables_implement_the_traits() {
        fn is_6d<T: Lookup6D>() {}
        fn is_3d<T: Lookup3D>() {}
        is_6d::<Table6DFlat<f32>>();
        is_6d::<Table6DAdaptive<f32>>();
        is_3d::<Table3DFlat<f32>>();
        is_3d::<Table3DAdaptive<f32>>();
    }
}
