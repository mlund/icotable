//! Scheme-independent lookup traits for tabulated interaction tables.
//!
//! A caller queries an energy table through these traits without knowing how it
//! is stored or which icosphere subdivision it uses. This keeps the subdivision
//! and interpolation scheme an implementation detail, so a table backed by a
//! different scheme can be substituted behind the same interface.

use crate::{PointGroup, TableMetadata, UnitQuaternion, Vector3};

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

    /// Energy of a rigid pair, Boltzmann-interpolated at `beta`, from B's
    /// lab-frame offset `separation` (b − a) and both lab orientations.
    ///
    /// Below `rmin` it is +∞, so bodies can't pass through each other, and
    /// beyond `rmax` the tail correction.
    ///
    /// For [`PointGroup::Exchange`] tables E(A,B) = E(B,A), but the two lookup
    /// orders read different dihedral bins and faces, and near walls can differ
    /// by thousands of kT. Their Boltzmann factors are averaged: the result is
    /// the same whichever body comes first, as detailed balance needs, and keeps
    /// each order's orientational average of exp(−βu), which the table's PMF is.
    /// Averaging energies would underweight every pose where they disagree.
    ///
    /// # Panics
    /// If `separation` isn't finite, or `beta` isn't finite and positive.
    fn pair_energy(
        &self,
        separation: &Vector3,
        q_a: &UnitQuaternion,
        q_b: &UnitQuaternion,
        beta: f64,
    ) -> PairEnergy {
        assert!(
            beta.is_finite() && beta > 0.0,
            "beta must be finite and positive, got {beta}"
        );
        let r = separation.norm();
        // NaN would pass both range checks and read the contact bin.
        assert!(r.is_finite(), "pair separation must be finite, got {r}");
        let (rmin, rmax) = self.r_range();
        let only = |energy| PairEnergy {
            energy,
            order_mismatch: 0.0,
        };
        if r < rmin {
            return only(f64::INFINITY);
        }
        if r > rmax {
            return only(self.tail_energy(r));
        }
        let ordered = |sep: &Vector3, q_a, q_b| {
            let (r, omega, dir_a, dir_b) = crate::inverse_orient(sep, q_a, q_b);
            self.lookup(r, omega, &dir_a, &dir_b, Some(beta))
        };
        let forward = ordered(separation, q_a, q_b);
        let exchange = self
            .metadata()
            .is_some_and(|m| m.point_group == PointGroup::Exchange);
        if !exchange {
            return only(forward);
        }
        let reverse = ordered(&-separation, q_b, q_a);
        let low = forward.min(reverse);
        if low == f64::INFINITY {
            return only(low);
        }
        // Relative to the lower energy, so deep wells don't overflow.
        let (wf, wr) = (
            (-beta * (forward - low)).exp(),
            (-beta * (reverse - low)).exp(),
        );
        let gap = (wf - wr).abs();
        PairEnergy {
            energy: low - (0.5 * (wf + wr)).ln() / beta,
            // Equal deep wells would give ∞ × 0.
            order_mismatch: if gap == 0.0 {
                0.0
            } else {
                (-beta * low).exp() * gap
            },
        }
    }
}

/// Result of [`Lookup6D::pair_energy`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PairEnergy {
    /// In the table's energy unit.
    pub energy: f64,
    /// |exp(−βu₁) − exp(−βu₂)| between the two lookup orders of an exchange
    /// table: how much they disagree at this pose. Zero for one lookup. Orders
    /// that agree can still both be wrong, so this bounds no error.
    pub order_mismatch: f64,
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

    use super::PairEnergy;
    use crate::{
        orient, AdaptiveBuilder, PointGroup, TableMetadata, TailCorrectionTerm, UnitQuaternion,
        Vector3,
    };
    use approx::assert_relative_eq;
    use rand::{Rng, SeedableRng};

    const BETA: f64 = 1.0; // energies in kT
    const RMIN: f64 = 5.0;
    const RMAX: f64 = 8.0;

    /// Exchange symmetric; steep in the directions and in the dihedral, so the
    /// two lookup orders disagree by many kT on a coarse table.
    fn exact(separation: &Vector3, q_a: &UnitQuaternion, q_b: &UnitQuaternion) -> f64 {
        let g = |v: &Vector3| v.x + 2.0 * v.y + 3.0 * v.z;
        let d = separation.normalize();
        let twist = (q_a * Vector3::x()).dot(&(q_b * Vector3::x()));
        2.0 * (g(&(q_a.inverse() * d)) + g(&(q_b.inverse() * -d))) + 12.0 * twist
    }

    /// Each entry is the energy of the poses `orient` builds for it, as a scan
    /// fills a table, so both lookup orders describe one configuration.
    fn table(point_group: PointGroup) -> crate::Table6DAdaptive<f32> {
        let omega_step = std::f64::consts::TAU / 64.0;
        let mut builder = AdaptiveBuilder::new(RMIN, RMAX, 1.0, omega_step, 2, 1e-9, BETA);
        for ri in 0..builder.n_r() {
            let r = builder.r_value(ri);
            let verts: Vec<Vector3> = builder
                .vertex_directions(builder.current_level())
                .iter()
                .map(|&v| Vector3::from(v))
                .collect();
            for oi in 0..builder.n_omega() {
                let omega = builder.omega_value(oi);
                let data: Vec<f64> = verts
                    .iter()
                    .flat_map(|a| {
                        verts.iter().map(move |b| {
                            let (q_a, q_b, sep) = orient(r, omega, a, b);
                            exact(&sep, &q_a, &q_b)
                        })
                    })
                    .collect();
                builder.set_slab(ri, oi, &data);
            }
            builder.finish_r_slice(ri);
        }
        let mut table = builder.build();
        table.metadata = Some(TableMetadata {
            tail_terms: vec![TailCorrectionTerm {
                coefficient: -10.0,
                kappa: 0.0,
                power: 1,
            }],
            electric_prefactor: Some(1.0),
            point_group,
            ..Default::default()
        });
        table
    }

    fn random_pose(rng: &mut impl Rng, r: f64) -> (Vector3, UnitQuaternion, UnitQuaternion) {
        let mut quat = || {
            let (u1, u2, u3): (f64, f64, f64) = (rng.gen(), rng.gen(), rng.gen());
            let tau = std::f64::consts::TAU;
            let (a, b) = ((1.0 - u1).sqrt(), u1.sqrt());
            UnitQuaternion::from_quaternion(nalgebra::Quaternion::new(
                a * (tau * u2).sin(),
                a * (tau * u2).cos(),
                b * (tau * u3).sin(),
                b * (tau * u3).cos(),
            ))
        };
        let (q_a, q_b, q_dir) = (quat(), quat(), quat());
        (q_dir * Vector3::z() * r, q_a, q_b)
    }

    /// Random poses at fixed R must average exp(−βu) to the exact value, as the
    /// table's PMF does. Averaging the two orders' energies instead reads 0.81 of it.
    #[test]
    fn exchange_pair_energy_keeps_the_boltzmann_average() {
        let t = table(PointGroup::Exchange);
        let mut rng = rand::rngs::StdRng::seed_from_u64(7);
        let (mut table_sum, mut exact_sum) = (0.0, 0.0);
        for _ in 0..20_000 {
            let (sep, q_a, q_b) = random_pose(&mut rng, 6.0);
            table_sum += (-BETA * t.pair_energy(&sep, &q_a, &q_b, BETA).energy).exp();
            exact_sum += (-BETA * exact(&sep, &q_a, &q_b)).exp();
        }
        assert_relative_eq!(table_sum, exact_sum, max_relative = 0.05);
    }

    /// Detailed balance needs one energy per pair, whichever body comes first.
    #[test]
    fn exchange_pair_energy_ignores_which_body_comes_first() {
        let t = table(PointGroup::Exchange);
        let mut rng = rand::rngs::StdRng::seed_from_u64(3);
        for _ in 0..200 {
            let (sep, q_a, q_b) = random_pose(&mut rng, 6.3);
            let ab = t.pair_energy(&sep, &q_a, &q_b, BETA);
            let ba = t.pair_energy(&-sep, &q_b, &q_a, BETA);
            assert_relative_eq!(ab.energy, ba.energy, max_relative = 1e-12);
            assert_relative_eq!(ab.order_mismatch, ba.order_mismatch, max_relative = 1e-12);
        }
    }

    /// Different bodies have no exchange partner: one lookup, as tabulated.
    #[test]
    fn asymmetric_pair_energy_is_the_single_lookup() {
        let t = table(PointGroup::Asymmetric);
        let mut rng = rand::rngs::StdRng::seed_from_u64(5);
        let (sep, q_a, q_b) = random_pose(&mut rng, 6.0);
        let (r, omega, dir_a, dir_b) = crate::inverse_orient(&sep, &q_a, &q_b);
        let single = Lookup6D::lookup(&t, r, omega, &dir_a, &dir_b, Some(BETA));
        let pair = t.pair_energy(&sep, &q_a, &q_b, BETA);
        assert_eq!(
            pair,
            PairEnergy {
                energy: single,
                order_mismatch: 0.0
            }
        );
    }

    /// The table's own lookup returns 0 outside its range, which would let
    /// bodies pass through each other.
    #[test]
    fn pair_energy_is_a_wall_below_and_the_tail_beyond_the_range() {
        let t = table(PointGroup::Exchange);
        let q = UnitQuaternion::identity();
        let below = t.pair_energy(&Vector3::new(0.0, 0.0, 4.0), &q, &q, BETA);
        assert_eq!(below.energy, f64::INFINITY);
        let beyond = t.pair_energy(&Vector3::new(0.0, 0.0, 20.0), &q, &q, BETA);
        assert_relative_eq!(beyond.energy, -10.0 / 20.0);
        assert_eq!(beyond.order_mismatch, 0.0);
    }

    /// A NaN distance would pass both range checks and read the contact bin.
    #[test]
    #[should_panic(expected = "finite")]
    fn non_finite_separation_is_a_bug_not_an_energy() {
        let t = table(PointGroup::Exchange);
        let q = UnitQuaternion::identity();
        t.pair_energy(&Vector3::new(f64::NAN, 0.0, 6.0), &q, &q, BETA);
    }

    /// Exchange table whose energy depends only on which way the separation
    /// points in the first body's frame: `ahead` along +z, `behind` otherwise.
    /// With both bodies unrotated and B on +z, the two lookup orders read one each.
    struct ByOrder {
        ahead: f64,
        behind: f64,
        metadata: TableMetadata,
    }

    impl ByOrder {
        fn new(ahead: f64, behind: f64) -> Self {
            let metadata = TableMetadata {
                point_group: PointGroup::Exchange,
                ..Default::default()
            };
            Self {
                ahead,
                behind,
                metadata,
            }
        }

        fn at_contact(&self, beta: f64) -> PairEnergy {
            let q = UnitQuaternion::identity();
            self.pair_energy(&Vector3::new(0.0, 0.0, 6.0), &q, &q, beta)
        }
    }

    impl super::TabulatedInteraction for ByOrder {
        fn r_range(&self) -> (f64, f64) {
            (RMIN, RMAX)
        }
        fn metadata(&self) -> Option<&TableMetadata> {
            Some(&self.metadata)
        }
    }

    impl Lookup6D for ByOrder {
        fn lookup(&self, _: f64, _: f64, dir_a: &Vector3, _: &Vector3, _: Option<f64>) -> f64 {
            if dir_a.z > 0.0 {
                self.ahead
            } else {
                self.behind
            }
        }
    }

    /// Half the orders see a wall, so half the Boltzmann weight survives:
    /// u = ln 2 / β, in either order. An energy mean would give +∞.
    #[test]
    fn a_wall_in_one_order_halves_the_boltzmann_factor() {
        let beta = 2.0;
        for t in [
            ByOrder::new(0.0, f64::INFINITY),
            ByOrder::new(f64::INFINITY, 0.0),
        ] {
            let pair = t.at_contact(beta);
            assert_relative_eq!(pair.energy, std::f64::consts::LN_2 / beta);
            assert_relative_eq!(pair.order_mismatch, 1.0);
        }
    }

    #[test]
    fn walls_in_both_orders_stay_a_wall() {
        let pair = ByOrder::new(f64::INFINITY, f64::INFINITY).at_contact(BETA);
        assert_eq!(pair.energy, f64::INFINITY);
        assert_eq!(pair.order_mismatch, 0.0);
    }

    /// exp(−βu) overflows here; the shifted sum must not.
    #[test]
    fn equal_deep_wells_agree_without_overflow() {
        let pair = ByOrder::new(-1000.0, -1000.0).at_contact(BETA);
        assert_relative_eq!(pair.energy, -1000.0);
        assert_eq!(pair.order_mismatch, 0.0);
    }

    #[test]
    #[should_panic(expected = "beta")]
    fn non_positive_beta_is_a_bug() {
        ByOrder::new(0.0, 0.0).at_contact(0.0);
    }

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
