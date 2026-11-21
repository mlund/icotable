//! Scheme-independent lookup traits for tabulated interaction tables.
//!
//! A caller queries an energy table through these traits without knowing how it
//! is stored or which icosphere subdivision it uses. This keeps the subdivision
//! and interpolation scheme an implementation detail, so a table backed by a
//! different scheme can be substituted behind the same interface.

use crate::boltzmann::combine;
pub use crate::boltzmann::Interpolated;
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
    /// Interpolated free and mean energy at separation `r`, dihedral angle
    /// `omega`, and body-frame directions `dir_a`, `dir_b`.
    ///
    /// `beta = Some(β)` applies Boltzmann weighting across the interpolation
    /// stencil; `None` is plain linear interpolation, where the two energies are
    /// equal. Boltzmann weighting reduces to linear interpolation as β→0.
    fn interpolate(
        &self,
        r: f64,
        omega: f64,
        dir_a: &Vector3,
        dir_b: &Vector3,
        beta: Option<f64>,
    ) -> Interpolated;

    /// The free energy of [`Self::interpolate`].
    fn lookup(
        &self,
        r: f64,
        omega: f64,
        dir_a: &Vector3,
        dir_b: &Vector3,
        beta: Option<f64>,
    ) -> f64 {
        self.interpolate(r, omega, dir_a, dir_b, beta).energy
    }

    /// Energy of a rigid pair, Boltzmann-interpolated at `beta`, from B's
    /// lab-frame offset `separation` (b − a) and both lab orientations.
    ///
    /// Below `rmin` it is +∞, so bodies can't pass through each other, and
    /// beyond `rmax` the tail correction.
    ///
    /// For [`PointGroup::Exchange`] tables E(A,B) = E(B,A), but the two lookup
    /// orders read different dihedral bins and faces, and near walls can differ
    /// by thousands of kT. Both are combined as two equally likely stencils: the
    /// result is the same whichever body comes first, as detailed balance needs,
    /// and keeps each order's orientational average of exp(−βu), which the
    /// table's PMF is. Averaging energies would underweight every pose where
    /// they disagree.
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
        if let Some(outside) = outside_table(self, separation, beta) {
            return outside;
        }
        let ordered = |sep: &Vector3, q_a, q_b| {
            let (r, omega, dir_a, dir_b) = crate::inverse_orient(sep, q_a, q_b);
            self.interpolate(r, omega, &dir_a, &dir_b, Some(beta))
        };
        let forward = ordered(separation, q_a, q_b);
        let exchange = self
            .metadata()
            .is_some_and(|m| m.point_group == PointGroup::Exchange);
        if !exchange {
            return PairEnergy::single(forward);
        }
        let reverse = ordered(&-separation, q_b, q_a);
        let both = combine(&[0.5, 0.5], &[forward, reverse], beta);
        PairEnergy {
            energy: both.energy,
            mean_energy: both.mean_energy,
            order_mismatch: order_mismatch(forward.energy, reverse.energy, beta),
        }
    }
}

/// Result of [`Lookup6D::pair_energy`] and [`Lookup3D::pair_energy`], in the
/// table's energy unit.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PairEnergy {
    /// Free energy w. Use for MC acceptance, not for ⟨U⟩.
    pub energy: f64,
    /// ⟨u⟩ = ∂(βw)/∂β. Sum this to report ⟨U⟩.
    pub mean_energy: f64,
    /// |exp(−βu₁) − exp(−βu₂)| between the two lookup orders of an exchange
    /// table: how much they disagree at this pose. Zero for one lookup. Orders
    /// that agree can still both be wrong, so this bounds no error.
    pub order_mismatch: f64,
}

impl PairEnergy {
    const fn single(e: Interpolated) -> Self {
        Self {
            energy: e.energy,
            mean_energy: e.mean_energy,
            order_mismatch: 0.0,
        }
    }
}

/// Lookup over a 3D molecule–atom table: separation and one body-frame direction.
pub trait Lookup3D: TabulatedInteraction {
    /// Interpolated free and mean energy at separation `r` and body-frame
    /// direction `dir`. `beta` behaves as in [`Lookup6D::interpolate`].
    fn interpolate(&self, r: f64, dir: &Vector3, beta: Option<f64>) -> Interpolated;

    /// The free energy of [`Self::interpolate`].
    fn lookup(&self, r: f64, dir: &Vector3, beta: Option<f64>) -> f64 {
        self.interpolate(r, dir, beta).energy
    }

    /// Energy of a rigid molecule and an atom at lab-frame offset `separation`
    /// (atom − molecule), with the molecule's lab orientation `q`, Boltzmann-
    /// interpolated at `beta`. Range rules as in [`Lookup6D::pair_energy`].
    ///
    /// # Panics
    /// If `separation` isn't finite, or `beta` isn't finite and positive.
    fn pair_energy(&self, separation: &Vector3, q: &UnitQuaternion, beta: f64) -> PairEnergy {
        if let Some(outside) = outside_table(self, separation, beta) {
            return outside;
        }
        let r = separation.norm();
        let dir = q.inverse_transform_vector(separation) / r;
        PairEnergy::single(self.interpolate(r, &dir, Some(beta)))
    }
}

/// The pair energy where no lookup is needed: +∞ below the table, so bodies
/// can't pass through each other, and the tail beyond it.
fn outside_table<T: TabulatedInteraction + ?Sized>(
    table: &T,
    separation: &Vector3,
    beta: f64,
) -> Option<PairEnergy> {
    assert!(
        beta.is_finite() && beta > 0.0,
        "beta must be finite and positive, got {beta}"
    );
    let r = separation.norm();
    // NaN would pass both range checks and read the contact bin.
    assert!(r.is_finite(), "pair separation must be finite, got {r}");
    let (rmin, rmax) = table.r_range();
    let exact = |u| Some(PairEnergy::single(Interpolated::exact(u)));
    if r < rmin {
        exact(f64::INFINITY)
    } else if r > rmax {
        exact(table.tail_energy(r))
    } else {
        None
    }
}

/// See [`PairEnergy::order_mismatch`].
fn order_mismatch(forward: f64, reverse: f64, beta: f64) -> f64 {
    let low = forward.min(reverse);
    if low == f64::INFINITY {
        return 0.0;
    }
    // Relative to the lower energy, so deep wells don't overflow.
    let gap = ((-beta * (forward - low)).exp() - (-beta * (reverse - low)).exp()).abs();
    // Equal deep wells would give ∞ × 0.
    if gap == 0.0 {
        0.0
    } else {
        (-beta * low).exp() * gap
    }
}

#[cfg(test)]
mod tests {
    use super::{Lookup3D, Lookup6D};
    use crate::{Table3DAdaptive, Table3DFlat, Table6DAdaptive, Table6DFlat};

    use super::{Interpolated, PairEnergy};
    use crate::{
        orient, AdaptiveBuilder, PointGroup, TableMetadata, TailCorrectionTerm, UnitQuaternion,
        Vector3,
    };
    use approx::assert_relative_eq;
    use rand::{Rng, SeedableRng};

    /// Energies are in kT at the temperature the test tables were scanned at.
    const SCAN_BETA: f64 = 1.0;
    /// Lookups may run at any temperature; β = 1 alone would hide a stray factor of β.
    const BETAS: [f64; 3] = [0.4, 1.0, 2.5];
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
        let mut builder = AdaptiveBuilder::new(RMIN, RMAX, 1.0, omega_step, 2, 1e-9, SCAN_BETA);
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
            table_sum += (-SCAN_BETA * t.pair_energy(&sep, &q_a, &q_b, SCAN_BETA).energy).exp();
            exact_sum += (-SCAN_BETA * exact(&sep, &q_a, &q_b)).exp();
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
            let ab = t.pair_energy(&sep, &q_a, &q_b, SCAN_BETA);
            let ba = t.pair_energy(&-sep, &q_b, &q_a, SCAN_BETA);
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
        let single = Lookup6D::interpolate(&t, r, omega, &dir_a, &dir_b, Some(SCAN_BETA));
        let pair = t.pair_energy(&sep, &q_a, &q_b, SCAN_BETA);
        assert_eq!(
            pair,
            PairEnergy {
                energy: single.energy,
                mean_energy: single.mean_energy,
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
        let below = t.pair_energy(&Vector3::new(0.0, 0.0, 4.0), &q, &q, SCAN_BETA);
        assert_eq!(below.energy, f64::INFINITY);
        let beyond = t.pair_energy(&Vector3::new(0.0, 0.0, 20.0), &q, &q, SCAN_BETA);
        assert_relative_eq!(beyond.energy, -10.0 / 20.0);
        assert_eq!(beyond.order_mismatch, 0.0);
    }

    /// A NaN distance would pass both range checks and read the contact bin.
    #[test]
    #[should_panic(expected = "finite")]
    fn non_finite_separation_is_a_bug_not_an_energy() {
        let t = table(PointGroup::Exchange);
        let q = UnitQuaternion::identity();
        t.pair_energy(&Vector3::new(f64::NAN, 0.0, 6.0), &q, &q, SCAN_BETA);
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
        fn interpolate(
            &self,
            _: f64,
            _: f64,
            dir_a: &Vector3,
            _: &Vector3,
            _: Option<f64>,
        ) -> Interpolated {
            Interpolated::exact(if dir_a.z > 0.0 {
                self.ahead
            } else {
                self.behind
            })
        }
    }

    /// The two orders are two states of equal prior weight, so the mean energy
    /// is their Boltzmann-weighted average, whichever order comes first.
    #[test]
    fn exchange_mean_energy_weights_each_order_by_its_boltzmann_factor() {
        let (a, b) = (0.0, 1.5);
        for beta in BETAS {
            let (xa, xb) = ((-beta * a).exp(), (-beta * b).exp());
            let expected = (a * xa + b * xb) / (xa + xb);
            for t in [ByOrder::new(a, b), ByOrder::new(b, a)] {
                let got = t.at_contact(beta).mean_energy;
                assert_relative_eq!(got, expected, max_relative = 1e-14);
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
        let pair = ByOrder::new(f64::INFINITY, f64::INFINITY).at_contact(SCAN_BETA);
        assert_eq!(pair.energy, f64::INFINITY);
        assert_eq!(pair.order_mismatch, 0.0);
    }

    /// exp(−βu) overflows here; the shifted sum must not.
    #[test]
    fn equal_deep_wells_agree_without_overflow() {
        let pair = ByOrder::new(-1000.0, -1000.0).at_contact(SCAN_BETA);
        assert_relative_eq!(pair.energy, -1000.0);
        assert_eq!(pair.order_mismatch, 0.0);
    }

    #[test]
    #[should_panic(expected = "beta")]
    fn non_positive_beta_is_a_bug() {
        ByOrder::new(0.0, 0.0).at_contact(0.0);
    }

    /// ∂(βw)/∂β by central difference: the mean energy, independently of how
    /// the lookup computes it.
    fn beta_derivative(beta: f64, beta_w: impl Fn(f64) -> f64) -> f64 {
        let h = 1e-4;
        (beta_w(beta + h) - beta_w(beta - h)) / (2.0 * h)
    }

    /// The mean energy at `beta` is the β-derivative of βw, and never above w (Gibbs).
    fn assert_mean_energy(beta: f64, pair: PairEnergy, free_energy_at: impl Fn(f64) -> f64) {
        let expected = beta_derivative(beta, |b| b * free_energy_at(b));
        assert_relative_eq!(pair.mean_energy, expected, epsilon = 1e-6);
        assert!(pair.mean_energy <= pair.energy + 1e-12);
    }

    #[test]
    fn exchange_mean_energy_is_the_beta_derivative_in_either_order() {
        let t = table(PointGroup::Exchange);
        let mut rng = rand::rngs::StdRng::seed_from_u64(11);
        for _ in 0..50 {
            let r = rng.gen_range(5.5..7.5);
            let (sep, q_a, q_b) = random_pose(&mut rng, r);
            for (sep, q_a, q_b) in [(sep, q_a, q_b), (-sep, q_b, q_a)] {
                for beta in BETAS {
                    let pair = t.pair_energy(&sep, &q_a, &q_b, beta);
                    assert_mean_energy(beta, pair, |b| t.pair_energy(&sep, &q_a, &q_b, b).energy);
                }
            }
        }
    }

    /// Steep in the body-frame direction, like the 6D test table.
    fn table_3d() -> crate::Table3DAdaptive<f32> {
        let mut builder = crate::Adaptive3DBuilder::new(RMIN, RMAX, 1.0, 2, 1e-9, SCAN_BETA);
        for ri in 0..builder.n_r() {
            let data: Vec<f64> = builder
                .vertex_directions(builder.current_level())
                .iter()
                .map(|v| 2.0 * (v[0] + 2.0 * v[1] + 3.0 * v[2]))
                .collect();
            builder.set_slab(ri, &data);
            builder.finish_r_slice(ri);
        }
        let mut t = builder.build();
        t.metadata = table(PointGroup::Asymmetric).metadata;
        t
    }

    #[test]
    fn three_d_mean_energy_is_the_beta_derivative() {
        let t = table_3d();
        let mut rng = rand::rngs::StdRng::seed_from_u64(13);
        for _ in 0..50 {
            let r = rng.gen_range(5.5..7.5);
            let (sep, q, _) = random_pose(&mut rng, r);
            for beta in BETAS {
                let pair = t.pair_energy(&sep, &q, beta);
                assert_mean_energy(beta, pair, |b| t.pair_energy(&sep, &q, b).energy);
                assert_eq!(pair.order_mismatch, 0.0);
            }
        }
    }

    #[test]
    fn three_d_pair_energy_is_a_wall_below_and_the_tail_beyond_the_range() {
        let t = table_3d();
        let q = UnitQuaternion::identity();
        let below = t.pair_energy(&Vector3::new(0.0, 0.0, 4.0), &q, SCAN_BETA);
        assert_eq!(below.energy, f64::INFINITY);
        assert_eq!(below.mean_energy, f64::INFINITY);
        let beyond = t.pair_energy(&Vector3::new(0.0, 0.0, 20.0), &q, SCAN_BETA);
        assert_relative_eq!(beyond.energy, -10.0 / 20.0);
        assert_relative_eq!(beyond.mean_energy, -10.0 / 20.0);
    }

    /// Linear interpolation has no Boltzmann weights, so nothing separates the two.
    #[test]
    fn linear_interpolation_has_equal_free_and_mean_energy() {
        let (t6, t3) = (table(PointGroup::Exchange), table_3d());
        let dir = Vector3::new(0.3, -0.5, 0.8).normalize();
        let six = Lookup6D::interpolate(&t6, 6.2, 0.7, &dir, &-dir, None);
        let three = Lookup3D::interpolate(&t3, 6.2, &dir, None);
        for e in [six, three] {
            assert_eq!(e.mean_energy, e.energy);
        }
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
