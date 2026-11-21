//! Boltzmann-weighted combination of interpolation stencils.
//!
//! Interpolating exp(−βu) rather than u avoids the Jensen bias of linear
//! interpolation, but the result w = −ln Σ aₖ e^{−βuₖ} / β is a free energy over
//! the stencil. The ensemble mean energy needs ∂(βw)/∂β = Σ pₖ uₖ instead, which
//! is ≤ w. Both come from the same weights, so they are computed together.

/// Free energy and mean energy of a Boltzmann-weighted set of states.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Interpolated {
    /// Free energy w. Use for MC acceptance, not for ⟨U⟩.
    pub energy: f64,
    /// ⟨u⟩ = ∂(βw)/∂β. Sum this to report ⟨U⟩.
    pub mean_energy: f64,
}

impl Interpolated {
    /// A single state, such as a grid value: both energies are `u`.
    pub(crate) const fn exact(u: f64) -> Self {
        Self {
            energy: u,
            mean_energy: u,
        }
    }
}

/// Combines `parts` with prior weights `weights` at inverse temperature `beta`.
///
/// Parts may themselves be combinations, so stencils and lookup orders nest:
/// the result equals one combination over all underlying states. Shifting by the
/// lowest weighted energy keeps the exponentials in range. Zero-weight terms are
/// dropped, not multiplied by zero, since their exponential may be ∞. All walls,
/// or a sum that underflows, give +∞. `beta` must be positive.
pub(crate) fn combine(weights: &[f64], parts: &[Interpolated], beta: f64) -> Interpolated {
    debug_assert_eq!(weights.len(), parts.len());
    let weighted = || weights.iter().zip(parts).filter(|(&w, _)| w > 0.0);
    let low = weighted()
        .map(|(_, p)| p.energy)
        .fold(f64::INFINITY, f64::min);
    if low.is_infinite() {
        return Interpolated::exact(f64::INFINITY);
    }
    let (mut sum, mut moment) = (0.0, 0.0);
    for (w, p) in weighted() {
        let x = w * (-beta * (p.energy - low)).exp();
        // A wall carries no weight; skipping it avoids 0 · ∞.
        if x > 0.0 {
            sum += x;
            moment += x * p.mean_energy;
        }
    }
    if sum <= 0.0 {
        return Interpolated::exact(f64::INFINITY);
    }
    Interpolated {
        energy: low - sum.ln() / beta,
        mean_energy: moment / sum,
    }
}

#[cfg(test)]
mod tests {
    use super::{combine, Interpolated};
    use approx::assert_relative_eq;

    /// Any positive value; the closed-form test also scans β.
    const BETA: f64 = 2.0;

    #[test]
    fn two_states_match_the_closed_form() {
        let delta = 1.5;
        for beta in [0.4_f64, 1.0, 2.5] {
            let x = (-beta * delta).exp();
            let states = [Interpolated::exact(0.0), Interpolated::exact(delta)];
            let got = combine(&[0.5, 0.5], &states, beta);
            let free = -((1.0 + x) / 2.0).ln() / beta;
            assert_relative_eq!(got.energy, free, max_relative = 1e-14);
            assert_relative_eq!(got.mean_energy, delta * x / (1.0 + x), max_relative = 1e-14);
        }
    }

    #[test]
    fn equal_states_are_that_state() {
        let u = Interpolated::exact(-3.0);
        let got = combine(&[0.2, 0.3, 0.5], &[u, u, u], BETA);
        assert_relative_eq!(got.energy, -3.0, max_relative = 1e-14);
        assert_relative_eq!(got.mean_energy, -3.0, max_relative = 1e-14);
    }

    /// Half the weight is a wall: the free energy pays ln 2 / β, the mean energy nothing.
    #[test]
    fn a_wall_carries_no_weight() {
        let wall = Interpolated::exact(f64::INFINITY);
        let got = combine(&[0.5, 0.5], &[Interpolated::exact(0.0), wall], BETA);
        assert_relative_eq!(got.energy, std::f64::consts::LN_2 / BETA);
        assert_eq!(got.mean_energy, 0.0);
        assert_eq!(combine(&[0.5, 0.5], &[wall, wall], BETA), wall);
    }

    /// Why lookup orders can be combined like stencil points: a combination of
    /// combinations is one combination over all states, with weights multiplied.
    #[test]
    fn nested_combinations_flatten() {
        let (wa, ua) = ([0.2, 0.8], [0.0, 2.0].map(Interpolated::exact));
        let (wb, ub) = ([0.6, 0.4], [-1.0, 4.0].map(Interpolated::exact));
        let nested = combine(
            &[0.5, 0.5],
            &[combine(&wa, &ua, BETA), combine(&wb, &ub, BETA)],
            BETA,
        );
        let weights: Vec<f64> = wa.iter().chain(&wb).map(|w| 0.5 * w).collect();
        let states: Vec<Interpolated> = ua.iter().chain(&ub).copied().collect();
        let flat = combine(&weights, &states, BETA);
        assert_relative_eq!(nested.energy, flat.energy, max_relative = 1e-13);
        assert_relative_eq!(nested.mean_energy, flat.mean_energy, max_relative = 1e-13);
    }
}
