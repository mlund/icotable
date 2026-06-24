//! Direction→cell `locate` throughput: geodesic search vs. lattice closed-form,
//! at matched vertex counts.
//!
//! Run with: `cargo bench --bench locate`
//!
//! Both schemes produce `10·(n_div+1)²+2` vertices at a given `n_div`, so each
//! `n_div` row compares the two locators on identical mesh sizes. Each timed
//! sample runs a tight batch (no per-call `black_box` barrier); `ItemsCount`
//! reports throughput — invert for ns/lookup (100 Mitem/s = 10 ns/lookup).
//!
//! Note: absolute ns/lookup are only representative on a non-throttled CPU
//! (disable power-saving / keep the machine plugged in). The geodesic-vs-lattice
//! *ratio* is robust to throttling, since both run under identical conditions.

use divan::{black_box, counter::ItemsCount, Bencher};
use icotable::{MeshLevel, Subdivision, Vector3};
use rand::{rngs::StdRng, Rng, SeedableRng};

/// Subdivision levels → vertex counts 42 / 92 / 162 / 252 / 362.
const NDIVS: &[usize] = &[1, 2, 3, 4, 5];

/// Distinct directions per timed batch.
const N_DIRS: usize = 1024;

fn main() {
    divan::main();
}

/// Deterministic batch of normalized directions (seeded for run-to-run stability).
fn sample_dirs() -> Vec<Vector3> {
    let mut rng = StdRng::seed_from_u64(0xC0FFEE);
    let mut out = Vec::with_capacity(N_DIRS);
    while out.len() < N_DIRS {
        let x: f64 = rng.gen_range(-1.0..1.0);
        let y: f64 = rng.gen_range(-1.0..1.0);
        let z: f64 = rng.gen_range(-1.0..1.0);
        let r2 = x * x + y * y + z * z;
        if r2 > 0.01 && r2 < 1.0 {
            out.push(Vector3::new(x, y, z).normalize());
        }
    }
    out
}

fn bench_scheme(bencher: Bencher, scheme: Subdivision, n_div: usize) {
    let mesh = MeshLevel::with_subdivision(scheme, n_div);
    let dirs = sample_dirs();
    let _ = mesh.locate(&dirs[0]); // warm the lazily-built locator
    bencher
        .counter(ItemsCount::new(N_DIRS))
        .bench_local(|| {
            let mut acc = 0usize;
            for d in &dirs {
                acc ^= mesh.locate(d).0[0];
            }
            black_box(acc)
        });
}

/// Geodesic icosphere — spatial-search (`FaceGrid`) locator.
#[divan::bench(args = NDIVS)]
fn geodesic(bencher: Bencher, n_div: usize) {
    bench_scheme(bencher, Subdivision::Geodesic, n_div);
}

/// Regular lattice — closed-form (`AnalyticLattice`) locator.
#[divan::bench(args = NDIVS)]
fn lattice(bencher: Bencher, n_div: usize) {
    bench_scheme(bencher, Subdivision::Lattice, n_div);
}
