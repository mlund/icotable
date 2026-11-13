//! GPU port of [`Table6DAdaptive`] lookups: packed buffers plus WGSL.
//!
//! The table is flattened into one `u32` and one `f32` array, since WebGPU
//! allows only a few storage buffers per shader. The WGSL from [`wgsl`] reads
//! them through two functions the caller defines, so it fixes no bindings:
//!
//! ```wgsl
//! @group(0) @binding(0) var<storage, read> words: array<u32>;
//! @group(0) @binding(1) var<storage, read> floats: array<f32>;
//! fn icotable_word(i: u32) -> u32 { return words[i]; }
//! fn icotable_float(i: u32) -> f32 { return floats[i]; }
//! ```
//!
//! Word layout: a header (`HEADER_LEN` words, see the `H_*` indices), one
//! slab record of two words per (R, ω) slab, then one level record of
//! `LEVEL_LEN` words per mesh level, followed by the levels' grid arrays.
//! Floats hold the slab energies (indexed by slab offsets as on the CPU),
//! then the levels' vertex positions.
//!
//! The lookup runs in f32. It agrees with the CPU except in rare edge cases:
//! directions within ~1e-3 rad of a body's −z axis (where ω is ill-defined
//! and the anti-parallel branch differs), R or ω within an f32 ulp of a bin
//! edge, and near-ties between nearest vertices.

use crate::adaptive::{SlabResolution, Table6DAdaptive};
use anyhow::{anyhow, ensure, Result};

/// Energies at or above this are repulsive. WGSL may not represent infinity,
/// so packing replaces +∞ and larger values by it.
pub const REPULSIVE: f32 = 1e30;

const H_N_R: usize = 0;
const H_N_OMEGA: usize = 1;
const H_RMIN: usize = 2;
const H_RMAX: usize = 3;
const H_DR: usize = 4;
const H_OMEGA_STEP: usize = 5;
const H_SLABS: usize = 6;
const H_LEVELS: usize = 7;
const HEADER_LEN: usize = 8;
/// Fields as returned by `FaceGrid::pack`.
const LEVEL_LEN: usize = 7;

/// Slab kinds in the low bits of each slab record's first word; the mesh
/// level sits above `LEVEL_SHIFT`.
const KIND_REPULSIVE: u32 = 0;
const KIND_SCALAR: u32 = 1;
const KIND_NEAREST: u32 = 2;
const KIND_INTERPOLATE: u32 = 3;
const LEVEL_SHIFT: u32 = 8;

/// The lookup functions, to append to a shader that defines the accessors.
/// The layout constants are generated from this file's, so the two sides
/// can't drift apart.
pub fn wgsl() -> String {
    let constants = [
        ("H_N_R", H_N_R as u32),
        ("H_N_OMEGA", H_N_OMEGA as u32),
        ("H_RMIN", H_RMIN as u32),
        ("H_RMAX", H_RMAX as u32),
        ("H_DR", H_DR as u32),
        ("H_OMEGA_STEP", H_OMEGA_STEP as u32),
        ("H_SLABS", H_SLABS as u32),
        ("H_LEVELS", H_LEVELS as u32),
        ("LEVEL_LEN", LEVEL_LEN as u32),
        ("KIND_REPULSIVE", KIND_REPULSIVE),
        ("KIND_SCALAR", KIND_SCALAR),
        ("KIND_NEAREST", KIND_NEAREST),
        ("KIND_INTERPOLATE", KIND_INTERPOLATE),
        ("LEVEL_SHIFT", LEVEL_SHIFT),
        ("KIND_MASK", (1 << LEVEL_SHIFT) - 1),
    ];
    let mut prelude: String = constants
        .iter()
        .map(|(name, value)| format!("const ICOTABLE_{name}: u32 = {value}u;\n"))
        .collect();
    prelude += &format!("const ICOTABLE_REPULSIVE: f32 = {REPULSIVE:e};\n");
    prelude + include_str!("gpu.wgsl")
}

/// A table packed for the GPU; upload both arrays as storage buffers. Large
/// tables can exceed WebGPU's default 128 MiB binding limit, so check
/// `floats.len() * 4` against the adapter's `max_storage_buffer_binding_size`.
#[derive(Clone, Debug)]
pub struct GpuTable {
    /// Header, slab records, level records and locator grids.
    pub words: Vec<u32>,
    /// Slab energies, then mesh vertex positions.
    pub floats: Vec<f32>,
}

impl GpuTable {
    /// Fails for tables on the lattice subdivision, whose locator has no GPU
    /// port yet, and for tables holding NaN or −∞, which would otherwise pass
    /// as repulsive and hide broken data.
    pub fn new(table: &Table6DAdaptive<f32>) -> Result<Self> {
        let scalars = table.slab_res.iter().filter_map(|res| match res {
            SlabResolution::Scalar(u) => Some(*u),
            _ => None,
        });
        let invalid = |u: f32| u.is_nan() || u == f32::NEG_INFINITY;
        ensure!(
            !table.data.iter().copied().chain(scalars).any(invalid),
            "table holds NaN or -inf energies"
        );
        let mut words = vec![0u32; HEADER_LEN];
        let mut floats: Vec<f32> = table.data.iter().map(|&u| sanitize(u)).collect();
        write_header(&mut words, table);
        words[H_SLABS] = words.len() as u32;
        for (res, &offset) in table.slab_res.iter().zip(&table.slab_offsets) {
            words.extend(slab_record(res, offset));
        }
        words[H_LEVELS] = words.len() as u32;
        let levels_start = words.len();
        words.resize(levels_start + LEVEL_LEN * table.levels.len(), 0);
        for (i, level) in table.levels.iter().enumerate() {
            let grid = level
                .face_grid()
                .ok_or_else(|| anyhow!("GPU lookup supports only the geodesic subdivision"))?;
            let record = grid.pack(&mut words, &mut floats);
            let at = levels_start + LEVEL_LEN * i;
            words[at..at + LEVEL_LEN].copy_from_slice(&record);
        }
        Ok(Self { words, floats })
    }
}

fn write_header(words: &mut [u32], table: &Table6DAdaptive<f32>) {
    words[H_N_R] = table.n_r as u32;
    words[H_N_OMEGA] = table.n_omega as u32;
    words[H_RMIN] = (table.rmin as f32).to_bits();
    words[H_RMAX] = (table.rmax as f32).to_bits();
    words[H_DR] = (table.dr as f32).to_bits();
    words[H_OMEGA_STEP] = (table.omega_step as f32).to_bits();
}

fn slab_record(res: &SlabResolution, offset: u32) -> [u32; 2] {
    match *res {
        SlabResolution::Repulsive => [KIND_REPULSIVE, 0],
        SlabResolution::Scalar(u) => [KIND_SCALAR, sanitize(u).to_bits()],
        SlabResolution::Mesh { level, interpolate } => {
            let kind = if interpolate {
                KIND_INTERPOLATE
            } else {
                KIND_NEAREST
            };
            [kind | (level as u32) << LEVEL_SHIFT, offset]
        }
    }
}

/// +∞ and huge energies become the repulsive sentinel (NaN and −∞ are
/// rejected earlier).
fn sanitize(u: f32) -> f32 {
    if u < REPULSIVE {
        u
    } else {
        REPULSIVE
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::adaptive::AdaptiveBuilder;
    use crate::Vector3;
    use wgpu::util::DeviceExt;

    /// Evaluates `icotable_lookup` for a batch of queries on the GPU.
    struct Runner {
        device: wgpu::Device,
        queue: wgpu::Queue,
        pipeline: wgpu::ComputePipeline,
    }

    /// One query: (r, ω, β) and the two body-frame directions.
    struct Query {
        r: f32,
        omega: f32,
        beta: f32,
        dir_a: [f32; 3],
        dir_b: [f32; 3],
    }

    /// Bindings shared by every test shader; each adds its own `main`.
    const BINDINGS: &str = "
        @group(0) @binding(0) var<storage, read> words: array<u32>;
        @group(0) @binding(1) var<storage, read> floats: array<f32>;
        @group(0) @binding(2) var<storage, read> queries: array<vec4<f32>>;
        @group(0) @binding(3) var<storage, read_write> out: array<f32>;
        fn icotable_word(i: u32) -> u32 { return words[i]; }
        fn icotable_float(i: u32) -> f32 { return floats[i]; }
    ";

    const LOOKUP_MAIN: &str = "
        @compute @workgroup_size(64)
        fn main(@builtin(global_invocation_id) id: vec3<u32>) {
            let i = id.x;
            if 2u * i >= arrayLength(&out) { return; }
            let q = queries[3u * i];
            let v = icotable_lookup(q.x, q.y, queries[3u * i + 1u].xyz, queries[3u * i + 2u].xyz, q.z);
            out[2u * i] = v.energy;
            out[2u * i + 1u] = f32(v.repulsive);
        }
    ";

    impl Runner {
        /// `None` without a GPU adapter, so tests can skip.
        fn new(main: &str) -> Option<Self> {
            let instance = wgpu::Instance::default();
            let adapter = pollster::block_on(instance.request_adapter(&Default::default())).ok()?;
            let (device, queue) =
                pollster::block_on(adapter.request_device(&Default::default())).ok()?;
            let source = format!("{BINDINGS}\n{main}\n{}", wgsl());
            let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("icotable-test"),
                source: wgpu::ShaderSource::Wgsl(source.into()),
            });
            let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("icotable-test"),
                layout: None,
                module: &module,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });
            Some(Self {
                device,
                queue,
                pipeline,
            })
        }

        fn storage(&self, contents: &[u8], usage: wgpu::BufferUsages) -> wgpu::Buffer {
            self.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: None,
                    contents,
                    usage: wgpu::BufferUsages::STORAGE | usage,
                })
        }

        /// (energy, repulsive) per query.
        fn run(&self, table: &GpuTable, queries: &[Query]) -> Vec<(f32, bool)> {
            let packed: Vec<f32> = queries
                .iter()
                .flat_map(|q| {
                    let [ax, ay, az] = q.dir_a;
                    let [bx, by, bz] = q.dir_b;
                    [q.r, q.omega, q.beta, 0.0, ax, ay, az, 0.0, bx, by, bz, 0.0]
                })
                .collect();
            let out = self.run_raw(table, &packed, queries.len(), 2);
            out.chunks(2).map(|v| (v[0], v[1] != 0.0)).collect()
        }

        /// Runs the shader over `n` queries packed as three vec4 each and
        /// returns `width` floats per query.
        fn run_raw(&self, table: &GpuTable, packed: &[f32], n: usize, width: usize) -> Vec<f32> {
            let out_size = (n * width * 4) as u64;
            let none = wgpu::BufferUsages::empty();
            let words = self.storage(bytemuck::cast_slice(&table.words), none);
            let floats = self.storage(bytemuck::cast_slice(&table.floats), none);
            let input = self.storage(bytemuck::cast_slice(packed), none);
            let out = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: out_size,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            let read = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: out_size,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let layout = self.pipeline.get_bind_group_layout(0);
            fn entry(binding: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
                wgpu::BindGroupEntry {
                    binding,
                    resource: buffer.as_entire_binding(),
                }
            }
            let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &layout,
                entries: &[
                    entry(0, &words),
                    entry(1, &floats),
                    entry(2, &input),
                    entry(3, &out),
                ],
            });
            let mut encoder = self.device.create_command_encoder(&Default::default());
            {
                let mut pass = encoder.begin_compute_pass(&Default::default());
                pass.set_pipeline(&self.pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.dispatch_workgroups((n as u32).div_ceil(64), 1, 1);
            }
            encoder.copy_buffer_to_buffer(&out, 0, &read, 0, out_size);
            self.queue.submit([encoder.finish()]);
            read.slice(..).map_async(wgpu::MapMode::Read, |_| {});
            self.device
                .poll(wgpu::PollType::wait_indefinitely())
                .unwrap();
            let data = read.slice(..).get_mapped_range();
            bytemuck::cast_slice(&data).to_vec()
        }
    }

    fn runner() -> Option<Runner> {
        runner_with(LOOKUP_MAIN)
    }

    fn runner_with(main: &str) -> Option<Runner> {
        let runner = Runner::new(main);
        if runner.is_none() {
            eprintln!("no GPU adapter; skipping");
        }
        runner
    }

    fn query(r: f64, omega: f64, a: Vector3, b: Vector3, beta: f64) -> Query {
        let (a, b) = (a.normalize(), b.normalize());
        Query {
            r: r as f32,
            omega: omega as f32,
            beta: beta as f32,
            dir_a: [a.x as f32, a.y as f32, a.z as f32],
            dir_b: [b.x as f32, b.y as f32, b.z as f32],
        }
    }

    /// Constant energy with a huge gradient threshold: every slab is scalar.
    fn constant_table(value: f64) -> Table6DAdaptive<f32> {
        let mut builder =
            AdaptiveBuilder::new(5.0, 10.0, 1.0, std::f64::consts::TAU / 8.0, 2, 1e10, 0.4);
        for ri in 0..builder.n_r() {
            let n_v = builder.current_n_vertices();
            for oi in 0..builder.n_omega() {
                builder.set_slab(ri, oi, &vec![value; n_v * n_v]);
            }
            builder.finish_r_slice(ri);
        }
        builder.build()
    }

    /// Smooth anisotropic energy, repulsive in the innermost R bin; the
    /// gradient threshold decides between interpolated and nearest slabs.
    fn anisotropic_table(gradient_threshold: f64) -> Table6DAdaptive<f32> {
        let u = |r: f64, a: &Vector3, b: &Vector3| {
            if r < 5.5 {
                1e4
            } else {
                2.0 * a.x - 1.5 * b.y + a.z * b.z + 0.3 * r
            }
        };
        let omega_step = std::f64::consts::TAU / 8.0;
        let mut builder =
            AdaptiveBuilder::new(5.0, 10.0, 1.0, omega_step, 2, gradient_threshold, 0.4);
        for ri in 0..builder.n_r() {
            let r = 5.0 + ri as f64;
            let n_v = builder.current_n_vertices();
            let verts = builder.vertex_directions(builder.current_level()).to_vec();
            let data: Vec<f64> = (0..n_v * n_v)
                .map(|k| {
                    u(
                        r,
                        &Vector3::from(verts[k / n_v]),
                        &Vector3::from(verts[k % n_v]),
                    )
                })
                .collect();
            for oi in 0..builder.n_omega() {
                builder.set_slab(ri, oi, &data);
            }
            builder.finish_r_slice(ri);
        }
        builder.build()
    }

    fn has_kind(table: &Table6DAdaptive<f32>, kind: impl Fn(&SlabResolution) -> bool) -> bool {
        table.slab_res.iter().any(kind)
    }

    /// GPU and CPU lookups agree on random poses, to f32 precision.
    fn assert_matches_cpu(table: &Table6DAdaptive<f32>) {
        use rand::{Rng, SeedableRng};
        let Some(gpu) = runner() else { return };
        let mut rng = rand::rngs::StdRng::seed_from_u64(3);
        let mut dir = || Vector3::from_fn(|_, _| rng.gen_range(-1.0..1.0)).normalize();
        let mut cases = Vec::new();
        for i in 0..2000 {
            let r = 5.0 + 5.0 * (i as f64 / 2000.0);
            cases.push((r, 0.37 * i as f64, dir(), dir()));
        }
        let queries: Vec<Query> = cases
            .iter()
            .map(|&(r, w, a, b)| query(r, w, a, b, 0.4))
            .collect();
        let results = gpu.run(&GpuTable::new(table).unwrap(), &queries);
        for ((r, w, a, b), (energy, repulsive)) in cases.iter().zip(results) {
            let expected = table.lookup_boltzmann(*r, *w, a, b, 0.4);
            if expected.is_infinite() {
                assert!(repulsive, "r={r} ω={w}: GPU {energy}, CPU repulsive");
            } else {
                assert!(!repulsive, "r={r} ω={w}: GPU repulsive, CPU {expected}");
                approx::assert_relative_eq!(
                    energy as f64,
                    expected,
                    epsilon = 2e-4,
                    max_relative = 1e-4
                );
            }
        }
    }

    #[test]
    fn interpolated_slabs_match_the_cpu() {
        let table = anisotropic_table(1e-9);
        assert!(has_kind(&table, |s| matches!(
            s,
            SlabResolution::Mesh {
                interpolate: true,
                ..
            }
        )));
        assert!(has_kind(&table, |s| matches!(s, SlabResolution::Repulsive)));
        assert_matches_cpu(&table);
    }

    #[test]
    fn nearest_vertex_slabs_match_the_cpu() {
        let table = anisotropic_table(5.0);
        assert!(has_kind(&table, |s| matches!(
            s,
            SlabResolution::Mesh {
                interpolate: false,
                ..
            }
        )));
        assert_matches_cpu(&table);
    }

    const ORIENT_MAIN: &str = "
        @compute @workgroup_size(64)
        fn main(@builtin(global_invocation_id) id: vec3<u32>) {
            let i = id.x;
            if 8u * i >= arrayLength(&out) { return; }
            _ = icotable_word(0u) + u32(icotable_float(0u)); // keep the shared bindings
            let p = icotable_inverse_orient(
                queries[3u * i].xyz, queries[3u * i + 1u], queries[3u * i + 2u],
            );
            let values = array<f32, 8>(p.r, p.omega, p.dir_a.x, p.dir_a.y, p.dir_a.z,
                                       p.dir_b.x, p.dir_b.y, p.dir_b.z);
            for (var k = 0u; k < 8u; k++) { out[8u * i + k] = values[k]; }
        }
    ";

    /// The GPU recovers the same table coordinates as `inverse_orient`, so
    /// lab-frame poses look up the same entries on both.
    #[test]
    fn inverse_orient_matches_the_cpu() {
        use crate::UnitQuaternion;
        use rand::{Rng, SeedableRng};
        let Some(gpu) = runner_with(ORIENT_MAIN) else {
            return;
        };
        let mut rng = rand::rngs::StdRng::seed_from_u64(5);
        let mut turn =
            || UnitQuaternion::from_scaled_axis(Vector3::from_fn(|_, _| rng.gen_range(-2.0..2.0)));
        let cases: Vec<(Vector3, UnitQuaternion, UnitQuaternion)> = (0..1000)
            .map(|i| {
                let sep = Vector3::new((i as f64 * 0.7).sin(), (i as f64 * 1.3).cos(), 0.4) * 7.0;
                (sep, turn(), turn())
            })
            .collect();
        let packed: Vec<f32> = cases
            .iter()
            .flat_map(|(sep, qa, qb)| {
                let (a, b) = (qa.quaternion(), qb.quaternion());
                [
                    sep.x, sep.y, sep.z, 0.0, a.i, a.j, a.k, a.w, b.i, b.j, b.k, b.w,
                ]
                .map(|x| x as f32)
            })
            .collect();
        let table = GpuTable::new(&constant_table(1.0)).unwrap();
        let out = gpu.run_raw(&table, &packed, cases.len(), 8);
        for ((sep, qa, qb), got) in cases.iter().zip(out.chunks(8)) {
            let (r, omega, dir_a, dir_b) = crate::inverse_orient(sep, qa, qb);
            approx::assert_relative_eq!(got[0] as f64, r, max_relative = 1e-5);
            let dw = (got[1] as f64 - omega).rem_euclid(std::f64::consts::TAU);
            assert!(
                dw.min(std::f64::consts::TAU - dw) < 1e-3,
                "ω {} vs {omega}",
                got[1]
            );
            for k in 0..3 {
                approx::assert_abs_diff_eq!(got[2 + k] as f64, dir_a[k], epsilon = 1e-4);
                approx::assert_abs_diff_eq!(got[5 + k] as f64, dir_b[k], epsilon = 1e-4);
            }
        }
    }

    /// NaN or −∞ means broken data; packing it as repulsive would hide that.
    #[test]
    fn broken_energies_are_rejected() {
        for bad in [f32::NAN, f32::NEG_INFINITY] {
            let mut table = anisotropic_table(1e-9);
            *table.data.last_mut().unwrap() = bad;
            assert!(GpuTable::new(&table).is_err());
        }
    }

    #[test]
    fn scalar_slabs_and_out_of_range() {
        let Some(gpu) = runner() else { return };
        // Small enough at β = 0.4 to stay scalar rather than repulsive.
        let cpu = constant_table(2.5);
        let (x, y) = (Vector3::x(), Vector3::y());
        assert_eq!(cpu.lookup_boltzmann(7.0, 1.0, &x, &y, 0.4), 2.5);
        let table = GpuTable::new(&cpu).unwrap();
        let results = gpu.run(
            &table,
            &[query(7.0, 1.0, x, y, 0.4), query(3.0, 1.0, x, y, 0.4)],
        );
        assert_eq!(results, [(2.5, false), (0.0, false)]);
    }

    /// At a mesh vertex the other face vertices weigh nothing; a deep well
    /// there must not shift the f32 exponentials into underflow.
    #[test]
    fn zero_weight_vertices_do_not_shift_the_minimum() {
        let Some(gpu) = runner() else { return };
        let mut cpu = anisotropic_table(1e-9);
        cpu.data.fill(0.0);
        let slab = 2 * cpu.n_omega; // r = 7
        let SlabResolution::Mesh { level, .. } = cpu.slab_res[slab] else {
            panic!("slab should be a mesh")
        };
        let lv = &cpu.levels[level as usize];
        let a = Vector3::from(lv.vertices[0]);
        let (face, bary) = lv.locate(&a);
        let unweighted = bary.iter().position(|&w| w == 0.0).expect("a vertex");
        let base = cpu.slab_offsets[slab] as usize;
        let row = base + face[unweighted] * lv.n_vertices;
        cpu.data[row..row + lv.n_vertices].fill(-300.0);
        let b = Vector3::y();
        assert_eq!(cpu.lookup_boltzmann(7.0, 0.0, &a, &b, 0.4), 0.0);
        let table = GpuTable::new(&cpu).unwrap();
        let results = gpu.run(&table, &[query(7.0, 0.0, a, b, 0.4)]);
        assert_eq!(results, [(0.0, false)]);
    }
}
