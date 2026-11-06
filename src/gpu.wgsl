// icotable GPU lookup. The including shader defines:
//   fn icotable_word(i: u32) -> u32
//   fn icotable_float(i: u32) -> f32
// over the arrays of `GpuTable`. Mirrors Table6DAdaptive::lookup with β
// (Boltzmann-weighted interpolation) in f32. See gpu.rs for the layout.

// Layout constants (ICOTABLE_H_*, ICOTABLE_KIND_*, ...) are prepended by
// gpu.rs from its own definitions.

const ICOTABLE_TAU: f32 = 6.283185307179586;

/// Energy in the table's units, plus a flag: 1 for a repulsive (forbidden)
/// configuration, else 0. Out of the table's R range gives (0, 0), as on the
/// CPU; range rules (contact, tail) are the caller's.
struct IcotableValue {
    energy: f32,
    repulsive: u32,
}

struct IcotableLevel {
    n_vertices: u32,
    grid: u32,
    cells: u32,
    cell_vertices: u32,
    adjacency: u32,
    faces: u32,
    positions: u32,
}

struct IcotableFace {
    vertices: vec3<u32>,
    bary: vec3<f32>,
}

fn icotable_header_f32(i: u32) -> f32 {
    return bitcast<f32>(icotable_word(i));
}

fn icotable_level(level: u32) -> IcotableLevel {
    let at = icotable_word(ICOTABLE_H_LEVELS) + ICOTABLE_LEVEL_LEN * level;
    return IcotableLevel(
        icotable_word(at), icotable_word(at + 1u), icotable_word(at + 2u),
        icotable_word(at + 3u), icotable_word(at + 4u), icotable_word(at + 5u),
        icotable_word(at + 6u),
    );
}

fn icotable_vertex(lv: IcotableLevel, i: u32) -> vec3<f32> {
    let at = lv.positions + 3u * i;
    return vec3<f32>(icotable_float(at), icotable_float(at + 1u), icotable_float(at + 2u));
}

/// Cube face and in-face (u, w) of a direction, as flat.rs cube_project.
fn icotable_cube_cell(lv: IcotableLevel, v: vec3<f32>) -> u32 {
    let a = abs(v);
    var face = 0u;
    var u = 0.0;
    var w = 0.0;
    var d = 1.0;
    if a.x >= a.y && a.x >= a.z {
        if v.x > 0.0 { face = 0u; u = v.y; w = v.z; d = v.x; }
        else { face = 1u; u = -v.y; w = v.z; d = -v.x; }
    } else if a.y >= a.z {
        if v.y > 0.0 { face = 2u; u = v.x; w = v.z; d = v.y; }
        else { face = 3u; u = -v.x; w = v.z; d = -v.y; }
    } else if v.z > 0.0 {
        face = 4u; u = v.x; w = v.y; d = v.z;
    } else {
        face = 5u; u = -v.x; w = v.y; d = -v.z;
    }
    let g = f32(lv.grid);
    let ci = u32(clamp((u / d * 0.5 + 0.5) * g, 0.0, g - 1.0));
    let cj = u32(clamp((w / d * 0.5 + 0.5) * g, 0.0, g - 1.0));
    return face * lv.grid * lv.grid + ci * lv.grid + cj;
}

fn icotable_nearest_vertex(lv: IcotableLevel, dir: vec3<f32>) -> u32 {
    let cell = icotable_cube_cell(lv, dir);
    let start = icotable_word(lv.cells + cell);
    let end = icotable_word(lv.cells + cell + 1u);
    var best = 0u;
    var best_d = 1e30;
    for (var k = start; k < end; k++) {
        let v = icotable_word(lv.cell_vertices + k);
        let diff = dir - icotable_vertex(lv, v);
        let d = dot(diff, diff);
        if d < best_d { best_d = d; best = v; }
    }
    return best;
}

/// Projected barycentric coordinates of p in triangle abc, clamped to the
/// triangle (as flat.rs projected_barycentric).
fn icotable_barycentric(p: vec3<f32>, a: vec3<f32>, b: vec3<f32>, c: vec3<f32>) -> vec3<f32> {
    let ab = b - a;
    let ac = c - a;
    let ap = p - a;
    let d1 = dot(ab, ap);
    let d2 = dot(ac, ap);
    if d1 <= 0.0 && d2 <= 0.0 { return vec3<f32>(1.0, 0.0, 0.0); }
    let bp = p - b;
    let d3 = dot(ab, bp);
    let d4 = dot(ac, bp);
    if d3 >= 0.0 && d4 <= d3 { return vec3<f32>(0.0, 1.0, 0.0); }
    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        let v = d1 / (d1 - d3);
        return vec3<f32>(1.0 - v, v, 0.0);
    }
    let cp = p - c;
    let d5 = dot(ab, cp);
    let d6 = dot(ac, cp);
    if d6 >= 0.0 && d5 <= d6 { return vec3<f32>(0.0, 0.0, 1.0); }
    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        let w = d2 / (d2 - d6);
        return vec3<f32>(1.0 - w, 0.0, w);
    }
    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
        let w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return vec3<f32>(0.0, 1.0 - w, w);
    }
    let denom = 1.0 / (va + vb + vc);
    let v = vb * denom;
    let w = vc * denom;
    return vec3<f32>(1.0 - v - w, v, w);
}

/// Face around the nearest vertex that best contains `dir`, as FaceGrid::locate.
fn icotable_locate(lv: IcotableLevel, dir: vec3<f32>) -> IcotableFace {
    let nearest = icotable_nearest_vertex(lv, dir);
    let start = icotable_word(lv.adjacency + nearest);
    let end = icotable_word(lv.adjacency + nearest + 1u);
    var best = IcotableFace(vec3<u32>(0u), vec3<f32>(0.0));
    var best_min = -1e30;
    for (var f = start; f < end; f++) {
        let at = lv.faces + 3u * f;
        let ids = vec3<u32>(icotable_word(at), icotable_word(at + 1u), icotable_word(at + 2u));
        let bary = icotable_barycentric(
            dir, icotable_vertex(lv, ids.x), icotable_vertex(lv, ids.y), icotable_vertex(lv, ids.z),
        );
        let m = min(bary.x, min(bary.y, bary.z));
        if m > best_min { best_min = m; best = IcotableFace(ids, bary); }
    }
    return best;
}

fn icotable_repulsive() -> IcotableValue {
    return IcotableValue(ICOTABLE_REPULSIVE, 1u);
}

fn icotable_value(u: f32) -> IcotableValue {
    if u >= ICOTABLE_REPULSIVE { return icotable_repulsive(); }
    return IcotableValue(u, 0u);
}

/// Nearest R and ω bin, as Table6DAdaptive::resolve_slab_idx; -1 outside R.
fn icotable_slab(r: f32, omega: f32) -> i32 {
    let rmin = icotable_header_f32(ICOTABLE_H_RMIN);
    let rmax = icotable_header_f32(ICOTABLE_H_RMAX);
    if r < rmin || r > rmax { return -1; }
    let n_r = icotable_word(ICOTABLE_H_N_R);
    let n_omega = icotable_word(ICOTABLE_H_N_OMEGA);
    let ri = min(u32((r - rmin) / icotable_header_f32(ICOTABLE_H_DR) + 0.5), n_r - 1u);
    let wrapped = omega - ICOTABLE_TAU * floor(omega / ICOTABLE_TAU);
    let oi = u32(wrapped / icotable_header_f32(ICOTABLE_H_OMEGA_STEP) + 0.5) % n_omega;
    return i32(ri * n_omega + oi);
}

fn icotable_mesh_value(base: u32, n_v: u32, a: u32, b: u32) -> f32 {
    return icotable_float(base + a * n_v + b);
}

/// Boltzmann-weighted interpolation over the 3×3 face-vertex values, shifted
/// by the minimum so the exponentials stay in range.
fn icotable_interpolate(lv: IcotableLevel, base: u32, dir_a: vec3<f32>, dir_b: vec3<f32>, beta: f32) -> IcotableValue {
    let fa = icotable_locate(lv, dir_a);
    let fb = icotable_locate(lv, dir_b);
    var vals: array<f32, 9>;
    var u_min = ICOTABLE_REPULSIVE;
    for (var i = 0u; i < 3u; i++) {
        for (var j = 0u; j < 3u; j++) {
            let v = icotable_mesh_value(base, lv.n_vertices, fa.vertices[i], fb.vertices[j]);
            vals[i * 3u + j] = v;
            u_min = min(u_min, v);
        }
    }
    if u_min >= ICOTABLE_REPULSIVE { return icotable_repulsive(); }
    var sum = 0.0;
    for (var i = 0u; i < 3u; i++) {
        for (var j = 0u; j < 3u; j++) {
            sum += fa.bary[i] * fb.bary[j] * exp(-beta * (vals[i * 3u + j] - u_min));
        }
    }
    if sum <= 0.0 { return icotable_repulsive(); }
    return icotable_value(u_min - log(sum) / beta);
}

/// Table energy at (r, ω, dir_a, dir_b) with Boltzmann weighting at `beta`.
fn icotable_lookup(r: f32, omega: f32, dir_a: vec3<f32>, dir_b: vec3<f32>, beta: f32) -> IcotableValue {
    let slab = icotable_slab(r, omega);
    if slab < 0 { return IcotableValue(0.0, 0u); }
    let at = icotable_word(ICOTABLE_H_SLABS) + 2u * u32(slab);
    let tag = icotable_word(at);
    let kind = tag & ICOTABLE_KIND_MASK;
    if kind == ICOTABLE_KIND_REPULSIVE { return icotable_repulsive(); }
    if kind == ICOTABLE_KIND_SCALAR { return icotable_value(bitcast<f32>(icotable_word(at + 1u))); }
    let lv = icotable_level(tag >> ICOTABLE_LEVEL_SHIFT);
    let base = icotable_word(at + 1u);
    if kind == ICOTABLE_KIND_NEAREST {
        let a = icotable_nearest_vertex(lv, normalize(dir_a));
        let b = icotable_nearest_vertex(lv, normalize(dir_b));
        return icotable_value(icotable_mesh_value(base, lv.n_vertices, a, b));
    }
    return icotable_interpolate(lv, base, normalize(dir_a), normalize(dir_b), beta);
}

// --- Lab frame to table coordinates (orient.rs inverse_orient) ---
// Quaternions are vec4(x, y, z, w), i.e. (i, j, k, w).

/// Table coordinates of a pair: separation, dihedral ω, and the two
/// body-frame directions.
struct IcotablePose {
    r: f32,
    omega: f32,
    dir_a: vec3<f32>,
    dir_b: vec3<f32>,
}

fn icotable_qmul(a: vec4<f32>, b: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(a.w * b.xyz + b.w * a.xyz + cross(a.xyz, b.xyz), a.w * b.w - dot(a.xyz, b.xyz));
}

fn icotable_qconj(q: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(-q.xyz, q.w);
}

fn icotable_rotate(q: vec4<f32>, v: vec3<f32>) -> vec3<f32> {
    let t = 2.0 * cross(q.xyz, v);
    return v + q.w * t + cross(q.xyz, t);
}

/// Shortest rotation taking `u` to `v`; a half turn about a
/// perpendicular axis when they are anti-parallel.
fn icotable_rotation_between(u: vec3<f32>, v: vec3<f32>) -> vec4<f32> {
    let d = dot(u, v);
    if d < -1.0 + 1e-6 {
        var perp = vec3<f32>(0.0, 1.0, 0.0);
        if abs(u.x) < 0.9 { perp = vec3<f32>(1.0, 0.0, 0.0); }
        return vec4<f32>(normalize(cross(u, perp)), 0.0);
    }
    return normalize(vec4<f32>(cross(u, v), 1.0 + d));
}

/// Swing-twist decomposition about z gives ω as the twist and dir_b from
/// the swing, as on the CPU.
fn icotable_inverse_orient(separation: vec3<f32>, q_a: vec4<f32>, q_b: vec4<f32>) -> IcotablePose {
    let z = vec3<f32>(0.0, 0.0, 1.0);
    let r = length(separation);
    if r < 1e-12 { return IcotablePose(0.0, 0.0, z, z); }
    let q_a_inv = icotable_qconj(q_a);
    let dir_a = icotable_rotate(q_a_inv, separation) / r;
    let q3 = icotable_rotation_between(z, dir_a);
    let q_rel = icotable_qmul(icotable_qmul(icotable_qconj(q3), q_a_inv), q_b);
    let twist_raw = vec4<f32>(0.0, 0.0, q_rel.z, q_rel.w);
    var twist = vec4<f32>(0.0, 0.0, 0.0, 1.0);
    var omega = 0.0;
    if length(twist_raw) > 1e-12 {
        twist = normalize(twist_raw);
        omega = 2.0 * atan2(q_rel.z, q_rel.w);
    }
    omega = omega - ICOTABLE_TAU * floor(omega / ICOTABLE_TAU);
    let q1 = icotable_qmul(q_rel, icotable_qconj(twist));
    let dir_b = normalize(icotable_rotate(icotable_qconj(q1), -z));
    return IcotablePose(r, omega, dir_a, dir_b);
}
