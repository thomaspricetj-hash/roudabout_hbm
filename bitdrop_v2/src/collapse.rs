use crate::blocks::{Cube, CubeData, Orientation, rotate_cube_data};
use crate::metrics::score_bytes_for_zlib;
use crate::transform::{Transform, TransformLog};
use crate::gpu::backend::gpu_available;
use crate::predictor::{predict_for_cluster, train_from_cluster};
use crate::model::global::GlobalModel;
use rayon::prelude::*;

#[inline]
pub fn cube_signature(c: &Cube) -> u64 {
    use std::hash::Hasher;
    let mut h = ahash::AHasher::default();
    h.write(c.bytes());
    h.write_u64(
        ((c.shape.0 as u64) << 40)
            | ((c.shape.1 as u64) << 20)
            | (c.shape.2 as u64),
    );
    h.write_u8(c.quant_bits);
    h.write_u8(c.quant_vmin);
    h.write_u8(c.quant_vmax);
    h.finish()
}

/// Stable, ID‑independent cluster signature.
/// - independent of cube IDs
/// - independent of positions
/// - independent of base_id
/// - depends only on cube content, shape, quantization and target layer
#[inline]
pub fn compute_cluster_signature(cluster: &[Cube], target_layer: u16) -> u64 {
    use std::hash::Hasher;

    // Collect per‑cube stable signatures
    let mut sigs: Vec<u64> = cluster.iter().map(cube_signature).collect();

    // Make order‑independent
    sigs.sort_unstable();

    let mut h = ahash::AHasher::default();
    h.write_u16(target_layer);
    h.write_u64(cluster.len() as u64);

    for s in sigs {
        h.write_u64(s);
    }

    h.finish()
}

#[inline]
pub fn finalize_cube_for_cache(_c: &mut Cube) {}

#[inline]
fn compute_vmin_vmax(data: &[u8]) -> (u8, u8) {
    let mut vmin = 255u8;
    let mut vmax = 0u8;
    for &v in data {
        if v < vmin { vmin = v; }
        if v > vmax { vmax = v; }
    }
    (vmin, vmax)
}

#[inline]
fn quantize_block_linear(data: &[u8], bits: u8, vmin: u8, vmax: u8) -> Vec<u8> {
    if bits == 0 || vmin == vmax {
        return data.to_vec();
    }
    let levels = (1u32 << bits) - 1;
    let range = (vmax as f32 - vmin as f32).max(1.0);
    let scale = range / levels as f32;
    let mut out = Vec::with_capacity(data.len());
    for &v in data {
        let dv = v as f32 - vmin as f32;
        let q = (dv / scale).round().clamp(0.0, levels as f32) as u8;
        out.push(q);
    }
    out
}

#[inline]
fn entropy_from_hist_u32(hist: &[u32; 256], len: f32) -> f32 {
    if len == 0.0 { return 0.0; }
    let mut e = 0.0;
    for &h in hist {
        if h > 0 {
            let p = h as f32 / len;
            e -= p * p.log2();
        }
    }
    e
}

#[inline]
fn quick_entropy_scalar(bytes: &[u8]) -> f32 {
    let mut hist = [0u32; 256];
    for &b in bytes { hist[b as usize] += 1; }
    entropy_from_hist_u32(&hist, bytes.len() as f32)
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[inline]
unsafe fn quick_entropy_avx2(bytes: &[u8]) -> f32 {
    use core::arch::x86_64::*;
    if bytes.is_empty() { return 0.0; }
    let mut hist = [0u32; 256];
    let mut i = 0;
    let len = bytes.len();
    while i + 32 <= len {
        let ptr = bytes.as_ptr().add(i) as *const __m256i;
        let v = _mm256_loadu_si256(ptr);
        let mut tmp = [0u8; 32];
        _mm256_storeu_si256(tmp.as_mut_ptr() as *mut __m256i, v);
        for &b in &tmp { hist[b as usize] += 1; }
        i += 32;
    }
    for &b in &bytes[i..] { hist[b as usize] += 1; }
    entropy_from_hist_u32(&hist, len as f32)
}

#[inline]
fn quick_entropy(c: &Cube) -> f32 {
    let bytes = c.bytes();
    if bytes.is_empty() { return 0.0; }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if is_x86_feature_detected!("avx2") {
        unsafe { return quick_entropy_avx2(bytes); }
    }
    quick_entropy_scalar(bytes)
}

fn is_structured_data(cluster: &[Cube]) -> bool {
    if cluster.is_empty() { return false; }
    let avg_entropy =
        cluster.iter().map(quick_entropy).sum::<f32>() / cluster.len() as f32;
    if avg_entropy < 4.0 || avg_entropy > 7.0 {
        return false;
    }
    let mut skew_sum = 0u32;
    for c in cluster.iter().take(32) {
        let bytes = c.bytes();
        let mut minb = 255u8;
        let mut maxb = 0u8;
        for &b in bytes {
            if b < minb { minb = b; }
            if b > maxb { maxb = b; }
        }
        skew_sum += (maxb - minb) as u32;
    }
    let avg_skew = skew_sum as f32 / cluster.len().min(32) as f32;
    avg_skew > 20.0
}

fn shift_cube_data(c: &Cube, shift: usize) -> Vec<u8> {
    let mut out = c.bytes().to_vec();
    out.rotate_left(shift & 31);
    out
}

thread_local! {
    static MERGE_SCRATCH: std::cell::RefCell<Vec<u8>> =
        std::cell::RefCell::new(Vec::new());
}

#[inline]
fn base_zlib_delta(a_data: &[u8], b_data: &[u8]) -> i64 {
    let score_before =
        score_bytes_for_zlib(a_data) as i64 +
        score_bytes_for_zlib(b_data) as i64;

    let score_after = MERGE_SCRATCH.with(|buf_cell| {
        let cap = buf_cell.borrow().capacity();
        let mut buf = buf_cell.borrow_mut();
        let needed = a_data.len() + b_data.len();
        if cap < needed {
            buf.reserve(needed - cap);
        }
        buf.clear();
        buf.extend_from_slice(a_data);
        buf.extend_from_slice(b_data);
        score_bytes_for_zlib(&buf) as i64
    });

    score_after - score_before
}

#[inline]
fn hybrid_merge_score(a: &Cube, b: &Cube, structured: bool) -> i64 {
    let quant_bonus = match (a.quant_bits, b.quant_bits) {
        (4,4) => -32,
        (8,8) => -16,
        (4,8)|(8,4) => -12,
        _ => 0,
    };

    let a_bytes = a.bytes();
    let b_bytes = b.bytes();

    let mut best_delta = base_zlib_delta(a_bytes, b_bytes);

    if structured && b_bytes.len() <= 32 * 1024 {
        let mut no_improve = 0;
        for shift in 0..32 {
            let shifted = shift_cube_data(b, shift);
            let delta = base_zlib_delta(a_bytes, &shifted);
            if delta < best_delta {
                best_delta = delta;
                no_improve = 0;
            } else {
                no_improve += 1;
                if no_improve >= 16 {
                    break;
                }
            }
        }
    }

    if a.shape == b.shape {
        let (sx, sy, sz) = a.shape;
        if sx == sy && sy == sz && a_bytes.len() <= 16 * 1024 {
            for ori in Orientation::all().iter().take(6) {
                let rotated = rotate_cube_data(a.shape, b_bytes, *ori);
                let delta = base_zlib_delta(a_bytes, &rotated);
                if delta < best_delta {
                    best_delta = delta;
                }
            }
        }
    }

    let mut delta = best_delta + quant_bonus;

    let avg = (a_bytes.len() + b_bytes.len()) / 2;
    if avg < 4 * 1024 { delta += 8; }
    else if avg > 64 * 1024 { delta -= 16; }

    delta
}

fn merge_multi_cube(
    mut cluster: Vec<Cube>,
    group: Vec<usize>,
    target_layer: u16,
    next_id: &mut u32,
    log: &mut TransformLog,
) -> Vec<Cube> {

    let mut cubes_to_merge = Vec::new();
    for &idx in group.iter().rev() {
        cubes_to_merge.push(cluster.remove(idx));
    }

    cubes_to_merge.sort_by_key(|c| c.id);

    let mut merged_data = Vec::new();
    let mut offsets = Vec::new();
    let mut members = Vec::new();
    let mut shapes = Vec::new();
    let mut positions = Vec::new();
    let mut layers = Vec::new();

    let mut offset = 0;
    for c in &cubes_to_merge {
        let bytes = c.bytes();
        offsets.push(offset);
        offset += bytes.len() as u32;
        merged_data.extend_from_slice(bytes);
        members.push(c.id);
        shapes.push(c.shape);
        positions.push(c.pos);
        layers.push(c.layer);
    }

    let new_id = crate::blocks::CubeId(*next_id);
    *next_id += 1;

    let mut merged = Cube::new(
        new_id,
        target_layer,
        cubes_to_merge[0].pos,
        cubes_to_merge[0].shape,
        merged_data,
    );

    merged.quant_bits = cubes_to_merge.iter().map(|c| c.quant_bits).max().unwrap_or(0);
    merged.quant_vmin = cubes_to_merge.iter().map(|c| c.quant_vmin).min().unwrap_or(0);
    merged.quant_vmax = cubes_to_merge.iter().map(|c| c.quant_vmax).max().unwrap_or(0);

    log.push(Transform::Merge {
        new_cube_id: merged.id,
        layer_from: cubes_to_merge[0].layer,
        layer_to: target_layer,
        members,
        offsets,
        original_positions: positions,
        original_shapes: shapes,
        original_layers: layers,
    });

    finalize_cube_for_cache(&mut merged);
    cluster.push(merged);
    cluster
}

#[inline]
fn adaptive_pass_limit(cluster: &[Cube]) -> usize {
    if cluster.is_empty() { return 0; }
    let avg_size =
        cluster.iter().map(|c| c.bytes().len()).sum::<usize>() / cluster.len();
    let avg_entropy =
        cluster.iter().map(quick_entropy).sum::<f32>() / cluster.len() as f32;

    if avg_size < 4 * 1024 {
        if avg_entropy < 3.0 { 2 } else { 1 }
    } else if avg_size < 16 * 1024 {
        if avg_entropy < 3.0 { 4 } else { 3 }
    } else if avg_size < 64 * 1024 {
        if avg_entropy < 3.0 { 10 } else { 8 }
    } else {
        usize::MAX
    }
}

#[inline]
fn dynamic_merge_threshold(cluster: &[Cube]) -> i64 {
    if cluster.is_empty() { return 0; }
    let avg_size =
        cluster.iter().map(|c| c.bytes().len()).sum::<usize>() / cluster.len();
    let avg_entropy =
        cluster.iter().map(quick_entropy).sum::<f32>() / cluster.len() as f32;

    let mut base = if avg_entropy > 7.0 { -8 }
    else if avg_entropy > 5.0 { -4 }
    else if avg_entropy > 3.0 { -2 }
    else { 0 };

    if avg_size > 64 * 1024 { base -= 8; }
    else if avg_size > 16 * 1024 { base -= 4; }

    base
}

#[inline]
fn choose_quant_bits_for_cluster(cluster: &[Cube]) -> u8 {
    if cluster.is_empty() { return 0; }

    let mut sample = Vec::new();
    for c in cluster.iter().take(4) {
        sample.extend_from_slice(c.bytes());
        if sample.len() >= 16 * 1024 {
            break;
        }
    }
    if sample.len() < 2048 {
        return 0;
    }

    let mut freq = [0u32; 256];
    for &b in &sample {
        freq[b as usize] += 1;
    }

    let len = sample.len() as f64;
    if len == 0.0 {
        return 0;
    }

    let mut h = 0.0;
    for &f in &freq {
        if f != 0 {
            let p = f as f64 / len;
            h -= p * p.log2();
        }
    }

    if h > 7.2 { 0 }
    else if h > 6.2 { 8 }
    else if h > 5.2 { 6 }
    else if h > 4.2 { 5 }
    else { 4 }
}

#[inline]
fn should_use_gpu_for_cluster(cluster: &[Cube]) -> bool {
    if !gpu_available() {
        return false;
    }

    let n = cluster.len();
    if n < 8 {
        return false;
    }

    let pair_count = (n as u64 * (n as u64 - 1)) / 2;
    if pair_count > 5000 {
        return true;
    }

    let total_bytes: usize = cluster.iter().map(|c| c.bytes().len()).sum();
    total_bytes > 256 * 1024
}

#[inline]
fn compute_merge_candidates_parallel(
    cluster: &[Cube],
    structured: bool,
    merge_threshold: i64,
) -> Vec<(usize, usize, i64)> {
    let n = cluster.len();
    if n < 2 {
        return Vec::new();
    }

    (0..n)
        .into_par_iter()
        .map(|i| {
            let mut local = Vec::new();
            for j in (i + 1)..n {
                let s = hybrid_merge_score(&cluster[i], &cluster[j], structured);
                if s < merge_threshold {
                    local.push((i, j, s));
                }
            }
            local
        })
        .reduce(Vec::new, |mut a, mut b| { a.append(&mut b); a })
}

#[inline]
fn greedy_deterministic_matching(
    n: usize,
    mut edges: Vec<(usize, usize, i64)>,
) -> Vec<(usize, usize, i64)> {
    if n < 2 || edges.is_empty() {
        return Vec::new();
    }

    edges.sort_by(|a, b| {
        let sa = a.2;
        let sb = b.2;
        if sa != sb {
            sa.cmp(&sb)
        } else if a.0 != b.0 {
            a.0.cmp(&b.0)
        } else {
            a.1.cmp(&b.1)
        }
    });

    let mut used = vec![false; n];
    let mut matches = Vec::new();

    for (i, j, s) in edges {
        if !used[i] && !used[j] {
            used[i] = true;
            used[j] = true;
            matches.push((i, j, s));
        }
    }

    matches
}

pub fn collapse_cluster(
    mut cluster: Vec<Cube>,
    target_layer: u16,
    next_id: &mut u32,
    log: &mut TransformLog,
    global_model: &GlobalModel,
) -> Vec<Cube> {

    if cluster.is_empty() {
        return Vec::new();
    }

    if cluster.len() == 1 {
        let mut c = cluster.remove(0);
        if c.layer != target_layer {
            log.push(Transform::DropLayer {
                cube_id: c.id,
                from_layer: c.layer,
                to_layer: target_layer,
            });
            c.layer = target_layer;
        }
        finalize_cube_for_cache(&mut c);
        return vec![c];
    }

    let total_bytes: usize = cluster.iter().map(|c| c.bytes().len()).sum();
    let small_cluster = total_bytes < 64 * 1024;

    let avg_entropy = if small_cluster {
        0.0
    } else {
        cluster.iter().map(quick_entropy).sum::<f32>() / cluster.len() as f32
    };

    let avg_size = if cluster.is_empty() {
        0
    } else {
        total_bytes / cluster.len()
    };

    let mut structured = if small_cluster {
        false
    } else {
        is_structured_data(&cluster)
    };

    let mut pass_limit_hint: Option<usize> = None;
    let mut merge_threshold_hint: Option<i64> = None;
    let mut quant_bits_override: Option<u8> = None;

    if !small_cluster {
        if let Some((pl, th, qb)) = global_model.get_hint(avg_entropy, avg_size) {
            if pl > 0 {
                pass_limit_hint = Some(pl);
            }
            merge_threshold_hint = Some(th);
            if qb > 0 {
                quant_bits_override = Some(qb);
            }
        }

        if let Some(hint) = predict_for_cluster(&cluster) {
            if hint.confidence >= 0.75 {
                structured = hint.structured_hint;
                if hint.pass_limit_hint > 0 {
                    pass_limit_hint = Some(hint.pass_limit_hint);
                }
                if merge_threshold_hint.is_none() {
                    merge_threshold_hint = Some(hint.merge_threshold_hint);
                }
                if hint.quant_bits_hint > 0 && quant_bits_override.is_none() {
                    quant_bits_override = Some(hint.quant_bits_hint);
                }
            }
        }
    }

    let mut quant_bits = if !small_cluster && avg_entropy <= 6.5 {
        if let Some(qb) = global_model.suggest_quant_bits(avg_entropy) {
            qb
        } else {
            choose_quant_bits_for_cluster(&cluster)
        }
    } else {
        0
    };

    if let Some(qb) = quant_bits_override {
        quant_bits = qb;
    }

    if quant_bits != 0 {
        let mut global_vmin = 255u8;
        let mut global_vmax = 0u8;

        for c in &cluster {
            let (vmin, vmax) = compute_vmin_vmax(c.bytes());
            if vmin < global_vmin { global_vmin = vmin; }
            if vmax > global_vmax { global_vmax = vmax; }
        }

        for c in &mut cluster {
            let q = quantize_block_linear(c.bytes(), quant_bits, global_vmin, global_vmax);
            c.quant_bits = quant_bits;
            c.quant_vmin = global_vmin;
            c.quant_vmax = global_vmax;
            c.data = CubeData::Raw(q);
        }
    }

    let mut merge_threshold = dynamic_merge_threshold(&cluster);
    if structured {
        merge_threshold -= 6;
    }

    if let Some(th) = merge_threshold_hint {
        merge_threshold = th;
    }

    let mut pass_limit = adaptive_pass_limit(&cluster);
    if structured {
        pass_limit = pass_limit * 5 / 4;
    }

    if let Some(pl) = pass_limit_hint {
        pass_limit = pl;
    }

    let mut passes = 0;

    loop {
        if cluster.len() <= 1 {
            break;
        }
        if passes >= pass_limit {
            break;
        }
        passes += 1;

        let _use_gpu = should_use_gpu_for_cluster(&cluster);

        let candidates = compute_merge_candidates_parallel(&cluster, structured, merge_threshold);
        if candidates.is_empty() {
            break;
        }

        let matches = greedy_deterministic_matching(cluster.len(), candidates);
        if matches.is_empty() {
            break;
        }

        let old_cluster = cluster;
        let n = old_cluster.len();
        let mut consumed = vec![false; n];
        let mut new_cluster = Vec::with_capacity(n);

        for (i, j, _score) in matches {
            if consumed[i] || consumed[j] {
                continue;
            }
            consumed[i] = true;
            consumed[j] = true;

            let a = &old_cluster[i];
            let b = &old_cluster[j];

            let a_bytes = a.bytes();
            let b_bytes = b.bytes();

            let mut merged_data = Vec::with_capacity(a_bytes.len() + b_bytes.len());
            merged_data.extend_from_slice(a_bytes);
            merged_data.extend_from_slice(b_bytes);

            let new_id = crate::blocks::CubeId(*next_id);
            *next_id += 1;

            let mut merged = Cube::new(new_id, target_layer, a.pos, a.shape, merged_data);

            merged.quant_bits = a.quant_bits.max(b.quant_bits);
            merged.quant_vmin = a.quant_vmin.min(b.quant_vmin);
            merged.quant_vmax = a.quant_vmax.max(b.quant_vmax);

            log.push(Transform::Merge {
                new_cube_id: merged.id,
                layer_from: a.layer,
                layer_to: target_layer,
                members: vec![a.id, b.id],
                offsets: vec![0, a_bytes.len() as u32],
                original_positions: vec![a.pos, b.pos],
                original_shapes: vec![a.shape, b.shape],
                original_layers: vec![a.layer, b.layer],
            });

            finalize_cube_for_cache(&mut merged);
            new_cluster.push(merged);
        }

        for idx in 0..n {
            if !consumed[idx] {
                new_cluster.push(old_cluster[idx].clone());
            }
        }

        cluster = new_cluster;
    }

    train_from_cluster(&cluster, structured, quant_bits, pass_limit, merge_threshold);

    global_model.update_cluster_observation(
        avg_entropy,
        avg_size,
        pass_limit,
        merge_threshold,
        quant_bits,
    );

    cluster
}
