// src/predictor.rs
//
// BitDrop v2 predictor engine (MAX‑UNIFIED compatible)
// - Zero global state, zero I/O, zero locks
// - Pure heuristics: cannot hurt correctness
// - collapse.rs expects:
//       predict_for_cluster(&cluster) -> Option<ClusterHint>
//       train_from_cluster(&cluster, structured, quant_bits, pass_limit, merge_threshold)

use crate::blocks::Cube;
use crate::metrics::score_bytes_for_zlib;

/// Hint object used by collapse.rs
#[derive(Clone, Copy, Debug)]
pub struct ClusterHint {
    pub confidence: f32,
    pub structured_hint: bool,
    pub quant_bits_hint: u8,
    pub pass_limit_hint: usize,
    pub merge_threshold_hint: i64,
}

#[inline]
fn sample_cluster_bytes(cubes: &[Cube], max_bytes: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(max_bytes.min(4096));
    for c in cubes {
        let bytes = c.bytes();
        if out.len() >= max_bytes {
            break;
        }

        let remaining = max_bytes - out.len();
        if bytes.len() <= remaining {
            out.extend_from_slice(bytes);
        } else {
            out.extend_from_slice(&bytes[..remaining]);
        }
    }
    out
}

#[inline]
fn cluster_size_bytes(cubes: &[Cube]) -> usize {
    cubes.iter().map(|c| c.bytes().len()).sum()
}

#[inline]
fn cluster_compressibility_score(cubes: &[Cube]) -> i32 {
    const MAX_SAMPLE: usize = 4096;
    let sample = sample_cluster_bytes(cubes, MAX_SAMPLE);
    if sample.is_empty() {
        return 0;
    }
    score_bytes_for_zlib(&sample)
}

/// Main prediction entry point.
pub fn predict_for_cluster(cubes: &[Cube]) -> Option<ClusterHint> {
    let total_bytes = cluster_size_bytes(cubes);

    // ============================================================
    // Disable predictor for small/mid clusters (< 64 KB)
    // This restores speed for:
    //   - small text
    //   - random 64 KB
    //   - mid-size buffers
    // ============================================================
    if total_bytes < 64 * 1024 {
        return None;
    }

    let score = cluster_compressibility_score(cubes);
    let is_huge = total_bytes >= 8 * 1024 * 1024;

    let mut hint = ClusterHint {
        confidence: 0.0,
        structured_hint: false,
        quant_bits_hint: 0,
        pass_limit_hint: 0,
        merge_threshold_hint: 0,
    };

    // EXTREMELY COMPRESSIBLE (very strong signal)
    if score <= 0 {
        hint.confidence = if is_huge { 0.99 } else { 0.98 };
        hint.structured_hint = true;
        hint.quant_bits_hint = 0;
        hint.pass_limit_hint = if is_huge { 10 } else { 8 }; // slightly deeper on huge clusters
        hint.merge_threshold_hint = -6;
        return Some(hint);
    }

    // MODERATELY COMPRESSIBLE (safe middle band)
    if score <= 20 {
        hint.confidence = if is_huge { 0.95 } else { 0.92 };
        hint.structured_hint = true;
        hint.quant_bits_hint = 0;
        hint.pass_limit_hint = if is_huge { 6 } else { 4 }; // a bit more depth for big structured clusters
        hint.merge_threshold_hint = -3;
        return Some(hint);
    }

    // LIGHTLY STRUCTURED (weak but useful signal)
    if score <= 40 {
        hint.confidence = 0.80;
        hint.structured_hint = true;
        hint.quant_bits_hint = 0;
        hint.pass_limit_hint = if is_huge { 3 } else { 2 };
        hint.merge_threshold_hint = -1;
        return Some(hint);
    }

    // VERY RANDOM (strong skip signal)
    if score > 60 {
        hint.confidence = 0.95;
        hint.structured_hint = false;
        hint.quant_bits_hint = 0;
        hint.pass_limit_hint = 1;      // shallow collapse
        hint.merge_threshold_hint = 0;
        return Some(hint);
    }

    // MILDLY RANDOM (middle-high band)
    if score > 40 {
        hint.confidence = 0.90;
        hint.structured_hint = false;
        hint.quant_bits_hint = 0;
        hint.pass_limit_hint = 2;      // small speed boost
        hint.merge_threshold_hint = 0;
        return Some(hint);
    }

    // Weak signal → let collapse.rs decide
    None
}

/// Training hook (currently a no‑op).
pub fn train_from_cluster(
    _cubes: &[Cube],
    _structured: bool,
    _quant_bits: u8,
    _pass_limit: usize,
    _merge_threshold: i64,
) {
    // Intentionally empty. Stable API surface for future learning.
}







