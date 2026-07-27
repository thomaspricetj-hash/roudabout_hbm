// ============================================================
// BitDrop3DEngine — Hybrid BitDrop3D + zstd (parallel, adaptive)
// ============================================================

use std::{collections::HashMap, fs, path::PathBuf};

use dashmap::DashMap;
use rayon::prelude::*;
use chrono::Utc;
use directories::ProjectDirs;
use serde::{Deserialize, Serialize};

use crate::blocks::{
    flatten_from_cubes,
    inverse_orientation,
    lift_to_cubes,
    rotate_cube_data,
    Cube,
    CubeId,
    CubeData,
};
use crate::cluster::cluster_cubes;
use crate::collapse::{collapse_cluster, cube_signature};
use crate::container::{pack_cubes_and_log, unpack_cubes_and_log};
use crate::metrics::choose_best_orientation;
use crate::transform::{apply_shift, inverse_shift, PatternTag, Transform, TransformLog};
use crate::librarian::{Librarian, LibrarianDecision, PayloadStats};
use crate::predictor::{predict_for_cluster, train_from_cluster, ClusterHint};

// Hybrid backend
use zstd::stream::{encode_all as zstd_enc, decode_all as zstd_dec};

// ---------------------------------------------------------
// Debug flag
// ---------------------------------------------------------
const DEBUG_LOG: bool = false;

macro_rules! debug {
    ($($arg:tt)*) => {
        if DEBUG_LOG {
            println!($($arg)*);
        }
    };
}

// ---------------------------------------------------------
// Internal auto-adaptive compression profile
// ---------------------------------------------------------
#[derive(Clone, Copy, Debug)]
enum AutoProfile {
    ZstdFast,
    LosslessBD3D,
    SemBD3D,
}

// ---------------------------------------------------------
// Skimming decision
// ---------------------------------------------------------
#[derive(Clone, Copy, Debug)]
enum SkimDecision {
    ForceZstd,
    PreferZstd,
    Normal,
}

// ---------------------------------------------------------
// Persistent collapse cache types
// ---------------------------------------------------------
const BITDROP_CACHE_VERSION: &str = "bitdrop-cache-v1";

#[derive(Serialize, Deserialize, Clone)]
struct CollapseCacheEntry {
    cubes: Vec<Cube>,
    transforms: Vec<Transform>,
    created_ts: i64,
    last_used_ts: i64,
    hits: u32,
}

#[derive(Serialize, Deserialize, Clone)]
struct CubeCollapseCacheEntry {
    cube: Cube,
    transforms: Vec<Transform>,
    created_ts: i64,
    last_used_ts: i64,
    hits: u32,
}

#[derive(Serialize, Deserialize)]
struct CollapseCacheFile {
    version: String,
    entries: Vec<(u64, CollapseCacheEntry)>,
}

// ---------------------------------------------------------
// Engine struct
// ---------------------------------------------------------
pub struct BitDrop3DEngine {
    pub block_shape: (usize, usize, usize),
    pub max_layers: u16,
    pub vector_stride: Option<usize>,
    collapse_cache: DashMap<u64, CollapseCacheEntry>,
    cube_cache: DashMap<u64, CubeCollapseCacheEntry>,
}

// -----------------------------
// Byte metrics helpers
// -----------------------------
#[inline]
fn byte_entropy_score(data: &[u8]) -> f64 {
    if data.is_empty() {
        return 0.0;
    }
    let mut freq = [0u32; 256];
    for &b in data {
        freq[b as usize] += 1;
    }
    let len = data.len() as f64;
    let mut h = 0.0;
    for &c in &freq {
        if c != 0 {
            let p = c as f64 / len;
            h -= p * p.log2();
        }
    }
    h
}

#[inline]
fn size_tier(len: usize) -> u8 {
    if len < 512 * 1024 {
        0
    } else if len < 8 * 1024 * 1024 {
        1
    } else {
        2
    }
}

// ---------------------------------------------------------
// Lightweight payload index
// ---------------------------------------------------------
#[derive(Clone, Copy, Debug)]
struct PayloadIndex {
    len: usize,
    entropy: f64,
    tier: u8,
}

#[inline]
fn build_payload_index(payload: &[u8]) -> PayloadIndex {
    let len = payload.len();
    let entropy = byte_entropy_score(payload);
    let tier = size_tier(len);
    PayloadIndex { len, entropy, tier }
}

#[inline]
fn looks_already_compressed(payload: &[u8], idx: &PayloadIndex) -> bool {
    if idx.len < 4 || idx.entropy < 7.0 {
        return false;
    }
    let magic = &payload[..4];
    matches!(
        magic,
        [0x1F, 0x8B, ..] |
        [0x78, 0x01, ..] |
        [0x78, 0x9C, ..] |
        [0x78, 0xDA, ..] |
        [0x28, 0xB5, 0x2F, 0xFD]
    )
}

#[inline]
fn cube_entropy_score(bytes: &[u8]) -> f64 {
    let n = bytes.len().min(1024);
    if n == 0 {
        return 0.0;
    }
    let mut freq = [0u32; 256];
    for &b in &bytes[..n] {
        freq[b as usize] += 1;
    }
    let len = n as f64;
    let mut h = 0.0;
    for &c in &freq {
        if c != 0 {
            let p = c as f64 / len;
            h -= p * p.log2();
        }
    }
    h
}

#[inline]
fn compute_cluster_signature(cluster: &[Cube], target_layer: u16, base_id: u32) -> u64 {
    use std::hash::Hasher;

    let mut sigs: Vec<u64> = cluster
        .iter()
        .map(|c| {
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
        })
        .collect();

    sigs.sort_unstable();

    let mut h = ahash::AHasher::default();
    h.write_u16(target_layer);
    h.write_u64(cluster.len() as u64);
    h.write_u32(base_id);

    for s in sigs {
        h.write_u64(s);
    }

    h.finish()
}

// ---------------------------------------------------------
// Cache file path helper
// ---------------------------------------------------------
fn cache_file_path() -> Option<PathBuf> {
    let proj = ProjectDirs::from("ai", "SyntheticMind", "BitDrop3D")?;
    let dir = proj.cache_dir();
    let _ = fs::create_dir_all(dir);
    Some(dir.join("collapse_cache.bin"))
}

// ---------------------------------------------------------
// Engine Implementation
// ---------------------------------------------------------
impl BitDrop3DEngine {
    pub fn new(block_shape: (usize, usize, usize), max_layers: u16) -> Self {
        let engine = Self {
            block_shape,
            max_layers,
            vector_stride: Some(128),
            collapse_cache: DashMap::new(),
            cube_cache: DashMap::new(),
        };
        engine.load_collapse_cache();
        engine
    }

    fn load_collapse_cache(&self) {
        let Some(path) = cache_file_path() else {
            return;
        };
        let Ok(bytes) = fs::read(&path) else {
            return;
        };
        let Ok(file): Result<CollapseCacheFile, _> = bincode::deserialize(&bytes) else {
            return;
        };
        if file.version != BITDROP_CACHE_VERSION {
            return;
        }
        for (key, entry) in file.entries {
            self.collapse_cache.insert(key, entry);
        }
    }

    fn decay_and_collect(&self) -> Vec<(u64, CollapseCacheEntry)> {
        let now = Utc::now().timestamp();
        let mut kept = Vec::new();
        for kv in self.collapse_cache.iter() {
            let key = *kv.key();
            let entry = kv.value();
            let age_days = ((now - entry.last_used_ts) as f64 / 86400.0).max(0.0);
            let score = (entry.hits as f64) / (1.0 + age_days);
            if score >= 0.01 {
                kept.push((key, entry.clone()));
            }
        }
        kept
    }

    pub fn flush_collapse_cache(&self) {
        let Some(path) = cache_file_path() else {
            return;
        };
        let entries = self.decay_and_collect();
        let file = CollapseCacheFile {
            version: BITDROP_CACHE_VERSION.to_string(),
            entries,
        };
        if let Ok(bytes) = bincode::serialize(&file) {
            let _ = fs::write(path, bytes);
        }
    }

    #[inline]
    fn auto_block_shape(&self, _payload: &[u8], idx: &PayloadIndex) -> (usize, usize, usize) {
        let h = idx.entropy;
        let n = idx.len;
        let tier = idx.tier;
        let (bx, by, bz) = self.block_shape;

        match tier {
            0 => {
                if n < 16 * 1024 {
                    (bx.min(4).max(2), by.min(4).max(2), bz.min(64).max(16))
                } else if h < 4.0 {
                    (bx.max(4), by.max(4), (bz * 2).min(256).max(64))
                } else {
                    (bx.max(4), by.max(4), bz.max(64).min(128))
                }
            }
            1 => {
                if h < 4.0 {
                    (bx.max(4), by.max(4), (bz * 2).min(256).max(64))
                } else if h < 6.0 {
                    (bx.max(4), by.max(4), bz.max(64).min(128))
                } else {
                    (bx.max(4), by.max(4), bz.min(64).max(32))
                }
            }
            _ => {
                if h < 4.0 {
                    (bx.max(8), by.max(8), (bz * 4).min(512).max(128))
                } else {
                    (bx.max(8), by.max(8), bz.max(128).min(256))
                }
            }
        }
    }

    #[inline]
    fn auto_max_layers(&self, idx: &PayloadIndex) -> u16 {
        let h = idx.entropy;
        let n = idx.len;
        let tier = idx.tier;

        match tier {
            0 => {
                if n < 32 * 1024 {
                    self.max_layers.min(2)
                } else if h < 4.0 {
                    self.max_layers
                } else if h < 6.0 {
                    self.max_layers.min(4)
                } else {
                    self.max_layers.min(3)
                }
            }
            1 => {
                if n < 32 * 1024 {
                    self.max_layers.min(2)
                } else if h < 4.0 {
                    self.max_layers
                } else if h < 6.0 {
                    self.max_layers.min(4)
                } else {
                    self.max_layers.min(2)
                }
            }
            _ => {
                if h < 4.0 {
                    self.max_layers.min(3)
                } else if h < 6.0 {
                    self.max_layers.min(2)
                } else {
                    self.max_layers.min(1)
                }
            }
        }
    }

    #[inline]
    fn score_bytes_for_zlib_local(&self, data: &[u8]) -> i32 {
        use crate::metrics::score_bytes_for_zlib;
        score_bytes_for_zlib(data)
    }

    #[inline]
    fn score_bytes_for_zlib_global(&self, data: &[u8]) -> i32 {
        const MAX_SAMPLE: usize = 64 * 1024;
        if data.is_empty() {
            return 0;
        }
        let sample_len = data.len().min(MAX_SAMPLE);
        self.score_bytes_for_zlib_local(&data[..sample_len])
    }

    #[inline]
    fn vector_stride_or_zero(&self) -> usize {
        self.vector_stride.unwrap_or(0)
    }

    #[inline]
    fn forward_vector_delta(&self, data: &[u8]) -> Vec<u8> {
        let stride = self.vector_stride_or_zero();
        if data.is_empty() || stride == 0 {
            return data.to_vec();
        }
        let n = data.len();
        let mut out = vec![0u8; n];
        let mut base = 0;
        while base < n {
            let end = (base + stride).min(n);
            let mut prev = 0u8;
            for i in base..end {
                let v = data[i];
                let d = v.wrapping_sub(prev);
                out[i] = d;
                prev = v;
            }
            base += stride;
        }
        out
    }

    #[inline]
    fn inverse_vector_delta(&self, data: &[u8]) -> Vec<u8> {
        let stride = self.vector_stride_or_zero();
        if data.is_empty() || stride == 0 {
            return data.to_vec();
        }
        let n = data.len();
        let mut out = vec![0u8; n];
        let mut base = 0;
        while base < n {
            let end = (base + stride).min(n);
            let mut prev = 0u8;
            for i in base..end {
                let d = data[i];
                let v = d.wrapping_add(prev);
                out[i] = v;
                prev = v;
            }
            base += stride;
        }
        out
    }

    #[inline]
    fn build_dim_permutation(&self, data: &[u8]) -> Vec<usize> {
        let stride = self.vector_stride_or_zero();
        if stride == 0 {
            return Vec::new();
        }
        if data.len() < 2 * stride {
            return (0..stride).collect();
        }

        let mut counts = vec![0usize; stride];
        let mut sums = vec![0.0f64; stride];
        let mut sums_sq = vec![0.0f64; stride];

        let n = data.len();
        let mut i = 0;
        while i + stride <= n {
            let row = &data[i..i + stride];
            for (j, &v) in row.iter().enumerate() {
                counts[j] += 1;
                let fv = v as f64;
                sums[j] += fv;
                sums_sq[j] += fv * fv;
            }
            i += stride;
        }

        let mut variances: Vec<(f64, usize)> = Vec::with_capacity(stride);
        for j in 0..stride {
            let c = counts[j];
            if c == 0 {
                variances.push((0.0, j));
            } else {
                let mean = sums[j] / c as f64;
                let mut var = (sums_sq[j] / c as f64) - mean * mean;
                if var < 0.0 {
                    var = 0.0;
                }
                variances.push((var, j));
            }
        }

        variances.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        variances.into_iter().map(|(_, idx)| idx).collect()
    }

    #[inline]
    fn apply_dim_permutation(&self, data: &[u8], perm: &[usize]) -> Vec<u8> {
        let stride = self.vector_stride_or_zero();
        if data.is_empty() || perm.is_empty() || stride == 0 {
            return data.to_vec();
        }
        let n = data.len();
        let mut out = vec![0u8; n];
        let plen = perm.len();

        let mut base = 0;
        while base < n {
            let end = (base + stride).min(n);
            if end - base < plen {
                out[base..end].copy_from_slice(&data[base..end]);
            } else {
                for (new_pos, &old_pos) in perm.iter().enumerate() {
                    out[base + new_pos] = data[base + old_pos];
                }
            }
            base += stride;
        }
        out
    }

    #[inline]
    fn apply_dim_inverse_permutation(&self, data: &[u8], perm: &[usize]) -> Vec<u8> {
        let stride = self.vector_stride_or_zero();
        if data.is_empty() || perm.is_empty() || stride == 0 {
            return data.to_vec();
        }
        let n = data.len();
        let mut out = vec![0u8; n];
        let plen = perm.len();

        let mut inv = vec![0usize; plen];
        for (new_pos, &old_pos) in perm.iter().enumerate() {
            inv[old_pos] = new_pos;
        }

        let mut base = 0;
        while base < n {
            let end = (base + stride).min(n);
            if end - base < plen {
                out[base..end].copy_from_slice(&data[base..end]);
            } else {
                for (old_pos, &new_pos) in inv.iter().enumerate() {
                    out[base + old_pos] = data[base + new_pos];
                }
            }
            base += stride;
        }
        out
    }

    #[inline]
    fn semantic_forward_auto(&self, data: &[u8]) -> (Vec<u8>, Vec<u8>, u8) {
        let stride = self.vector_stride_or_zero();
        if data.is_empty() || stride == 0 {
            return (data.to_vec(), Vec::new(), 0);
        }

        let sample_len = data.len().min(512 * stride);
        let sample = &data[..sample_len];

        let mut best_mode: u8 = 0;
        let mut best_perm: Vec<usize> = Vec::new();
        let mut best_score = self.score_bytes_for_zlib_local(sample);

        let delta_sample = self.forward_vector_delta(sample);
        let score_delta = self.score_bytes_for_zlib_local(&delta_sample);
        if score_delta < best_score {
            best_score = score_delta;
            best_mode = 1;
            best_perm.clear();
        }

        let perm = self.build_dim_permutation(sample);
        if !perm.is_empty() {
            let perm_sample = self.apply_dim_permutation(sample, &perm);
            let perm_delta_sample = self.forward_vector_delta(&perm_sample);
            let score_perm_delta = self.score_bytes_for_zlib_local(&perm_delta_sample);

            if score_perm_delta < best_score {
                best_score = score_perm_delta;
                best_mode = 2;
                best_perm = perm;
            }
        }

        match best_mode {
            0 => (data.to_vec(), Vec::new(), 0),
            1 => {
                let full = self.forward_vector_delta(data);
                (full, Vec::new(), 1)
            }
            2 => {
                let mut d = data.to_vec();
                if best_perm.is_empty() {
                    best_perm = self.build_dim_permutation(&d);
                }
                if !best_perm.is_empty() {
                    d = self.apply_dim_permutation(&d, &best_perm);
                }
                let full = self.forward_vector_delta(&d);
                let perm_bytes: Vec<u8> = best_perm.iter().map(|&p| p as u8).collect();
                (full, perm_bytes, 2)
            }
            _ => (data.to_vec(), Vec::new(), 0),
        }
    }

    #[inline]
    fn semantic_inverse_with_mode(&self, data: &[u8], perm_bytes: &[u8], mode: u8) -> Vec<u8> {
        if data.is_empty() {
            return data.to_vec();
        }
        match mode {
            0 => data.to_vec(),
            1 => self.inverse_vector_delta(data),
            2 => {
                let d = self.inverse_vector_delta(data);
                if perm_bytes.is_empty() {
                    return d;
                }
                let perm: Vec<usize> = perm_bytes.iter().map(|&b| b as usize).collect();
                self.apply_dim_inverse_permutation(&d, &perm)
            }
            _ => data.to_vec(),
        }
    }

    #[inline]
    fn choose_auto_profile(&self, payload: &[u8], idx: &PayloadIndex) -> AutoProfile {
        let n = idx.len;
        let h = idx.entropy;

        if n < 64 * 1024 || h > 7.2 {
            return AutoProfile::ZstdFast;
        }

        let zscore = self.score_bytes_for_zlib_global(payload);

        if zscore <= 10 {
            return AutoProfile::SemBD3D;
        }

        if h > 6.0 {
            AutoProfile::LosslessBD3D
        } else {
            AutoProfile::SemBD3D
        }
    }

    // ---------------------------------------------------------
    // Skimming: cheap early decision to avoid useless BD3D work
    // ---------------------------------------------------------
    #[inline]
    fn skim_payload(&self, payload: &[u8], idx: &PayloadIndex) -> SkimDecision {
        let n = idx.len;
        let h = idx.entropy;

        // Very small payloads: always zstd
        if n <= 256 {
            return SkimDecision::ForceZstd;
        }

        // Already compressed: force zstd
        if looks_already_compressed(payload, idx) {
            return SkimDecision::ForceZstd;
        }

        // Very high entropy: BD3D unlikely to win
        if h > 7.6 {
            return SkimDecision::PreferZstd;
        }

        // Global zlib score probe
        let zscore = self.score_bytes_for_zlib_global(payload);

        // If zlib sees almost no structure, BD3D won't help much
        if zscore > 40 {
            return SkimDecision::PreferZstd;
        }

        SkimDecision::Normal
    }

    #[inline]
    fn rubik_forward_blocks(&self, data: &[u8]) -> (Vec<u8>, Vec<u8>) {
        const BLOCK: usize = 1024;

        if data.len() <= BLOCK {
            return (data.to_vec(), Vec::new());
        }

        let n = data.len();
        let blocks = (n + BLOCK - 1) / BLOCK;

        let mut scores: Vec<(f64, usize)> = Vec::with_capacity(blocks);
        for i in 0..blocks {
            let start = i * BLOCK;
            let end = ((i + 1) * BLOCK).min(n);
            let slice = &data[start..end];
            let h = byte_entropy_score(slice);
            scores.push((h, i));
        }

        scores.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        let mut perm: Vec<usize> = Vec::with_capacity(blocks);
        for (_, idx) in &scores {
            perm.push(*idx);
        }

        let mut out = vec![0u8; n];
        for (new_idx, &old_idx) in perm.iter().enumerate() {
            let src_start = old_idx * BLOCK;
            let src_end = ((old_idx + 1) * BLOCK).min(n);
            let dst_start = new_idx * BLOCK;
            let dst_end = ((new_idx + 1) * BLOCK).min(n);
            let len = src_end - src_start;
            out[dst_start..dst_start + len].copy_from_slice(&data[src_start..src_end]);
            if dst_end > dst_start + len {
                for b in &mut out[dst_start + len..dst_end] {
                    *b = 0;
                }
            }
        }

        if blocks > u16::MAX as usize {
            return (data.to_vec(), Vec::new());
        }
        let mut perm_bytes = Vec::with_capacity(blocks * 2);
        for idx in perm {
            let v = idx as u16;
            perm_bytes.extend_from_slice(&v.to_be_bytes());
        }

        (out, perm_bytes)
    }

    #[inline]
    fn rubik_inverse_blocks(&self, data: &[u8], perm_bytes: &[u8]) -> Vec<u8> {
        const BLOCK: usize = 1024;

        if perm_bytes.is_empty() || data.len() <= BLOCK {
            return data.to_vec();
        }

        if perm_bytes.len() % 2 != 0 {
            return data.to_vec();
        }

        let blocks = perm_bytes.len() / 2;
        let n = data.len();
        let mut perm: Vec<usize> = Vec::with_capacity(blocks);
        for i in 0..blocks {
            let b0 = perm_bytes[2 * i];
            let b1 = perm_bytes[2 * i + 1];
            let v = u16::from_be_bytes([b0, b1]) as usize;
            perm.push(v);
        }

        let mut out = vec![0u8; n];

        for (new_idx, &old_idx) in perm.iter().enumerate() {
            let src_start = new_idx * BLOCK;
            let src_end = ((new_idx + 1) * BLOCK).min(n);
            let dst_start = old_idx * BLOCK;
            let dst_end = ((old_idx + 1) * BLOCK).min(n);
            let len = src_end - src_start;
            if dst_start >= n {
                continue;
            }
            let dst_len = dst_end - dst_start;
            let copy_len = len.min(dst_len);
            out[dst_start..dst_start + copy_len]
                .copy_from_slice(&data[src_start..src_start + copy_len]);
        }

        out
    }

    #[inline]
    fn should_micro_merge(cubes: &[Cube]) -> bool {
        if cubes.len() < 2 {
            return false;
        }
        let total: usize = cubes
            .iter()
            .map(|c| match &c.data {
                CubeData::Raw(b) => b.len(),
                CubeData::Ref(id) => {
                    panic!("should_micro_merge saw unresolved cube reference {:?}", id);
                }
            })
            .sum();
        let avg = total / cubes.len();
        avg < 4 * 1024
    }

    #[inline]
    fn apply_predictor_to_layers(
        &self,
        effective_max_layers: u16,
        cubes: &[Cube],
    ) -> (u16, Option<ClusterHint>) {
        if cubes.is_empty() {
            return (effective_max_layers, None);
        }

        let hint = predict_for_cluster(cubes);
        if let Some(h) = hint {
            let mut layers = effective_max_layers;
            if h.pass_limit_hint > 0 {
                let cap = h.pass_limit_hint as u16;
                layers = layers.min(cap).max(1);
            }
            (layers, hint)
        } else {
            (effective_max_layers, None)
        }
    }

    fn encode_inner(
        &self,
        payload: &[u8],
        idx: &PayloadIndex,
        decision: &LibrarianDecision,
    ) -> Vec<u8> {
        debug!("ENCODE_FIRST16: {:?}", &payload[..payload.len().min(16)]);

        let mut effective_block_shape = self.auto_block_shape(payload, idx);
        let mut effective_max_layers = self.auto_max_layers(idx);

        if let Some(cap) = decision.max_layers_cap {
            effective_max_layers = effective_max_layers.min(cap);
        }

        if decision.block_depth_scale != 1.0 {
            let scaled = ((effective_block_shape.2 as f32) * decision.block_depth_scale)
                .round()
                .clamp(8.0, 512.0) as usize;
            effective_block_shape.2 = scaled;
        }

        let low_entropy = idx.entropy < 3.0 && idx.len >= 8 * 1024 * 1024;
        if low_entropy {
            effective_block_shape.2 = (effective_block_shape.2 * 2).min(512);
            effective_max_layers = effective_max_layers.min(2);
        }

        let (mut cubes, original_len) = lift_to_cubes(payload, effective_block_shape);

        let (effective_max_layers, predictor_hint) = if low_entropy {
            (effective_max_layers.min(2), None)
        } else {
            self.apply_predictor_to_layers(effective_max_layers, &cubes)
        };

        let mut all_layers = Vec::with_capacity(cubes.len() * 4);
        let mut log = TransformLog::new();
        let mut next_id: u32 = cubes.len() as u32;
        let mut current_layer: u16 = 0;

        let global_model = crate::get_global_model();

        let rotate_logs: Vec<Transform> = cubes
            .par_iter_mut()
            .map(|cube| {
                let best_ori = choose_best_orientation(cube);
                let bytes = match &cube.data {
                    CubeData::Raw(b) => b.as_slice(),
                    CubeData::Ref(id) => {
                        panic!("encode_inner rotate saw unresolved cube reference {:?}", id);
                    }
                };
                let rotated = rotate_cube_data(cube.shape, bytes, best_ori);
                cube.data = CubeData::Raw(rotated);
                Transform::Rotate {
                    cube_id: cube.id,
                    layer: cube.layer,
                    orientation: best_ori,
                }
            })
            .collect();

        for t in rotate_logs {
            log.push(t);
        }

        if !low_entropy && !decision.skip_pts {
            let pattern_logs: Vec<Transform> = cubes
                .par_iter_mut()
                .filter_map(|cube| {
                    let bytes = match &cube.data {
                        CubeData::Raw(b) => b.as_slice(),
                        CubeData::Ref(id) => {
                            panic!("encode_inner PTS saw unresolved cube reference {:?}", id);
                        }
                    };

                    let h = cube_entropy_score(bytes);
                    if h > 7.0 {
                        return None;
                    }

                    let (compressed, tags) = pts_compress_cube(bytes);
                    if tags.is_empty() {
                        None
                    } else {
                        cube.data = CubeData::Raw(compressed);
                        Some(Transform::PatternTag {
                            cube_id: cube.id,
                            layer: cube.layer,
                            tags,
                        })
                    }
                })
                .collect();

            for t in pattern_logs {
                log.push(t);
            }
        }

        all_layers.extend(cubes.iter().cloned());

        while current_layer < effective_max_layers {
            let clusters = cluster_cubes(&cubes);
            if clusters.len() <= 1 {
                break;
            }

            let mut id_cursor = next_id;
            let mut ranges = Vec::with_capacity(clusters.len());
            for cluster in &clusters {
                ranges.push(id_cursor);
                let span = (cluster.len() as u32) * 2 + 16;
                id_cursor = id_cursor.wrapping_add(span);
            }
            next_id = id_cursor;

            let results: Vec<(Vec<Cube>, TransformLog)> = clusters
                .into_par_iter()
                .zip(ranges.into_par_iter())
                .map(|(cluster, base_id)| {
                    let mut local_next = base_id;
                    let mut local_log = TransformLog::new();

                    if cluster.len() == 1 {
                        let csig = cube_signature(&cluster[0]);
                        if let Some(mut centry) = self.cube_cache.get_mut(&csig) {
                            let collapsed = vec![centry.cube.clone()];
                            for t in centry.transforms.iter() {
                                local_log.push(t.clone());
                            }
                            centry.hits = centry.hits.saturating_add(1);
                            centry.last_used_ts = Utc::now().timestamp();
                            return (collapsed, local_log);
                        }
                    }

                    let sig = compute_cluster_signature(&cluster, current_layer + 1, base_id);

                    if let Some(mut entry) = self.collapse_cache.get_mut(&sig) {
                        let collapsed = entry.cubes.clone();
                        for t in entry.transforms.iter() {
                            local_log.push(t.clone());
                        }
                        entry.hits = entry.hits.saturating_add(1);
                        entry.last_used_ts = Utc::now().timestamp();
                        (collapsed, local_log)
                    } else {
                        let collapsed = collapse_cluster(
                            cluster,
                            current_layer + 1,
                            &mut local_next,
                            &mut local_log,
                            global_model,
                        );
                        let now_ts = Utc::now().timestamp();
                        let cached_transforms = local_log.transforms.clone();
                        let entry = CollapseCacheEntry {
                            cubes: collapsed.clone(),
                            transforms: cached_transforms,
                            created_ts: now_ts,
                            last_used_ts: now_ts,
                            hits: 1,
                        };
                        self.collapse_cache.insert(sig, entry);

                        if collapsed.len() == 1 {
                            let csig = cube_signature(&collapsed[0]);
                            let centry = CubeCollapseCacheEntry {
                                cube: collapsed[0].clone(),
                                transforms: local_log.transforms.clone(),
                                created_ts: now_ts,
                                last_used_ts: now_ts,
                                hits: 1,
                            };
                            self.cube_cache.insert(csig, centry);
                        }

                        (collapsed, local_log)
                    }
                })
                .collect();

            let mut next_cubes = Vec::new();
            for (mut collapsed, local_log) in results {
                next_cubes.append(&mut collapsed);
                for t in local_log.transforms {
                    log.push(t);
                }
            }

            if next_cubes.len() == cubes.len() {
                break;
            }

            all_layers.extend(next_cubes.iter().cloned());
            cubes = next_cubes;
            current_layer += 1;

            if low_entropy {
                break;
            }
        }

        if !low_entropy && Self::should_micro_merge(&cubes) {
            let sig = compute_cluster_signature(&cubes, current_layer + 1, 0);

            if let Some(mut entry) = self.collapse_cache.get_mut(&sig) {
                let merged = entry.cubes.clone();
                for t in entry.transforms.iter() {
                    log.push(t.clone());
                }
                entry.hits = entry.hits.saturating_add(1);
                entry.last_used_ts = Utc::now().timestamp();
                all_layers.extend(merged.iter().cloned());
            } else {
                let merged = collapse_cluster(
                    cubes,
                    current_layer + 1,
                    &mut next_id,
                    &mut log,
                    global_model,
                );
                let now_ts = Utc::now().timestamp();
                let cached_transforms = log.transforms.clone();
                let entry = CollapseCacheEntry {
                    cubes: merged.clone(),
                    transforms: cached_transforms,
                    created_ts: now_ts,
                    last_used_ts: now_ts,
                    hits: 1,
                };
                self.collapse_cache.insert(sig, entry);
                all_layers.extend(merged.iter().cloned());
            }
        }

        if let Some(h) = predictor_hint {
            train_from_cluster(
                &all_layers,
                h.structured_hint,
                h.quant_bits_hint,
                h.pass_limit_hint,
                h.merge_threshold_hint,
            );
        }

        pack_cubes_and_log(&all_layers, original_len, &log)
    }

    fn encode_inner_light(
        &self,
        payload: &[u8],
        idx: &PayloadIndex,
        decision: &LibrarianDecision,
    ) -> Vec<u8> {
        debug!(
            "ENCODE_LIGHT_FIRST16: {:?}",
            &payload[..payload.len().min(16)]
        );

        let mut effective_block_shape = self.auto_block_shape(payload, idx);
        let mut effective_max_layers = self.auto_max_layers(idx);

        if effective_max_layers > 2 {
            effective_max_layers = 2;
        }

        if let Some(cap) = decision.max_layers_cap {
            effective_max_layers = effective_max_layers.min(cap);
        }

        if decision.block_depth_scale != 1.0 {
            let scaled = ((effective_block_shape.2 as f32) * decision.block_depth_scale)
                .round()
                .clamp(8.0, 512.0) as usize;
            effective_block_shape.2 = scaled;
        }

        let low_entropy = idx.entropy < 4.0 && idx.len >= 8 * 1024 * 1024;
        if low_entropy {
            effective_block_shape.2 = (effective_block_shape.2 * 2).min(512);
            effective_max_layers = 1;
        }

        let (mut cubes, original_len) = lift_to_cubes(payload, effective_block_shape);

        let (effective_max_layers, predictor_hint) = if low_entropy {
            (effective_max_layers.min(1), None)
        } else {
            self.apply_predictor_to_layers(effective_max_layers, &cubes)
        };

        let mut all_layers = Vec::with_capacity(cubes.len() * 2);
        let mut log = TransformLog::new();
        let mut next_id: u32 = cubes.len() as u32;
        let mut current_layer: u16 = 0;

        let global_model = crate::get_global_model();

        let rotate_logs: Vec<Transform> = cubes
            .par_iter_mut()
            .map(|cube| {
                let best_ori = choose_best_orientation(cube);
                let bytes = match &cube.data {
                    CubeData::Raw(b) => b.as_slice(),
                    CubeData::Ref(id) => {
                        panic!(
                            "encode_inner_light rotate saw unresolved cube reference {:?}",
                            id
                        );
                    }
                };
                let rotated = rotate_cube_data(cube.shape, bytes, best_ori);
                cube.data = CubeData::Raw(rotated);
                Transform::Rotate {
                    cube_id: cube.id,
                    layer: cube.layer,
                    orientation: best_ori,
                }
            })
            .collect();

        for t in rotate_logs {
            log.push(t);
        }

        all_layers.extend(cubes.iter().cloned());

        while current_layer < effective_max_layers {
            let clusters = cluster_cubes(&cubes);
            if clusters.len() <= 1 {
                break;
            }

            let mut id_cursor = next_id;
            let mut ranges = Vec::with_capacity(clusters.len());
            for cluster in &clusters {
                ranges.push(id_cursor);
                let span = (cluster.len() as u32) * 2 + 16;
                id_cursor = id_cursor.wrapping_add(span);
            }
            next_id = id_cursor;

            let results: Vec<(Vec<Cube>, TransformLog)> = clusters
                .into_par_iter()
                .zip(ranges.into_par_iter())
                .map(|(cluster, base_id)| {
                    let mut local_next = base_id;
                    let mut local_log = TransformLog::new();

                    if cluster.len() == 1 {
                        let csig = cube_signature(&cluster[0]);
                        if let Some(mut centry) = self.cube_cache.get_mut(&csig) {
                            let collapsed = vec![centry.cube.clone()];
                            for t in centry.transforms.iter() {
                                local_log.push(t.clone());
                            }
                            centry.hits = centry.hits.saturating_add(1);
                            centry.last_used_ts = Utc::now().timestamp();
                            return (collapsed, local_log);
                        }
                    }

                    let sig = compute_cluster_signature(&cluster, current_layer + 1, base_id);

                    if let Some(mut entry) = self.collapse_cache.get_mut(&sig) {
                        let collapsed = entry.cubes.clone();
                        for t in entry.transforms.iter() {
                            local_log.push(t.clone());
                        }
                        entry.hits = entry.hits.saturating_add(1);
                        entry.last_used_ts = Utc::now().timestamp();
                        (collapsed, local_log)
                    } else {
                        let collapsed = collapse_cluster(
                            cluster,
                            current_layer + 1,
                            &mut local_next,
                            &mut local_log,
                            global_model,
                        );
                        let now_ts = Utc::now().timestamp();
                        let cached_transforms = local_log.transforms.clone();
                        let entry = CollapseCacheEntry {
                            cubes: collapsed.clone(),
                            transforms: cached_transforms,
                            created_ts: now_ts,
                            last_used_ts: now_ts,
                            hits: 1,
                        };
                        self.collapse_cache.insert(sig, entry);

                        if collapsed.len() == 1 {
                            let csig = cube_signature(&collapsed[0]);
                            let centry = CubeCollapseCacheEntry {
                                cube: collapsed[0].clone(),
                                transforms: local_log.transforms.clone(),
                                created_ts: now_ts,
                                last_used_ts: now_ts,
                                hits: 1,
                            };
                            self.cube_cache.insert(csig, centry);
                        }

                        (collapsed, local_log)
                    }
                })
                .collect();

            let mut next_cubes = Vec::new();
            for (mut collapsed, local_log) in results {
                next_cubes.append(&mut collapsed);
                for t in local_log.transforms {
                    log.push(t);
                }
            }

            if next_cubes.len() == cubes.len() {
                break;
            }

            all_layers.extend(next_cubes.iter().cloned());
            cubes = next_cubes;
            current_layer += 1;

            if low_entropy {
                break;
            }
        }

        if let Some(h) = predictor_hint {
            train_from_cluster(
                &all_layers,
                h.structured_hint,
                h.quant_bits_hint,
                h.pass_limit_hint,
                h.merge_threshold_hint,
            );
        }

        pack_cubes_and_log(&all_layers, original_len, &log)
    }

    fn decode_inner(&self, blob: &[u8]) -> Vec<u8> {
        let (final_cubes, original_len, log) = unpack_cubes_and_log(blob);

        let mut cubes: HashMap<CubeId, Cube> =
            final_cubes.into_iter().map(|c| (c.id, c)).collect();

        for t in log.transforms.iter().rev() {
            match t {
                Transform::DropLayer { .. } => {}

                Transform::Merge {
                    new_cube_id,
                    members,
                    offsets,
                    original_positions,
                    original_shapes,
                    original_layers,
                    ..
                } => {
                    if let Some(merged) = cubes.remove(new_cube_id) {
                        let data = match merged.data {
                            CubeData::Raw(b) => b,
                            CubeData::Ref(id) => panic!(
                                "decode_inner Merge saw unresolved cube reference {:?}",
                                id
                            ),
                        };

                        for idx in 0..members.len() {
                            let start = offsets[idx] as usize;
                            let end = if idx + 1 < offsets.len() {
                                offsets[idx + 1] as usize
                            } else {
                                data.len()
                            };

                            let slice = data[start..end].to_vec();
                            let mut cube = Cube::new(
                                members[idx],
                                original_layers[idx],
                                original_positions[idx],
                                original_shapes[idx],
                                slice,
                            );

                            cube.quant_bits = merged.quant_bits;
                            cube.quant_vmin = merged.quant_vmin;
                            cube.quant_vmax = merged.quant_vmax;

                            cubes.insert(members[idx], cube);
                        }
                    }
                }

                Transform::Shift { cube_id, dx, dy, dz, .. } => {
                    if let Some(c) = cubes.get_mut(cube_id) {
                        let (ix, iy, iz) = inverse_shift(*dx, *dy, *dz);
                        c.pos = apply_shift(c.pos, ix, iy, iz);
                    }
                }

                Transform::PatternTag { cube_id, tags, .. } => {
                    if let Some(c) = cubes.get_mut(cube_id) {
                        let bytes = match &c.data {
                            CubeData::Raw(b) => b.as_slice(),
                            CubeData::Ref(id) => panic!(
                                "decode_inner PTS saw unresolved cube reference {:?}",
                                id
                            ),
                        };
                        let out = pts_decompress_cube(bytes, tags);
                        c.data = CubeData::Raw(out);
                    }
                }

                Transform::Rotate { cube_id, orientation, .. } => {
                    if let Some(c) = cubes.get_mut(cube_id) {
                        let inv = inverse_orientation(*orientation);
                        let bytes = match &c.data {
                            CubeData::Raw(b) => b.as_slice(),
                            CubeData::Ref(id) => panic!(
                                "decode_inner Rotate saw unresolved cube reference {:?}",
                                id
                            ),
                        };
                        let rotated = rotate_cube_data(c.shape, bytes, inv);
                        c.data = CubeData::Raw(rotated);
                    }
                }

                Transform::PatternRef { cube_id, ref_id, .. } => {
                    let src_id = CubeId(*ref_id);
                    if let Some(src) = cubes.get(&src_id) {
                        let cloned = src.data.clone();
                        if let Some(dst) = cubes.get_mut(cube_id) {
                            dst.data = cloned;
                        }
                    }
                }
            }
        }

        let mut all_cubes: Vec<Cube> = cubes.into_values().collect();

        for c in &mut all_cubes {
            if c.quant_bits == 0 {
                continue;
            }

            let bits = c.quant_bits;
            let vmin = c.quant_vmin;
            let vmax = c.quant_vmax;

            if vmin == vmax || bits == 0 {
                continue;
            }

            let levels = (1u32 << bits) - 1;
            let range = (vmax as f32 - vmin as f32).max(1.0);
            let scale = range / levels as f32;

            let src = match &c.data {
                CubeData::Raw(b) => b.as_slice(),
                CubeData::Ref(id) => panic!(
                    "decode_inner quantization saw unresolved cube reference {:?}",
                    id
                ),
            };

            let mut out = Vec::with_capacity(src.len());
            for &q in src {
                let v = vmin as f32 + (q as f32) * scale;
                out.push(v.round().clamp(0.0, 255.0) as u8);
            }

            c.data = CubeData::Raw(out);
        }

        flatten_from_cubes(&all_cubes, original_len)
    }

    pub fn encode(&self, payload: &[u8]) -> Vec<u8> {
        if payload.is_empty() {
            return vec![1];
        }

        let idx = build_payload_index(payload);

        // Skimming: cheap early exit
        match self.skim_payload(payload, &idx) {
            SkimDecision::ForceZstd => {
                let zstd = zstd_enc(&payload[..], 1).unwrap();
                let mut out = Vec::with_capacity(1 + zstd.len());
                out.push(1);
                out.extend_from_slice(&zstd);
                return out;
            }
            SkimDecision::PreferZstd => {
                // We'll bias profile toward ZstdFast below
            }
            SkimDecision::Normal => {}
        }

        if idx.len <= 256 {
            let zstd = zstd_enc(&payload[..], 1).unwrap();
            let mut out = Vec::with_capacity(1 + zstd.len());
            out.push(1);
            out.extend_from_slice(&zstd);
            return out;
        }

        if looks_already_compressed(payload, &idx) {
            let zstd = zstd_enc(&payload[..], 1).unwrap();
            let mut out = Vec::with_capacity(1 + zstd.len());
            out.push(1);
            out.extend_from_slice(&zstd);
            return out;
        }

        let stats = PayloadStats {
            len: idx.len,
            entropy: idx.entropy,
            tier: idx.tier,
        };
        let librarian = Librarian::new_with_default_library();
        let decision = librarian.advise(payload, &stats);

        if idx.entropy < 1.0 && idx.len >= 8 * 1024 * 1024 {
            let zstd = zstd_enc(&payload[..], 1).unwrap();
            let mut out = Vec::with_capacity(1 + zstd.len());
            out.push(1);
            out.extend_from_slice(&zstd);
            return out;
        }

        let mut profile = self.choose_auto_profile(payload, &idx);

        // If skimming preferred zstd, override profile to ZstdFast
        if matches!(self.skim_payload(payload, &idx), SkimDecision::PreferZstd) {
            profile = AutoProfile::ZstdFast;
        }

        let n = idx.len;
        let h = idx.entropy;
        let tier = idx.tier;

        let zstd_level: i32 = if n < 64 * 1024 {
            1
        } else if h > 7.5 {
            3
        } else if h > 6.0 {
            4
        } else {
            5
        };

        match profile {
            AutoProfile::ZstdFast => {
                let zstd = zstd_enc(&payload[..], zstd_level).unwrap();
                let mut out = Vec::with_capacity(1 + zstd.len());
                out.push(1);
                out.extend_from_slice(&zstd);
                out
            }

            AutoProfile::LosslessBD3D => {
                let use_light = match tier {
                    2 => true,
                    _ => decision.use_light_bd3d,
                };

                let bd = if use_light {
                    self.encode_inner_light(payload, &idx, &decision)
                } else {
                    self.encode_inner(payload, &idx, &decision)
                };

                let zstd = zstd_enc(&payload[..], zstd_level).unwrap();

                let bd_total = 1 + 1 + 2 + 0 + 1 + 1 + 1 + bd.len();
                let zstd_total = 1 + zstd.len();

                if zstd_total <= bd_total {
                    let mut out = Vec::with_capacity(zstd_total);
                    out.push(1);
                    out.extend_from_slice(&zstd);
                    out
                } else {
                    let mut out = Vec::with_capacity(bd_total);
                    out.push(0);
                    out.push(0);
                    out.extend_from_slice(&0u16.to_be_bytes());
                    out.push(0);
                    out.push(0);
                    out.push(0);
                    out.extend_from_slice(&bd);
                    out
                }
            }

            AutoProfile::SemBD3D => {
                if n < 64 * 1024
                    || h > 6.8
                    || decision.skip_semantic
                    || (h < 3.0 && n >= 8 * 1024 * 1024)
                {
                    let use_light = match tier {
                        2 => true,
                        _ => decision.use_light_bd3d,
                    };

                    let bd = if use_light {
                        self.encode_inner_light(payload, &idx, &decision)
                    } else {
                        self.encode_inner(payload, &idx, &decision)
                    };

                    let zstd = zstd_enc(&payload[..], zstd_level).unwrap();

                    let bd_total = 1 + 1 + 2 + 0 + 1 + 1 + 1 + bd.len();
                    let zstd_total = 1 + zstd.len();

                    if zstd_total <= bd_total {
                        let mut out = Vec::with_capacity(zstd_total);
                        out.push(1);
                        out.extend_from_slice(&zstd);
                        out
                    } else {
                        let mut out = Vec::with_capacity(bd_total);
                        out.push(0);
                        out.push(0);
                        out.extend_from_slice(&0u16.to_be_bytes());
                        out.push(0);
                        out.push(0);
                        out.push(0);
                        out.extend_from_slice(&bd);
                        out
                    }
                } else {
                    let (sem_payload, perm_bytes, mut semantic_mode) =
                        self.semantic_forward_auto(payload);

                    if tier == 2 && semantic_mode == 2 {
                        semantic_mode = 1;
                    }

                    let use_light = match tier {
                        2 => true,
                        _ => decision.use_light_bd3d,
                    };

                    let bd = if use_light {
                        self.encode_inner_light(&sem_payload, &idx, &decision)
                    } else {
                        self.encode_inner(&sem_payload, &idx, &decision)
                    };

                    let zstd = zstd_enc(&payload[..], zstd_level).unwrap();

                    let bd_total =
                        1 + 1 + 2 + perm_bytes.len() + 1 + 1 + 1 + bd.len();
                    let zstd_total = 1 + zstd.len();

                    if zstd_total <= bd_total {
                        let mut out = Vec::with_capacity(zstd_total);
                        out.push(1);
                        out.extend_from_slice(&zstd);
                        out
                    } else {
                        let mut out = Vec::with_capacity(bd_total);
                        out.push(0);
                        out.push(semantic_mode);
                        let perm_len = perm_bytes.len() as u16;
                        out.extend_from_slice(&perm_len.to_be_bytes());
                        out.extend_from_slice(&perm_bytes);
                        out.push(0);
                        out.push(0);
                        out.push(0);
                        out.extend_from_slice(&bd);
                        out
                    }
                }
            }
        }
    }

    pub fn decode(&self, blob: &[u8]) -> Vec<u8> {
        if blob.is_empty() {
            return Vec::new();
        }

        let mode = blob[0];
        let body = &blob[1..];

        match mode {
            0 => {
                if body.len() < 6 {
                    return Vec::new();
                }
                let semantic_mode = body[0];
                let perm_len = u16::from_be_bytes([body[1], body[2]]) as usize;

                if body.len() < 3 + perm_len + 3 {
                    return Vec::new();
                }

                let perm_bytes = &body[3..3 + perm_len];
                let bd_body = &body[6 + perm_len..];

                let decoded_bd = self.decode_inner(bd_body);
                self.semantic_inverse_with_mode(&decoded_bd, perm_bytes, semantic_mode)
            }

            1 => zstd_dec(body).unwrap(),

            // Safe fallback for unknown/legacy mode bytes:
            // treat as raw zstd payload (most legacy frames will be zstd-wrapped)
            _ => {
                // If this fails, return the body unchanged instead of panicking.
                match zstd_dec(body) {
                    Ok(decoded) => decoded,
                    Err(_) => body.to_vec(),
                }
            }
        }
    }

    pub fn encode_origami(&self, payload: &[u8]) -> Vec<u8> {
        if payload.is_empty() {
            return vec![2];
        }

        let (sem_payload, sem_perm_bytes, semantic_mode) = self.semantic_forward_auto(payload);
        let (rubik_payload, rubik_perm_bytes) = self.rubik_forward_blocks(&sem_payload);
        let idx = build_payload_index(&rubik_payload);

        let stats = PayloadStats {
            len: idx.len,
            entropy: idx.entropy,
            tier: idx.tier,
        };
        let librarian = Librarian::new_with_default_library();
        let decision = librarian.advise(&rubik_payload, &stats);

        let bd = self.encode_inner(&rubik_payload, &idx, &decision);

        let sem_perm_len = sem_perm_bytes.len() as u16;
        let rubik_perm_len = rubik_perm_bytes.len() as u16;

        let mut out = Vec::with_capacity(
            1 + 1 + 2 + sem_perm_bytes.len() + 2 + rubik_perm_bytes.len() + bd.len(),
        );
        out.push(2);
        out.push(semantic_mode);
        out.extend_from_slice(&sem_perm_len.to_be_bytes());
        out.extend_from_slice(&sem_perm_bytes);
        out.extend_from_slice(&rubik_perm_len.to_be_bytes());
        out.extend_from_slice(&rubik_perm_bytes);
        out.extend_from_slice(&bd);
        out
    }

    pub fn decode_origami(&self, blob: &[u8]) -> Vec<u8> {
        if blob.is_empty() {
            return Vec::new();
        }
        if blob[0] != 2 {
            return self.decode(blob);
        }

        let body = &blob[1..];
        if body.len() < 4 {
            return Vec::new();
        }

        let semantic_mode = body[0];
        let sem_perm_len = u16::from_be_bytes([body[1], body[2]]) as usize;

        if body.len() < 3 + sem_perm_len + 2 {
            return Vec::new();
        }

        let sem_perm_bytes = &body[3..3 + sem_perm_len];
        let rubik_len_offset = 3 + sem_perm_len;
        let rubik_perm_len = u16::from_be_bytes([
            body[rubik_len_offset],
            body[rubik_len_offset + 1],
        ]) as usize;

        if body.len() < rubik_len_offset + 2 + rubik_perm_len {
            return Vec::new();
        }

        let rubik_perm_bytes =
            &body[rubik_len_offset + 2..rubik_len_offset + 2 + rubik_perm_len];
        let bd_body = &body[rubik_len_offset + 2 + rubik_perm_len..];

        let decoded_bd = self.decode_inner(bd_body);
        let sem_payload = self.rubik_inverse_blocks(&decoded_bd, rubik_perm_bytes);
        self.semantic_inverse_with_mode(&sem_payload, sem_perm_bytes, semantic_mode)
    }
}

// ---------------------------------------------------------
// PTS helpers
// ---------------------------------------------------------
#[inline]
fn max_patterns_for_len(len: usize) -> usize {
    match len {
        0..=1023 => 0,
        1024..=8191 => 8,
        8192..=65535 => 16,
        65536..=262143 => 32,
        _ => 64,
    }
}

#[inline]
fn build_pattern_histogram(data: &[u8]) -> HashMap<u32, u32> {
    let mut hist = HashMap::with_capacity(data.len() / 4);
    if data.len() < 8 {
        return hist;
    }
    for i in 0..=(data.len() - 4) {
        let p = u32::from_le_bytes([data[i], data[i + 1], data[i + 2], data[i + 3]]);
        *hist.entry(p).or_insert(0) += 1;
    }
    hist
}

#[inline]
fn select_top_patterns(hist: &HashMap<u32, u32>, max: usize) -> Vec<(u32, u8)> {
    let mut v: Vec<(u32, u32)> = hist.iter().map(|(p, c)| (*p, *c)).collect();
    v.sort_unstable_by(|a, b| b.1.cmp(&a.1));
    v.into_iter()
        .filter(|(_, c)| *c >= 2)
        .take(max)
        .enumerate()
        .map(|(idx, (p, _))| (p, idx as u8))
        .collect()
}

#[inline]
fn pts_compress_cube(data: &[u8]) -> (Vec<u8>, Vec<PatternTag>) {
    let max_patterns = max_patterns_for_len(data.len());
    if max_patterns == 0 || data.len() < 8 {
        return (data.to_vec(), Vec::new());
    }

    let hist = build_pattern_histogram(data);
    let top = select_top_patterns(&hist, max_patterns);
    if top.is_empty() {
        return (data.to_vec(), Vec::new());
    }

    let mut pattern_to_tag = HashMap::with_capacity(top.len());
    let mut tags = Vec::with_capacity(top.len());
    for (pattern, tag_idx) in top {
        pattern_to_tag.insert(pattern, tag_idx);
        tags.push(PatternTag { pattern, tag: tag_idx as u32 });
    }

    let mut out = Vec::with_capacity(data.len());
    let mut i = 0;

    while i < data.len() {
        if i + 4 <= data.len() {
            let p = u32::from_le_bytes([data[i], data[i + 1], data[i + 2], data[i + 3]]);
            if let Some(&tag_idx) = pattern_to_tag.get(&p) {
                out.push(0xFF);
                out.push(tag_idx);
                i += 4;
                continue;
            }
        }

        let b = data[i];
        if b == 0xFF {
            out.push(0xFF);
            out.push(0xFF);
        } else {
            out.push(b);
        }
        i += 1;
    }

    if out.len() >= data.len() {
        return (data.to_vec(), Vec::new());
    }

    (out, tags)
}

#[inline]
fn pts_decompress_cube(data: &[u8], tags: &[PatternTag]) -> Vec<u8> {
    if tags.is_empty() {
        return data.to_vec();
    }

    let max_tag = tags.iter().map(|t| t.tag as usize).max().unwrap_or(0);
    let mut tag_to_pattern = vec![[0u8; 4]; max_tag + 1];

    for t in tags {
        tag_to_pattern[t.tag as usize] = t.pattern.to_le_bytes();
    }

    let mut out = Vec::with_capacity(data.len() * 2);
    let mut i = 0;

    while i < data.len() {
        let b = data[i];
        if b != 0xFF {
            out.push(b);
            i += 1;
            continue;
        }

        if i + 1 >= data.len() {
            out.push(0xFF);
            break;
        }

        let t = data[i + 1];
        if t == 0xFF {
            out.push(0xFF);
        } else {
            let idx = t as usize;
            if idx < tag_to_pattern.len() {
                out.extend_from_slice(&tag_to_pattern[idx]);
            } else {
                out.push(0xFF);
                out.push(t);
            }
        }
        i += 2;
    }

    out
}

