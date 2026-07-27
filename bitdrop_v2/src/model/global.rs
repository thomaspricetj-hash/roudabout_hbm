use serde::{Serialize, Deserialize};
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ClusterStats {
    pub count: u64,
    pub avg_entropy: f32,
    pub avg_size: f32,
    pub avg_pass_limit: f32,
    pub avg_merge_threshold: f32,
    pub avg_quant_bits: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct BlockShapeStats {
    pub count: u64,
    pub total_ratio: f64, // sum(output_len / input_len)
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PatternStats {
    pub count: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct GlobalModelData {
    // keyed by coarse (entropy_bucket, size_bucket)
    pub clusters: HashMap<(u8, u8), ClusterStats>,

    // keyed by (tier, entropy_bucket, shape_id)
    pub block_shapes: HashMap<(u8, u8, u8), BlockShapeStats>,

    // global pattern reinforcement (PTS)
    pub global_patterns: HashMap<u32, PatternStats>,
}

#[derive(Clone)]
pub struct GlobalModel {
    inner: Arc<RwLock<GlobalModelData>>,
    path: PathBuf,
}

impl GlobalModel {
    pub fn load(path: PathBuf) -> Self {
        let data = std::fs::read(&path)
            .ok()
            .and_then(|b| bincode::deserialize::<GlobalModelData>(&b).ok())
            .unwrap_or_default();
        Self {
            inner: Arc::new(RwLock::new(data)),
            path,
        }
    }

    pub fn save(&self) {
        if let Ok(guard) = self.inner.read() {
            if let Ok(bytes) = bincode::serialize(&*guard) {
                let parent: &Path = self.path.parent().unwrap_or_else(|| Path::new("."));
                let _ = std::fs::create_dir_all(parent);
                let _ = std::fs::write(&self.path, bytes);
            }
        }
    }

    #[inline]
    fn bucket_entropy(e: f32) -> u8 {
        (e.clamp(0.0, 8.0) * 4.0) as u8 // 0.25-bit buckets
    }

    #[inline]
    fn bucket_size(bytes: usize) -> u8 {
        let kb = (bytes as f32 / 1024.0).log2().clamp(0.0, 16.0);
        kb as u8
    }

    #[inline]
    fn entropy_bucket_f64(e: f64) -> u8 {
        (e.clamp(0.0, 8.0) * 4.0) as u8
    }

    #[inline]
    fn encode_shape_id(bx: usize, by: usize, bz: usize) -> u8 {
        // compact, stable ID; we only care about relative performance
        let bx = (bx.min(32) as u8) & 0x1F;
        let by = (by.min(32) as u8) & 0x1F;
        let bz = (bz.min(512) as u16) as u8; // coarse
        bx ^ (by << 1) ^ (bz >> 2)
    }

    // --------------------------------------------------------------------
    // Cluster-level hints (existing behavior, now backed by GlobalModel)
    // --------------------------------------------------------------------
    pub fn get_hint(
        &self,
        avg_entropy: f32,
        avg_size: usize,
    ) -> Option<(usize, i64, u8)> {
        let key = (Self::bucket_entropy(avg_entropy), Self::bucket_size(avg_size));
        let guard = self.inner.read().ok()?;
        let stats = guard.clusters.get(&key)?;
        if stats.count < 8 {
            return None;
        }
        let pass_limit = stats.avg_pass_limit.round() as usize;
        let merge_threshold = stats.avg_merge_threshold.round() as i64;
        let quant_bits = stats.avg_quant_bits.round() as u8;
        Some((pass_limit, merge_threshold, quant_bits))
    }

    pub fn update_cluster_observation(
        &self,
        avg_entropy: f32,
        avg_size: usize,
        pass_limit: usize,
        merge_threshold: i64,
        quant_bits: u8,
    ) {
        let key = (Self::bucket_entropy(avg_entropy), Self::bucket_size(avg_size));
        let mut guard = match self.inner.write() {
            Ok(g) => g,
            Err(_) => return,
        };
        let entry = guard.clusters.entry(key).or_default();
        let n = entry.count as f32;
        entry.count += 1;
        entry.avg_entropy = (entry.avg_entropy * n + avg_entropy) / (n + 1.0);
        entry.avg_size = (entry.avg_size * n + avg_size as f32) / (n + 1.0);
        entry.avg_pass_limit = (entry.avg_pass_limit * n + pass_limit as f32) / (n + 1.0);
        entry.avg_merge_threshold =
            (entry.avg_merge_threshold * n + merge_threshold as f32) / (n + 1.0);
        entry.avg_quant_bits =
            (entry.avg_quant_bits * n + quant_bits as f32) / (n + 1.0);
    }

    // --------------------------------------------------------------------
    // Adaptive block-shape learning
    // --------------------------------------------------------------------
    pub fn update_block_shape_observation(
        &self,
        tier: u8,
        entropy: f64,
        bx: usize,
        by: usize,
        bz: usize,
        input_len: usize,
        output_len: usize,
    ) {
        if input_len == 0 {
            return;
        }
        let ent_bucket = Self::entropy_bucket_f64(entropy);
        let shape_id = Self::encode_shape_id(bx, by, bz);
        let ratio = output_len as f64 / input_len as f64;

        let mut guard = match self.inner.write() {
            Ok(g) => g,
            Err(_) => return,
        };
        let key = (tier, ent_bucket, shape_id);
        let entry = guard.block_shapes.entry(key).or_default();
        entry.count += 1;
        entry.total_ratio += ratio;
    }

    pub fn suggest_block_shape(
        &self,
        tier: u8,
        entropy: f64,
        candidates: &[(usize, usize, usize)],
    ) -> Option<(usize, usize, usize)> {
        let ent_bucket = Self::entropy_bucket_f64(entropy);
        let guard = self.inner.read().ok()?;
        let mut best: Option<((usize, usize, usize), f64)> = None;

        for &(bx, by, bz) in candidates {
            let sid = Self::encode_shape_id(bx, by, bz);
            if let Some(stats) = guard.block_shapes.get(&(tier, ent_bucket, sid)) {
                if stats.count >= 8 {
                    let avg = stats.total_ratio / stats.count as f64;
                    match best {
                        None => best = Some(((bx, by, bz), avg)),
                        Some((_, best_avg)) if avg < best_avg => {
                            best = Some(((bx, by, bz), avg));
                        }
                        _ => {}
                    }
                }
            }
        }

        best.map(|(shape, _)| shape)
    }

    // --------------------------------------------------------------------
    // Cross-file pattern reinforcement (PTS)
    // --------------------------------------------------------------------
    pub fn update_pattern(&self, pattern: u32) {
        let mut guard = match self.inner.write() {
            Ok(g) => g,
            Err(_) => return,
        };
        let entry = guard.global_patterns.entry(pattern).or_default();
        entry.count += 1;
    }

    pub fn top_global_patterns(&self, max: usize) -> Vec<u32> {
        let guard = match self.inner.read() {
            Ok(g) => g,
            Err(_) => return Vec::new(),
        };
        let mut v: Vec<(u32, u64)> =
            guard.global_patterns.iter().map(|(p, s)| (*p, s.count)).collect();
        v.sort_by(|a, b| b.1.cmp(&a.1));
        v.into_iter().take(max).map(|(p, _)| p).collect()
    }

    // --------------------------------------------------------------------
    // Entropy-tiered quantization suggestion (future-extensible)
    // --------------------------------------------------------------------
    pub fn suggest_quant_bits(&self, avg_entropy: f32) -> Option<u8> {
        let key = (Self::bucket_entropy(avg_entropy), 0u8);
        let guard = self.inner.read().ok()?;
        let stats = guard.clusters.get(&key)?;
        if stats.count < 16 {
            return None;
        }
        let qb = stats.avg_quant_bits.round() as u8;
        if qb == 0 {
            None
        } else {
            Some(qb)
        }
    }
}

