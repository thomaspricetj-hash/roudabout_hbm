// ============================================================
// Pattern Library + Global Structure Map
// - Fast structural fingerprinting for payloads
// - Lazy-usable by higher-level "Librarian" logic
// - Optimized for large inputs, cheap on small ones
// ============================================================

use std::collections::HashMap;

// ----------------------------------------------------------
// Global Structure Map (GSM)
// ----------------------------------------------------------
#[derive(Clone, Copy, Debug)]
pub struct GlobalStructureMap {
    pub longest_run: u32,
    pub repeat_ratio: f64,
}

#[inline]
pub fn build_global_structure_map(payload: &[u8]) -> GlobalStructureMap {
    if payload.is_empty() {
        return GlobalStructureMap {
            longest_run: 0,
            repeat_ratio: 0.0,
        };
    }

    let mut longest_run: u32 = 1;
    let mut current_run: u32 = 1;
    let mut repeats: usize = 0;

    for w in payload.windows(2) {
        if w[0] == w[1] {
            current_run += 1;
            repeats += 1;
            if current_run > longest_run {
                longest_run = current_run;
            }
        } else {
            current_run = 1;
        }
    }

    let repeat_ratio = repeats as f64 / payload.len().max(1) as f64;

    GlobalStructureMap {
        longest_run,
        repeat_ratio,
    }
}

// ----------------------------------------------------------
// Fast structural hashing
// ----------------------------------------------------------
#[inline]
pub fn fast_hash64(data: &[u8]) -> u64 {
    // Simple but strong-enough 64-bit hash (xorshift-mix over chunks)
    let mut hash: u64 = 0x9E37_79B9_7F4A_7C15;
    let mut i = 0;

    while i + 8 <= data.len() {
        let chunk = u64::from_le_bytes([
            data[i],
            data[i + 1],
            data[i + 2],
            data[i + 3],
            data[i + 4],
            data[i + 5],
            data[i + 6],
            data[i + 7],
        ]);
        hash ^= chunk.wrapping_mul(0xBF58_476D_1CE4_E5B9);
        hash = hash.rotate_left(27).wrapping_mul(0x94D0_49BB_1331_11EB);
        i += 8;
    }

    if i < data.len() {
        let mut tail = [0u8; 8];
        let rem = &data[i..];
        tail[..rem.len()].copy_from_slice(rem);
        let chunk = u64::from_le_bytes(tail);
        hash ^= chunk.wrapping_mul(0xBF58_476D_1CE4_E5B9);
        hash = hash.rotate_left(31).wrapping_mul(0x94D0_49BB_1331_11EB);
    }

    hash ^= (data.len() as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    hash ^= hash >> 33;
    hash = hash.wrapping_mul(0xFF51_AFDC_ED55_8CCD);
    hash ^= hash >> 33;
    hash = hash.wrapping_mul(0xC4CE_B9FE_1A85_EC53);
    hash ^= hash >> 33;
    hash
}

// ----------------------------------------------------------
// Pattern shapes and entries
// ----------------------------------------------------------
#[derive(Clone, Copy, Debug)]
pub struct PatternShape {
    pub signature: u64,
    pub length_bucket: u8,
    pub entropy_bucket: u8,
    pub structure_type: u8, // 0=flat,1=grid,2=tree,3=blocky,4=repetitive,5=other
    pub repeat_score: f32,
}

#[derive(Clone, Debug)]
pub struct PatternEntry {
    pub shape: PatternShape,
    pub name: &'static str,
}

// ----------------------------------------------------------
// Pattern Library
// ----------------------------------------------------------
#[derive(Debug)]
pub struct PatternLibrary {
    by_signature: HashMap<u64, PatternEntry>,
    by_bucket: HashMap<(u8, u8, u8), Vec<PatternEntry>>,
}

impl PatternLibrary {
    pub fn new() -> Self {
        Self {
            by_signature: HashMap::new(),
            by_bucket: HashMap::new(),
        }
    }

    pub fn with_default_patterns() -> Self {
        let mut lib = Self::new();

        // A few generic structural archetypes; you can extend these later.
        let defaults: &[PatternEntry] = &[
            PatternEntry {
                name: "highly_repetitive_stream",
                shape: PatternShape {
                    signature: 0, // wildcard
                    length_bucket: 3,
                    entropy_bucket: 0,
                    structure_type: 4,
                    repeat_score: 0.9,
                },
            },
            PatternEntry {
                name: "grid_like_numeric",
                shape: PatternShape {
                    signature: 0,
                    length_bucket: 2,
                    entropy_bucket: 2,
                    structure_type: 1,
                    repeat_score: 0.4,
                },
            },
            PatternEntry {
                name: "medium_entropy_blocky",
                shape: PatternShape {
                    signature: 0,
                    length_bucket: 2,
                    entropy_bucket: 3,
                    structure_type: 3,
                    repeat_score: 0.3,
                },
            },
        ];

        for entry in defaults {
            lib.insert_pattern(entry.clone());
        }

        lib
    }

    pub fn insert_pattern(&mut self, entry: PatternEntry) {
        if entry.shape.signature != 0 {
            self.by_signature.insert(entry.shape.signature, entry.clone());
        }
        let key = (
            entry.shape.length_bucket,
            entry.shape.entropy_bucket,
            entry.shape.structure_type,
        );
        self.by_bucket.entry(key).or_default().push(entry);
    }

    #[inline]
    fn bucket_for_len(len: usize) -> u8 {
        if len < 64 * 1024 {
            0
        } else if len < 512 * 1024 {
            1
        } else if len < 8 * 1024 * 1024 {
            2
        } else {
            3
        }
    }

    #[inline]
    fn bucket_for_entropy(entropy: f64) -> u8 {
        if entropy < 2.0 {
            0
        } else if entropy < 4.0 {
            1
        } else if entropy < 6.0 {
            2
        } else if entropy < 7.5 {
            3
        } else {
            4
        }
    }

    #[inline]
    fn structure_type_from_gsm(gsm: &GlobalStructureMap) -> u8 {
        if gsm.repeat_ratio > 0.6 {
            4 // repetitive
        } else if gsm.repeat_ratio > 0.3 {
            3 // blocky
        } else if gsm.longest_run > 32 {
            3 // blocky-ish
        } else {
            5 // other/unknown
        }
    }

    pub fn lookup(
        &self,
        signature: u64,
        len: usize,
        entropy: f64,
        gsm: &GlobalStructureMap,
    ) -> Option<PatternEntry> {
        if let Some(e) = self.by_signature.get(&signature) {
            return Some(e.clone());
        }

        let lb = Self::bucket_for_len(len);
        let eb = Self::bucket_for_entropy(entropy);
        let st = Self::structure_type_from_gsm(gsm);
        let key = (lb, eb, st);

        if let Some(candidates) = self.by_bucket.get(&key) {
            // For now, just pick the highest repeat_score.
            let mut best: Option<PatternEntry> = None;
            let mut best_score = -1.0f32;
            for e in candidates {
                if e.shape.repeat_score > best_score {
                    best_score = e.shape.repeat_score;
                    best = Some(e.clone());
                }
            }
            return best;
        }

        None
    }
}

// ----------------------------------------------------------
// Analysis result for a payload
// ----------------------------------------------------------
#[derive(Clone, Debug)]
pub struct PatternAnalysis {
    pub signature: u64,
    pub gsm: GlobalStructureMap,
    pub matched: Option<PatternEntry>,
}

impl PatternLibrary {
    pub fn analyze(&self, payload: &[u8], entropy: f64) -> PatternAnalysis {
        let gsm = build_global_structure_map(payload);
        let signature = fast_hash64(payload);
        let matched = self.lookup(signature, payload.len(), entropy, &gsm);
        PatternAnalysis {
            signature,
            gsm,
            matched,
        }
    }
}
