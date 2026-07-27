// ============================================================================
// PyMid v4 — Simple Dictionary Builder (no clustering)
// ============================================================================

use std::collections::HashMap;

// ============================================================================
// SCORE A TOKEN (length + frequency + entropy-lite)
// ============================================================================

fn byte_entropy(data: &[u8]) -> f32 {
    if data.is_empty() {
        return 0.0;
    }

    let mut freq = [0u32; 256];
    for &b in data {
        freq[b as usize] += 1;
    }

    let n = data.len() as f32;
    let mut ent = 0.0;

    for &c in freq.iter() {
        if c == 0 {
            continue;
        }
        let p = c as f32 / n;
        ent -= p * p.log2();
    }

    ent
}

fn score_token(token: &[u8], freq: u32) -> f32 {
    if token.len() < 3 || freq < 3 {
        return 0.0;
    }

    let len = token.len() as f32;
    let ent = byte_entropy(token);

    if ent < 1.0 {
        return 0.0;
    }

    freq as f32 * (len - 2.0) * (1.0 + ent / 6.0)
}

// ============================================================================
// BUILD DICTIONARY (single-frame mode)
// ============================================================================

pub fn build_dictionary(tokens: &[Vec<u8>], max_dict: usize) -> Vec<Vec<u8>> {
    let mut freq: HashMap<&[u8], u32> = HashMap::new();

    for t in tokens {
        *freq.entry(t.as_slice()).or_insert(0) += 1;
    }

    let mut scored: Vec<(Vec<u8>, f32)> = freq
        .into_iter()
        .map(|(tok, count)| {
            let score = score_token(tok, count);
            (tok.to_vec(), score)
        })
        .collect();

    scored.retain(|(_, score)| *score > 0.0);

    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    let limit = max_dict.min(4096);
    if scored.len() > limit {
        scored.truncate(limit);
    }

    let mut dict: Vec<Vec<u8>> = scored.into_iter().map(|(tok, _)| tok).collect();

    dict.sort();
    dict.dedup();

    dict
}
