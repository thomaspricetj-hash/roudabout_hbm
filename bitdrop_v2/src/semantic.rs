use std::cmp::min;

pub fn semantic_forward_auto(
    data: &[u8],
    vector_stride: Option<usize>,
) -> (Vec<u8>, Vec<u8>, u8) {
    let stride = match vector_stride {
        Some(s) if s > 0 => s,
        _ => return (data.to_vec(), Vec::new(), 0),
    };

    if data.is_empty() {
        return (Vec::new(), Vec::new(), 0);
    }

    // sample prefix for scoring
    let sample_len = min(data.len(), 256 * stride);
    let sample = &data[..sample_len];

    // mode 0: none
    let mut best_mode = 0u8;
    let mut best_perm: Vec<u8> = Vec::new();
    let mut best_score = score_bytes_for_zlib(sample);

    // mode 1: delta only
    let delta_sample = forward_vector_delta(sample, stride);
    let score_delta = score_bytes_for_zlib(&delta_sample);
    if score_delta < best_score {
        best_score = score_delta;
        best_mode = 1;
        best_perm.clear();
    }

    // mode 2: perm + delta
    let perm = build_dim_permutation(sample, stride);
    if !perm.is_empty() {
        let perm_sample = apply_dim_permutation(sample, &perm, stride);
        let perm_delta_sample = forward_vector_delta(&perm_sample, stride);
        let score_perm_delta = score_bytes_for_zlib(&perm_delta_sample);
        if score_perm_delta < best_score {
            best_score = score_perm_delta;
            best_mode = 2;
            best_perm = perm.clone();
        }
    }

    // apply chosen mode to full data
    match best_mode {
        0 => (data.to_vec(), Vec::new(), 0),
        1 => (forward_vector_delta(data, stride), Vec::new(), 1),
        2 => {
            let mut tmp = data.to_vec();
            if !best_perm.is_empty() {
                tmp = apply_dim_permutation(&tmp, &best_perm, stride);
            }
            let full = forward_vector_delta(&tmp, stride);
            (full, best_perm, 2)
        }
        _ => data.to_vec().into(),
    }
}

pub fn semantic_inverse_with_mode(
    data: &[u8],
    perm: &[u8],
    mode: u8,
    vector_stride: Option<usize>,
) -> Vec<u8> {
    if data.is_empty() {
        return Vec::new();
    }

    match mode {
        0 => data.to_vec(),
        1 => {
            let stride = match vector_stride {
                Some(s) if s > 0 => s,
                _ => return data.to_vec(),
            };
            inverse_vector_delta(data, stride)
        }
        2 => {
            let stride = match vector_stride {
                Some(s) if s > 0 => s,
                _ => return data.to_vec(),
            };
            let mut tmp = inverse_vector_delta(data, stride);
            if !perm.is_empty() {
                tmp = apply_dim_inverse_permutation(&tmp, perm, stride);
            }
            tmp
        }
        _ => data.to_vec(),
    }
}

pub fn build_dim_permutation(data: &[u8], stride: usize) -> Vec<u8> {
    if stride == 0 {
        return Vec::new();
    }
    if data.len() < stride * 2 {
        return (0..stride as u8).collect();
    }

    let mut counts = vec![0usize; stride];
    let mut sums = vec![0f64; stride];
    let mut sums_sq = vec![0f64; stride];

    let mut i = 0;
    while i + stride <= data.len() {
        let row = &data[i..i + stride];
        for (j, &v) in row.iter().enumerate() {
            counts[j] += 1;
            let fv = v as f64;
            sums[j] += fv;
            sums_sq[j] += fv * fv;
        }
        i += stride;
    }

    let mut variances: Vec<(f64, u8)> = Vec::with_capacity(stride);
    for j in 0..stride {
        let c = counts[j];
        if c == 0 {
            variances.push((0.0, j as u8));
        } else {
            let mean = sums[j] / c as f64;
            let var = (sums_sq[j] / c as f64) - (mean * mean);
            variances.push((var.max(0.0), j as u8));
        }
    }

    variances.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    variances.into_iter().map(|(_, idx)| idx).collect()
}

pub fn apply_dim_permutation(data: &[u8], perm: &[u8], stride: usize) -> Vec<u8> {
    if perm.is_empty() || stride == 0 {
        return data.to_vec();
    }

    let mut out = vec![0u8; data.len()];
    let plen = perm.len();

    let mut base = 0;
    while base < data.len() {
        let end = min(base + stride, data.len());
        if end - base < plen {
            out[base..end].copy_from_slice(&data[base..end]);
        } else {
            for (new_pos, &old_pos) in perm.iter().enumerate() {
                out[base + new_pos] = data[base + old_pos as usize];
            }
        }
        base += stride;
    }
    out
}

pub fn apply_dim_inverse_permutation(data: &[u8], perm: &[u8], stride: usize) -> Vec<u8> {
    if perm.is_empty() || stride == 0 {
        return data.to_vec();
    }

    let mut out = vec![0u8; data.len()];
    let plen = perm.len();

    let mut inv = vec![0usize; plen];
    for (new_pos, &old_pos) in perm.iter().enumerate() {
        inv[old_pos as usize] = new_pos;
    }

    let mut base = 0;
    while base < data.len() {
        let end = min(base + stride, data.len());
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

pub fn forward_vector_delta(data: &[u8], stride: usize) -> Vec<u8> {
    if stride == 0 {
        return data.to_vec();
    }

    let mut out = vec![0u8; data.len()];
    let mut base = 0;

    while base < data.len() {
        let end = min(base + stride, data.len());
        let mut prev = 0u8;
        for i in base..end {
            let v = data[i];
            out[i] = v.wrapping_sub(prev);
            prev = v;
        }
        base += stride;
    }
    out
}

pub fn inverse_vector_delta(data: &[u8], stride: usize) -> Vec<u8> {
    if stride == 0 {
        return data.to_vec();
    }

    let mut out = vec![0u8; data.len()];
    let mut base = 0;

    while base < data.len() {
        let end = min(base + stride, data.len());
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

pub fn score_bytes_for_zlib(data: &[u8]) -> i32 {
    if data.is_empty() {
        return 0;
    }
    let mut transitions = 0i32;
    let mut zeros = 0i32;
    let mut prev = data[0];
    if prev == 0 {
        zeros += 1;
    }
    for &v in &data[1..] {
        if v != prev {
            transitions += 1;
        }
        if v == 0 {
            zeros += 1;
        }
        prev = v;
    }
    transitions - zeros
}
