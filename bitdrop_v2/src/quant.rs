use crate::blocks::BinaryBlock3D;
use std::arch::x86_64::*;

//
// Quantization: quantize_block_with_range (4‑bit packed, SIMD‑accelerated when available)
//
pub fn quantize_block_with_range(
    block: &BinaryBlock3D,
    vmin: u8,
    vmax: u8,
) -> (Vec<u8>, f32, f32) {
    let data = &block.data;

    if data.is_empty() {
        return (Vec::new(), 0.0, 1.0);
    }

    // If vmax <= vmin → all zeros, scale=1, zero=vmin (Python parity)
    if vmax <= vmin {
        let n = data.len();
        let packed_len = (n + 1) / 2;
        return (vec![0u8; packed_len], 1.0, vmin as f32);
    }

    let scale = (vmax as f32 - vmin as f32) / 15.0;
    let zero = vmin as f32;
    let inv_scale = 1.0 / scale;

    let n = data.len();
    let packed_len = (n + 1) / 2;
    let mut packed = vec![0u8; packed_len];

    // SIMD path if available
    if is_x86_feature_detected!("sse2") && n >= 16 {
        unsafe {
            quantize_block_with_range_simd(data, vmin, inv_scale, &mut packed);
        }
    } else {
        quantize_block_with_range_scalar(data, vmin, inv_scale, &mut packed);
    }

    (packed, scale, zero)
}

fn quantize_block_with_range_scalar(
    data: &[u8],
    vmin: u8,
    inv_scale: f32,
    packed: &mut [u8],
) {
    for (i, &v) in data.iter().enumerate() {
        let qf = (v as f32 - vmin as f32) * inv_scale + 0.5;
        let mut q = qf as i32;

        if q < 0 {
            q = 0;
        } else if q > 15 {
            q = 15;
        }

        let byte_index = i / 2;
        if (i & 1) == 0 {
            packed[byte_index] = (q & 0x0F) as u8;
        } else {
            packed[byte_index] |= ((q & 0x0F) as u8) << 4;
        }
    }
}

unsafe fn quantize_block_with_range_simd(
    data: &[u8],
    vmin: u8,
    inv_scale: f32,
    packed: &mut [u8],
) {
    let v_vmin = _mm_set1_ps(vmin as f32);
    let v_inv_scale = _mm_set1_ps(inv_scale);

    let mut i = 0usize;
    let n = data.len();

    // Process 16 elements at a time
    while i + 16 <= n {
        let chunk = &data[i..i + 16];

        let mut fbuf = [0f32; 16];
        for (k, &b) in chunk.iter().enumerate() {
            fbuf[k] = b as f32;
        }

        let p0 = _mm_loadu_ps(&fbuf[0]);
        let p1 = _mm_loadu_ps(&fbuf[4]);
        let p2 = _mm_loadu_ps(&fbuf[8]);
        let p3 = _mm_loadu_ps(&fbuf[12]);

        let d0 = _mm_mul_ps(_mm_sub_ps(p0, v_vmin), v_inv_scale);
        let d1 = _mm_mul_ps(_mm_sub_ps(p1, v_vmin), v_inv_scale);
        let d2 = _mm_mul_ps(_mm_sub_ps(p2, v_vmin), v_inv_scale);
        let d3 = _mm_mul_ps(_mm_sub_ps(p3, v_vmin), v_inv_scale);

        _mm_storeu_ps(&mut fbuf[0], d0);
        _mm_storeu_ps(&mut fbuf[4], d1);
        _mm_storeu_ps(&mut fbuf[8], d2);
        _mm_storeu_ps(&mut fbuf[12], d3);

        for k in 0..16 {
            let qf = fbuf[k] + 0.5;
            let mut q = qf as i32;
            if q < 0 {
                q = 0;
            } else if q > 15 {
                q = 15;
            }

            let idx = i + k;
            let byte_index = idx / 2;
            if (idx & 1) == 0 {
                packed[byte_index] = (q & 0x0F) as u8;
            } else {
                packed[byte_index] |= ((q & 0x0F) as u8) << 4;
            }
        }

        i += 16;
    }

    // Tail
    if i < n {
        quantize_block_with_range_scalar(&data[i..], vmin, inv_scale, &mut packed[i / 2..]);
    }
}

//
// Dequantization: dequantize_block
//
pub fn dequantize_block(qdata: &[u8], scale: f32, zero: f32, n_elems: usize) -> Vec<u8> {
    if qdata.is_empty() || n_elems == 0 {
        return Vec::new();
    }

    let mut out = vec![0u8; n_elems];

    for i in 0..n_elems {
        let byte_index = i / 2;
        if byte_index >= qdata.len() {
            break;
        }

        let b = qdata[byte_index];

        let q = if (i & 1) == 0 {
            b & 0x0F
        } else {
            (b >> 4) & 0x0F
        };

        let mut v = (zero + q as f32 * scale + 0.5) as i32;

        if v < 0 {
            v = 0;
        } else if v > 255 {
            v = 255;
        }

        out[i] = v as u8;
    }

    out
}

//
// Mode selection: choose_mode_for_block
//
pub fn choose_mode_for_block(data: &[u8]) -> u8 {
    let raw_score = score_bytes_for_zlib(data);
    let delta_data = forward_delta(data);
    let delta_score = score_bytes_for_zlib(&delta_data);

    if delta_score < raw_score {
        1
    } else {
        0
    }
}

//
// Forward mode: apply_mode_forward
//
pub fn apply_mode_forward(data: &[u8], mode: u8) -> Vec<u8> {
    if mode == 1 {
        forward_delta(data)
    } else {
        data.to_vec()
    }
}

//
// Inverse mode: apply_mode_inverse
//
pub fn apply_mode_inverse(data: &[u8], mode: u8) -> Vec<u8> {
    if mode == 1 {
        inverse_delta(data)
    } else {
        data.to_vec()
    }
}

//
// Forward delta
//
fn forward_delta(data: &[u8]) -> Vec<u8> {
    if data.is_empty() {
        return Vec::new();
    }

    let mut out = Vec::with_capacity(data.len());
    let mut prev = 0u8;

    for &v in data {
        let d = v.wrapping_sub(prev);
        out.push(d);
        prev = v;
    }

    out
}

//
// Inverse delta
//
fn inverse_delta(data: &[u8]) -> Vec<u8> {
    if data.is_empty() {
        return Vec::new();
    }

    let mut out = Vec::with_capacity(data.len());
    let mut prev = 0u8;

    for &d in data {
        let v = d.wrapping_add(prev);
        out.push(v);
        prev = v;
    }

    out
}

//
// Heuristic: score_bytes_for_zlib
//
fn score_bytes_for_zlib(data: &[u8]) -> i32 {
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


