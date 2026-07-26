use std::cmp::Ordering;

use crate::blocks::{Cube, Orientation};
use crate::blocks::rotate_cube_data;

/// ============================================================
/// Low-level byte metrics + micro-helpers
/// ============================================================

#[inline]
fn entropy_from_hist_u32(hist: &[u32; 256], len: f32) -> f32 {
    if len == 0.0 {
        return 0.0;
    }
    let mut e = 0.0;
    for &c in hist {
        if c > 0 {
            let p = c as f32 / len;
            e -= p * p.log2();
        }
    }
    e
}

#[inline]
fn entropy_from_bytes_scalar(data: &[u8]) -> f32 {
    if data.is_empty() {
        return 0.0;
    }

    let mut hist = [0u32; 256];
    for &b in data {
        hist[b as usize] += 1;
    }

    entropy_from_hist_u32(&hist, data.len() as f32)
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[inline]
unsafe fn entropy_from_bytes_avx2(data: &[u8]) -> f32 {
    use core::arch::x86_64::*;

    if data.is_empty() {
        return 0.0;
    }

    let mut hist = [0u32; 256];

    let mut i = 0;
    let len = data.len();
    let chunk = 32;

    while i + chunk <= len {
        let ptr = data.as_ptr().add(i) as *const __m256i;
        let v = _mm256_loadu_si256(ptr);
        let mut tmp = [0u8; 32];
        _mm256_storeu_si256(tmp.as_mut_ptr() as *mut __m256i, v);
        for &b in &tmp {
            hist[b as usize] += 1;
        }
        i += chunk;
    }

    for &b in &data[i..] {
        hist[b as usize] += 1;
    }

    entropy_from_hist_u32(&hist, len as f32)
}

#[inline]
pub fn entropy_from_bytes(data: &[u8]) -> f32 {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe {
                return entropy_from_bytes_avx2(data);
            }
        }
    }
    entropy_from_bytes_scalar(data)
}

#[inline]
pub fn byte_skew_from_bytes(data: &[u8]) -> u8 {
    if data.is_empty() {
        return 0;
    }

    let mut minb = 255u8;
    let mut maxb = 0u8;
    for &b in data {
        if b < minb {
            minb = b;
        }
        if b > maxb {
            maxb = b;
        }
    }
    maxb.wrapping_sub(minb)
}

#[inline]
pub fn zero_ratio_from_bytes(data: &[u8]) -> f32 {
    if data.is_empty() {
        return 0.0;
    }
    let zeros = data.iter().filter(|&&b| b == 0).count();
    zeros as f32 / data.len() as f32
}

#[inline]
fn entropy_bucket_from_bytes(data: &[u8]) -> u64 {
    let e = entropy_from_bytes(data);
    ((e * 4.0).min(15.0).max(0.0) as u64) & 0xF
}

#[inline]
fn skew_bucket_from_bytes(data: &[u8]) -> u64 {
    byte_skew_from_bytes(data) as u64 & 0xFF
}

#[inline]
fn zero_bucket_from_bytes(data: &[u8]) -> u64 {
    let zr = zero_ratio_from_bytes(data);
    ((zr * 16.0).min(15.0).max(0.0) as u64) & 0xF
}

/// Micro-helper: run-length structure score.
#[inline]
fn run_length_score(data: &[u8]) -> i64 {
    if data.len() < 2 {
        return 0;
    }

    let mut score: i64 = 0;
    let mut current = data[0];
    let mut run_len: i64 = 1;

    for &b in &data[1..] {
        if b == current {
            run_len += 1;
        } else {
            if run_len > 1 {
                score += run_len * run_len;
            }
            current = b;
            run_len = 1;
        }
    }
    if run_len > 1 {
        score += run_len * run_len;
    }

    score
}

#[inline]
fn frequency_spread_penalty_scalar(data: &[u8]) -> i64 {
    if data.is_empty() {
        return 0;
    }

    let mut freq = [0u32; 256];
    for &b in data {
        freq[b as usize] += 1;
    }

    let mut nonzero = 0u32;
    let mut max_f = 0u32;
    for &f in &freq {
        if f > 0 {
            nonzero += 1;
            if f > max_f {
                max_f = f;
            }
        }
    }

    if nonzero == 0 {
        0
    } else {
        (nonzero as i64) * (data.len() as i64 - max_f as i64)
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[inline]
unsafe fn frequency_spread_penalty_avx2(data: &[u8]) -> i64 {
    use core::arch::x86_64::*;

    if data.is_empty() {
        return 0;
    }

    let mut freq = [0u32; 256];

    let mut i = 0;
    let len = data.len();
    let chunk = 32;

    while i + chunk <= len {
        let ptr = data.as_ptr().add(i) as *const __m256i;
        let v = _mm256_loadu_si256(ptr);
        let mut tmp = [0u8; 32];
        _mm256_storeu_si256(tmp.as_mut_ptr() as *mut __m256i, v);
        for &b in &tmp {
            freq[b as usize] += 1;
        }
        i += chunk;
    }

    for &b in &data[i..] {
        freq[b as usize] += 1;
    }

    let mut nonzero = 0u32;
    let mut max_f = 0u32;
    for &f in &freq {
        if f > 0 {
            nonzero += 1;
            if f > max_f {
                max_f = f;
            }
        }
    }

    if nonzero == 0 {
        0
    } else {
        (nonzero as i64) * (len as i64 - max_f as i64)
    }
}

#[inline]
fn frequency_spread_penalty(data: &[u8]) -> i64 {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe {
                return frequency_spread_penalty_avx2(data);
            }
        }
    }
    frequency_spread_penalty_scalar(data)
}

/// ============================================================
/// Zlib-style scoring
/// ============================================================

#[inline]
pub fn score_bytes_for_zlib(data: &[u8]) -> i32 {
    if data.is_empty() {
        return 0;
    }

    if data.len() < 32 {
        return (data.len() as i32) * 10;
    }

    let run_score = run_length_score(data);
    let spread_penalty = frequency_spread_penalty(data);

    let entropy = entropy_from_bytes(data);
    let skew = byte_skew_from_bytes(data) as i64;
    let zero_ratio = zero_ratio_from_bytes(data);

    let entropy_term = (entropy * 50.0) as i64;
    let skew_bonus = skew * 4;
    let zero_bonus = (zero_ratio * 1000.0) as i64;

    let base = (data.len() as i64) * 10;
    let score = base - run_score + spread_penalty + entropy_term - skew_bonus - zero_bonus;

    score.clamp(i32::MIN as i64, i32::MAX as i64) as i32
}

#[inline]
fn cube_bytes(cube: &Cube) -> &[u8] {
    cube.bytes()
}

#[inline]
pub fn score_cube_for_zlib(cube: &Cube) -> i32 {
    score_bytes_for_zlib(cube_bytes(cube))
}

#[inline]
pub fn cube_entropy(cube: &Cube) -> f32 {
    entropy_from_bytes(cube_bytes(cube))
}

#[inline]
pub fn cube_skew(cube: &Cube) -> u8 {
    byte_skew_from_bytes(cube_bytes(cube))
}

#[inline]
pub fn cube_zero_ratio(cube: &Cube) -> f32 {
    zero_ratio_from_bytes(cube_bytes(cube))
}

/// ============================================================
/// Signatures
/// ============================================================

#[inline]
pub fn cube_signature(cube: &Cube) -> u64 {
    let data = cube_bytes(cube);
    let mut h: u64 = 0xcbf29ce484222325;

    for (i, &b) in data.iter().enumerate() {
        let v = b as u64;
        h ^= v.wrapping_add((i as u64).wrapping_mul(0x100000001b3));
        h = h.wrapping_mul(0x100000001b3);
    }

    let (sx, sy, sz) = cube.shape;
    h ^= (sx as u64) << 40 ^ (sy as u64) << 20 ^ (sz as u64);

    h
}

#[inline]
pub fn rich_cube_signature(cube: &Cube) -> u64 {
    let base = cube_signature(cube);
    let data = cube_bytes(cube);

    let e_bucket = entropy_bucket_from_bytes(data);
    let skew_bucket = skew_bucket_from_bytes(data);
    let zr_bucket = zero_bucket_from_bytes(data);

    base
        ^ (e_bucket << 48)
        ^ (skew_bucket << 32)
        ^ (zr_bucket << 28)
}

/// ============================================================
/// Orientation scoring
/// ============================================================

pub fn choose_best_orientation(cube: &Cube) -> Orientation {
    let (sx, sy, sz) = cube.shape;
    let data_len = cube_bytes(cube).len();

    if sx == 0 || sy == 0 || sz == 0 {
        return Orientation(0);
    }
    if sx != sy || sy != sz {
        return Orientation(0);
    }
    if data_len > 16 * 1024 {
        return Orientation(0);
    }

    let shape = cube.shape;
    let original = cube_bytes(cube);

    let mut best_ori = Orientation(0);
    let mut best_score = i32::MAX;

    for ori in Orientation::all() {
        let rotated = rotate_cube_data(shape, original, ori);
        let score = score_bytes_for_zlib(&rotated);

        if score.cmp(&best_score) == Ordering::Less {
            best_score = score;
            best_ori = ori;
        }
    }

    best_ori
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::blocks::{Cube, CubeId, CubePos};

    #[test]
    fn orientation_scoring_prefers_valid_orientation() {
        let shape = (4usize, 4usize, 4usize);
        let n = shape.0 * shape.1 * shape.2;

        let mut structured = vec![0u8; n];
        for z in 0..shape.2 {
            let v = (z % 4) as u8;
            for y in 0..shape.1 {
                for x in 0..shape.0 {
                    let idx = (z * shape.1 + y) * shape.0 + x;
                    structured[idx] = v;
                }
            }
        }

        let cube_struct = Cube::new(
            CubeId(0),
            0,
            CubePos { x: 0, y: 0, z: 0 },
            shape,
            structured,
        );

        let ori_struct = choose_best_orientation(&cube_struct);
        assert!(ori_struct.0 < 24);
    }

    #[test]
    fn cube_signature_is_stable() {
        let shape = (4usize, 4usize, 4usize);
        let data = vec![1u8; shape.0 * shape.1 * shape.2];

        let cube = Cube::new(
            CubeId(0),
            0,
            CubePos { x: 0, y: 0, z: 0 },
            shape,
            data,
        );

        let sig1 = cube_signature(&cube);
        let sig2 = cube_signature(&cube);

        assert_eq!(sig1, sig2);
    }

    #[test]
    fn rich_signature_varies_with_structure() {
        let shape = (4usize, 4usize, 4usize);
        let n = shape.0 * shape.1 * shape.2;

        let data_a = vec![0u8; n];
        let mut data_b = vec![0u8; n];
        for i in 0..n {
            data_b[i] = (i as u8).wrapping_mul(37);
        }

        let cube_a = Cube::new(
            CubeId(0),
            0,
            CubePos { x: 0, y: 0, z: 0 },
            shape,
            data_a,
        );
        let cube_b = Cube::new(
            CubeId(1),
            0,
            CubePos { x: 1, y: 0, z: 0 },
            shape,
            data_b,
        );

        let sig_a = rich_cube_signature(&cube_a);
        let sig_b = rich_cube_signature(&cube_b);

        assert_ne!(sig_a, sig_b);
    }

    #[test]
    fn cube_micro_helpers_match_byte_helpers() {
        let shape = (4usize, 4usize, 4usize);
        let n = shape.0 * shape.1 * shape.2;
        let mut data = vec![0u8; n];
        for i in 0..n {
            data[i] = (i as u8).wrapping_mul(17);
        }

        let cube = Cube::new(
            CubeId(0),
            0,
            CubePos { x: 0, y: 0, z: 0 },
            shape,
            data.clone(),
        );

        assert!((cube_entropy(&cube) - entropy_from_bytes(&data)).abs() < 1e-5);
        assert_eq!(cube_skew(&cube), byte_skew_from_bytes(&data));
        assert!((cube_zero_ratio(&cube) - zero_ratio_from_bytes(&data)).abs() < 1e-5);
    }
}







