// blocks.rs

use std::collections::HashMap;
use serde::{Serialize, Deserialize};

/// ============================================================
/// BinaryBlock3D — richer structural metadata + micro‑helpers
/// ============================================================
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BinaryBlock3D {
    pub data: Vec<u8>,
    pub shape: (usize, usize, usize),

    pub index: usize,
    pub region_id: usize,
    pub cluster_id: usize,
    pub local_index: usize,

    pub tags: HashMap<String, i64>,
}

impl BinaryBlock3D {
    pub fn new(data: Vec<u8>, shape: (usize, usize, usize), index: usize) -> Self {
        let mut tags = HashMap::new();

        if !data.is_empty() {
            let mut hist = [0u32; 256];
            let mut zeros = 0u32;
            let mut last = data[0];
            let mut runs = 1u32;

            for &b in &data {
                hist[b as usize] += 1;
                if b == 0 {
                    zeros += 1;
                }
                if b != last {
                    runs += 1;
                    last = b;
                }
            }

            let len = data.len() as f64;

            // Entropy
            let mut entropy = 0.0;
            for &c in &hist {
                if c > 0 {
                    let p = c as f64 / len;
                    entropy -= p * p.log2();
                }
            }

            // Byte skew
            let mut minb = 255u8;
            let mut maxb = 0u8;
            for &b in &data {
                if b < minb {
                    minb = b;
                }
                if b > maxb {
                    maxb = b;
                }
            }
            let skew = (maxb as i32 - minb as i32) as i64;

            tags.insert("entropy_x1000".into(), (entropy * 1000.0) as i64);
            tags.insert("size".into(), data.len() as i64);
            tags.insert(
                "zero_ratio_x1000".into(),
                ((zeros as f64 / len) * 1000.0) as i64,
            );
            tags.insert("skew".into(), skew);
            tags.insert("run_count".into(), runs as i64);
        }

        Self {
            data,
            shape,
            index,
            region_id: 0,
            cluster_id: 0,
            local_index: 0,
            tags,
        }
    }

    #[inline]
    pub fn voxel_count(&self) -> usize {
        self.shape.0 * self.shape.1 * self.shape.2
    }

    #[inline]
    pub fn is_cubic(&self) -> bool {
        let (x, y, z) = self.shape;
        x == y && y == z
    }

    #[inline]
    pub fn tag(&self, key: &str) -> Option<i64> {
        self.tags.get(key).copied()
    }

    #[inline]
    pub fn entropy(&self) -> f32 {
        self.tag("entropy_x1000")
            .map(|v| v as f32 / 1000.0)
            .unwrap_or(0.0)
    }

    #[inline]
    pub fn zero_ratio(&self) -> f32 {
        self.tag("zero_ratio_x1000")
            .map(|v| v as f32 / 1000.0)
            .unwrap_or(0.0)
    }

    #[inline]
    pub fn skew(&self) -> i64 {
        self.tag("skew").unwrap_or(0)
    }

    #[inline]
    pub fn run_count(&self) -> i64 {
        self.tag("run_count").unwrap_or(0)
    }

    #[inline]
    pub fn is_mostly_zero(&self, threshold: f32) -> bool {
        self.zero_ratio() >= threshold
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

/// ============================================================
/// Cube identity + ordering
/// ============================================================
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Ord, PartialOrd, Serialize, Deserialize)]
pub struct CubeId(pub u32);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CubePos {
    pub x: i32,
    pub y: i32,
    pub z: i32,
}

/// ============================================================
/// Cube data representation (raw bytes or reference)
/// ============================================================
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum CubeData {
    /// Fully materialized cube payload.
    Raw(Vec<u8>),

    /// Reference into a shared pattern buffer (engine‑resolved).
    Ref(u32),
}

/// ============================================================
/// Cube — with micro‑helpers for structured detection
/// ============================================================
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Cube {
    pub id: CubeId,
    pub layer: u16,
    pub pos: CubePos,
    pub shape: (usize, usize, usize),
    pub data: CubeData,

    pub quant_bits: u8,
    pub quant_vmin: u8,
    pub quant_vmax: u8,
}

impl Cube {
    pub fn new(
        id: CubeId,
        layer: u16,
        pos: CubePos,
        shape: (usize, usize, usize),
        data: Vec<u8>,
    ) -> Self {
        Self {
            id,
            layer,
            pos,
            shape,
            data: CubeData::Raw(data),
            quant_bits: 0,
            quant_vmin: 0,
            quant_vmax: 0,
        }
    }

    #[inline]
    fn idx(&self, x: usize, y: usize, z: usize) -> usize {
        let (bx, by, _bz) = self.shape;
        (z * by + y) * bx + x
    }

    #[inline]
    pub fn get(&self, x: usize, y: usize, z: usize) -> u8 {
        let i = self.idx(x, y, z);
        self.bytes()[i]
    }

    #[inline]
    pub fn set(&mut self, x: usize, y: usize, z: usize, v: u8) {
        let i = self.idx(x, y, z);
        let bytes = self.bytes_mut();
        bytes[i] = v;
    }

    #[inline]
    pub fn voxel_count(&self) -> usize {
        self.shape.0 * self.shape.1 * self.shape.2
    }

    #[inline]
    pub fn is_cubic(&self) -> bool {
        let (x, y, z) = self.shape;
        x == y && y == z
    }

    /// Unified immutable byte view over CubeData.
    #[inline]
    pub fn bytes(&self) -> &[u8] {
        match &self.data {
            CubeData::Raw(bytes) => bytes.as_slice(),
            CubeData::Ref(id) => {
                panic!("attempted to access bytes of unresolved cube reference {:?}", id);
            }
        }
    }

    /// Unified mutable byte view over CubeData::Raw.
    #[inline]
    pub fn bytes_mut(&mut self) -> &mut Vec<u8> {
        match &mut self.data {
            CubeData::Raw(bytes) => bytes,
            CubeData::Ref(id) => {
                panic!("attempted to mutate bytes of unresolved cube reference {:?}", id);
            }
        }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.bytes().len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Micro‑helper: entropy
    #[inline]
    pub fn entropy(&self) -> f32 {
        let bytes = self.bytes();

        let mut hist = [0u32; 256];
        for &b in bytes {
            hist[b as usize] += 1;
        }
        let len = bytes.len() as f32;
        if len == 0.0 {
            return 0.0;
        }
        let mut e = 0.0;
        for &c in &hist {
            if c > 0 {
                let p = c as f32 / len;
                e -= p * p.log2();
            }
        }
        e
    }

    /// Micro‑helper: skew
    #[inline]
    pub fn skew(&self) -> u8 {
        let bytes = self.bytes();

        let mut minb = 255u8;
        let mut maxb = 0u8;
        for &b in bytes {
            if b < minb {
                minb = b;
            }
            if b > maxb {
                maxb = b;
            }
        }
        maxb - minb
    }

    /// Micro‑helper: zero ratio
    #[inline]
    pub fn zero_ratio(&self) -> f32 {
        let bytes = self.bytes();

        if bytes.is_empty() {
            return 0.0;
        }
        let zeros = bytes.iter().filter(|&&b| b == 0).count();
        zeros as f32 / bytes.len() as f32
    }
}

/// ============================================================
/// Orientation + rotation
/// ============================================================
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Orientation(pub u8);

impl Orientation {
    #[inline]
    pub fn all() -> [Orientation; 24] {
        let mut out = [Orientation(0); 24];
        for i in 0..24 {
            out[i] = Orientation(i as u8);
        }
        out
    }
}

#[inline]
fn clamp_index(v: i32, n: usize) -> usize {
    if v <= 0 {
        0
    } else if v >= n as i32 {
        n - 1
    } else {
        v as usize
    }
}

#[inline]
fn apply_orientation(
    ori: Orientation,
    x: usize,
    y: usize,
    z: usize,
    n: usize,
) -> (usize, usize, usize) {
    let x = x as i32;
    let y = y as i32;
    let z = z as i32;
    let n1 = (n as i32) - 1;

    let (rx, ry, rz) = match ori.0 % 24 {
        0 => (x, y, z),
        1 => (x, z, n1 - y),
        2 => (x, n1 - y, n1 - z),
        3 => (x, n1 - z, y),

        4 => (y, x, n1 - z),
        5 => (y, z, x),
        6 => (y, n1 - x, z),
        7 => (y, n1 - z, n1 - x),

        8 => (z, x, y),
        9 => (z, y, n1 - x),
        10 => (z, n1 - x, n1 - y),
        11 => (z, n1 - y, x),

        12 => (n1 - x, y, n1 - z),
        13 => (n1 - x, z, y),
        14 => (n1 - x, n1 - y, z),
        15 => (n1 - x, n1 - z, n1 - y),

        16 => (n1 - y, x, z),
        17 => (n1 - y, z, n1 - x),
        18 => (n1 - y, n1 - x, n1 - z),
        19 => (n1 - y, n1 - z, x),

        20 => (n1 - z, x, n1 - y),
        21 => (n1 - z, y, x),
        22 => (n1 - z, n1 - x, y),
        23 => (n1 - z, n1 - y, n1 - x),

        _ => (x, y, z),
    };

    (
        clamp_index(rx, n),
        clamp_index(ry, n),
        clamp_index(rz, n),
    )
}

#[inline]
pub fn rotate_cube_data(
    shape: (usize, usize, usize),
    data: &[u8],
    ori: Orientation,
) -> Vec<u8> {
    let (sx, sy, sz) = shape;

    if sx == 0 || sy == 0 || sz == 0 {
        return data.to_vec();
    }

    // Only fully support rotations for cubic shapes (N×N×N).
    if sx != sy || sy != sz {
        return data.to_vec();
    }

    let n = sx;
    let mut out = vec![0u8; data.len()];

    for z in 0..n {
        for y in 0..n {
            for x in 0..n {
                let src_idx = (z * n + y) * n + x;
                let (nx, ny, nz) = apply_orientation(ori, x, y, z, n);
                let dst_idx = (nz * n + ny) * n + nx;
                out[dst_idx] = data[src_idx];
            }
        }
    }

    out
}

#[inline]
pub fn inverse_orientation(ori: Orientation) -> Orientation {
    ori
}

/// ============================================================
/// Block shape selection — smarter, payload‑aware
/// ============================================================
#[inline]
fn choose_block_shape(
    payload_len: usize,
    preferred: (usize, usize, usize),
) -> (usize, usize, usize) {
    let (bx, by, mut bz) = preferred;
    let base_block = bx * by * bz;

    if base_block == 0 {
        return preferred;
    }

    // Very small payloads → shrink depth but never below 4 and only if bz > 4
    if payload_len < base_block / 2 {
        if bz > 4 {
            bz = (bz / 2).max(4);
        }
    }

    // Very large payloads → allow deeper blocks to reduce cube count
    if payload_len > base_block * 16 {
        if bz < 128 {
            bz = (bz * 2).min(128);
        }
    }

    (bx, by, bz)
}

/// ============================================================
/// Block splitting
/// ============================================================
#[inline]
pub fn to_blocks(payload: &[u8], shape: (usize, usize, usize)) -> Vec<BinaryBlock3D> {
    let shape = choose_block_shape(payload.len(), shape);
    let block_size = shape.0 * shape.1 * shape.2;

    let mut blocks = Vec::new();
    let mut i = 0;

    while i < payload.len() {
        let end = (i + block_size).min(payload.len());
        let mut chunk = payload[i..end].to_vec();
        if chunk.len() < block_size {
            chunk.resize(block_size, 0);
        }
        blocks.push(BinaryBlock3D::new(chunk, shape, blocks.len()));
        i += block_size;
    }

    blocks
}

/// ============================================================
/// Region grouping — stable, deterministic
/// ============================================================
#[inline]
pub fn group_blocks(
    blocks: &[BinaryBlock3D],
    region_block_target: usize,
) -> Vec<Vec<BinaryBlock3D>> {
    if blocks.is_empty() {
        return Vec::new();
    }

    let mut regions = Vec::new();
    let step = region_block_target.max(1);
    let mut i = 0;

    while i < blocks.len() {
        let end = (i + step).min(blocks.len());
        let mut region = Vec::with_capacity(end - i);

        for (local_idx, b) in blocks[i..end].iter().enumerate() {
            let mut bb = b.clone();
            bb.region_id = regions.len();
            bb.local_index = local_idx;
            region.push(bb);
        }

        regions.push(region);
        i += step;
    }

    regions
}

/// ============================================================
/// Lift payload into cubes
/// ============================================================
#[inline]
pub fn lift_to_cubes(
    payload: &[u8],
    block_shape: (usize, usize, usize),
) -> (Vec<Cube>, usize) {
    let block_shape = choose_block_shape(payload.len(), block_shape);
    let block_size = block_shape.0 * block_shape.1 * block_shape.2;
    let total_blocks = (payload.len() + block_size - 1) / block_size;

    let mut cubes = Vec::with_capacity(total_blocks);
    let mut offset = 0usize;
    let mut next_id = 0u32;

    for ix in 0..total_blocks {
        let remaining = payload.len().saturating_sub(offset);
        let take = remaining.min(block_size);

        let mut data = payload[offset..offset + take].to_vec();
        if data.len() < block_size {
            data.resize(block_size, 0);
        }

        cubes.push(Cube::new(
            CubeId(next_id),
            0,
            CubePos {
                x: ix as i32,
                y: 0,
                z: 0,
            },
            block_shape,
            data,
        ));

        next_id += 1;
        offset += take;
    }

    (cubes, payload.len())
}

/// ============================================================
/// Flatten cubes — stable, deterministic, shape‑aware
/// ============================================================
#[inline]
pub fn flatten_from_cubes(cubes: &[Cube], original_len: usize) -> Vec<u8> {
    let mut sorted = cubes.to_vec();

    sorted.sort_by_key(|c| (c.pos.z, c.pos.y, c.pos.x, c.id));

    let mut out = Vec::with_capacity(original_len);
    for cube in sorted {
        match cube.data {
            CubeData::Raw(bytes) => out.extend_from_slice(&bytes),
            CubeData::Ref(id) => {
                panic!("attempted to flatten unresolved cube reference {:?}", id);
            }
        }
    }

    out.truncate(original_len);
    out
}
















