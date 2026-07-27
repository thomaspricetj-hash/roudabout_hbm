// transform.rs

use crate::blocks::{CubeId, CubePos, Orientation};
use serde::{Serialize, Deserialize};

/// Axes used for conceptual transforms (not currently used in engine logic)
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Axis {
    X,
    Y,
    Z,
}

/// Simple 4‑byte pattern tag (PTS-style hook).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PatternTag {
    pub pattern: u32,
    pub tag: u32,
}

/// Full reversible transform log entry.
/// Every variant contains all metadata required for perfect reconstruction.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum Transform {
    /// Rotation in the 24‑orientation cube group.
    Rotate {
        cube_id: CubeId,
        layer: u16,
        orientation: Orientation,
    },

    /// Integer 3D shift.
    Shift {
        cube_id: CubeId,
        layer: u16,
        dx: i8,
        dy: i8,
        dz: i8,
    },

    /// Lossless merge of multiple cubes into a single cube.
    /// Contains full metadata for perfect inverse reconstruction.
    Merge {
        new_cube_id: CubeId,
        layer_from: u16,
        layer_to: u16,

        /// IDs of original cubes in merge order.
        members: Vec<CubeId>,

        /// Byte offsets into merged cube data for each member.
        offsets: Vec<u32>,

        /// Original cube positions.
        original_positions: Vec<CubePos>,

        /// Original cube shapes.
        original_shapes: Vec<(usize, usize, usize)>,

        /// Original cube layers.
        original_layers: Vec<u16>,
    },

    /// Layer bookkeeping (informational).
    DropLayer {
        cube_id: CubeId,
        from_layer: u16,
        to_layer: u16,
    },

    /// PTS-style 4‑byte pattern tagging (hook for v8‑Ultra rules).
    PatternTag {
        cube_id: CubeId,
        layer: u16,
        tags: Vec<PatternTag>,
    },

    /// Reference into a shared pattern buffer (for cached cube payloads).
    PatternRef {
        cube_id: CubeId,
        layer: u16,
        ref_id: u32,
    },
}

/// A complete ordered transform log.
/// Stored exactly as applied during encode; reversed during decode.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct TransformLog {
    pub transforms: Vec<Transform>,
}

impl TransformLog {
    #[inline]
    pub fn new() -> Self {
        Self { transforms: Vec::new() }
    }

    #[inline]
    pub fn push(&mut self, t: Transform) {
        self.transforms.push(t);
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.transforms.is_empty()
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.transforms.len()
    }

    #[inline]
    pub fn iter(&self) -> impl Iterator<Item = &Transform> {
        self.transforms.iter()
    }
}

/// Compute inverse shift vector.
#[inline]
pub fn inverse_shift(dx: i8, dy: i8, dz: i8) -> (i8, i8, i8) {
    (-dx, -dy, -dz)
}

/// Apply a shift to a cube position.
#[inline]
pub fn apply_shift(pos: CubePos, dx: i8, dy: i8, dz: i8) -> CubePos {
    CubePos {
        x: pos.x + dx as i32,
        y: pos.y + dy as i32,
        z: pos.z + dz as i32,
    }
}




