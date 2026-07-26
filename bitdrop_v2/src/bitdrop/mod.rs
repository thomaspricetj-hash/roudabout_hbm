// ============================================================================
// BitDrop Binary Compression Module
// Provides binary compression helpers for BitDrop3DEngine.
// This module now exposes encode_fast() and decode_fast() so that lib.rs
// can call the fast-path compressor without errors.
// ============================================================================

pub mod binary;

use crate::engine::BitDrop3DEngine;

// ---------------------------------------------------------------------------
// FAST PATH WRAPPERS
// These are intentionally shallow-collapse wrappers.
// They satisfy the Python API expected by lib.rs.
// ---------------------------------------------------------------------------

pub fn encode_fast(data: &[u8]) -> Vec<u8> {
    // Fast = shallow collapse, minimal overhead
    let engine = BitDrop3DEngine::new((4, 4, 64), 4);
    engine.encode(data)
}

pub fn decode_fast(data: &[u8]) -> Vec<u8> {
    let engine = BitDrop3DEngine::new((4, 4, 64), 4);
    engine.decode(data)
}
