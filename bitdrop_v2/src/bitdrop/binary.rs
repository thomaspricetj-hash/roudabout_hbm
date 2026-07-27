// ============================================================================
// BitDropBinaryEngine — lightweight binary compressor for PyMid structured data
// ============================================================================

use crate::engine::BitDrop3DEngine;
use crate::gpu;
use crate::model::global::GlobalModel;
use crate::get_global_model;
use crate::save_global_model;

pub struct BitDropBinaryEngine;

impl BitDropBinaryEngine {
    #[inline]
    pub fn compress(data: &[u8]) -> Vec<u8> {
        // Initialize GPU backend (safe to call multiple times)
        gpu::backend::init_gpu_backend();

        // Load global model
        let _model: &GlobalModel = get_global_model();

        // Use a small, fast BitDrop3D configuration for binary payloads
        let engine = BitDrop3DEngine::new((4, 4, 64), 8);

        let out = engine.encode(data);

        // Persist model updates
        save_global_model();

        out
    }

    #[inline]
    pub fn decompress(data: &[u8]) -> Option<Vec<u8>> {
        gpu::backend::init_gpu_backend();

        let _model: &GlobalModel = get_global_model();

        let engine = BitDrop3DEngine::new((4, 4, 64), 8);

        Some(engine.decode(data))
    }
}
