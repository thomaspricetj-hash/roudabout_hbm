// src/gpu/backend.rs

use crate::blocks::Cube;
use once_cell::sync::OnceCell;

// Import sibling modules (NOT child modules)
use super::wgpu_backend::WgpuBackend;



/// Represents whichever GPU backend is active.
enum BackendKind {
    Wgpu(WgpuBackend),
    // Cuda(CudaBackend),
}

/// Global GPU backend handle.
/// None = no GPU available.
/// Some = WGPU or CUDA backend active.
static GPU_BACKEND: OnceCell<Option<BackendKind>> = OnceCell::new();


/// Initialize the adaptive GPU backend.
///
/// Order:
/// 1. (future) CUDA
/// 2. WGPU
/// 3. CPU fallback
pub fn init_gpu_backend() {
    // Only run once
    if GPU_BACKEND.get().is_some() {
        return;
    }

    // --- FUTURE: Try CUDA first ---
    // if let Some(cuda) = CudaBackend::new() {
    //     GPU_BACKEND.set(Some(BackendKind::Cuda(cuda))).ok();
    //     return;
    // }

    // --- Try WGPU next ---
    let wgpu_backend = futures_lite::future::block_on(WgpuBackend::new());
    if let Some(wgpu) = wgpu_backend {
        GPU_BACKEND.set(Some(BackendKind::Wgpu(wgpu))).ok();
        return;
    }

    // --- No GPU available ---
    GPU_BACKEND.set(None).ok();
}


/// Returns true if any GPU backend is active.
#[inline]
pub fn gpu_available() -> bool {
    matches!(GPU_BACKEND.get(), Some(Some(_)))
}


/// Ask the GPU backend for the best merge pair.
///
/// Returns:
/// - Some((i, j, score)) if GPU computed a result
/// - None if no GPU backend is active
pub fn gpu_best_merge_pair(cluster: &[Cube]) -> Option<(usize, usize, i64)> {
    match GPU_BACKEND.get()? {
        Some(BackendKind::Wgpu(backend)) => backend.best_merge_pair(cluster),
        // Some(BackendKind::Cuda(backend)) => backend.best_merge_pair(cluster),
        None => None,
    }
}
