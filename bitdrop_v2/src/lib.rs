use pyo3::prelude::*;
use pyo3::types::PyBytes;

use std::path::PathBuf;
use std::sync::OnceLock;

// ============================================================
// CORE MODULES
// ============================================================

pub mod blocks;
pub mod collapse;
pub mod cluster;
pub mod container;
pub mod engine;
pub mod metrics;
pub mod transform;

// NEW MODULES (required for librarian system)
pub mod pattern_library;
pub mod librarian;

// GLOBAL MODEL (long‑term learning across files)
pub mod model;

// GPU backend (WGPU only)
pub mod gpu;

// Predictor module (required by collapse.rs + engine.rs)
pub mod predictor;

// NEW: PyMid text/structured transform
pub mod pymid;

// NEW: BitDrop binary engine (required for structured‑binary decode)
pub mod bitdrop;

pub use engine::BitDrop3DEngine;
use crate::model::global::GlobalModel;

// ============================================================
// GLOBAL MODEL SINGLETON
// ============================================================

static GLOBAL_MODEL: OnceLock<GlobalModel> = OnceLock::new();

pub fn get_global_model() -> &'static GlobalModel {
    GLOBAL_MODEL.get_or_init(|| {
        let default_path = PathBuf::from("bitdrop_global_model.bin");
        let path = std::env::var("BITDROP_MODEL_PATH")
            .map(PathBuf::from)
            .unwrap_or(default_path);
        GlobalModel::load(path)
    })
}

pub fn save_global_model() {
    if let Some(model) = GLOBAL_MODEL.get() {
        model.save();
    }
}

// NOTE: OnceLock cannot be reset safely without interior mutability.
// For now this is a no‑op placeholder.
pub fn reset_global_model() {
    // Future: switch to RwLock/Mutex‑wrapped model for true reset.
}

// ============================================================
// HELPER: FAST ASCII CHECK
// ============================================================

#[inline]
fn is_mostly_ascii(data: &[u8]) -> bool {
    if data.is_empty() {
        return false;
    }
    let mut ascii = 0usize;
    for &b in data {
        if matches!(b, 0x09 | 0x0A | 0x0D | 0x20..=0x7E) {
            ascii += 1;
        }
    }
    ascii as f32 / data.len() as f32 >= 0.97
}

// ============================================================
// HELPER: STRUCTURE CHECK (JSON / CSV / logs)
//  — density‑based to avoid misclassifying random ASCII
// ============================================================

#[inline]
fn has_structure_markers(data: &[u8]) -> bool {
    if data.len() < 2048 {
        return false;
    }

    let mut jsonish = 0usize;
    let mut csvish = 0usize;
    let mut newline = 0usize;

    for &b in data {
        match b {
            b'{' | b'}' | b'[' | b']' | b':' | b'"' => jsonish += 1,
            b',' | b';' | b'|' => csvish += 1,
            b'\n' | b'\r' => newline += 1,
            _ => {}
        }
    }

    let n = data.len() as f32;

    let json_density = jsonish as f32 / n;
    let csv_density  = csvish as f32 / n;
    let nl_density   = newline as f32 / n;

    json_density > 0.0005 || csv_density > 0.0005 || nl_density > 0.0005
}

// ============================================================
// HELPER: NUMERIC u32 COUNTER DETECTION (binary structured)
// ============================================================

const NUMBIN_MAGIC: &[u8; 4] = b"NBIN";

pub fn looks_like_u32_counter(data: &[u8]) -> bool {
    if data.len() < 4 || data.len() % 4 != 0 {
        return false;
    }

    let mut same_delta = 0usize;
    let mut samples = 0usize;

    let step = ((data.len() / 4) / 128).max(1); // sample up to ~128 steps
    let mut idx = 0usize;

    while idx + 8 <= data.len() && samples < 256 {
        let a = u32::from_le_bytes([
            data[idx],
            data[idx + 1],
            data[idx + 2],
            data[idx + 3],
        ]);
        let b = u32::from_le_bytes([
            data[idx + 4],
            data[idx + 5],
            data[idx + 6],
            data[idx + 7],
        ]);
        let d = b.wrapping_sub(a);
        if d == 1 {
            same_delta += 1;
        }
        samples += 1;
        idx += step * 4;
    }

    samples > 0 && same_delta * 2 >= samples
}

fn encode_u32_counter_frame(data: &[u8]) -> Vec<u8> {
    let count = (data.len() / 4) as u32;
    let first = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
    let delta: u32 = 1;

    let mut out = Vec::with_capacity(4 + 1 + 4 + 4 + 4);
    out.extend_from_slice(NUMBIN_MAGIC); // magic
    out.push(1);                         // version
    out.extend_from_slice(&first.to_le_bytes());
    out.extend_from_slice(&delta.to_le_bytes());
    out.extend_from_slice(&count.to_le_bytes());

    out
}

fn decode_u32_counter_frame(data: &[u8]) -> Option<Vec<u8>> {
    if data.len() < 4 + 1 + 4 + 4 + 4 {
        return None;
    }
    if &data[0..4] != NUMBIN_MAGIC {
        return None;
    }

    let version = data[4];
    if version != 1 {
        return None;
    }

    let first = u32::from_le_bytes([data[5], data[6], data[7], data[8]]);
    let delta = u32::from_le_bytes([data[9], data[10], data[11], data[12]]);
    let count = u32::from_le_bytes([data[13], data[14], data[15], data[16]]) as usize;

    let mut out = Vec::with_capacity(count * 4);
    let mut v = first;
    for _ in 0..count {
        out.extend_from_slice(&v.to_le_bytes());
        v = v.wrapping_add(delta);
    }

    Some(out)
}

// ============================================================
// SIMPLE HEURISTIC: TEXT / STRUCTURED DETECTION
// ============================================================

pub fn looks_like_text_or_structured(data: &[u8]) -> bool {
    if data.is_empty() {
        return false;
    }

    if data.len() <= 256 {
        return true;
    }

    if !is_mostly_ascii(data) {
        return false;
    }

    if !has_structure_markers(data) {
        return false;
    }

    true
}

// ============================================================
// ENTROPY ESTIMATION (Shannon)
// ============================================================

pub fn estimate_entropy(data: &[u8]) -> f32 {
    if data.is_empty() {
        return 0.0;
    }
    let mut counts = [0u32; 256];
    for &b in data {
        counts[b as usize] += 1;
    }
    let total = data.len() as f32;
    let mut h = 0.0f32;
    for &c in &counts {
        if c == 0 {
            continue;
        }
        let p = c as f32 / total;
        h -= p * p.log2();
    }
    h
}

// ============================================================
// RUST‑NATIVE PUBLIC API (BASE)
// ============================================================

pub fn compress(data: &[u8]) -> Vec<u8> {
    if looks_like_u32_counter(data) {
        return encode_u32_counter_frame(data);
    }

    if looks_like_text_or_structured(data) {
        return crate::pymid::pymid_encode(data);
    }

    gpu::backend::init_gpu_backend();
    let _model = get_global_model();
    let engine = BitDrop3DEngine::new((4, 4, 64), 8);
    let out = engine.encode(data);
    save_global_model();
    out
}

pub fn decompress(data: &[u8]) -> Vec<u8> {
    if data.len() >= 4 && &data[0..4] == NUMBIN_MAGIC {
        if let Some(out) = decode_u32_counter_frame(data) {
            return out;
        }
    }

    if data.len() >= 2 && (&data[0..2] == b"PH" || (data.len() >= 4 && &data[0..4] == b"PM2\0")) {
        if let Some(out) = crate::pymid::pymid_decode_segmented(data) {
            return out;
        }
    }

    gpu::backend::init_gpu_backend();
    let _model = get_global_model();
    let engine = BitDrop3DEngine::new((4, 4, 64), 8);
    engine.decode(data)
}

pub fn gpu_available() -> bool {
    gpu::backend::gpu_available()
}

pub fn init_gpu_backend() {
    gpu::backend::init_gpu_backend();
}

// ============================================================
// PROFILED COMPRESSION MODES (FAST / ADAPTIVE / ORIGAMI / GPU)
// ============================================================

// FAST: minimal overhead, shallow BD3D, still uses heuristics for text/numeric.
pub fn compress_fast(data: &[u8]) -> Vec<u8> {
    if looks_like_u32_counter(data) {
        return encode_u32_counter_frame(data);
    }
    if looks_like_text_or_structured(data) {
        return crate::pymid::pymid_encode(data);
    }
    gpu::backend::init_gpu_backend();
    let _model = get_global_model();
    // Fast profile: fewer layers for speed.
    let engine = BitDrop3DEngine::new((4, 4, 64), 3);
    let out = engine.encode(data);
    save_global_model();
    out
}

pub fn decompress_fast(data: &[u8]) -> Vec<u8> {
    if data.len() >= 4 && &data[0..4] == NUMBIN_MAGIC {
        if let Some(out) = decode_u32_counter_frame(data) {
            return out;
        }
    }
    if data.len() >= 2 && (&data[0..2] == b"PH" || (data.len() >= 4 && &data[0..4] == b"PM2\0")) {
        if let Some(out) = crate::pymid::pymid_decode_segmented(data) {
            return out;
        }
    }
    gpu::backend::init_gpu_backend();
    let _model = get_global_model();
    let engine = BitDrop3DEngine::new((4, 4, 64), 3);
    engine.decode(data)
}

pub fn compress_origami(data: &[u8]) -> Vec<u8> {
    gpu::backend::init_gpu_backend();
    let _model = get_global_model();
    let engine = BitDrop3DEngine::new((4, 4, 64), 8);
    let out = engine.encode_origami(data);
    save_global_model();
    out
}

pub fn decompress_origami(data: &[u8]) -> Vec<u8> {
    gpu::backend::init_gpu_backend();
    let _model = get_global_model();
    let engine = BitDrop3DEngine::new((4, 4, 64), 8);
    engine.decode_origami(data)
}

// ADAPTIVE: choose profile based on entropy/size, but use existing encode/decode.
pub fn compress_adaptive(data: &[u8]) -> Vec<u8> {
    let h = estimate_entropy(data);
    let size = data.len();

    if looks_like_u32_counter(data) {
        return encode_u32_counter_frame(data);
    }
    if looks_like_text_or_structured(data) {
        return crate::pymid::pymid_encode(data);
    }

    gpu::backend::init_gpu_backend();
    let _model = get_global_model();

    // Simple adaptive heuristic:
    // - very small or very high entropy: fast
    // - medium entropy / larger size: BD3D
    // - low entropy / large size: origami
    let out = if size < 64 * 1024 || h > 7.5 {
        let engine = BitDrop3DEngine::new((4, 4, 64), 3);
        engine.encode(data)
    } else if h < 3.0 && size >= 8 * 1024 * 1024 {
        let engine = BitDrop3DEngine::new((4, 4, 64), 8);
        engine.encode_origami(data)
    } else {
        let engine = BitDrop3DEngine::new((4, 4, 64), 8);
        engine.encode(data)
    };

    save_global_model();
    out
}

pub fn decompress_adaptive(data: &[u8]) -> Vec<u8> {
    // Decode path can safely use generic decompress() which auto‑detects
    // NUMBIN / PyMid / BD3D / Zstd‑wrapped frames.
    decompress(data)
}

// GPU: currently just ensures GPU backend is initialized and uses BD3D.
pub fn compress_gpu(data: &[u8]) -> Vec<u8> {
    gpu::backend::init_gpu_backend();
    let _model = get_global_model();
    let engine = BitDrop3DEngine::new((4, 4, 64), 8);
    let out = engine.encode(data);
    save_global_model();
    out
}

pub fn decompress_gpu(data: &[u8]) -> Vec<u8> {
    gpu::backend::init_gpu_backend();
    let _model = get_global_model();
    let engine = BitDrop3DEngine::new((4, 4, 64), 8);
    engine.decode(data)
}

pub fn compress_bd3d(data: &[u8]) -> Vec<u8> {
    gpu::backend::init_gpu_backend();
    let _model = get_global_model();
    let engine = BitDrop3DEngine::new((4, 4, 64), 8);
    let out = engine.encode(data);
    save_global_model();
    out
}

pub fn decompress_bd3d(data: &[u8]) -> Vec<u8> {
    gpu::backend::init_gpu_backend();
    let _model = get_global_model();
    let engine = BitDrop3DEngine::new((4, 4, 64), 8);
    engine.decode(data)
}

pub fn compress_pymid(data: &[u8]) -> Vec<u8> {
    crate::pymid::pymid_encode(data)
}

pub fn decompress_pymid(data: &[u8]) -> Vec<u8> {
    crate::pymid::pymid_decode_segmented(data).unwrap_or_else(|| data.to_vec())
}

pub fn compress_numbin(data: &[u8]) -> Vec<u8> {
    if looks_like_u32_counter(data) {
        encode_u32_counter_frame(data)
    } else {
        data.to_vec()
    }
}

pub fn decompress_numbin(data: &[u8]) -> Vec<u8> {
    if data.len() >= 4 && &data[0..4] == NUMBIN_MAGIC {
        if let Some(out) = decode_u32_counter_frame(data) {
            return out;
        }
    }
    data.to_vec()
}

// ============================================================
// PROFILED DISPATCH (compress_with_profile / decompress_with_profile)
// ============================================================

pub fn compress_with_profile(data: &[u8], profile: &str) -> Vec<u8> {
    match profile {
        "fast" => compress_fast(data),
        "adaptive" => compress_adaptive(data),
        "origami" => compress_origami(data),
        "gpu" => compress_gpu(data),
        "bd3d" => compress_bd3d(data),
        "pymid" => compress_pymid(data),
        "numbin" => compress_numbin(data),
        _ => compress(data),
    }
}

pub fn decompress_with_profile(data: &[u8], profile: &str) -> Vec<u8> {
    match profile {
        "fast" => decompress_fast(data),
        "adaptive" => decompress_adaptive(data),
        "origami" => decompress_origami(data),
        "gpu" => decompress_gpu(data),
        "bd3d" => decompress_bd3d(data),
        "pymid" => decompress_pymid(data),
        "numbin" => decompress_numbin(data),
        _ => decompress(data),
    }
}

// ============================================================
// PYTHON WRAPPER CLASS (HIGH‑LEVEL ENGINE)
// ============================================================

#[pyclass]
pub struct PyBitDropEngine {
    inner: BitDrop3DEngine,
}

#[pymethods]
impl PyBitDropEngine {
    #[new]
    pub fn new(
        block_shape: (usize, usize, usize),
        _level: u32,
        _max_clusters: usize,
        _auto_tune_block_shape: bool,
        _region_block_target: usize,
        _use_4d_pairs: bool,
        _vector_stride: Option<usize>,
    ) -> Self {
        gpu::backend::init_gpu_backend();
        let _model = get_global_model();
        Self {
            inner: BitDrop3DEngine::new(block_shape, 8),
        }
    }

    pub fn encode<'py>(
        &self,
        py: Python<'py>,
        payload: &Bound<'py, pyo3::types::PyAny>,
    ) -> PyResult<Bound<'py, PyBytes>> {
        let bytes: Vec<u8> = payload.extract::<Vec<u8>>()
            .map_err(|_| pyo3::exceptions::PyTypeError::new_err(
                "encode expects bytes or bytearray",
            ))?;
        let out = self.inner.encode(&bytes);
        Ok(PyBytes::new_bound(py, &out))
    }

    pub fn decode<'py>(
        &self,
        py: Python<'py>,
        blob: &Bound<'py, pyo3::types::PyAny>,
    ) -> PyResult<Bound<'py, PyBytes>> {
        let bytes: Vec<u8> = blob.extract::<Vec<u8>>()
            .map_err(|_| pyo3::exceptions::PyTypeError::new_err(
                "decode expects bytes or bytearray",
            ))?;
        let out = self.inner.decode(&bytes);
        Ok(PyBytes::new_bound(py, &out))
    }

    pub fn set_profile(&mut self, _profile: &str) {
        // hook for future profile wiring
    }

    pub fn set_block_shape(&mut self, _shape: (usize, usize, usize)) {
        // hook for future block shape tuning
    }

    pub fn set_gpu(&mut self, _enabled: bool) {
        // hook for future GPU toggle
    }
}

// ============================================================
// PYTHON FUNCTIONS (FLAT API)
// ============================================================

#[pyfunction]
fn compress_py<'py>(py: Python<'py>, data: &Bound<'py, pyo3::types::PyAny>) -> PyResult<Bound<'py, PyBytes>> {
    let bytes: Vec<u8> = data.extract::<Vec<u8>>()
        .map_err(|_| pyo3::exceptions::PyTypeError::new_err(
            "compress() expects bytes or bytearray",
        ))?;
    let out = compress(&bytes);
    Ok(PyBytes::new_bound(py, &out))
}

#[pyfunction]
fn decompress_py<'py>(py: Python<'py>, data: &Bound<'py, pyo3::types::PyAny>) -> PyResult<Bound<'py, PyBytes>> {
    let bytes: Vec<u8> = data.extract::<Vec<u8>>()
        .map_err(|_| pyo3::exceptions::PyTypeError::new_err(
            "decompress() expects bytes or bytearray",
        ))?;
    let out = decompress(&bytes);
    Ok(PyBytes::new_bound(py, &out))
}

#[pyfunction]
fn compress_fast_py<'py>(py: Python<'py>, data: &Bound<'py, pyo3::types::PyAny>) -> PyResult<Bound<'py, PyBytes>> {
    let bytes: Vec<u8> = data.extract::<Vec<u8>>()
        .map_err(|_| pyo3::exceptions::PyTypeError::new_err(
            "compress_fast() expects bytes or bytearray",
        ))?;
    let out = compress_fast(&bytes);
    Ok(PyBytes::new_bound(py, &out))
}

#[pyfunction]
fn decompress_fast_py<'py>(py: Python<'py>, data: &Bound<'py, pyo3::types::PyAny>) -> PyResult<Bound<'py, PyBytes>> {
    let bytes: Vec<u8> = data.extract::<Vec<u8>>()
        .map_err(|_| pyo3::exceptions::PyTypeError::new_err(
            "decompress_fast() expects bytes or bytearray",
        ))?;
    let out = decompress_fast(&bytes);
    Ok(PyBytes::new_bound(py, &out))
}

#[pyfunction]
fn compress_origami_py<'py>(py: Python<'py>, data: &Bound<'py, pyo3::types::PyAny>) -> PyResult<Bound<'py, PyBytes>> {
    let bytes: Vec<u8> = data.extract::<Vec<u8>>()
        .map_err(|_| pyo3::exceptions::PyTypeError::new_err(
            "compress_origami() expects bytes or bytearray",
        ))?;
    let out = compress_origami(&bytes);
    Ok(PyBytes::new_bound(py, &out))
}

#[pyfunction]
fn decompress_origami_py<'py>(py: Python<'py>, data: &Bound<'py, pyo3::types::PyAny>) -> PyResult<Bound<'py, PyBytes>> {
    let bytes: Vec<u8> = data.extract::<Vec<u8>>()
        .map_err(|_| pyo3::exceptions::PyTypeError::new_err(
            "decompress_origami() expects bytes or bytearray",
        ))?;
    let out = decompress_origami(&bytes);
    Ok(PyBytes::new_bound(py, &out))
}

#[pyfunction]
fn compress_adaptive_py<'py>(py: Python<'py>, data: &Bound<'py, pyo3::types::PyAny>) -> PyResult<Bound<'py, PyBytes>> {
    let bytes: Vec<u8> = data.extract::<Vec<u8>>()
        .map_err(|_| pyo3::exceptions::PyTypeError::new_err(
            "compress_adaptive() expects bytes or bytearray",
        ))?;
    let out = compress_adaptive(&bytes);
    Ok(PyBytes::new_bound(py, &out))
}

#[pyfunction]
fn decompress_adaptive_py<'py>(py: Python<'py>, data: &Bound<'py, pyo3::types::PyAny>) -> PyResult<Bound<'py, PyBytes>> {
    let bytes: Vec<u8> = data.extract::<Vec<u8>>()
        .map_err(|_| pyo3::exceptions::PyTypeError::new_err(
            "decompress_adaptive() expects bytes or bytearray",
        ))?;
    let out = decompress_adaptive(&bytes);
    Ok(PyBytes::new_bound(py, &out))
}

#[pyfunction]
fn compress_gpu_py<'py>(py: Python<'py>, data: &Bound<'py, pyo3::types::PyAny>) -> PyResult<Bound<'py, PyBytes>> {
    let bytes: Vec<u8> = data.extract::<Vec<u8>>()
        .map_err(|_| pyo3::exceptions::PyTypeError::new_err(
            "compress_gpu() expects bytes or bytearray",
        ))?;
    let out = compress_gpu(&bytes);
    Ok(PyBytes::new_bound(py, &out))
}

#[pyfunction]
fn decompress_gpu_py<'py>(py: Python<'py>, data: &Bound<'py, pyo3::types::PyAny>) -> PyResult<Bound<'py, PyBytes>> {
    let bytes: Vec<u8> = data.extract::<Vec<u8>>()
        .map_err(|_| pyo3::exceptions::PyTypeError::new_err(
            "decompress_gpu() expects bytes or bytearray",
        ))?;
    let out = decompress_gpu(&bytes);
    Ok(PyBytes::new_bound(py, &out))
}

#[pyfunction]
fn compress_bd3d_py<'py>(py: Python<'py>, data: &Bound<'py, pyo3::types::PyAny>) -> PyResult<Bound<'py, PyBytes>> {
    let bytes: Vec<u8> = data.extract::<Vec<u8>>()
        .map_err(|_| pyo3::exceptions::PyTypeError::new_err(
            "compress_bd3d() expects bytes or bytearray",
        ))?;
    let out = compress_bd3d(&bytes);
    Ok(PyBytes::new_bound(py, &out))
}

#[pyfunction]
fn decompress_bd3d_py<'py>(py: Python<'py>, data: &Bound<'py, pyo3::types::PyAny>) -> PyResult<Bound<'py, PyBytes>> {
    let bytes: Vec<u8> = data.extract::<Vec<u8>>()
        .map_err(|_| pyo3::exceptions::PyTypeError::new_err(
            "decompress_bd3d() expects bytes or bytearray",
        ))?;
    let out = decompress_bd3d(&bytes);
    Ok(PyBytes::new_bound(py, &out))
}

#[pyfunction]
fn compress_pymid_py<'py>(py: Python<'py>, data: &Bound<'py, pyo3::types::PyAny>) -> PyResult<Bound<'py, PyBytes>> {
    let bytes: Vec<u8> = data.extract::<Vec<u8>>()
        .map_err(|_| pyo3::exceptions::PyTypeError::new_err(
            "compress_pymid() expects bytes or bytearray",
        ))?;
    let out = compress_pymid(&bytes);
    Ok(PyBytes::new_bound(py, &out))
}

#[pyfunction]
fn decompress_pymid_py<'py>(py: Python<'py>, data: &Bound<'py, pyo3::types::PyAny>) -> PyResult<Bound<'py, PyBytes>> {
    let bytes: Vec<u8> = data.extract::<Vec<u8>>()
        .map_err(|_| pyo3::exceptions::PyTypeError::new_err(
            "decompress_pymid() expects bytes or bytearray",
        ))?;
    let out = decompress_pymid(&bytes);
    Ok(PyBytes::new_bound(py, &out))
}

#[pyfunction]
fn compress_numbin_py<'py>(py: Python<'py>, data: &Bound<'py, pyo3::types::PyAny>) -> PyResult<Bound<'py, PyBytes>> {
    let bytes: Vec<u8> = data.extract::<Vec<u8>>()
        .map_err(|_| pyo3::exceptions::PyTypeError::new_err(
            "compress_numbin() expects bytes or bytearray",
        ))?;
    let out = compress_numbin(&bytes);
    Ok(PyBytes::new_bound(py, &out))
}

#[pyfunction]
fn decompress_numbin_py<'py>(py: Python<'py>, data: &Bound<'py, pyo3::types::PyAny>) -> PyResult<Bound<'py, PyBytes>> {
    let bytes: Vec<u8> = data.extract::<Vec<u8>>()
        .map_err(|_| pyo3::exceptions::PyTypeError::new_err(
            "decompress_numbin() expects bytes or bytearray",
        ))?;
    let out = decompress_numbin(&bytes);
    Ok(PyBytes::new_bound(py, &out))
}

#[pyfunction]
fn compress_with_profile_py<'py>(
    py: Python<'py>,
    data: &Bound<'py, pyo3::types::PyAny>,
    profile: &str,
) -> PyResult<Bound<'py, PyBytes>> {
    let bytes: Vec<u8> = data.extract::<Vec<u8>>()
        .map_err(|_| pyo3::exceptions::PyTypeError::new_err(
            "compress_with_profile() expects bytes or bytearray",
        ))?;
    let out = compress_with_profile(&bytes, profile);
    Ok(PyBytes::new_bound(py, &out))
}

#[pyfunction]
fn decompress_with_profile_py<'py>(
    py: Python<'py>,
    data: &Bound<'py, pyo3::types::PyAny>,
    profile: &str,
) -> PyResult<Bound<'py, PyBytes>> {
    let bytes: Vec<u8> = data.extract::<Vec<u8>>()
        .map_err(|_| pyo3::exceptions::PyTypeError::new_err(
            "decompress_with_profile() expects bytes or bytearray",
        ))?;
    let out = decompress_with_profile(&bytes, profile);
    Ok(PyBytes::new_bound(py, &out))
}

#[pyfunction]
fn get_entropy_py(data: &Bound<pyo3::types::PyAny>) -> PyResult<f32> {
    let bytes: Vec<u8> = data.extract::<Vec<u8>>()
        .map_err(|_| pyo3::exceptions::PyTypeError::new_err(
            "get_entropy() expects bytes or bytearray",
        ))?;
    Ok(estimate_entropy(&bytes))
}

#[pyfunction]
fn detect_structure_py(data: &Bound<pyo3::types::PyAny>) -> PyResult<bool> {
    let bytes: Vec<u8> = data.extract::<Vec<u8>>()
        .map_err(|_| pyo3::exceptions::PyTypeError::new_err(
            "detect_structure() expects bytes or bytearray",
        ))?;
    Ok(looks_like_text_or_structured(&bytes))
}

#[pyfunction]
fn detect_numeric_counter_py(data: &Bound<pyo3::types::PyAny>) -> PyResult<bool> {
    let bytes: Vec<u8> = data.extract::<Vec<u8>>()
        .map_err(|_| pyo3::exceptions::PyTypeError::new_err(
            "detect_numeric_counter() expects bytes or bytearray",
        ))?;
    Ok(looks_like_u32_counter(&bytes))
}

#[pyfunction]
fn gpu_available_py() -> PyResult<bool> {
    Ok(gpu_available())
}

#[pyfunction]
fn init_gpu_backend_py() -> PyResult<()> {
    init_gpu_backend();
    Ok(())
}

#[pyfunction]
fn save_global_model_py() -> PyResult<()> {
    save_global_model();
    Ok(())
}

#[pyfunction]
fn reset_global_model_py() -> PyResult<()> {
    reset_global_model();
    Ok(())
}

// ============================================================
// PYTHON MODULE
// ============================================================

#[pymodule]
fn bitdrop_v2(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyBitDropEngine>()?;

    m.add_function(wrap_pyfunction!(compress_py, m)?)?;
    m.add_function(wrap_pyfunction!(decompress_py, m)?)?;

    m.add_function(wrap_pyfunction!(compress_fast_py, m)?)?;
    m.add_function(wrap_pyfunction!(decompress_fast_py, m)?)?;

    m.add_function(wrap_pyfunction!(compress_origami_py, m)?)?;
    m.add_function(wrap_pyfunction!(decompress_origami_py, m)?)?;

    m.add_function(wrap_pyfunction!(compress_adaptive_py, m)?)?;
    m.add_function(wrap_pyfunction!(decompress_adaptive_py, m)?)?;

    m.add_function(wrap_pyfunction!(compress_gpu_py, m)?)?;
    m.add_function(wrap_pyfunction!(decompress_gpu_py, m)?)?;

    m.add_function(wrap_pyfunction!(compress_bd3d_py, m)?)?;
    m.add_function(wrap_pyfunction!(decompress_bd3d_py, m)?)?;

    m.add_function(wrap_pyfunction!(compress_pymid_py, m)?)?;
    m.add_function(wrap_pyfunction!(decompress_pymid_py, m)?)?;

    m.add_function(wrap_pyfunction!(compress_numbin_py, m)?)?;
    m.add_function(wrap_pyfunction!(decompress_numbin_py, m)?)?;

    m.add_function(wrap_pyfunction!(compress_with_profile_py, m)?)?;
    m.add_function(wrap_pyfunction!(decompress_with_profile_py, m)?)?;

    m.add_function(wrap_pyfunction!(get_entropy_py, m)?)?;
    m.add_function(wrap_pyfunction!(detect_structure_py, m)?)?;
    m.add_function(wrap_pyfunction!(detect_numeric_counter_py, m)?)?;

    m.add_function(wrap_pyfunction!(gpu_available_py, m)?)?;
    m.add_function(wrap_pyfunction!(init_gpu_backend_py, m)?)?;

    m.add_function(wrap_pyfunction!(save_global_model_py, m)?)?;
    m.add_function(wrap_pyfunction!(reset_global_model_py, m)?)?;

    Ok(())
}












