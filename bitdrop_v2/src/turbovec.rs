// src/turbovec.rs
use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;

/// Python wrapper for TurboVec-like vector encoder.
/// This version is safe, predictable, and ready for SIMD drop-in.
#[pyclass]
pub struct PyTurboVecEncoder {
    dim: usize,
    bit_width: usize,
}

#[pymethods]
impl PyTurboVecEncoder {
    /// Create a new encoder with a fixed vector dimension and bit width.
    #[new]
    pub fn new(dim: usize, bit_width: usize) -> PyResult<Self> {
        if dim == 0 {
            return Err(PyValueError::new_err("dim must be > 0"));
        }
        if bit_width == 0 || bit_width > 32 {
            return Err(PyValueError::new_err("bit_width must be in 1..=32"));
        }

        Ok(Self { dim, bit_width })
    }

    /// Encode a list of vectors (list[list[float]]) into bytes.
    /// This is a placeholder linear quantizer; replace with SIMD later.
    pub fn encode(&self, vectors: Vec<Vec<f32>>) -> PyResult<Vec<u8>> {
        // Preallocate for performance
        let total = vectors.len() * self.dim;
        let mut out = Vec::with_capacity(total);

        for (idx, v) in vectors.iter().enumerate() {
            if v.len() != self.dim {
                return Err(PyValueError::new_err(format!(
                    "Vector {} has length {}, expected {}",
                    idx,
                    v.len(),
                    self.dim
                )));
            }

            // Simple symmetric quantization to 0..255
            for &f in v {
                // Map [-1, 1] → [0, 255]
                let q = ((f + 1.0) * 127.5)
                    .clamp(0.0, 255.0)
                    as u8;

                out.push(q);
            }
        }

        Ok(out)
    }

    /// Expose dim to Python
    #[getter]
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Expose bit_width to Python
    #[getter]
    pub fn bit_width(&self) -> usize {
        self.bit_width
    }
}
