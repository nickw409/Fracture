use crate::DType;

/// Opaque identifier for a tensor stored on a device.
/// Only the backend knows how to resolve this to actual device memory.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TensorId(pub u64);

/// An opaque handle to a tensor on a GPU device.
///
/// Contains metadata (shape, dtype) but no device pointers.
/// The engine can inspect shape and dtype but cannot access device memory directly.
/// Only the backend that created this tensor can operate on it.
#[derive(Debug, Clone)]
pub struct DeviceTensor {
    pub id: TensorId,
    pub shape: Vec<usize>,
    pub dtype: DType,
}

impl DeviceTensor {
    pub fn new(id: TensorId, shape: Vec<usize>, dtype: DType) -> Self {
        Self { id, shape, dtype }
    }

    /// Total number of elements in the tensor.
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    /// Total size in bytes on device.
    pub fn size_bytes(&self) -> usize {
        if self.dtype.is_packed() {
            // INT4: 2 elements per byte
            (self.numel() + 1) / 2
        } else {
            self.numel() * self.dtype.size_bytes()
        }
    }
}
