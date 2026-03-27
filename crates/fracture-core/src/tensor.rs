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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_numel_1d() {
        let t = DeviceTensor::new(TensorId(0), vec![10], DType::FP32);
        assert_eq!(t.numel(), 10);
    }

    #[test]
    fn test_numel_2d() {
        let t = DeviceTensor::new(TensorId(0), vec![3, 4], DType::FP32);
        assert_eq!(t.numel(), 12);
    }

    #[test]
    fn test_numel_3d() {
        let t = DeviceTensor::new(TensorId(0), vec![2, 3, 4], DType::FP32);
        assert_eq!(t.numel(), 24);
    }

    #[test]
    fn test_size_bytes_fp16() {
        let t = DeviceTensor::new(TensorId(0), vec![1024], DType::FP16);
        assert_eq!(t.size_bytes(), 1024 * 2);
    }

    #[test]
    fn test_size_bytes_int4_even() {
        // 1024 elements packed as INT4 => 512 bytes
        let t = DeviceTensor::new(TensorId(0), vec![1024], DType::INT4);
        assert_eq!(t.size_bytes(), 512);
    }

    #[test]
    fn test_size_bytes_int4_odd() {
        // 1023 elements packed as INT4 => (1023+1)/2 = 512 bytes
        let t = DeviceTensor::new(TensorId(0), vec![1023], DType::INT4);
        assert_eq!(t.size_bytes(), 512);
    }
}
