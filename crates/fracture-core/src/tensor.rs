use crate::{DType, FractureError};

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

    /// Validated constructor that rejects invalid shapes.
    /// Returns error for empty shape vectors or shapes with zero-sized dimensions.
    pub fn try_new(
        id: TensorId,
        shape: Vec<usize>,
        dtype: DType,
    ) -> Result<Self, FractureError> {
        if shape.is_empty() {
            return Err(FractureError::InvalidShape(
                "shape must have at least one dimension".into(),
            ));
        }
        for (i, &dim) in shape.iter().enumerate() {
            if dim == 0 {
                return Err(FractureError::InvalidShape(format!(
                    "dimension {i} is zero in shape {shape:?}"
                )));
            }
        }
        Ok(Self { id, shape, dtype })
    }

    /// Total number of elements in the tensor.
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    /// Total size in bytes on device.
    pub fn size_bytes(&self) -> usize {
        if self.dtype.is_packed() {
            // INT4: 2 elements per byte
            self.numel().div_ceil(2)
        } else {
            self.numel() * self.dtype.size_bytes()
        }
    }

    /// Reshape this tensor to a new shape, validating that the total element
    /// count is preserved. Returns error if numel differs.
    pub fn reshape(&self, new_shape: Vec<usize>) -> Result<Self, FractureError> {
        let new_numel: usize = new_shape.iter().product();
        if new_numel != self.numel() {
            return Err(FractureError::InvalidShape(format!(
                "cannot reshape tensor with {} elements to shape {:?} ({} elements)",
                self.numel(),
                new_shape,
                new_numel,
            )));
        }
        Ok(Self {
            id: self.id,
            shape: new_shape,
            dtype: self.dtype,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::FractureError;

    #[test]
    fn test_try_new_valid_shapes() {
        assert!(DeviceTensor::try_new(TensorId(0), vec![10], DType::FP32).is_ok());
        assert!(DeviceTensor::try_new(TensorId(0), vec![3, 4], DType::FP16).is_ok());
        assert!(DeviceTensor::try_new(TensorId(0), vec![2, 3, 4], DType::BF16).is_ok());
    }

    #[test]
    fn test_try_new_empty_shape() {
        let err = DeviceTensor::try_new(TensorId(0), vec![], DType::FP32).unwrap_err();
        assert!(matches!(err, FractureError::InvalidShape(_)));
        assert!(err.to_string().contains("at least one dimension"));
    }

    #[test]
    fn test_try_new_zero_dimension() {
        let err = DeviceTensor::try_new(TensorId(0), vec![4096, 0], DType::FP16).unwrap_err();
        assert!(matches!(err, FractureError::InvalidShape(_)));
        assert!(err.to_string().contains("dimension 1 is zero"));
    }

    #[test]
    fn test_try_new_zero_first_dimension() {
        let err = DeviceTensor::try_new(TensorId(0), vec![0, 128], DType::FP16).unwrap_err();
        assert!(matches!(err, FractureError::InvalidShape(_)));
        assert!(err.to_string().contains("dimension 0 is zero"));
    }

    #[test]
    fn test_reshape_valid() {
        let t = DeviceTensor::new(TensorId(0), vec![3, 4], DType::FP16);
        let reshaped = t.reshape(vec![12]).unwrap();
        assert_eq!(reshaped.shape, vec![12]);
        assert_eq!(reshaped.dtype, DType::FP16);
        assert_eq!(reshaped.id, TensorId(0));
    }

    #[test]
    fn test_reshape_numel_mismatch() {
        let t = DeviceTensor::new(TensorId(0), vec![4096], DType::FP16);
        let err = t.reshape(vec![64, 65]).unwrap_err();
        assert!(matches!(err, FractureError::InvalidShape(_)));
        assert!(err.to_string().contains("4096 elements"));
        assert!(err.to_string().contains("4160 elements"));
    }

    #[test]
    fn test_reshape_preserves_dtype() {
        let t = DeviceTensor::new(TensorId(5), vec![2, 3, 4], DType::INT8);
        let reshaped = t.reshape(vec![6, 4]).unwrap();
        assert_eq!(reshaped.dtype, DType::INT8);
        assert_eq!(reshaped.numel(), 24);
    }

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

    #[test]
    fn test_device_tensor_is_opaque() {
        // Verify DeviceTensor is composed of TensorId + shape + dtype (no raw pointers).
        let t = DeviceTensor::new(TensorId(42), vec![3, 4], DType::FP16);
        assert_eq!(t.id, TensorId(42));
        assert_eq!(t.shape, vec![3, 4]);
        assert_eq!(t.dtype, DType::FP16);
    }

    #[test]
    fn test_device_tensor_all_dtypes() {
        let dtypes = [DType::FP16, DType::FP32, DType::BF16, DType::INT8, DType::INT4];
        let shape = vec![4, 8]; // 32 elements

        for (i, dtype) in dtypes.iter().enumerate() {
            let t = DeviceTensor::new(TensorId(i as u64), shape.clone(), *dtype);
            assert_eq!(t.numel(), 32, "numel wrong for {dtype}");

            let expected_bytes = if dtype.is_packed() {
                32_usize.div_ceil(2) // INT4: ceil(32/2) = 16
            } else {
                32 * dtype.size_bytes()
            };
            assert_eq!(
                t.size_bytes(),
                expected_bytes,
                "size_bytes wrong for {dtype}: got {} expected {}",
                t.size_bytes(),
                expected_bytes
            );
        }

        // Verify specific expected values
        assert_eq!(DeviceTensor::new(TensorId(0), vec![32], DType::FP16).size_bytes(), 64);
        assert_eq!(DeviceTensor::new(TensorId(0), vec![32], DType::FP32).size_bytes(), 128);
        assert_eq!(DeviceTensor::new(TensorId(0), vec![32], DType::BF16).size_bytes(), 64);
        assert_eq!(DeviceTensor::new(TensorId(0), vec![32], DType::INT8).size_bytes(), 32);
        assert_eq!(DeviceTensor::new(TensorId(0), vec![32], DType::INT4).size_bytes(), 16);
    }
}
