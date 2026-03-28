//! Compact tensor wire format for activation transfer.
//!
//! This module handles only the byte-level encoding/decoding of tensor
//! metadata and raw data. GPU-to-host and host-to-GPU copies are the
//! responsibility of the caller (which has access to a Backend).
//!
//! Wire format:
//! ```text
//! [2 bytes: ndim (u16 BE)]
//! [4 bytes × ndim: shape (u32 BE per dim)]
//! [1 byte: dtype (0=FP16, 1=FP32, 2=BF16, 3=INT8, 4=INT4)]
//! [1 byte: compression (0=None, 1=LZ4 future)]
//! [4 bytes: data_len (u32 BE)]
//! [data_len bytes: raw tensor data]
//! ```

use fracture_core::{DType, FractureError, Result};
use serde::{Deserialize, Serialize};

/// Tensor metadata sent on the wire alongside raw data.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TensorWireHeader {
    pub ndim: u16,
    pub shape: Vec<u32>,
    pub dtype: u8,
    pub compression: u8,
    pub data_len: u32,
}

/// DType to wire tag.
pub fn dtype_to_wire(dtype: DType) -> u8 {
    match dtype {
        DType::FP16 => 0,
        DType::FP32 => 1,
        DType::BF16 => 2,
        DType::INT8 => 3,
        DType::INT4 => 4,
    }
}

/// Wire tag to DType.
pub fn wire_to_dtype(tag: u8) -> Result<DType> {
    match tag {
        0 => Ok(DType::FP16),
        1 => Ok(DType::FP32),
        2 => Ok(DType::BF16),
        3 => Ok(DType::INT8),
        4 => Ok(DType::INT4),
        _ => Err(FractureError::Protocol(format!(
            "unknown dtype wire tag: {tag}"
        ))),
    }
}

/// Build a TensorWireHeader from shape, dtype, and raw data length.
pub fn make_header(shape: &[usize], dtype: DType, data_len: usize) -> TensorWireHeader {
    TensorWireHeader {
        ndim: shape.len() as u16,
        shape: shape.iter().map(|&d| d as u32).collect(),
        dtype: dtype_to_wire(dtype),
        compression: 0, // No compression in Phase 3
        data_len: data_len as u32,
    }
}

/// Encode a TensorWireHeader to bytes (manual binary layout, not bincode).
/// Returns the header bytes only — caller appends raw tensor data.
pub fn encode_tensor_header(header: &TensorWireHeader) -> Vec<u8> {
    let size = 2 + 4 * header.ndim as usize + 1 + 1 + 4;
    let mut buf = Vec::with_capacity(size);
    buf.extend_from_slice(&header.ndim.to_be_bytes());
    for &dim in &header.shape {
        buf.extend_from_slice(&dim.to_be_bytes());
    }
    buf.push(header.dtype);
    buf.push(header.compression);
    buf.extend_from_slice(&header.data_len.to_be_bytes());
    buf
}

/// Decode a TensorWireHeader from a byte slice. Returns the header and
/// the number of bytes consumed (not including raw tensor data).
pub fn decode_tensor_header(data: &[u8]) -> Result<(TensorWireHeader, usize)> {
    if data.len() < 2 {
        return Err(FractureError::Protocol(
            "tensor header too short for ndim".into(),
        ));
    }
    let ndim = u16::from_be_bytes([data[0], data[1]]);
    let shape_bytes = 4 * ndim as usize;
    let header_size = 2 + shape_bytes + 1 + 1 + 4;

    if data.len() < header_size {
        return Err(FractureError::Protocol(format!(
            "tensor header too short: need {} bytes, have {}",
            header_size,
            data.len()
        )));
    }

    let mut offset = 2;
    let mut shape = Vec::with_capacity(ndim as usize);
    for _ in 0..ndim {
        let dim = u32::from_be_bytes(data[offset..offset + 4].try_into().unwrap());
        shape.push(dim);
        offset += 4;
    }

    let dtype = data[offset];
    offset += 1;
    let compression = data[offset];
    offset += 1;
    let data_len = u32::from_be_bytes(data[offset..offset + 4].try_into().unwrap());
    offset += 4;

    // Validate dtype tag
    wire_to_dtype(dtype)?;

    if compression != 0 {
        return Err(FractureError::Protocol(format!(
            "unsupported compression type: {compression}"
        )));
    }

    Ok((
        TensorWireHeader {
            ndim,
            shape,
            dtype,
            compression,
            data_len,
        },
        offset,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dtype_wire_roundtrip() {
        let types = [DType::FP16, DType::FP32, DType::BF16, DType::INT8, DType::INT4];
        for dt in types {
            let tag = dtype_to_wire(dt);
            let decoded = wire_to_dtype(tag).unwrap();
            assert_eq!(decoded, dt);
        }
    }

    #[test]
    fn test_dtype_wire_unknown() {
        assert!(wire_to_dtype(5).is_err());
        assert!(wire_to_dtype(255).is_err());
    }

    #[test]
    fn test_tensor_header_roundtrip() {
        let header = make_header(&[1, 4096], DType::FP16, 8192);
        let bytes = encode_tensor_header(&header);
        let (decoded, consumed) = decode_tensor_header(&bytes).unwrap();
        assert_eq!(consumed, bytes.len());
        assert_eq!(decoded, header);
    }

    #[test]
    fn test_tensor_header_3d() {
        let header = make_header(&[128, 32, 128], DType::FP16, 128 * 32 * 128 * 2);
        let bytes = encode_tensor_header(&header);
        let (decoded, _) = decode_tensor_header(&bytes).unwrap();
        assert_eq!(decoded.ndim, 3);
        assert_eq!(decoded.shape, vec![128, 32, 128]);
        assert_eq!(decoded.data_len, 128 * 32 * 128 * 2);
    }

    #[test]
    fn test_tensor_header_fp32() {
        let header = make_header(&[128256], DType::FP32, 128256 * 4);
        let bytes = encode_tensor_header(&header);
        let (decoded, _) = decode_tensor_header(&bytes).unwrap();
        assert_eq!(decoded.dtype, 1); // FP32
        assert_eq!(decoded.data_len, 128256 * 4);
    }

    #[test]
    fn test_tensor_header_truncated() {
        assert!(decode_tensor_header(&[]).is_err());
        assert!(decode_tensor_header(&[0]).is_err());
        // ndim=2 but not enough bytes for shape
        assert!(decode_tensor_header(&[0, 2, 0, 0]).is_err());
    }

    #[test]
    fn test_tensor_header_with_trailing_data() {
        let header = make_header(&[1, 4096], DType::FP16, 8192);
        let mut bytes = encode_tensor_header(&header);
        // Append some fake tensor data
        bytes.extend_from_slice(&[0xAB; 100]);
        let (decoded, consumed) = decode_tensor_header(&bytes).unwrap();
        assert_eq!(decoded, header);
        // consumed should not include the trailing data
        assert_eq!(consumed, bytes.len() - 100);
    }

    #[test]
    fn test_make_header() {
        let h = make_header(&[512, 4096], DType::FP16, 512 * 4096 * 2);
        assert_eq!(h.ndim, 2);
        assert_eq!(h.shape, vec![512, 4096]);
        assert_eq!(h.dtype, 0);
        assert_eq!(h.compression, 0);
        assert_eq!(h.data_len, 512 * 4096 * 2);
    }

    #[test]
    fn test_unsupported_compression_rejected() {
        let mut header = make_header(&[1, 4096], DType::FP16, 8192);
        header.compression = 1; // LZ4 — not supported in Phase 3
        let mut bytes = encode_tensor_header(&header);
        // Manually set compression byte to 1
        // ndim(2) + shape(8) = 10 bytes, then dtype(1), then compression at offset 11
        let result = decode_tensor_header(&bytes);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("compression"));
    }

    #[test]
    fn test_tensor_header_big_endian_layout() {
        let header = make_header(&[256, 4096], DType::FP32, 0x00ABCDEF);
        let bytes = encode_tensor_header(&header);

        // ndim = 2 (big-endian u16)
        assert_eq!(bytes[0], 0x00);
        assert_eq!(bytes[1], 0x02);
        // shape[0] = 256 (big-endian u32)
        assert_eq!(&bytes[2..6], &[0x00, 0x00, 0x01, 0x00]);
        // shape[1] = 4096 (big-endian u32)
        assert_eq!(&bytes[6..10], &[0x00, 0x00, 0x10, 0x00]);
        // dtype = 1 (FP32)
        assert_eq!(bytes[10], 0x01);
        // compression = 0
        assert_eq!(bytes[11], 0x00);
        // data_len = 0x00ABCDEF (big-endian u32)
        assert_eq!(&bytes[12..16], &[0x00, 0xAB, 0xCD, 0xEF]);
    }
}
