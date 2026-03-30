//! Wire protocol frame encoding and decoding.
//!
//! Frame format (big-endian):
//! ```text
//! Offset  Size   Field
//! ──────  ────   ─────
//! 0       2      Magic: 0x4652 ("FR")
//! 2       1      Version: 0x01
//! 3       1      Message Type: u8
//! 4       8      Sequence ID: u64
//! 12      4      Payload Length: u32
//! 16      N      Payload
//! 16+N    4      CRC32C over bytes [0..16+N)
//! ```

use fracture_core::{FractureError, Result};

/// Magic bytes: "FR" (0x46, 0x52).
pub const MAGIC: [u8; 2] = [0x46, 0x52];

/// Protocol version.
pub const VERSION: u8 = 0x01;

/// Fixed header size before the variable-length payload.
pub const HEADER_SIZE: usize = 16;

/// CRC32C trailer size.
pub const CRC_SIZE: usize = 4;

/// Maximum payload size: 256 MB. Prevents memory exhaustion from malformed frames.
pub const MAX_PAYLOAD_SIZE: u32 = 256 * 1024 * 1024;

/// Wire protocol message types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum MessageType {
    Register = 0x01,
    RegisterAck = 0x02,
    Forward = 0x03,
    ForwardResult = 0x04,
    Heartbeat = 0x05,
    HeartbeatAck = 0x06,
    CacheAlloc = 0x07,
    CacheFree = 0x08,
    Shutdown = 0x09,
    Error = 0x0A,
    CacheAllocAck = 0x0B,
    BatchedForward = 0x0C,
    BatchedForwardResult = 0x0D,
    WorkerReady = 0x0E,
}

impl MessageType {
    pub fn from_u8(v: u8) -> Result<Self> {
        match v {
            0x01 => Ok(Self::Register),
            0x02 => Ok(Self::RegisterAck),
            0x03 => Ok(Self::Forward),
            0x04 => Ok(Self::ForwardResult),
            0x05 => Ok(Self::Heartbeat),
            0x06 => Ok(Self::HeartbeatAck),
            0x07 => Ok(Self::CacheAlloc),
            0x08 => Ok(Self::CacheFree),
            0x09 => Ok(Self::Shutdown),
            0x0A => Ok(Self::Error),
            0x0B => Ok(Self::CacheAllocAck),
            0x0C => Ok(Self::BatchedForward),
            0x0D => Ok(Self::BatchedForwardResult),
            0x0E => Ok(Self::WorkerReady),
            _ => Err(FractureError::Protocol(format!(
                "unknown message type: 0x{v:02X}"
            ))),
        }
    }
}

/// Decoded frame header (without payload or CRC).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FrameHeader {
    pub msg_type: MessageType,
    pub seq_id: u64,
    pub payload_len: u32,
}

/// Encode a frame header into a 16-byte buffer.
pub fn encode_header(header: &FrameHeader, buf: &mut [u8; HEADER_SIZE]) {
    buf[0] = MAGIC[0];
    buf[1] = MAGIC[1];
    buf[2] = VERSION;
    buf[3] = header.msg_type as u8;
    buf[4..12].copy_from_slice(&header.seq_id.to_be_bytes());
    buf[12..16].copy_from_slice(&header.payload_len.to_be_bytes());
}

/// Decode a frame header from a 16-byte buffer.
pub fn decode_header(buf: &[u8; HEADER_SIZE]) -> Result<FrameHeader> {
    if buf[0] != MAGIC[0] || buf[1] != MAGIC[1] {
        return Err(FractureError::Protocol(format!(
            "invalid magic: 0x{:02X}{:02X}, expected 0x4652",
            buf[0], buf[1]
        )));
    }
    if buf[2] != VERSION {
        return Err(FractureError::Protocol(format!(
            "unsupported protocol version: {}, expected {}",
            buf[2], VERSION
        )));
    }
    let msg_type = MessageType::from_u8(buf[3])?;
    let seq_id = u64::from_be_bytes(buf[4..12].try_into().unwrap());
    let payload_len = u32::from_be_bytes(buf[12..16].try_into().unwrap());

    if payload_len > MAX_PAYLOAD_SIZE {
        return Err(FractureError::Protocol(format!(
            "payload length {} exceeds maximum {}",
            payload_len, MAX_PAYLOAD_SIZE
        )));
    }

    Ok(FrameHeader {
        msg_type,
        seq_id,
        payload_len,
    })
}

/// Encode a complete frame: header + payload + CRC32C.
pub fn encode_frame(header: &FrameHeader, payload: &[u8]) -> Vec<u8> {
    let total_size = HEADER_SIZE + payload.len() + CRC_SIZE;
    let mut buf = Vec::with_capacity(total_size);

    let mut hdr = [0u8; HEADER_SIZE];
    encode_header(header, &mut hdr);
    buf.extend_from_slice(&hdr);
    buf.extend_from_slice(payload);

    let crc = crc32c::crc32c(&buf);
    buf.extend_from_slice(&crc.to_be_bytes());

    buf
}

/// Verify CRC32C of a frame (header + payload bytes, excluding the CRC trailer).
pub fn verify_crc(header_and_payload: &[u8], expected_crc: u32) -> Result<()> {
    let actual = crc32c::crc32c(header_and_payload);
    if actual != expected_crc {
        return Err(FractureError::Protocol(format!(
            "CRC32C mismatch: expected 0x{expected_crc:08X}, got 0x{actual:08X}"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_message_type_roundtrip() {
        let types = [
            MessageType::Register,
            MessageType::RegisterAck,
            MessageType::Forward,
            MessageType::ForwardResult,
            MessageType::Heartbeat,
            MessageType::HeartbeatAck,
            MessageType::CacheAlloc,
            MessageType::CacheFree,
            MessageType::Shutdown,
            MessageType::Error,
            MessageType::CacheAllocAck,
            MessageType::BatchedForward,
            MessageType::BatchedForwardResult,
            MessageType::WorkerReady,
        ];
        for mt in types {
            let v = mt as u8;
            let decoded = MessageType::from_u8(v).unwrap();
            assert_eq!(decoded, mt);
        }
    }

    #[test]
    fn test_message_type_unknown() {
        assert!(MessageType::from_u8(0x00).is_err());
        assert!(MessageType::from_u8(0x0F).is_err());
        assert!(MessageType::from_u8(0xFF).is_err());
    }

    #[test]
    fn test_header_roundtrip() {
        let header = FrameHeader {
            msg_type: MessageType::Forward,
            seq_id: 0xDEADBEEF_CAFEBABE,
            payload_len: 1024,
        };
        let mut buf = [0u8; HEADER_SIZE];
        encode_header(&header, &mut buf);
        let decoded = decode_header(&buf).unwrap();
        assert_eq!(decoded, header);
    }

    #[test]
    fn test_header_bad_magic() {
        let mut buf = [0u8; HEADER_SIZE];
        buf[0] = 0x00;
        buf[1] = 0x00;
        buf[2] = VERSION;
        buf[3] = MessageType::Heartbeat as u8;
        assert!(decode_header(&buf).is_err());
    }

    #[test]
    fn test_header_bad_version() {
        let mut buf = [0u8; HEADER_SIZE];
        buf[0] = MAGIC[0];
        buf[1] = MAGIC[1];
        buf[2] = 0xFF;
        buf[3] = MessageType::Heartbeat as u8;
        assert!(decode_header(&buf).is_err());
    }

    #[test]
    fn test_header_payload_too_large() {
        let header = FrameHeader {
            msg_type: MessageType::Forward,
            seq_id: 0,
            payload_len: MAX_PAYLOAD_SIZE + 1,
        };
        let mut buf = [0u8; HEADER_SIZE];
        encode_header(&header, &mut buf);
        assert!(decode_header(&buf).is_err());
    }

    #[test]
    fn test_frame_encode_decode_with_crc() {
        let header = FrameHeader {
            msg_type: MessageType::Register,
            seq_id: 42,
            payload_len: 5,
        };
        let payload = b"hello";
        let frame = encode_frame(&header, payload);

        // Total: 16 header + 5 payload + 4 CRC = 25
        assert_eq!(frame.len(), 25);

        // Decode header
        let hdr_buf: [u8; HEADER_SIZE] = frame[..HEADER_SIZE].try_into().unwrap();
        let decoded = decode_header(&hdr_buf).unwrap();
        assert_eq!(decoded, header);

        // Extract payload
        let payload_end = HEADER_SIZE + decoded.payload_len as usize;
        let decoded_payload = &frame[HEADER_SIZE..payload_end];
        assert_eq!(decoded_payload, payload);

        // Verify CRC
        let crc_bytes: [u8; 4] = frame[payload_end..payload_end + CRC_SIZE]
            .try_into()
            .unwrap();
        let expected_crc = u32::from_be_bytes(crc_bytes);
        verify_crc(&frame[..payload_end], expected_crc).unwrap();
    }

    #[test]
    fn test_crc_detects_corruption() {
        let header = FrameHeader {
            msg_type: MessageType::Heartbeat,
            seq_id: 0,
            payload_len: 3,
        };
        let mut frame = encode_frame(&header, b"abc");

        // Corrupt one payload byte
        frame[HEADER_SIZE] ^= 0xFF;

        let payload_end = HEADER_SIZE + 3;
        let crc_bytes: [u8; 4] = frame[payload_end..payload_end + CRC_SIZE]
            .try_into()
            .unwrap();
        let expected_crc = u32::from_be_bytes(crc_bytes);
        assert!(verify_crc(&frame[..payload_end], expected_crc).is_err());
    }

    #[test]
    fn test_zero_payload_frame() {
        let header = FrameHeader {
            msg_type: MessageType::Shutdown,
            seq_id: 0,
            payload_len: 0,
        };
        let frame = encode_frame(&header, &[]);
        assert_eq!(frame.len(), HEADER_SIZE + CRC_SIZE);

        let hdr_buf: [u8; HEADER_SIZE] = frame[..HEADER_SIZE].try_into().unwrap();
        let decoded = decode_header(&hdr_buf).unwrap();
        assert_eq!(decoded.payload_len, 0);

        let crc_bytes: [u8; 4] = frame[HEADER_SIZE..HEADER_SIZE + CRC_SIZE]
            .try_into()
            .unwrap();
        let expected_crc = u32::from_be_bytes(crc_bytes);
        verify_crc(&frame[..HEADER_SIZE], expected_crc).unwrap();
    }

    #[test]
    fn test_header_big_endian_byte_layout() {
        let header = FrameHeader {
            msg_type: MessageType::Forward,
            seq_id: 0x0102030405060708,
            payload_len: 0x00010002,
        };
        let mut buf = [0u8; HEADER_SIZE];
        encode_header(&header, &mut buf);

        // Magic
        assert_eq!(buf[0], 0x46);
        assert_eq!(buf[1], 0x52);
        // Version
        assert_eq!(buf[2], 0x01);
        // Message type
        assert_eq!(buf[3], 0x03); // Forward = 0x03
        // Seq ID (big-endian u64)
        assert_eq!(&buf[4..12], &[0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08]);
        // Payload length (big-endian u32)
        assert_eq!(&buf[12..16], &[0x00, 0x01, 0x00, 0x02]);
    }

    #[test]
    fn test_crc_error_includes_values() {
        let result = verify_crc(&[0, 1, 2, 3], 0xDEADBEEF);
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("DEADBEEF"), "should show expected CRC: {msg}");
    }

    #[test]
    fn test_cache_free_empty_payload() {
        let seq_id = 0x00000000_0000002A; // cache sequence identifier
        let header = FrameHeader {
            msg_type: MessageType::CacheFree,
            seq_id,
            payload_len: 0,
        };
        let frame = encode_frame(&header, &[]);
        assert_eq!(frame.len(), HEADER_SIZE + CRC_SIZE);

        // Decode and verify header fields
        let hdr_buf: [u8; HEADER_SIZE] = frame[..HEADER_SIZE].try_into().unwrap();
        let decoded = decode_header(&hdr_buf).unwrap();
        assert_eq!(decoded.msg_type, MessageType::CacheFree);
        assert_eq!(decoded.seq_id, seq_id);
        assert_eq!(decoded.payload_len, 0);

        // Verify message type byte is 0x08
        assert_eq!(frame[3], 0x08);

        // Verify CRC
        let crc_bytes: [u8; 4] = frame[HEADER_SIZE..HEADER_SIZE + CRC_SIZE]
            .try_into()
            .unwrap();
        let expected_crc = u32::from_be_bytes(crc_bytes);
        verify_crc(&frame[..HEADER_SIZE], expected_crc).unwrap();
    }

    #[test]
    fn test_shutdown_empty_payload() {
        let seq_id = 0;
        let header = FrameHeader {
            msg_type: MessageType::Shutdown,
            seq_id,
            payload_len: 0,
        };
        let frame = encode_frame(&header, &[]);
        assert_eq!(frame.len(), HEADER_SIZE + CRC_SIZE);

        // Decode and verify header fields
        let hdr_buf: [u8; HEADER_SIZE] = frame[..HEADER_SIZE].try_into().unwrap();
        let decoded = decode_header(&hdr_buf).unwrap();
        assert_eq!(decoded.msg_type, MessageType::Shutdown);
        assert_eq!(decoded.seq_id, seq_id);
        assert_eq!(decoded.payload_len, 0);

        // Verify message type byte is 0x09
        assert_eq!(frame[3], 0x09);

        // Verify CRC
        let crc_bytes: [u8; 4] = frame[HEADER_SIZE..HEADER_SIZE + CRC_SIZE]
            .try_into()
            .unwrap();
        let expected_crc = u32::from_be_bytes(crc_bytes);
        verify_crc(&frame[..HEADER_SIZE], expected_crc).unwrap();
    }
}
