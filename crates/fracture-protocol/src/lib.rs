//! Fracture wire protocol for distributed inference.
//!
//! This crate defines the binary framing format, message types, tensor
//! serialization, and async connection abstraction used for communication
//! between the coordinator and worker nodes.
//!
//! ## Architecture
//!
//! - [`frame`] — Frame header encoding/decoding, CRC32C, message type enum
//! - [`messages`] — All message payload structs (bincode-serializable)
//! - [`tensor`] — Compact tensor wire format (shape + dtype + compression + data)
//! - [`connection`] — Async framed connection over TCP

pub mod connection;
pub mod frame;
pub mod messages;
pub mod tensor;

// Re-export key types for convenience
pub use connection::{FramedConnection, FramedReader, FramedWriter};
pub use frame::{decode_frame_from_bytes, FrameHeader, MessageType};
pub use messages::*;
pub use tensor::TensorWireHeader;
