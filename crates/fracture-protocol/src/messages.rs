//! Wire protocol message payload types.
//!
//! All payloads are serialized with bincode inside the frame's payload section.
//! Tensor raw data is embedded as `Vec<u8>` within the payload structs —
//! the protocol layer handles only bytes, not GPU memory.

use crate::tensor::TensorWireHeader;
use fracture_core::ModelConfig;
use serde::{Deserialize, Serialize};

// ── 0x01 Register (Worker → Coordinator) ────────────────────────────────

/// Worker registration payload sent on initial connection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegisterPayload {
    pub node_id: String,
    pub gpu_model: String,
    pub gpu_memory_total: u64,
    pub gpu_memory_available: u64,
    pub compute_capability: (u32, u32),
    pub decode_ms_per_layer: f32,
    pub prefill_ms_per_layer_128: f32,
}

// ── 0x02 RegisterAck (Coordinator → Worker) ─────────────────────────────

/// Layer assignment and session config sent back to a registered worker.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegisterAckPayload {
    pub layer_start: u32,
    pub layer_end: u32,
    pub total_layers: u32,
    pub max_seq_len: u32,
    pub model_config: ModelConfig,
}

// ── 0x03 Forward (Coordinator → Worker) ─────────────────────────────────

/// Forward pass request. Contains either token IDs (head node) or
/// serialized activation tensors (middle/tail nodes).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForwardPayload {
    pub is_prefill: bool,
    pub positions: Vec<u32>,
    pub input: ForwardInputWire,
}

/// Discriminated input: token IDs for the head node, or a serialized
/// activation tensor for middle/tail nodes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ForwardInputWire {
    TokenIds {
        ids: Vec<u32>,
    },
    Activations {
        tensor_header: TensorWireHeader,
        tensor_data: Vec<u8>,
    },
}

// ── 0x03 BatchedForward (Coordinator → Worker) — uses same MessageType ──

/// Per-sequence metadata within a batched forward request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SequenceMetadataWire {
    pub seq_id: u64,
    /// Number of tokens for this sequence in this batch.
    pub num_tokens: usize,
    /// Absolute positions for RoPE.
    pub positions: Vec<u32>,
    /// Block table for paged attention (physical block IDs).
    pub block_table: Vec<u32>,
    /// Total KV cache length (for attention masking).
    pub cache_seq_len: usize,
    /// Valid tokens in last block.
    pub last_block_tokens: usize,
}

/// Batched forward pass request. Contains multiple sequences'
/// token IDs concatenated together, plus per-sequence metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchedForwardPayload {
    pub is_prefill: bool,
    pub sequences: Vec<SequenceMetadataWire>,
    pub input: ForwardInputWire,
}

/// Batched forward result with per-sequence logit offsets.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchedForwardResultPayload {
    /// Per-sequence output. For intermediate workers, a single concatenated
    /// activation tensor. For the tail worker, per-sequence logits.
    pub output: ForwardOutputWire,
    /// Number of sequences in the batch.
    pub num_sequences: usize,
    /// Byte offsets into the logits data for each sequence (tail only).
    /// For activations, this is empty (single concatenated tensor).
    pub logit_offsets: Vec<usize>,
}

// ── 0x04 ForwardResult (Worker → Coordinator) ───────────────────────────

/// Forward pass result. Tail nodes return logits; head/middle nodes
/// return serialized activation tensors.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForwardResultPayload {
    pub output: ForwardOutputWire,
}

/// Discriminated output: logits from the tail node (raw f32 LE bytes),
/// or a serialized activation tensor from head/middle nodes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ForwardOutputWire {
    Logits {
        data: Vec<u8>,
    },
    Activations {
        tensor_header: TensorWireHeader,
        tensor_data: Vec<u8>,
    },
}

// ── 0x05 Heartbeat (Bidirectional) ──────────────────────────────────────

/// Heartbeat probe. The nonce correlates request with acknowledgement.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeartbeatPayload {
    pub timestamp_ns: u64,
    pub nonce: u64,
}

// ── 0x06 HeartbeatAck (Bidirectional) ───────────────────────────────────

/// Heartbeat acknowledgement with worker health stats.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeartbeatAckPayload {
    pub timestamp_echo: u64,
    pub nonce_echo: u64,
    pub gpu_memory_used: u64,
    pub active_sequences: u32,
    /// Free blocks in this worker's paged KV cache pool.
    /// 0 if paged mode is not active.
    #[serde(default)]
    pub free_blocks: u32,
}

// ── 0x07 CacheAlloc (Coordinator → Worker) ──────────────────────────────

// ── 0x07 CacheAlloc — no payload (seq_id in frame header) ──────────────

// ── 0x08 CacheFree — no payload (seq_id in frame header) ────────────────
// ── 0x09 Shutdown — no payload ──────────────────────────────────────────

// ── 0x0A Error (Bidirectional) ──────────────────────────────────────────

/// Error codes for the Error message type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u32)]
pub enum ErrorCode {
    Internal = 1,
    OutOfMemory = 2,
    InvalidSequence = 3,
    ProtocolViolation = 4,
}

/// Error payload with a machine-readable code and human-readable message.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorPayload {
    pub error_code: ErrorCode,
    pub message: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: bincode round-trip for any Serialize + Deserialize type.
    fn bincode_roundtrip<T: Serialize + for<'de> Deserialize<'de> + std::fmt::Debug>(val: &T) -> T {
        let bytes = bincode::serialize(val).unwrap();
        bincode::deserialize(&bytes).unwrap()
    }

    #[test]
    fn test_register_payload_roundtrip() {
        let payload = RegisterPayload {
            node_id: "worker-0".into(),
            gpu_model: "NVIDIA RTX 3090".into(),
            gpu_memory_total: 24 * 1024 * 1024 * 1024,
            gpu_memory_available: 22 * 1024 * 1024 * 1024,
            compute_capability: (8, 6),
            decode_ms_per_layer: 1.1,
            prefill_ms_per_layer_128: 3.5,
        };
        let decoded: RegisterPayload = bincode_roundtrip(&payload);
        assert_eq!(decoded.node_id, "worker-0");
        assert_eq!(decoded.compute_capability, (8, 6));
        assert!((decoded.decode_ms_per_layer - 1.1).abs() < 1e-6);
    }

    #[test]
    fn test_register_ack_payload_roundtrip() {
        let payload = RegisterAckPayload {
            layer_start: 0,
            layer_end: 16,
            total_layers: 32,
            max_seq_len: 4096,
            model_config: ModelConfig {
                hidden_size: 4096,
                num_layers: 32,
                num_q_heads: 32,
                num_kv_heads: 8,
                head_dim: 128,
                intermediate_size: 14336,
                vocab_size: 128256,
                rope_theta: 500000.0,
                rms_norm_eps: 1e-5,
                max_seq_len: 8192,
            },
        };
        let decoded: RegisterAckPayload = bincode_roundtrip(&payload);
        assert_eq!(decoded.layer_start, 0);
        assert_eq!(decoded.layer_end, 16);
        assert_eq!(decoded.model_config.hidden_size, 4096);
        assert_eq!(decoded.model_config.vocab_size, 128256);
    }

    #[test]
    fn test_forward_payload_token_ids_roundtrip() {
        let payload = ForwardPayload {
            is_prefill: true,
            positions: vec![0, 1, 2, 3],
            input: ForwardInputWire::TokenIds {
                ids: vec![128000, 791, 1401, 315],
            },
        };
        let decoded: ForwardPayload = bincode_roundtrip(&payload);
        assert!(decoded.is_prefill);
        assert_eq!(decoded.positions, vec![0, 1, 2, 3]);
        match decoded.input {
            ForwardInputWire::TokenIds { ids } => {
                assert_eq!(ids, vec![128000, 791, 1401, 315]);
            }
            _ => panic!("expected TokenIds"),
        }
    }

    #[test]
    fn test_forward_payload_activations_roundtrip() {
        let header = TensorWireHeader {
            ndim: 2,
            shape: vec![1, 4096],
            dtype: 0, // FP16
            compression: 0,
            data_len: 8192,
        };
        let tensor_data = vec![0xAB; 8192];
        let payload = ForwardPayload {
            is_prefill: false,
            positions: vec![47],
            input: ForwardInputWire::Activations {
                tensor_header: header,
                tensor_data: tensor_data.clone(),
            },
        };
        let decoded: ForwardPayload = bincode_roundtrip(&payload);
        assert!(!decoded.is_prefill);
        match decoded.input {
            ForwardInputWire::Activations {
                tensor_header,
                tensor_data: data,
            } => {
                assert_eq!(tensor_header.ndim, 2);
                assert_eq!(tensor_header.shape, vec![1, 4096]);
                assert_eq!(tensor_header.dtype, 0);
                assert_eq!(tensor_header.data_len, 8192);
                assert_eq!(data, tensor_data);
            }
            _ => panic!("expected Activations"),
        }
    }

    #[test]
    fn test_forward_result_logits_roundtrip() {
        // Simulate top-K logits: small payload
        let logit_bytes = vec![0u8; 128256 * 4]; // full vocab FP32
        let payload = ForwardResultPayload {
            output: ForwardOutputWire::Logits {
                data: logit_bytes.clone(),
            },
        };
        let decoded: ForwardResultPayload = bincode_roundtrip(&payload);
        match decoded.output {
            ForwardOutputWire::Logits { data } => {
                assert_eq!(data.len(), 128256 * 4);
            }
            _ => panic!("expected Logits"),
        }
    }

    #[test]
    fn test_heartbeat_roundtrip() {
        let payload = HeartbeatPayload {
            timestamp_ns: 1234567890,
            nonce: 42,
        };
        let decoded: HeartbeatPayload = bincode_roundtrip(&payload);
        assert_eq!(decoded.timestamp_ns, 1234567890);
        assert_eq!(decoded.nonce, 42);
    }

    #[test]
    fn test_heartbeat_ack_roundtrip() {
        let payload = HeartbeatAckPayload {
            timestamp_echo: 1234567890,
            nonce_echo: 42,
            gpu_memory_used: 15 * 1024 * 1024 * 1024,
            active_sequences: 3,
            free_blocks: 0,
        };
        let decoded: HeartbeatAckPayload = bincode_roundtrip(&payload);
        assert_eq!(decoded.nonce_echo, 42);
        assert_eq!(decoded.active_sequences, 3);
    }

    #[test]
    fn test_heartbeat_ack_free_blocks_nonzero_roundtrip() {
        let payload = HeartbeatAckPayload {
            timestamp_echo: 0,
            nonce_echo: 0,
            gpu_memory_used: 0,
            active_sequences: 5,
            free_blocks: 1024,
        };
        let decoded: HeartbeatAckPayload = bincode_roundtrip(&payload);
        assert_eq!(decoded.free_blocks, 1024);
        assert_eq!(decoded.active_sequences, 5);
    }

    #[test]
    fn test_batched_forward_payload_token_ids_roundtrip() {
        let payload = BatchedForwardPayload {
            is_prefill: true,
            sequences: vec![
                SequenceMetadataWire {
                    seq_id: 1,
                    num_tokens: 3,
                    positions: vec![0, 1, 2],
                    block_table: vec![0, 5],
                    cache_seq_len: 3,
                    last_block_tokens: 3,
                },
                SequenceMetadataWire {
                    seq_id: 2,
                    num_tokens: 1,
                    positions: vec![10],
                    block_table: vec![1, 2, 7],
                    cache_seq_len: 11,
                    last_block_tokens: 11,
                },
            ],
            input: ForwardInputWire::TokenIds {
                ids: vec![128000, 791, 1401, 42],
            },
        };
        let decoded: BatchedForwardPayload = bincode_roundtrip(&payload);
        assert!(decoded.is_prefill);
        assert_eq!(decoded.sequences.len(), 2);
        assert_eq!(decoded.sequences[0].seq_id, 1);
        assert_eq!(decoded.sequences[0].num_tokens, 3);
        assert_eq!(decoded.sequences[0].positions, vec![0, 1, 2]);
        assert_eq!(decoded.sequences[0].block_table, vec![0, 5]);
        assert_eq!(decoded.sequences[1].seq_id, 2);
        assert_eq!(decoded.sequences[1].cache_seq_len, 11);
        match decoded.input {
            ForwardInputWire::TokenIds { ids } => {
                assert_eq!(ids, vec![128000, 791, 1401, 42]);
            }
            _ => panic!("expected TokenIds"),
        }
    }

    #[test]
    fn test_batched_forward_payload_activations_roundtrip() {
        let header = TensorWireHeader {
            ndim: 2,
            shape: vec![4, 4096],
            dtype: 0,
            compression: 0,
            data_len: 4 * 4096 * 2,
        };
        let tensor_data = vec![0xCD; 4 * 4096 * 2];
        let payload = BatchedForwardPayload {
            is_prefill: false,
            sequences: vec![SequenceMetadataWire {
                seq_id: 99,
                num_tokens: 4,
                positions: vec![0, 1, 2, 3],
                block_table: vec![10],
                cache_seq_len: 4,
                last_block_tokens: 4,
            }],
            input: ForwardInputWire::Activations {
                tensor_header: header,
                tensor_data: tensor_data.clone(),
            },
        };
        let decoded: BatchedForwardPayload = bincode_roundtrip(&payload);
        assert!(!decoded.is_prefill);
        assert_eq!(decoded.sequences.len(), 1);
        assert_eq!(decoded.sequences[0].seq_id, 99);
        match decoded.input {
            ForwardInputWire::Activations {
                tensor_header,
                tensor_data: data,
            } => {
                assert_eq!(tensor_header.shape, vec![4, 4096]);
                assert_eq!(data, tensor_data);
            }
            _ => panic!("expected Activations"),
        }
    }

    #[test]
    fn test_batched_forward_result_logits_roundtrip() {
        let logit_bytes: Vec<u8> = (0..10u32)
            .flat_map(|i| (i as f32).to_le_bytes())
            .collect();
        let payload = BatchedForwardResultPayload {
            output: ForwardOutputWire::Logits {
                data: logit_bytes.clone(),
            },
            num_sequences: 2,
            logit_offsets: vec![0, 20],
        };
        let decoded: BatchedForwardResultPayload = bincode_roundtrip(&payload);
        assert_eq!(decoded.num_sequences, 2);
        assert_eq!(decoded.logit_offsets, vec![0, 20]);
        match decoded.output {
            ForwardOutputWire::Logits { data } => {
                assert_eq!(data, logit_bytes);
            }
            _ => panic!("expected Logits"),
        }
    }

    #[test]
    fn test_batched_forward_result_activations_roundtrip() {
        let header = TensorWireHeader {
            ndim: 2,
            shape: vec![6, 4096],
            dtype: 0,
            compression: 0,
            data_len: 6 * 4096 * 2,
        };
        let tensor_data = vec![0u8; 6 * 4096 * 2];
        let payload = BatchedForwardResultPayload {
            output: ForwardOutputWire::Activations {
                tensor_header: header,
                tensor_data: tensor_data.clone(),
            },
            num_sequences: 3,
            logit_offsets: Vec::new(),
        };
        let decoded: BatchedForwardResultPayload = bincode_roundtrip(&payload);
        assert_eq!(decoded.num_sequences, 3);
        assert!(decoded.logit_offsets.is_empty());
        match decoded.output {
            ForwardOutputWire::Activations {
                tensor_header,
                tensor_data: data,
            } => {
                assert_eq!(tensor_header.shape, vec![6, 4096]);
                assert_eq!(data, tensor_data);
            }
            _ => panic!("expected Activations"),
        }
    }

    #[test]
    #[test]
    fn test_error_payload_roundtrip() {
        let payload = ErrorPayload {
            error_code: ErrorCode::OutOfMemory,
            message: "GPU OOM: requested 2GB, available 512MB".into(),
        };
        let decoded: ErrorPayload = bincode_roundtrip(&payload);
        assert_eq!(decoded.error_code, ErrorCode::OutOfMemory);
        assert!(decoded.message.contains("GPU OOM"));
    }
}
