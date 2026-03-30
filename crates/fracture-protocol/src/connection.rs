//! Async framed connection over TCP.
//!
//! Wraps a `TcpStream` with buffered I/O and provides typed send/recv
//! for the Fracture wire protocol. Handles frame encoding, CRC32C
//! computation, and validation transparently.

use crate::frame::{
    decode_header, encode_header, verify_crc, FrameHeader, MessageType, CRC_SIZE, HEADER_SIZE,
};
use fracture_core::{FractureError, Result};
use serde::Serialize;
use tokio::io::{AsyncReadExt, AsyncWriteExt, BufReader, BufWriter};
use tokio::net::tcp::{OwnedReadHalf, OwnedWriteHalf};
use tokio::net::TcpStream;

/// Async framed connection for the Fracture wire protocol.
///
/// Splits the TCP stream into independent buffered reader/writer halves
/// so reads and writes can proceed without contention.
pub struct FramedConnection {
    reader: BufReader<OwnedReadHalf>,
    writer: BufWriter<OwnedWriteHalf>,
}

impl FramedConnection {
    /// Create a new framed connection from a TCP stream.
    pub fn new(stream: TcpStream) -> Self {
        let (read_half, write_half) = stream.into_split();
        Self {
            reader: BufReader::new(read_half),
            writer: BufWriter::new(write_half),
        }
    }

    /// Send a message with a bincode-serialized payload.
    pub async fn send<P: Serialize>(
        &mut self,
        msg_type: MessageType,
        seq_id: u64,
        payload: &P,
    ) -> Result<()> {
        let payload_bytes =
            bincode::serialize(payload).map_err(|e| FractureError::Protocol(e.to_string()))?;
        self.send_raw(msg_type, seq_id, &payload_bytes).await
    }

    /// Send a message with a pre-serialized payload.
    pub async fn send_raw(
        &mut self,
        msg_type: MessageType,
        seq_id: u64,
        payload: &[u8],
    ) -> Result<()> {
        let header = FrameHeader {
            msg_type,
            seq_id,
            payload_len: payload.len() as u32,
        };

        // Encode header
        let mut hdr_buf = [0u8; HEADER_SIZE];
        encode_header(&header, &mut hdr_buf);

        // Compute CRC over header + payload
        let mut crc_input = Vec::with_capacity(HEADER_SIZE + payload.len());
        crc_input.extend_from_slice(&hdr_buf);
        crc_input.extend_from_slice(payload);
        let crc = crc32c::crc32c(&crc_input);

        // Write header + payload + CRC
        self.writer.write_all(&hdr_buf).await?;
        self.writer.write_all(payload).await?;
        self.writer.write_all(&crc.to_be_bytes()).await?;
        self.writer.flush().await?;

        Ok(())
    }

    /// Send a message with no payload (e.g., Shutdown, CacheFree).
    pub async fn send_empty(&mut self, msg_type: MessageType, seq_id: u64) -> Result<()> {
        self.send_raw(msg_type, seq_id, &[]).await
    }

    /// Receive the next frame. Returns the header and raw payload bytes.
    /// The caller is responsible for deserializing the payload based on
    /// the message type in the header.
    pub async fn recv(&mut self) -> Result<(FrameHeader, Vec<u8>)> {
        // Read fixed header
        let mut hdr_buf = [0u8; HEADER_SIZE];
        self.reader.read_exact(&mut hdr_buf).await?;
        let header = decode_header(&hdr_buf)?;

        // Read payload
        let mut payload = vec![0u8; header.payload_len as usize];
        if !payload.is_empty() {
            self.reader.read_exact(&mut payload).await?;
        }

        // Read and verify CRC
        let mut crc_buf = [0u8; CRC_SIZE];
        self.reader.read_exact(&mut crc_buf).await?;
        let expected_crc = u32::from_be_bytes(crc_buf);

        // CRC is computed over header + payload
        let mut crc_input = Vec::with_capacity(HEADER_SIZE + payload.len());
        crc_input.extend_from_slice(&hdr_buf);
        crc_input.extend_from_slice(&payload);
        verify_crc(&crc_input, expected_crc)?;

        Ok((header, payload))
    }

    /// Deserialize a bincode payload. Convenience method for use after recv().
    pub fn deserialize_payload<T: serde::de::DeserializeOwned>(payload: &[u8]) -> Result<T> {
        bincode::deserialize(payload).map_err(|e| FractureError::Protocol(e.to_string()))
    }
}

impl std::fmt::Debug for FramedConnection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FramedConnection").finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::messages::*;
    use tokio::net::TcpListener;

    /// Spawn a TCP pair on localhost for testing.
    async fn tcp_pair() -> (TcpStream, TcpStream) {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let connect = TcpStream::connect(addr);
        let accept = listener.accept();
        let (client, server) = tokio::join!(connect, accept);
        (client.unwrap(), server.unwrap().0)
    }

    #[tokio::test]
    async fn test_send_recv_heartbeat() {
        let (client_stream, server_stream) = tcp_pair().await;
        let mut client = FramedConnection::new(client_stream);
        let mut server = FramedConnection::new(server_stream);

        let payload = HeartbeatPayload {
            timestamp_ns: 999_000_000,
            nonce: 77,
        };
        client
            .send(MessageType::Heartbeat, 0, &payload)
            .await
            .unwrap();

        let (header, data) = server.recv().await.unwrap();
        assert_eq!(header.msg_type, MessageType::Heartbeat);
        assert_eq!(header.seq_id, 0);

        let decoded: HeartbeatPayload = FramedConnection::deserialize_payload(&data).unwrap();
        assert_eq!(decoded.timestamp_ns, 999_000_000);
        assert_eq!(decoded.nonce, 77);
    }

    #[tokio::test]
    async fn test_send_recv_register() {
        let (client_stream, server_stream) = tcp_pair().await;
        let mut client = FramedConnection::new(client_stream);
        let mut server = FramedConnection::new(server_stream);

        let payload = RegisterPayload {
            node_id: "worker-0".into(),
            gpu_model: "RTX 3090".into(),
            gpu_memory_total: 24 * 1024 * 1024 * 1024,
            gpu_memory_available: 22 * 1024 * 1024 * 1024,
            compute_capability: (8, 6),
            decode_ms_per_layer: 1.1,
            prefill_ms_per_layer_128: 3.5,
        };
        client
            .send(MessageType::Register, 0, &payload)
            .await
            .unwrap();

        let (header, data) = server.recv().await.unwrap();
        assert_eq!(header.msg_type, MessageType::Register);

        let decoded: RegisterPayload = FramedConnection::deserialize_payload(&data).unwrap();
        assert_eq!(decoded.node_id, "worker-0");
        assert_eq!(decoded.gpu_model, "RTX 3090");
        assert_eq!(decoded.compute_capability, (8, 6));
    }

    #[tokio::test]
    async fn test_send_recv_empty_message() {
        let (client_stream, server_stream) = tcp_pair().await;
        let mut client = FramedConnection::new(client_stream);
        let mut server = FramedConnection::new(server_stream);

        client
            .send_empty(MessageType::Shutdown, 0)
            .await
            .unwrap();

        let (header, data) = server.recv().await.unwrap();
        assert_eq!(header.msg_type, MessageType::Shutdown);
        assert_eq!(header.seq_id, 0);
        assert!(data.is_empty());
    }

    #[tokio::test]
    async fn test_send_recv_cache_free() {
        let (client_stream, server_stream) = tcp_pair().await;
        let mut client = FramedConnection::new(client_stream);
        let mut server = FramedConnection::new(server_stream);

        let seq_id = 42;
        client
            .send_empty(MessageType::CacheFree, seq_id)
            .await
            .unwrap();

        let (header, data) = server.recv().await.unwrap();
        assert_eq!(header.msg_type, MessageType::CacheFree);
        assert_eq!(header.seq_id, seq_id);
        assert!(data.is_empty());
    }

    #[tokio::test]
    async fn test_send_recv_forward_with_activations() {
        let (client_stream, server_stream) = tcp_pair().await;
        let mut client = FramedConnection::new(client_stream);
        let mut server = FramedConnection::new(server_stream);

        let tensor_data = vec![0xCD; 8192]; // 1×4096 FP16
        let payload = ForwardPayload {
            is_prefill: false,
            positions: vec![47],
            input: ForwardInputWire::Activations {
                tensor_header: crate::tensor::TensorWireHeader {
                    ndim: 2,
                    shape: vec![1, 4096],
                    dtype: 0,
                    compression: 0,
                    data_len: 8192,
                },
                tensor_data: tensor_data.clone(),
            },
        };

        client
            .send(MessageType::Forward, 99, &payload)
            .await
            .unwrap();

        let (header, data) = server.recv().await.unwrap();
        assert_eq!(header.msg_type, MessageType::Forward);
        assert_eq!(header.seq_id, 99);

        let decoded: ForwardPayload = FramedConnection::deserialize_payload(&data).unwrap();
        assert!(!decoded.is_prefill);
        match decoded.input {
            ForwardInputWire::Activations {
                tensor_header,
                tensor_data: data,
            } => {
                assert_eq!(tensor_header.shape, vec![1, 4096]);
                assert_eq!(data, tensor_data);
            }
            _ => panic!("expected Activations"),
        }
    }

    #[tokio::test]
    async fn test_multiple_messages_sequential() {
        let (client_stream, server_stream) = tcp_pair().await;
        let mut client = FramedConnection::new(client_stream);
        let mut server = FramedConnection::new(server_stream);

        // Send 3 messages
        for i in 0..3u64 {
            let payload = HeartbeatPayload {
                timestamp_ns: i * 1000,
                nonce: i,
            };
            client
                .send(MessageType::Heartbeat, i, &payload)
                .await
                .unwrap();
        }

        // Receive all 3
        for i in 0..3u64 {
            let (header, data) = server.recv().await.unwrap();
            assert_eq!(header.msg_type, MessageType::Heartbeat);
            assert_eq!(header.seq_id, i);
            let decoded: HeartbeatPayload =
                FramedConnection::deserialize_payload(&data).unwrap();
            assert_eq!(decoded.nonce, i);
        }
    }

    #[tokio::test]
    async fn test_recv_detects_closed_connection() {
        let (client_stream, server_stream) = tcp_pair().await;
        let mut server = FramedConnection::new(server_stream);

        // Drop client to close the connection
        drop(client_stream);

        let result = server.recv().await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_send_raw_with_pre_serialized_payload() {
        let (client_stream, server_stream) = tcp_pair().await;
        let mut client = FramedConnection::new(client_stream);
        let mut server = FramedConnection::new(server_stream);

        // Pre-serialize a payload manually
        let payload = HeartbeatPayload {
            timestamp_ns: 42,
            nonce: 7,
        };
        let raw = bincode::serialize(&payload).unwrap();
        client
            .send_raw(MessageType::Heartbeat, 0, &raw)
            .await
            .unwrap();

        let (header, data) = server.recv().await.unwrap();
        assert_eq!(header.msg_type, MessageType::Heartbeat);
        let decoded: HeartbeatPayload = FramedConnection::deserialize_payload(&data).unwrap();
        assert_eq!(decoded.timestamp_ns, 42);
        assert_eq!(decoded.nonce, 7);
    }

    #[tokio::test]
    async fn test_forward_result_activations_roundtrip() {
        let (client_stream, server_stream) = tcp_pair().await;
        let mut client = FramedConnection::new(client_stream);
        let mut server = FramedConnection::new(server_stream);

        let tensor_data = vec![0xAB; 4096 * 2]; // [1, 4096] FP16
        let payload = ForwardResultPayload {
            output: ForwardOutputWire::Activations {
                tensor_header: crate::tensor::TensorWireHeader {
                    ndim: 2,
                    shape: vec![1, 4096],
                    dtype: 0,
                    compression: 0,
                    data_len: 4096 * 2,
                },
                tensor_data: tensor_data.clone(),
            },
        };

        client
            .send(MessageType::ForwardResult, 55, &payload)
            .await
            .unwrap();

        let (header, data) = server.recv().await.unwrap();
        assert_eq!(header.msg_type, MessageType::ForwardResult);
        assert_eq!(header.seq_id, 55);

        let decoded: ForwardResultPayload = FramedConnection::deserialize_payload(&data).unwrap();
        match decoded.output {
            ForwardOutputWire::Activations {
                tensor_header,
                tensor_data: data,
            } => {
                assert_eq!(tensor_header.shape, vec![1, 4096]);
                assert_eq!(tensor_header.dtype, 0);
                assert_eq!(data, tensor_data);
            }
            _ => panic!("expected Activations"),
        }
    }

    #[tokio::test]
    async fn test_send_recv_batched_forward() {
        let (client_stream, server_stream) = tcp_pair().await;
        let mut client = FramedConnection::new(client_stream);
        let mut server = FramedConnection::new(server_stream);

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
                    block_table: vec![1],
                    cache_seq_len: 11,
                    last_block_tokens: 11,
                },
            ],
            input: ForwardInputWire::TokenIds {
                ids: vec![128000, 791, 1401, 42],
            },
        };

        client
            .send(MessageType::BatchedForward, 1, &payload)
            .await
            .unwrap();

        let (header, data) = server.recv().await.unwrap();
        assert_eq!(header.msg_type, MessageType::BatchedForward);
        assert_eq!(header.seq_id, 1);

        let decoded: BatchedForwardPayload =
            FramedConnection::deserialize_payload(&data).unwrap();
        assert!(decoded.is_prefill);
        assert_eq!(decoded.sequences.len(), 2);
        assert_eq!(decoded.sequences[0].seq_id, 1);
        assert_eq!(decoded.sequences[1].seq_id, 2);
        match decoded.input {
            ForwardInputWire::TokenIds { ids } => {
                assert_eq!(ids, vec![128000, 791, 1401, 42]);
            }
            _ => panic!("expected TokenIds"),
        }
    }

    #[tokio::test]
    async fn test_send_recv_batched_forward_result() {
        let (client_stream, server_stream) = tcp_pair().await;
        let mut client = FramedConnection::new(client_stream);
        let mut server = FramedConnection::new(server_stream);

        let logit_data: Vec<u8> = (0..20u32)
            .flat_map(|i| (i as f32).to_le_bytes())
            .collect();
        let payload = BatchedForwardResultPayload {
            output: ForwardOutputWire::Logits {
                data: logit_data.clone(),
            },
            num_sequences: 2,
            logit_offsets: vec![0, 40],
        };

        client
            .send(MessageType::BatchedForwardResult, 1, &payload)
            .await
            .unwrap();

        let (header, data) = server.recv().await.unwrap();
        assert_eq!(header.msg_type, MessageType::BatchedForwardResult);

        let decoded: BatchedForwardResultPayload =
            FramedConnection::deserialize_payload(&data).unwrap();
        assert_eq!(decoded.num_sequences, 2);
        assert_eq!(decoded.logit_offsets, vec![0, 40]);
        match decoded.output {
            ForwardOutputWire::Logits { data } => {
                assert_eq!(data, logit_data);
            }
            _ => panic!("expected Logits"),
        }
    }
}
