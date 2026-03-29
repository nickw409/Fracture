//! Phase 2 IPC: Unix domain socket transport for pipeline communication.
//!
//! Provides framed message protocol and activation tensor serialization for
//! single-machine multi-process pipeline validation. Phase 3 replaces this
//! with TCP/QUIC networking.
//!
//! Wire format:
//! ```text
//! [4 bytes: message length (u32 big-endian)]
//! [N bytes: message payload (bincode-serialized)]
//! ```
//!
//! Activation tensor serialization:
//! ```text
//! [4 bytes: ndim (u32)]
//! [4 bytes × ndim: shape (u32 each)]
//! [4 bytes: dtype (u32, 0=FP16)]
//! [N bytes: raw tensor data]
//! ```

use byteorder::{BigEndian, ReadBytesExt, WriteBytesExt};
use fracture_core::{Backend, DType, DeviceTensor, FractureError, Result};
use serde::{Deserialize, Serialize};
use std::io::{Read, Write};

// ── Tensor serialization ─────────────────────────────────────────────────

/// Dtype to wire format tag.
fn dtype_to_tag(dtype: DType) -> u32 {
    match dtype {
        DType::FP16 => 0,
        DType::FP32 => 1,
        DType::BF16 => 2,
        DType::INT8 => 3,
        DType::INT4 => 4,
    }
}

/// Wire format tag to dtype.
fn tag_to_dtype(tag: u32) -> Result<DType> {
    match tag {
        0 => Ok(DType::FP16),
        1 => Ok(DType::FP32),
        2 => Ok(DType::BF16),
        3 => Ok(DType::INT8),
        4 => Ok(DType::INT4),
        _ => Err(FractureError::Pipeline(format!("unknown dtype tag: {tag}"))),
    }
}

/// Serialize an activation tensor: copy from GPU to host, then write header + data.
pub fn serialize_tensor<B: Backend>(
    backend: &B,
    tensor: &DeviceTensor,
    writer: &mut impl Write,
) -> Result<()> {
    // Header: ndim, shape, dtype
    let ndim = tensor.shape.len() as u32;
    writer.write_u32::<BigEndian>(ndim).map_err(FractureError::Io)?;
    for &dim in &tensor.shape {
        writer
            .write_u32::<BigEndian>(dim as u32)
            .map_err(FractureError::Io)?;
    }
    writer
        .write_u32::<BigEndian>(dtype_to_tag(tensor.dtype))
        .map_err(FractureError::Io)?;

    // Data: GPU → host → writer
    let size_bytes = tensor.size_bytes();
    let mut host_buf = vec![0u8; size_bytes];
    backend.copy_to_host(tensor, &mut host_buf)?;
    backend.synchronize()?;
    writer.write_all(&host_buf).map_err(FractureError::Io)?;

    Ok(())
}

/// Deserialize an activation tensor: read header + data, copy from host to GPU.
pub fn deserialize_tensor<B: Backend>(
    backend: &B,
    reader: &mut impl Read,
) -> Result<DeviceTensor> {
    // Header
    let ndim = reader.read_u32::<BigEndian>().map_err(FractureError::Io)? as usize;
    let mut shape = Vec::with_capacity(ndim);
    for _ in 0..ndim {
        shape.push(reader.read_u32::<BigEndian>().map_err(FractureError::Io)? as usize);
    }
    let dtype = tag_to_dtype(reader.read_u32::<BigEndian>().map_err(FractureError::Io)?)?;

    // Allocate on device
    let tensor = backend.alloc(&shape, dtype)?;
    let size_bytes = tensor.size_bytes();

    // Data: reader → host → GPU
    let mut host_buf = vec![0u8; size_bytes];
    reader.read_exact(&mut host_buf).map_err(FractureError::Io)?;
    backend.copy_to_device(&tensor, &host_buf)?;

    Ok(tensor)
}

// ── Framed protocol ──────────────────────────────────────────────────────

/// IPC message types exchanged over the Unix socket.
#[derive(Serialize, Deserialize)]
pub enum IpcMessage {
    /// Forward request: node should run its layers on this input.
    ForwardRequest {
        seq_id: u64,
        positions: Vec<u32>,
        is_prefill: bool,
        /// Raw serialized activation tensor (header + data).
        /// For head nodes, this is empty and token_ids is populated instead.
        tensor_data: Vec<u8>,
        /// Token IDs for head node input (empty for non-head).
        token_ids: Vec<u32>,
    },
    /// Forward response: node's output after running its layers.
    ForwardResponse {
        seq_id: u64,
        /// True if output is logits (tail node), false if activations.
        is_logits: bool,
        /// For logits: f32 values. For activations: raw serialized tensor.
        payload: Vec<u8>,
    },
    /// Request node info.
    InfoRequest,
    /// Node info response.
    InfoResponse {
        node_id: String,
        layer_start: usize,
        layer_end: usize,
        is_head: bool,
        is_tail: bool,
        gpu_memory_total: usize,
        gpu_memory_used: usize,
    },
}

/// Write a length-prefixed bincode message to a stream.
pub fn write_message(writer: &mut impl Write, msg: &IpcMessage) -> Result<()> {
    let payload = bincode::serialize(msg).map_err(|e| {
        FractureError::Pipeline(format!("bincode serialize: {e}"))
    })?;
    writer
        .write_u32::<BigEndian>(payload.len() as u32)
        .map_err(FractureError::Io)?;
    writer.write_all(&payload).map_err(FractureError::Io)?;
    writer.flush().map_err(FractureError::Io)?;
    Ok(())
}

/// Read a length-prefixed bincode message from a stream.
pub fn read_message(reader: &mut impl Read) -> Result<IpcMessage> {
    let len = reader.read_u32::<BigEndian>().map_err(FractureError::Io)? as usize;
    let mut buf = vec![0u8; len];
    reader.read_exact(&mut buf).map_err(FractureError::Io)?;
    bincode::deserialize(&buf).map_err(|e| {
        FractureError::Pipeline(format!("bincode deserialize: {e}"))
    })
}

// ── IPC Node Server ──────────────────────────────────────────────────────

use crate::kv_cache::{CacheHandle, KvCacheManager};
use crate::node::{ComputeNode, NodeInput, NodeOutput};
use std::collections::HashMap;
use std::os::unix::net::UnixListener;

/// Runs a node as a Unix socket server, handling forward requests.
///
/// The server listens on the given path and processes one connection at a time.
/// It manages KV cache allocation per sequence ID.
pub struct IpcNodeServer<N: ComputeNode> {
    node: N,
    cache: KvCacheManager,
    /// Maps seq_id to CacheHandle for cache reuse across decode steps.
    handles: HashMap<u64, CacheHandle>,
}

impl<N: ComputeNode> IpcNodeServer<N> {
    pub fn new(node: N, cache: KvCacheManager) -> Self {
        Self {
            node,
            cache,
            handles: HashMap::new(),
        }
    }

    /// Serve requests on a Unix socket. Blocks until the listener is dropped.
    pub fn serve<B: Backend>(
        &mut self,
        listener: &UnixListener,
        backend: &B,
    ) -> Result<()> {
        for stream in listener.incoming() {
            let mut stream = stream.map_err(FractureError::Io)?;
            self.handle_connection(&mut stream, backend)?;
        }
        Ok(())
    }

    /// Accept and serve a single connection, then return.
    pub fn serve_one<B: Backend>(
        &mut self,
        listener: &UnixListener,
        backend: &B,
    ) -> Result<()> {
        let (mut stream, _) = listener.accept().map_err(FractureError::Io)?;
        self.handle_connection(&mut stream, backend)
    }

    fn handle_connection<B: Backend, S: Read + Write>(
        &mut self,
        stream: &mut S,
        backend: &B,
    ) -> Result<()> {
        loop {
            let msg = match read_message(stream) {
                Ok(msg) => msg,
                Err(FractureError::Io(ref e))
                    if e.kind() == std::io::ErrorKind::UnexpectedEof =>
                {
                    break; // Connection closed
                }
                Err(e) => return Err(e),
            };

            let response = self.handle_message(msg, backend)?;
            write_message(stream, &response)?;
        }
        Ok(())
    }

    fn handle_message<B: Backend>(
        &mut self,
        msg: IpcMessage,
        backend: &B,
    ) -> Result<IpcMessage> {
        match msg {
            IpcMessage::ForwardRequest {
                seq_id,
                positions,
                is_prefill: _,
                tensor_data,
                token_ids,
            } => {
                // Get or create cache handle for this sequence
                let handle = if let Some(&h) = self.handles.get(&seq_id) {
                    h
                } else {
                    let h = self.cache.alloc(backend)?;
                    self.handles.insert(seq_id, h);
                    h
                };

                // Build NodeInput
                let input = if self.node.config().is_head() {
                    NodeInput::TokenIds {
                        ids: token_ids,
                        positions,
                    }
                } else {
                    // Deserialize activation tensor from bytes
                    let tensor =
                        deserialize_tensor(backend, &mut tensor_data.as_slice())?;
                    NodeInput::Activations {
                        hidden_states: tensor,
                        positions,
                    }
                };

                let output = self.node.forward(input, &mut self.cache, handle, None)?;

                // Serialize output
                let (is_logits, payload) = match output {
                    NodeOutput::Logits(logits) => {
                        // Logits as raw f32 bytes
                        let bytes: Vec<u8> = logits
                            .iter()
                            .flat_map(|f| f.to_le_bytes())
                            .collect();
                        (true, bytes)
                    }
                    NodeOutput::Activations(tensor) => {
                        // Serialize tensor: GPU → host → bytes
                        let mut buf = Vec::new();
                        serialize_tensor(backend, &tensor, &mut buf)?;
                        // Free the activation tensor (it's been serialized)
                        backend.free(&tensor)?;
                        (false, buf)
                    }
                };

                Ok(IpcMessage::ForwardResponse {
                    seq_id,
                    is_logits,
                    payload,
                })
            }
            IpcMessage::InfoRequest => {
                let cfg = self.node.config();
                Ok(IpcMessage::InfoResponse {
                    node_id: format!("node-{}-{}", cfg.layer_range.start, cfg.layer_range.end),
                    layer_start: cfg.layer_range.start,
                    layer_end: cfg.layer_range.end,
                    is_head: cfg.is_head(),
                    is_tail: cfg.is_tail(),
                    gpu_memory_total: 0, // Backend info not available through ComputeNode trait
                    gpu_memory_used: 0,
                })
            }
            _ => Err(FractureError::Pipeline(
                "unexpected message type for server".into(),
            )),
        }
    }
}

// ── IPC Node Client (implements ComputeNode) ─────────────────────────────

use std::os::unix::net::UnixStream;

/// A ComputeNode that forwards requests over a Unix socket to a remote
/// IpcNodeServer. Used by PipelineCoordinator to chain nodes across processes.
pub struct IpcNodeClient<B: Backend> {
    stream: std::sync::Mutex<UnixStream>,
    backend: B,
    node_config: crate::node::NodeConfig,
}

impl<B: Backend> IpcNodeClient<B> {
    /// Connect to a node server at the given Unix socket path.
    pub fn connect(
        path: &str,
        backend: B,
        node_config: crate::node::NodeConfig,
    ) -> Result<Self> {
        let stream = UnixStream::connect(path).map_err(FractureError::Io)?;
        Ok(Self {
            stream: std::sync::Mutex::new(stream),
            backend,
            node_config,
        })
    }
}

impl<B: Backend> ComputeNode for IpcNodeClient<B> {
    fn forward(
        &self,
        input: NodeInput,
        _cache: &mut KvCacheManager,
        cache_handle: CacheHandle,
        _profile: Option<&mut fracture_core::ForwardProfile>,
    ) -> Result<NodeOutput> {
        let seq_id = cache_handle.0;

        // Build IPC message from NodeInput
        let (token_ids, tensor_data, positions) = match input {
            NodeInput::TokenIds { ids, positions } => (ids, Vec::new(), positions),
            NodeInput::Activations {
                hidden_states,
                positions,
            } => {
                let mut buf = Vec::new();
                serialize_tensor(&self.backend, &hidden_states, &mut buf)?;
                (Vec::new(), buf, positions)
            }
        };

        let request = IpcMessage::ForwardRequest {
            seq_id,
            positions,
            is_prefill: false,
            tensor_data,
            token_ids,
        };

        let mut stream = self.stream.lock().map_err(|e| {
            FractureError::Pipeline(format!("stream lock poisoned: {e}"))
        })?;

        write_message(&mut *stream, &request)?;
        let response = read_message(&mut *stream)?;

        match response {
            IpcMessage::ForwardResponse {
                is_logits,
                payload,
                ..
            } => {
                if is_logits {
                    // Decode f32 logits from bytes
                    let logits: Vec<f32> = payload
                        .chunks_exact(4)
                        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                        .collect();
                    Ok(NodeOutput::Logits(logits))
                } else {
                    // Deserialize activation tensor: bytes → host → GPU
                    let tensor =
                        deserialize_tensor(&self.backend, &mut payload.as_slice())?;
                    Ok(NodeOutput::Activations(tensor))
                }
            }
            _ => Err(FractureError::Pipeline(
                "unexpected response type from server".into(),
            )),
        }
    }

    fn config(&self) -> &crate::node::NodeConfig {
        &self.node_config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dtype_roundtrip() {
        for dtype in [DType::FP16, DType::FP32, DType::BF16, DType::INT8, DType::INT4] {
            let tag = dtype_to_tag(dtype);
            let back = tag_to_dtype(tag).unwrap();
            assert_eq!(dtype, back);
        }
    }

    #[test]
    fn test_dtype_invalid_tag() {
        assert!(tag_to_dtype(99).is_err());
    }

    #[test]
    fn test_ipc_message_roundtrip() {
        let msg = IpcMessage::ForwardRequest {
            seq_id: 42,
            positions: vec![0, 1, 2],
            is_prefill: true,
            tensor_data: vec![1, 2, 3, 4],
            token_ids: vec![100, 200],
        };

        let mut buf = Vec::new();
        write_message(&mut buf, &msg).unwrap();

        let decoded = read_message(&mut buf.as_slice()).unwrap();
        match decoded {
            IpcMessage::ForwardRequest {
                seq_id,
                positions,
                is_prefill,
                tensor_data,
                token_ids,
            } => {
                assert_eq!(seq_id, 42);
                assert_eq!(positions, vec![0, 1, 2]);
                assert!(is_prefill);
                assert_eq!(tensor_data, vec![1, 2, 3, 4]);
                assert_eq!(token_ids, vec![100, 200]);
            }
            _ => panic!("wrong message type"),
        }
    }

    #[test]
    fn test_logits_response_roundtrip() {
        let logits = vec![1.0f32, 2.0, -3.5];
        let payload: Vec<u8> = logits.iter().flat_map(|f| f.to_le_bytes()).collect();

        let msg = IpcMessage::ForwardResponse {
            seq_id: 7,
            is_logits: true,
            payload,
        };

        let mut buf = Vec::new();
        write_message(&mut buf, &msg).unwrap();

        let decoded = read_message(&mut buf.as_slice()).unwrap();
        match decoded {
            IpcMessage::ForwardResponse {
                seq_id,
                is_logits,
                payload,
            } => {
                assert_eq!(seq_id, 7);
                assert!(is_logits);
                let decoded_logits: Vec<f32> = payload
                    .chunks_exact(4)
                    .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                    .collect();
                assert_eq!(decoded_logits, vec![1.0, 2.0, -3.5]);
            }
            _ => panic!("wrong message type"),
        }
    }

    #[test]
    fn test_tensor_serialization_roundtrip() {
        use fracture_core::{DeviceTimer, TensorId};
        use std::sync::atomic::{AtomicU64, Ordering};

        /// Mock backend that stores tensor data in memory for roundtrip testing.
        struct MemBackend {
            next_id: AtomicU64,
            store: std::sync::Mutex<HashMap<u64, Vec<u8>>>,
        }

        impl MemBackend {
            fn new() -> Self {
                Self {
                    next_id: AtomicU64::new(1),
                    store: std::sync::Mutex::new(HashMap::new()),
                }
            }
        }

        impl Backend for MemBackend {
            fn alloc(&self, shape: &[usize], dtype: DType) -> Result<DeviceTensor> {
                let id = self.next_id.fetch_add(1, Ordering::Relaxed);
                let numel: usize = shape.iter().product();
                let size = numel * dtype.size_bytes();
                self.store.lock().unwrap().insert(id, vec![0u8; size]);
                Ok(DeviceTensor::new(TensorId(id), shape.to_vec(), dtype))
            }
            fn free(&self, tensor: &DeviceTensor) -> Result<()> {
                self.store.lock().unwrap().remove(&tensor.id.0);
                Ok(())
            }
            fn copy_to_device(&self, dst: &DeviceTensor, src: &[u8]) -> Result<()> {
                let mut store = self.store.lock().unwrap();
                if let Some(buf) = store.get_mut(&dst.id.0) {
                    buf.copy_from_slice(src);
                }
                Ok(())
            }
            fn copy_to_host(&self, src: &DeviceTensor, dst: &mut [u8]) -> Result<()> {
                let store = self.store.lock().unwrap();
                if let Some(buf) = store.get(&src.id.0) {
                    dst.copy_from_slice(buf);
                }
                Ok(())
            }
            fn matmul(&self, _a: &DeviceTensor, _b: &DeviceTensor, _o: &DeviceTensor) -> Result<()> { Ok(()) }
            fn rmsnorm(&self, _i: &DeviceTensor, _w: &DeviceTensor, _e: f64, _o: &DeviceTensor) -> Result<()> { Ok(()) }
            fn rope(&self, _q: &DeviceTensor, _k: &DeviceTensor, _p: &[u32], _t: f64, _h: usize) -> Result<()> { Ok(()) }
            fn attention(&self, _q: &DeviceTensor, _k: &DeviceTensor, _v: &DeviceTensor, _n: usize, _s: usize, _o: &DeviceTensor) -> Result<()> { Ok(()) }
            fn silu_mul(&self, _g: &DeviceTensor, _u: &DeviceTensor, _o: &DeviceTensor) -> Result<()> { Ok(()) }
            fn embedding(&self, _t: &[u32], _e: &DeviceTensor, _o: &DeviceTensor) -> Result<()> { Ok(()) }
            fn add(&self, _a: &DeviceTensor, _b: &DeviceTensor, _o: &DeviceTensor) -> Result<()> { Ok(()) }
            fn copy_rows(&self, _s: &DeviceTensor, _d: &DeviceTensor, _so: usize, _do_: usize, _c: usize) -> Result<()> { Ok(()) }
            fn device_name(&self) -> &str { "mem" }
            fn total_memory(&self) -> usize { 1 << 30 }
            fn available_memory(&self) -> usize { 1 << 30 }
            fn synchronize(&self) -> Result<()> { Ok(()) }
            fn create_timer(&self) -> Result<DeviceTimer> { Ok(DeviceTimer(0)) }
            fn start_timer(&self, _t: &DeviceTimer) -> Result<()> { Ok(()) }
            fn stop_timer(&self, _t: &DeviceTimer) -> Result<f32> { Ok(0.0) }
            fn destroy_timer(&self, _t: &DeviceTimer) -> Result<()> { Ok(()) }
        }

        let backend = MemBackend::new();

        // Create a tensor with known data
        let tensor = backend.alloc(&[2, 4], DType::FP16).unwrap();
        let data: Vec<u8> = (0..16).collect(); // 2*4 elements * 2 bytes = 16 bytes
        backend.copy_to_device(&tensor, &data).unwrap();

        // Serialize
        let mut buf = Vec::new();
        serialize_tensor(&backend, &tensor, &mut buf).unwrap();

        // Deserialize into a new tensor
        let restored = deserialize_tensor(&backend, &mut buf.as_slice()).unwrap();

        assert_eq!(restored.shape, vec![2, 4]);
        assert_eq!(restored.dtype, DType::FP16);

        // Verify data survived the roundtrip
        let mut host = vec![0u8; 16];
        backend.copy_to_host(&restored, &mut host).unwrap();
        assert_eq!(host, data);
    }

    #[test]
    fn test_info_request_response_roundtrip() {
        let msg = IpcMessage::InfoRequest;
        let mut buf = Vec::new();
        write_message(&mut buf, &msg).unwrap();
        let decoded = read_message(&mut buf.as_slice()).unwrap();
        assert!(matches!(decoded, IpcMessage::InfoRequest));

        let resp = IpcMessage::InfoResponse {
            node_id: "test-node".into(),
            layer_start: 0,
            layer_end: 16,
            is_head: true,
            is_tail: false,
            gpu_memory_total: 24_000_000_000,
            gpu_memory_used: 12_000_000_000,
        };
        let mut buf = Vec::new();
        write_message(&mut buf, &resp).unwrap();
        let decoded = read_message(&mut buf.as_slice()).unwrap();
        match decoded {
            IpcMessage::InfoResponse {
                node_id,
                layer_start,
                layer_end,
                is_head,
                is_tail,
                gpu_memory_total,
                gpu_memory_used,
            } => {
                assert_eq!(node_id, "test-node");
                assert_eq!(layer_start, 0);
                assert_eq!(layer_end, 16);
                assert!(is_head);
                assert!(!is_tail);
                assert_eq!(gpu_memory_total, 24_000_000_000);
                assert_eq!(gpu_memory_used, 12_000_000_000);
            }
            _ => panic!("expected InfoResponse"),
        }
    }

    #[test]
    fn test_ipc_client_connect_nonexistent_path() {
        use fracture_core::{DeviceTimer, TensorId};
        use std::sync::atomic::{AtomicU64, Ordering};

        struct DummyBackend {
            next_id: AtomicU64,
        }
        impl DummyBackend {
            fn new() -> Self {
                Self { next_id: AtomicU64::new(1) }
            }
        }
        impl Backend for DummyBackend {
            fn alloc(&self, shape: &[usize], dtype: DType) -> Result<DeviceTensor> {
                let id = self.next_id.fetch_add(1, Ordering::SeqCst);
                Ok(DeviceTensor::new(TensorId(id), shape.to_vec(), dtype))
            }
            fn free(&self, _: &DeviceTensor) -> Result<()> { Ok(()) }
            fn copy_to_device(&self, _: &DeviceTensor, _: &[u8]) -> Result<()> { Ok(()) }
            fn copy_to_host(&self, _: &DeviceTensor, _: &mut [u8]) -> Result<()> { Ok(()) }
            fn matmul(&self, _: &DeviceTensor, _: &DeviceTensor, _: &DeviceTensor) -> Result<()> { Ok(()) }
            fn rmsnorm(&self, _: &DeviceTensor, _: &DeviceTensor, _: f64, _: &DeviceTensor) -> Result<()> { Ok(()) }
            fn rope(&self, _: &DeviceTensor, _: &DeviceTensor, _: &[u32], _: f64, _: usize) -> Result<()> { Ok(()) }
            fn attention(&self, _: &DeviceTensor, _: &DeviceTensor, _: &DeviceTensor, _: usize, _: usize, _: &DeviceTensor) -> Result<()> { Ok(()) }
            fn silu_mul(&self, _: &DeviceTensor, _: &DeviceTensor, _: &DeviceTensor) -> Result<()> { Ok(()) }
            fn embedding(&self, _: &[u32], _: &DeviceTensor, _: &DeviceTensor) -> Result<()> { Ok(()) }
            fn add(&self, _: &DeviceTensor, _: &DeviceTensor, _: &DeviceTensor) -> Result<()> { Ok(()) }
            fn copy_rows(&self, _: &DeviceTensor, _: &DeviceTensor, _: usize, _: usize, _: usize) -> Result<()> { Ok(()) }
            fn device_name(&self) -> &str { "dummy" }
            fn total_memory(&self) -> usize { 1 << 30 }
            fn available_memory(&self) -> usize { 1 << 30 }
            fn synchronize(&self) -> Result<()> { Ok(()) }
            fn create_timer(&self) -> Result<DeviceTimer> { Ok(DeviceTimer(0)) }
            fn start_timer(&self, _: &DeviceTimer) -> Result<()> { Ok(()) }
            fn stop_timer(&self, _: &DeviceTimer) -> Result<f32> { Ok(0.0) }
            fn destroy_timer(&self, _: &DeviceTimer) -> Result<()> { Ok(()) }
        }

        let backend = DummyBackend::new();
        let node_config = crate::node::NodeConfig::new(0..4, 4).unwrap();
        let result = IpcNodeClient::connect(
            "/tmp/fracture_nonexistent_socket_path_12345.sock",
            backend,
            node_config,
        );
        let err = match result {
            Err(e) => e,
            Ok(_) => panic!("connecting to nonexistent socket should fail"),
        };
        assert!(
            matches!(err, FractureError::Io(_)),
            "expected Io error, got: {err:?}"
        );
    }

    #[test]
    fn test_activation_response_roundtrip() {
        // ForwardResponse with is_logits=false (activation tensor payload)
        let tensor_data = vec![0xAB; 64];
        let msg = IpcMessage::ForwardResponse {
            seq_id: 99,
            is_logits: false,
            payload: tensor_data.clone(),
        };

        let mut buf = Vec::new();
        write_message(&mut buf, &msg).unwrap();

        let decoded = read_message(&mut buf.as_slice()).unwrap();
        match decoded {
            IpcMessage::ForwardResponse {
                seq_id,
                is_logits,
                payload,
            } => {
                assert_eq!(seq_id, 99);
                assert!(!is_logits);
                assert_eq!(payload, tensor_data);
            }
            _ => panic!("expected ForwardResponse"),
        }
    }
}
