//! Fracture worker binary (CUDA backend).
//!
//! Connects to a coordinator, benchmarks the local GPU, registers with
//! calibration data, receives a layer assignment, loads weights, and
//! serves Forward/Cache/Heartbeat requests over the wire protocol.

use anyhow::Result;
use fracture_core::Backend;
use fracture_cuda::CudaBackend;
use fracture_engine::{
    CacheHandle, ComputeNode, ComputeNodeImpl, Engine, KvCacheManager, NodeConfig, NodeInput,
    NodeOutput,
};
use fracture_gguf::{GgufParser, WeightStore};
use fracture_protocol::{
    connection::FramedConnection,
    frame::MessageType,
    messages::*,
    tensor::{make_header, wire_to_dtype},
};
use std::collections::HashMap;
use tokio::net::TcpStream;
use tracing_subscriber::EnvFilter;

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .init();

    // Parse CLI args
    let args: Vec<String> = std::env::args().collect();
    let coordinator_addr = args
        .iter()
        .position(|a| a == "--coordinator")
        .and_then(|i| args.get(i + 1))
        .expect(
            "usage: fracture-worker-cuda --coordinator <host:port> --model <path-to-gguf> \
             [--gpu <device_id>] [--node-id <name>]",
        )
        .clone();

    let model_path = args
        .iter()
        .position(|a| a == "--model")
        .and_then(|i| args.get(i + 1))
        .expect("--model <path-to-gguf> is required")
        .clone();

    let gpu_device: i32 = args
        .iter()
        .position(|a| a == "--gpu")
        .and_then(|i| args.get(i + 1))
        .and_then(|p| p.parse().ok())
        .unwrap_or(0);

    let node_id = args
        .iter()
        .position(|a| a == "--node-id")
        .and_then(|i| args.get(i + 1).cloned())
        .unwrap_or_else(|| format!("worker-gpu{gpu_device}"));

    tracing::info!("Fracture worker (CUDA backend)");
    tracing::info!("node_id: {node_id}");
    tracing::info!("coordinator: {coordinator_addr}");
    tracing::info!("model: {model_path}");

    // Parse GGUF metadata (no weight loading yet)
    let gguf = GgufParser::parse(std::path::Path::new(&model_path))?;
    let config = gguf.config.clone();

    // Initialize CUDA backend
    let mut backend = CudaBackend::new(gpu_device)?;
    let gpu_model = backend.device_name().to_string();
    let gpu_memory_total = backend.total_memory() as u64;
    let gpu_memory_available = backend.available_memory() as u64;
    tracing::info!(
        "GPU: {} ({:.1} GB total, {:.1} GB available)",
        gpu_model,
        gpu_memory_total as f64 / 1e9,
        gpu_memory_available as f64 / 1e9,
    );

    // Run calibration benchmark on a temporary backend instance
    tracing::info!("running calibration benchmark...");
    let (decode_ms, prefill_ms) =
        run_calibration(gpu_device, &model_path, &config)?;
    tracing::info!(
        "calibration: decode={decode_ms:.2} ms/layer, prefill(128)={prefill_ms:.2} ms/layer"
    );

    // Validate calibration results are plausible before sending to coordinator
    use fracture_coordinator::scheduler::WorkerCapabilities;
    let cal_check = WorkerCapabilities {
        node_id: node_id.clone(),
        gpu_model: String::new(),
        gpu_memory_available: 0,
        compute_capability: (0, 0),
        decode_ms_per_layer: decode_ms,
        prefill_ms_per_layer_128: prefill_ms,
    };
    cal_check.validate_calibration()?;

    // Precompute RoPE frequencies for inference
    backend.precompute_rope_freqs(config.head_dim, config.rope_theta)?;

    // Update available memory after calibration
    let gpu_memory_available = backend.available_memory() as u64;

    // Connect to coordinator
    tracing::info!("connecting to coordinator at {coordinator_addr}...");
    let stream = TcpStream::connect(&coordinator_addr).await?;
    let mut conn = FramedConnection::new(stream);

    // Send Register
    let register = RegisterPayload {
        node_id: node_id.clone(),
        gpu_model,
        gpu_memory_total,
        gpu_memory_available,
        compute_capability: (0, 0), // TODO: query from CUDA
        decode_ms_per_layer: decode_ms,
        prefill_ms_per_layer_128: prefill_ms,
    };
    conn.send(MessageType::Register, 0, &register).await?;
    tracing::info!("sent Register, waiting for layer assignment...");

    // Receive RegisterAck
    let (header, payload) = conn.recv().await?;
    if header.msg_type != MessageType::RegisterAck {
        anyhow::bail!("expected RegisterAck, got {:?}", header.msg_type);
    }
    let ack: RegisterAckPayload = FramedConnection::deserialize_payload(&payload)?;
    let layer_range = ack.layer_start as usize..ack.layer_end as usize;
    tracing::info!(
        "assigned layers {:?} (total {}), max_seq_len={}",
        layer_range,
        ack.total_layers,
        ack.max_seq_len
    );

    // Load assigned layer weights
    tracing::info!("loading weights for layers {:?}...", layer_range);
    let weights = WeightStore::load(
        std::path::Path::new(&model_path),
        &backend,
        Some(layer_range.clone()),
    )?;

    // Create engine and node
    let node_config = NodeConfig::new(layer_range.clone(), ack.total_layers as usize)?;
    let engine = Engine::new(backend, weights, layer_range.clone());
    let node = ComputeNodeImpl::new(engine, node_config);

    // Create KV cache manager for our layer range
    let mut cache = KvCacheManager::new(
        layer_range.len(),
        ack.model_config.num_kv_heads,
        ack.model_config.head_dim,
        ack.max_seq_len as usize,
    );

    // Sequence tracking: seq_id -> CacheHandle
    let mut handles: HashMap<u64, CacheHandle> = HashMap::new();

    tracing::info!("ready — entering serve loop");

    // Serve loop
    loop {
        let (header, payload) = match conn.recv().await {
            Ok(frame) => frame,
            Err(e) => {
                tracing::error!("connection lost: {e}");
                break;
            }
        };

        match header.msg_type {
            MessageType::Forward => {
                let result = handle_forward(
                    &header,
                    &payload,
                    &node,
                    &mut cache,
                    &mut handles,
                );
                match result {
                    Ok(result_payload) => {
                        conn.send(MessageType::ForwardResult, header.seq_id, &result_payload)
                            .await?;
                    }
                    Err(e) => {
                        tracing::error!("forward error for seq {}: {e}", header.seq_id);
                        let err = ErrorPayload {
                            error_code: ErrorCode::Internal,
                            message: e.to_string(),
                        };
                        conn.send(MessageType::Error, header.seq_id, &err).await?;
                    }
                }
            }

            MessageType::CacheAlloc => {
                let seq_id = header.seq_id;
                if handles.contains_key(&seq_id) {
                    let err = ErrorPayload {
                        error_code: ErrorCode::InvalidSequence,
                        message: format!(
                            "CacheAlloc for seq {seq_id}: cache already allocated"
                        ),
                    };
                    tracing::warn!("duplicate CacheAlloc for seq {seq_id}");
                    conn.send(MessageType::Error, seq_id, &err).await?;
                } else {
                    match cache.alloc(node.engine().backend()) {
                        Ok(h) => {
                            handles.insert(seq_id, h);
                            tracing::debug!("allocated cache for seq {seq_id}");
                        }
                        Err(e) => {
                            let err = ErrorPayload {
                                error_code: ErrorCode::OutOfMemory,
                                message: format!(
                                    "CacheAlloc for seq {seq_id} failed: {e}"
                                ),
                            };
                            tracing::error!("cache alloc OOM for seq {seq_id}: {e}");
                            conn.send(MessageType::Error, seq_id, &err).await?;
                        }
                    }
                }
            }

            MessageType::CacheFree => {
                let seq_id = header.seq_id;
                if let Some(h) = handles.remove(&seq_id) {
                    cache.free(h, node.engine().backend())?;
                    tracing::debug!("freed cache for seq {seq_id}");
                } else {
                    let err = ErrorPayload {
                        error_code: ErrorCode::InvalidSequence,
                        message: format!(
                            "CacheFree for seq {seq_id}: no cache allocated"
                        ),
                    };
                    tracing::warn!("CacheFree for unknown seq {seq_id}");
                    conn.send(MessageType::Error, seq_id, &err).await?;
                }
            }

            MessageType::Heartbeat => {
                let hb: HeartbeatPayload = FramedConnection::deserialize_payload(&payload)?;
                let ack = HeartbeatAckPayload {
                    timestamp_echo: hb.timestamp_ns,
                    nonce_echo: hb.nonce,
                    gpu_memory_used: node.engine().backend().total_memory() as u64
                        - node.engine().backend().available_memory() as u64,
                    active_sequences: handles.len() as u32,
                    free_blocks: 0,
                };
                conn.send(MessageType::HeartbeatAck, 0, &ack).await?;
            }

            MessageType::Shutdown => {
                tracing::info!("received Shutdown — exiting");
                for (_, h) in handles.drain() {
                    let _ = cache.free(h, node.engine().backend());
                }
                break;
            }

            other => {
                tracing::warn!("unexpected message type: {other:?}");
                let err = ErrorPayload {
                    error_code: ErrorCode::ProtocolViolation,
                    message: format!("unexpected message type: {other:?}"),
                };
                conn.send(MessageType::Error, header.seq_id, &err).await?;
            }
        }
    }

    tracing::info!("worker shut down");
    Ok(())
}

/// Handle a Forward message: deserialize input, run forward pass, serialize output.
fn handle_forward(
    header: &fracture_protocol::FrameHeader,
    payload: &[u8],
    node: &ComputeNodeImpl<CudaBackend>,
    cache: &mut KvCacheManager,
    handles: &mut HashMap<u64, CacheHandle>,
) -> fracture_core::Result<ForwardResultPayload> {
    let req: ForwardPayload = FramedConnection::deserialize_payload(payload)?;
    let seq_id = header.seq_id;

    // Get cache handle — require prior CacheAlloc
    let handle = if let Some(&h) = handles.get(&seq_id) {
        h
    } else {
        return Err(fracture_core::FractureError::Pipeline(format!(
            "Forward for seq {seq_id}: no cache allocated (missing CacheAlloc)"
        )));
    };

    // Build NodeInput
    let input = match req.input {
        ForwardInputWire::TokenIds { ids } => NodeInput::TokenIds {
            ids,
            positions: req.positions,
        },
        ForwardInputWire::Activations {
            tensor_header,
            tensor_data,
        } => {
            let dtype = wire_to_dtype(tensor_header.dtype)?;
            let shape: Vec<usize> = tensor_header.shape.iter().map(|&d| d as usize).collect();
            let tensor = node.engine().backend().alloc(&shape, dtype)?;
            node.engine().backend().copy_to_device(&tensor, &tensor_data)?;
            NodeInput::Activations {
                hidden_states: tensor,
                positions: req.positions,
            }
        }
    };

    // Run forward pass
    let output = node.forward(input, cache, handle, None)?;

    // Serialize output
    match output {
        NodeOutput::Logits(logits) => {
            let data: Vec<u8> = logits.iter().flat_map(|f: &f32| f.to_le_bytes()).collect();
            Ok(ForwardResultPayload {
                output: ForwardOutputWire::Logits { data },
            })
        }
        NodeOutput::Activations(tensor) => {
            let mut host_buf = vec![0u8; tensor.size_bytes()];
            node.engine().backend().copy_to_host(&tensor, &mut host_buf)?;
            node.engine().backend().synchronize()?;
            let th = make_header(&tensor.shape, tensor.dtype, host_buf.len());
            node.engine().backend().free(&tensor)?;
            Ok(ForwardResultPayload {
                output: ForwardOutputWire::Activations {
                    tensor_header: th,
                    tensor_data: host_buf,
                },
            })
        }
    }
}

/// Run calibration: load a single layer, run forward passes, measure timing.
/// Returns (decode_ms_per_layer, prefill_ms_per_layer_128).
fn run_calibration(
    gpu_device: i32,
    model_path: &str,
    config: &fracture_core::ModelConfig,
) -> Result<(f32, f32)> {
    // Use a temporary backend for calibration — it will be dropped when done
    let mut cal_backend = CudaBackend::new(gpu_device)?;
    cal_backend.precompute_rope_freqs(config.head_dim, config.rope_theta)?;

    // Load weights for layer 0 only
    let weights = WeightStore::load(std::path::Path::new(model_path), &cal_backend, Some(0..1))?;

    let node_config = NodeConfig::new(0..1, config.num_layers)?;
    let engine = Engine::new(cal_backend, weights, 0..1);
    let node = ComputeNodeImpl::new(engine, node_config);

    let mut cache =
        KvCacheManager::new(1, config.num_kv_heads, config.head_dim, config.max_seq_len);

    const WARMUP: usize = 5;
    const TOTAL: usize = 20;

    // Decode benchmark (N=1)
    let mut decode_times = Vec::with_capacity(TOTAL);
    for _ in 0..TOTAL {
        let handle = cache.alloc(node.engine().backend())?;
        let input = NodeInput::TokenIds {
            ids: vec![1],
            positions: vec![0],
        };
        let start = std::time::Instant::now();
        let _ = node.forward(input, &mut cache, handle, None)?;
        node.engine().backend().synchronize()?;
        decode_times.push(start.elapsed().as_secs_f32() * 1000.0);
        cache.free(handle, node.engine().backend())?;
    }

    // Prefill benchmark (N=128)
    let mut prefill_times = Vec::with_capacity(TOTAL);
    for _ in 0..TOTAL {
        let handle = cache.alloc(node.engine().backend())?;
        let input = NodeInput::TokenIds {
            ids: vec![1; 128],
            positions: (0..128).collect(),
        };
        let start = std::time::Instant::now();
        let _ = node.forward(input, &mut cache, handle, None)?;
        node.engine().backend().synchronize()?;
        prefill_times.push(start.elapsed().as_secs_f32() * 1000.0);
        cache.free(handle, node.engine().backend())?;
    }

    // Average last (TOTAL - WARMUP) runs
    let decode_avg: f32 = decode_times[WARMUP..].iter().sum::<f32>() / (TOTAL - WARMUP) as f32;
    let prefill_avg: f32 = prefill_times[WARMUP..].iter().sum::<f32>() / (TOTAL - WARMUP) as f32;

    // Free calibration weights by dropping engine/node (they own the weights)
    // The backend's free calls happen in WeightStore's Drop impl.

    Ok((decode_avg, prefill_avg))
}
