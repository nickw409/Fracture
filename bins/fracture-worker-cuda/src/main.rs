//! Fracture worker binary (CUDA backend).
//!
//! Connects to a coordinator, benchmarks the local GPU, registers with
//! calibration data, receives a layer assignment, loads weights, and
//! serves Forward/Cache/Heartbeat requests over the wire protocol.
//!
//! Heartbeat handling runs in a dedicated router task so that heartbeat
//! responses are never blocked by long-running forward passes.

/// Always-on status output for production monitoring.
/// Prints to stderr regardless of RUST_LOG level.
macro_rules! status {
    ($($arg:tt)*) => {
        eprintln!("[worker] {}", format!($($arg)*));
    };
}

use anyhow::Result;
use fracture_core::Backend;
use fracture_core::TurboQuantConfig;
use fracture_cuda::CudaBackend;
use fracture_engine::{
    batched_forward_node, CacheHandle, ComputeNode, ComputeNodeImpl, Engine, KvCacheManager,
    NodeConfig, NodeInput, NodeOutput, PagedKvCacheManager, QuantizedKvCacheManager,
    SequenceSlice, BLOCK_SIZE,
    paged_kv_cache::compute_num_blocks,
};
use fracture_gguf::{GgufParser, WeightStore};
use fracture_protocol::{
    connection::FramedConnection,
    frame::{FrameHeader, MessageType},
    messages::*,
    tensor::{make_header, wire_to_dtype},
};
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::time::Duration;
use tokio::net::TcpStream;
use tokio::sync::Mutex;
use tracing_subscriber::EnvFilter;

// ---------------------------------------------------------------------------
// Worker state machine (FT-1)
// ---------------------------------------------------------------------------

/// Tracks the worker's lifecycle state. Transitions:
///
/// ```text
/// Starting -> Connected -> Ready <-> DisconnectedStandby
///                                         |
///                           Shutdown --> Exited
///                           (from any connected state)
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)] // Starting/Connected used by FT-2 (reconnection)
enum WorkerState {
    /// Initial state: calibrating, not yet connected to coordinator.
    Starting,
    /// TCP connected and registered, waiting for layer assignment + weight load.
    Connected,
    /// Weights loaded, serving forward/cache/heartbeat requests.
    Ready,
    /// Coordinator connection lost. GPU state (weights, KV caches) retained.
    /// Worker will attempt reconnection.
    DisconnectedStandby,
    /// Explicit shutdown received -- worker will exit.
    Exited,
}

impl std::fmt::Display for WorkerState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            WorkerState::Starting => write!(f, "Starting"),
            WorkerState::Connected => write!(f, "Connected"),
            WorkerState::Ready => write!(f, "Ready"),
            WorkerState::DisconnectedStandby => write!(f, "DisconnectedStandby"),
            WorkerState::Exited => write!(f, "Exited"),
        }
    }
}

/// Shared worker stats read by the heartbeat responder and updated
/// by the main serve loop. Lock-free via atomics.
struct WorkerStats {
    gpu_memory_used: AtomicU64,
    active_sequences: AtomicU32,
    free_blocks: AtomicU32,
}

impl WorkerStats {
    fn new() -> Self {
        Self {
            gpu_memory_used: AtomicU64::new(0),
            active_sequences: AtomicU32::new(0),
            free_blocks: AtomicU32::new(0),
        }
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .init();

    // Load config file (fracture.env) -> CLI flags as fallback.
    let args: Vec<String> = std::env::args().collect();
    let (cfg, cfg_path) = fracture_core::env_config::load_config(&args);
    if let Some(path) = cfg_path {
        tracing::info!("loaded config from {path}");
    }

    let coordinator_addr_arg = cfg
        .get_or_flag("FRACTURE_COORDINATOR", &args, "--coordinator")
        .map(|s| s.to_string());

    let seed_addrs: Vec<String> = cfg
        .get_or_flag("FRACTURE_SEEDS", &args, "--seed")
        .map(|s| s.split(',').map(|a| a.trim().to_string()).collect())
        .unwrap_or_default();

    let peer_port: u16 = cfg
        .get_or_flag("FRACTURE_PEER_PORT", &args, "--peer-port")
        .and_then(|p| p.parse().ok())
        .unwrap_or(0);

    if coordinator_addr_arg.is_none() && seed_addrs.is_empty() {
        eprintln!(
            "usage: Set FRACTURE_COORDINATOR or FRACTURE_SEEDS in fracture.env, or use:\n  \
             fracture-worker-cuda (--coordinator <host:port> | --seed <h1:p,h2:p,...>) \
             --model <path-to-gguf> [--gpu <id>] [--node-id <name>] [--peer-port <port>] \
             [--kv-quant turboquant]"
        );
        std::process::exit(1);
    }

    let model_path = cfg
        .get_or_flag("FRACTURE_MODEL", &args, "--model")
        .expect("--model or FRACTURE_MODEL is required")
        .to_string();

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

    let election_priority: u32 = args
        .iter()
        .position(|a| a == "--election-priority")
        .and_then(|i| args.get(i + 1))
        .and_then(|p| p.parse().ok())
        .unwrap_or(u32::MAX); // default: lowest priority (won't win elections)

    let coordinator_capable = !args.iter().any(|a| a == "--no-coordinator");

    // TurboQuant KV cache compression (opt-in)
    let kv_quant = args
        .iter()
        .position(|a| a == "--kv-quant")
        .and_then(|i| args.get(i + 1).cloned());
    let tq_config = if kv_quant.as_deref() == Some("turboquant") {
        let key_bits: u8 = args.iter().position(|a| a == "--tq-key-bits")
            .and_then(|i| args.get(i + 1)).and_then(|p| p.parse().ok()).unwrap_or(4);
        let value_bits: u8 = args.iter().position(|a| a == "--tq-value-bits")
            .and_then(|i| args.get(i + 1)).and_then(|p| p.parse().ok()).unwrap_or(2);
        let protected_bits: u8 = args.iter().position(|a| a == "--tq-protected-bits")
            .and_then(|i| args.get(i + 1)).and_then(|p| p.parse().ok()).unwrap_or(8);
        let protected_layers: usize = args.iter().position(|a| a == "--tq-protected-layers")
            .and_then(|i| args.get(i + 1)).and_then(|p| p.parse().ok()).unwrap_or(0);
        let seed: u64 = args.iter().position(|a| a == "--tq-seed")
            .and_then(|i| args.get(i + 1)).and_then(|p| p.parse().ok()).unwrap_or(42);
        let config = TurboQuantConfig {
            key_bits, value_bits, protected_bits, protected_layers,
            residual_tokens: 0, seed,
        };
        config.validate().expect("invalid TurboQuant config");
        Some(config)
    } else {
        None
    };

    status!("{node_id} starting (gpu {gpu_device})");

    tracing::info!("Fracture worker (CUDA backend)");
    if coordinator_capable {
        tracing::info!("election priority: {election_priority}");
    } else {
        tracing::info!("coordinator-capable: false (--no-coordinator)");
    }
    tracing::info!("node_id: {node_id}");
    if let Some(ref addr) = coordinator_addr_arg {
        tracing::info!("coordinator: {addr}");
    }
    if !seed_addrs.is_empty() {
        tracing::info!("seed nodes: {}", seed_addrs.join(", "));
    }
    tracing::info!("model: {model_path}");

    // Parse GGUF metadata (no weight loading yet)
    let gguf = GgufParser::parse(std::path::Path::new(&model_path))?;
    let config = gguf.config.clone();

    // Initialize CUDA backend
    let mut backend = CudaBackend::new(gpu_device)?;
    let gpu_model = backend.device_name().to_string();
    let gpu_memory_total = backend.total_memory() as u64;
    let gpu_memory_available = backend.available_memory() as u64;
    status!("gpu: {} ({:.1} GB)", gpu_model, gpu_memory_total as f64 / 1e9);
    tracing::info!(
        "GPU: {} ({:.1} GB total, {:.1} GB available)",
        gpu_model,
        gpu_memory_total as f64 / 1e9,
        gpu_memory_available as f64 / 1e9,
    );

    status!("calibrating...");
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

    // Resolve coordinator address: either from --coordinator or via seed discovery.
    let mut coordinator_addr = if let Some(addr) = coordinator_addr_arg {
        addr
    } else {
        tracing::info!("discovering coordinator via seed nodes...");
        let (addr, term, manifest) =
            discover_coordinator(&seed_addrs, &node_id).await?;
        tracing::info!("discovered coordinator at {addr} (term={term})");
        let _ = (term, manifest); // used later when we have the manifest variable
        addr
    };

    status!("connecting to {coordinator_addr}");
    // Connect to coordinator
    tracing::info!("connecting to coordinator at {coordinator_addr}...");
    let stream = TcpStream::connect(&coordinator_addr).await?;
    let mut conn = FramedConnection::new(stream);
    status!("connected");

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
    status!("assigned layers {:?} of {}", layer_range, ack.total_layers);
    tracing::info!(
        "assigned layers {:?} (total {}), max_seq_len={}",
        layer_range,
        ack.total_layers,
        ack.max_seq_len
    );

    status!("loading weights...");
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
    let mut node = ComputeNodeImpl::new(engine, node_config);

    // Create KV cache manager for our layer range (contiguous -- used by Forward)
    let mut cache = KvCacheManager::new(
        layer_range.len(),
        ack.model_config.num_kv_heads,
        ack.model_config.head_dim,
        ack.max_seq_len as usize,
    );

    // Create paged KV cache (used by BatchedForward) -- either TQ or FP16
    let gpu_avail = node.engine().backend().available_memory();
    // Reserve 2 GB for forward pass scratch tensors (activations, projections,
    // FFN intermediates) and cuBLAS workspace. 512 MB was insufficient for
    // prompts longer than ~10 tokens.
    let scratch_reserve = 2 * 1024 * 1024 * 1024;

    let mut quantized_cache: Option<QuantizedKvCacheManager> = None;
    let mut paged_cache: Option<PagedKvCacheManager> = None;

    if let Some(ref tq_cfg) = tq_config {
        let num_blocks = tq_cfg.compute_num_blocks(
            gpu_avail, scratch_reserve, layer_range.len(),
            ack.model_config.num_kv_heads, ack.model_config.head_dim,
        );
        let bytes_per_block = tq_cfg.bytes_per_block_total(
            layer_range.len(), ack.model_config.num_kv_heads, ack.model_config.head_dim,
        );
        tracing::info!(
            "TurboQuant KV cache: K{}V{} ({} blocks, {:.1} MB, ~{} tokens)",
            tq_cfg.key_bits, tq_cfg.value_bits, num_blocks,
            (num_blocks * bytes_per_block) as f64 / 1e6, num_blocks * BLOCK_SIZE,
        );
        quantized_cache = Some(QuantizedKvCacheManager::new(
            num_blocks, layer_range.len(), ack.model_config.num_kv_heads,
            ack.model_config.head_dim, 512, tq_cfg.clone(), node.engine().backend(),
        )?);
    } else {
        let num_blocks = compute_num_blocks(
            gpu_avail, scratch_reserve, layer_range.len(),
            ack.model_config.num_kv_heads, ack.model_config.head_dim,
        );
        let bytes_per_block = BLOCK_SIZE * ack.model_config.num_kv_heads
            * ack.model_config.head_dim * 2 * 2 * layer_range.len();
        tracing::info!(
            "paged KV cache: {} blocks ({:.1} MB), ~{} tokens capacity",
            num_blocks, (num_blocks * bytes_per_block) as f64 / 1e6, num_blocks * BLOCK_SIZE,
        );
        paged_cache = Some(PagedKvCacheManager::new(
            num_blocks, layer_range.len(), ack.model_config.num_kv_heads,
            ack.model_config.head_dim, node.engine().backend(),
        )?);
    }

    // Cache dispatch helpers -- route to whichever cache is active
    macro_rules! paged_alloc {
        () => {
            if let Some(ref mut qc) = quantized_cache {
                fracture_engine::PagedCache::alloc(qc)
            } else {
                fracture_engine::PagedCache::alloc(paged_cache.as_mut().unwrap())
            }
        };
    }
    macro_rules! paged_free {
        ($handle:expr) => {
            if let Some(ref mut qc) = quantized_cache {
                fracture_engine::PagedCache::free(qc, $handle)
            } else {
                fracture_engine::PagedCache::free(paged_cache.as_mut().unwrap(), $handle)
            }
        };
    }
    macro_rules! paged_free_blocks {
        () => {
            if let Some(ref qc) = quantized_cache {
                qc.num_free_blocks()
            } else {
                paged_cache.as_ref().unwrap().num_free_blocks()
            }
        };
    }
    macro_rules! destroy_paged_caches {
        ($backend:expr) => {
            if let Some(ref qc) = quantized_cache {
                let _ = qc.destroy($backend);
            }
            if let Some(ref pc) = paged_cache {
                let _ = pc.destroy($backend);
            }
        };
    }

    // Rebuild paged cache for a new layer range (used by Reconfigure handlers).
    // Returns (new_quantized_cache, new_paged_cache).
    macro_rules! rebuild_paged_caches {
        ($backend:expr, $new_range:expr, $model_config:expr) => {{
            let gpu_avail = $backend.available_memory();
            let scratch_reserve: usize = 2 * 1024 * 1024 * 1024;
            if let Some(ref tq_cfg) = tq_config {
                let num_blocks = tq_cfg.compute_num_blocks(
                    gpu_avail, scratch_reserve, $new_range.len(),
                    $model_config.num_kv_heads, $model_config.head_dim,
                );
                let new_qc = QuantizedKvCacheManager::new(
                    num_blocks, $new_range.len(), $model_config.num_kv_heads,
                    $model_config.head_dim, 512, tq_cfg.clone(), $backend,
                )?;
                tracing::info!("reconfigured: TQ {} blocks, layers {:?}", num_blocks, $new_range);
                (Some(new_qc), None)
            } else {
                let num_blocks = compute_num_blocks(
                    gpu_avail, scratch_reserve, $new_range.len(),
                    $model_config.num_kv_heads, $model_config.head_dim,
                );
                let new_pc = PagedKvCacheManager::new(
                    num_blocks, $new_range.len(), $model_config.num_kv_heads,
                    $model_config.head_dim, $backend,
                )?;
                tracing::info!("reconfigured: {} blocks, layers {:?}", num_blocks, $new_range);
                (None, Some(new_pc))
            }
        }};
    }

    // Sequence tracking: seq_id -> CacheHandle
    let mut handles: HashMap<u64, CacheHandle> = HashMap::new();
    // Paged cache handles: seq_id -> CacheHandle (separate namespace)
    let mut paged_handles: HashMap<u64, CacheHandle> = HashMap::new();

    // Split connection: reader goes to the message router task,
    // writer is shared between heartbeat responder and main loop.
    let (writer, reader) = conn.into_split();
    let writer = Arc::new(Mutex::new(writer));

    // Shared stats for heartbeat responses (updated by main loop, read by router).
    let stats = Arc::new(WorkerStats::new());
    stats.gpu_memory_used.store(
        node.engine().backend().total_memory() as u64
            - node.engine().backend().available_memory() as u64,
        Ordering::Relaxed,
    );
    stats.free_blocks.store(paged_free_blocks!() as u32, Ordering::Relaxed);

    // Signal the coordinator that weight loading is complete.
    writer.lock().await.send_empty(MessageType::WorkerReady, 0).await?;
    let mut state = WorkerState::Ready;
    status!(
        "ready ({} cache blocks, {:.1} GB used)",
        paged_free_blocks!(),
        (node.engine().backend().total_memory() - node.engine().backend().available_memory()) as f64 / 1e9,
    );
    tracing::info!("ready -- entering serve loop (state: {state})");

    // Macros to send a message via the shared writer, transitioning to
    // DisconnectedStandby on failure. `break` exits the 'serve loop.
    macro_rules! send_or_standby {
        ($writer:expr, $msg_type:expr, $seq_id:expr, $payload:expr, $state:expr) => {
            if let Err(e) = $writer.lock().await.send($msg_type, $seq_id, $payload).await {
                tracing::error!("send {:?} failed: {e}", $msg_type);
                $state = WorkerState::DisconnectedStandby;
                break;
            }
        };
    }
    macro_rules! send_empty_or_standby {
        ($writer:expr, $msg_type:expr, $seq_id:expr, $state:expr) => {
            if let Err(e) = $writer.lock().await.send_empty($msg_type, $seq_id).await {
                tracing::error!("send {:?} failed: {e}", $msg_type);
                $state = WorkerState::DisconnectedStandby;
                break;
            }
        };
    }

    // Shared state for peer listener access.
    let cluster_manifest: Arc<tokio::sync::Mutex<Option<ClusterManifestPayload>>> =
        Arc::new(tokio::sync::Mutex::new(None));
    let current_term = Arc::new(AtomicU64::new(0));

    // Start peer listener (for seed discovery and election).
    let _peer_addr = spawn_peer_listener(
        peer_port,
        Arc::clone(&cluster_manifest),
        Arc::clone(&current_term),
    )
    .await?;

    // Election agent -- initialized if coordinator-capable.
    let mut election_agent = if coordinator_capable {
        Some(fracture_election::state_machine::ElectionAgent::new(
            fracture_election::state_machine::ElectionConfig {
                node_id: node_id.clone(),
                priority: election_priority,
                election_window: Duration::from_secs(5),
            },
            0, // initial term -- updated from cluster manifest
        ))
    } else {
        None
    };

    // SIGTERM handler: set flag so serve loop can send LeaveIntent gracefully.
    let leave_requested = Arc::new(AtomicBool::new(false));
    let leave_flag = Arc::clone(&leave_requested);
    tokio::spawn(async move {
        #[cfg(unix)]
        {
            use tokio::signal::unix::{signal, SignalKind};
            if let Ok(mut sigterm) = signal(SignalKind::terminate()) {
                sigterm.recv().await;
                tracing::info!("received SIGTERM -- requesting graceful leave");
                leave_flag.store(true, Ordering::SeqCst);
            }
        }
    });

    // Spawn message router: reads all messages, handles heartbeats immediately,
    // forwards everything else to the main loop via channel.
    let (msg_tx, mut msg_rx) = tokio::sync::mpsc::unbounded_channel::<(FrameHeader, Vec<u8>)>();
    let router_writer = writer.clone();
    let router_stats = stats.clone();
    let mut router_handle = tokio::spawn(async move {
        let mut reader = reader;
        loop {
            let (header, payload) = match reader.recv().await {
                Ok(frame) => frame,
                Err(e) => {
                    status!("connection lost: {e}");
                    tracing::error!("connection lost: {e}");
                    break;
                }
            };

            if header.msg_type == MessageType::Heartbeat {
                let ack = match FramedConnection::deserialize_payload::<HeartbeatPayload>(&payload) {
                    Ok(hb) => HeartbeatAckPayload {
                        timestamp_echo: hb.timestamp_ns,
                        nonce_echo: hb.nonce,
                        gpu_memory_used: router_stats.gpu_memory_used.load(Ordering::Relaxed),
                        active_sequences: router_stats.active_sequences.load(Ordering::Relaxed),
                        free_blocks: router_stats.free_blocks.load(Ordering::Relaxed),
                    },
                    Err(e) => {
                        tracing::warn!("malformed Heartbeat payload: {e}");
                        continue;
                    }
                };
                let mut w = router_writer.lock().await;
                if let Err(e) = w.send(MessageType::HeartbeatAck, 0, &ack).await {
                    tracing::error!("failed to send HeartbeatAck: {e}");
                    break;
                }
                continue;
            }

            // Forward non-heartbeat messages to the main loop.
            if msg_tx.send((header, payload)).is_err() {
                break; // Main loop exited
            }
        }
    });

    // Outer loop: cycles between serving and reconnection until Exited.
    while state != WorkerState::Exited {

    // Serve loop -- receives forwarded messages from the router.
    'serve: loop {
        if state != WorkerState::Ready {
            break;
        }

        // Check for graceful leave request (SIGTERM).
        if leave_requested.load(Ordering::SeqCst) {
            tracing::info!("sending LeaveIntent to coordinator");
            let payload = LeaveIntentPayload {
                reason: "SIGTERM received".into(),
            };
            if let Err(e) = writer.lock().await.send(MessageType::LeaveIntent, 0, &payload).await {
                tracing::error!("failed to send LeaveIntent: {e}");
                state = WorkerState::DisconnectedStandby;
                break;
            }
            // Continue serving until coordinator sends Shutdown.
            leave_requested.store(false, Ordering::SeqCst);
        }

        let (header, payload) = match msg_rx.recv().await {
            Some(frame) => frame,
            None => {
                tracing::error!("message router closed");
                state = WorkerState::DisconnectedStandby;
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
                        send_or_standby!(writer, MessageType::ForwardResult, header.seq_id, &result_payload, state);
                    }
                    Err(e) => {
                        tracing::error!("forward error for seq {}: {e}", header.seq_id);
                        let err = ErrorPayload {
                            error_code: ErrorCode::Internal,
                            message: e.to_string(),
                        };
                        send_or_standby!(writer, MessageType::Error, header.seq_id, &err, state);
                    }
                }
            }

            MessageType::CacheAlloc => {
                let seq_id = header.seq_id;
                if let std::collections::hash_map::Entry::Vacant(e) = handles.entry(seq_id) {
                    match cache.alloc(node.engine().backend()) {
                        Ok(h) => {
                            e.insert(h);
                            // Also allocate in paged cache for BatchedForward support.
                            // If paged alloc fails, fail the entire CacheAlloc -- don't
                            // send a false CacheAllocAck. (Fix #2)
                            match paged_alloc!() {
                                Ok(ph) => {
                                    paged_handles.insert(seq_id, ph);
                                }
                                Err(alloc_err) => {
                                    // Rollback the contiguous alloc
                                    let contiguous_handle = handles.remove(&seq_id).unwrap();
                                    let _ = cache.free(contiguous_handle, node.engine().backend());
                                    let err = ErrorPayload {
                                        error_code: ErrorCode::OutOfMemory,
                                        message: format!(
                                            "CacheAlloc for seq {seq_id} failed (paged): {alloc_err}"
                                        ),
                                    };
                                    tracing::error!("paged cache alloc OOM for seq {seq_id}: {alloc_err}");
                                    send_or_standby!(writer, MessageType::Error, seq_id, &err, state);
                                    // Update stats
                                    stats.active_sequences.store(handles.len() as u32, Ordering::Relaxed);
                                    stats.free_blocks.store(paged_free_blocks!() as u32, Ordering::Relaxed);
                                    continue;
                                }
                            }
                            send_empty_or_standby!(writer, MessageType::CacheAllocAck, seq_id, state);
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
                            send_or_standby!(writer, MessageType::Error, seq_id, &err, state);
                        }
                    }
                } else {
                    let err = ErrorPayload {
                        error_code: ErrorCode::InvalidSequence,
                        message: format!(
                            "CacheAlloc for seq {seq_id}: cache already allocated"
                        ),
                    };
                    tracing::warn!("duplicate CacheAlloc for seq {seq_id}");
                    send_or_standby!(writer, MessageType::Error, seq_id, &err, state);
                }
                // Update stats after alloc changes
                stats.active_sequences.store(handles.len() as u32, Ordering::Relaxed);
                stats.free_blocks.store(paged_free_blocks!() as u32, Ordering::Relaxed);
                stats.gpu_memory_used.store(
                    node.engine().backend().total_memory() as u64
                        - node.engine().backend().available_memory() as u64,
                    Ordering::Relaxed,
                );
            }

            MessageType::CacheFree => {
                let seq_id = header.seq_id;
                if let Some(h) = handles.remove(&seq_id) {
                    cache.free(h, node.engine().backend())?;
                    if let Some(ph) = paged_handles.remove(&seq_id) {
                        paged_free!(ph)?;
                    }
                    tracing::debug!("freed cache for seq {seq_id}");
                } else {
                    let err = ErrorPayload {
                        error_code: ErrorCode::InvalidSequence,
                        message: format!(
                            "CacheFree for seq {seq_id}: no cache allocated"
                        ),
                    };
                    tracing::warn!("CacheFree for unknown seq {seq_id}");
                    send_or_standby!(writer, MessageType::Error, header.seq_id, &err, state);
                }
                // Update stats after free
                stats.active_sequences.store(handles.len() as u32, Ordering::Relaxed);
                stats.free_blocks.store(paged_free_blocks!() as u32, Ordering::Relaxed);
            }

            MessageType::Reconfigure => {
                let reconf: RegisterAckPayload =
                    FramedConnection::deserialize_payload(&payload)?;
                let new_range = reconf.layer_start as usize..reconf.layer_end as usize;
                status!("reconfiguring: layers {:?} -> {:?}", node.config().layer_range, new_range);
                tracing::info!(
                    "reconfiguring: layers {:?} -> {:?}",
                    node.config().layer_range,
                    new_range
                );

                // Free all existing caches.
                for (_, h) in handles.drain() {
                    let _ = cache.free(h, node.engine().backend());
                }
                for (_, ph) in paged_handles.drain() {
                    let _ = paged_free!(ph);
                }
                destroy_paged_caches!(node.engine().backend());

                // Reload weights for the new layer range.
                tracing::info!("reloading weights for layers {:?}...", new_range);
                let new_weights = WeightStore::load(
                    std::path::Path::new(&model_path),
                    node.engine().backend(),
                    Some(new_range.clone()),
                )?;
                let new_node_config =
                    NodeConfig::new(new_range.clone(), reconf.total_layers as usize)?;
                let reused_backend = node.into_backend();
                let new_engine =
                    Engine::new(reused_backend, new_weights, new_range.clone());
                node = ComputeNodeImpl::new(new_engine, new_node_config);

                // Rebuild caches.
                cache = KvCacheManager::new(
                    new_range.len(),
                    reconf.model_config.num_kv_heads,
                    reconf.model_config.head_dim,
                    reconf.max_seq_len as usize,
                );
                let (new_qc, new_pc) = rebuild_paged_caches!(
                    node.engine().backend(), new_range, reconf.model_config
                );
                quantized_cache = new_qc;
                paged_cache = new_pc;

                // Update stats
                stats.active_sequences.store(0, Ordering::Relaxed);
                stats.free_blocks.store(paged_free_blocks!() as u32, Ordering::Relaxed);
                stats.gpu_memory_used.store(
                    node.engine().backend().total_memory() as u64
                        - node.engine().backend().available_memory() as u64,
                    Ordering::Relaxed,
                );

                send_empty_or_standby!(writer, MessageType::WorkerReady, 0, state);
                status!("ready after reconfigure ({} blocks, layers {:?})", paged_free_blocks!(), new_range);
                tracing::info!("ready after reconfigure");
            }

            MessageType::Shutdown => {
                status!("shutting down");
                tracing::info!("received Shutdown -- exiting");
                for (_, h) in handles.drain() {
                    let _ = cache.free(h, node.engine().backend());
                }
                for (_, ph) in paged_handles.drain() {
                    let _ = paged_free!(ph);
                }
                destroy_paged_caches!(node.engine().backend());
                state = WorkerState::Exited;
                break 'serve;
            }

            MessageType::BatchedForward => {
                let result = if let Some(ref mut qc) = quantized_cache {
                    handle_batched_forward(&header, &payload, &node, qc, &paged_handles)
                } else {
                    handle_batched_forward(&header, &payload, &node, paged_cache.as_mut().unwrap(), &paged_handles)
                };
                match result {
                    Ok(result_payload) => {
                        send_or_standby!(writer, MessageType::BatchedForwardResult, header.seq_id, &result_payload, state);
                    }
                    Err(e) => {
                        tracing::error!("batched forward error: {e}");
                        let err = ErrorPayload {
                            error_code: ErrorCode::Internal,
                            message: e.to_string(),
                        };
                        send_or_standby!(writer, MessageType::Error, header.seq_id, &err, state);
                    }
                }
            }

            MessageType::Victory => {
                match FramedConnection::deserialize_payload::<VictoryPayload>(&payload) {
                    Ok(victory) => {
                        let cur_term = current_term.load(Ordering::SeqCst);
                        if victory.term > cur_term {
                            tracing::info!(
                                "received Victory from '{}' at term {} (coordinator: {})",
                                victory.leader_id, victory.term, victory.coordinator_addr
                            );
                            current_term.store(victory.term, Ordering::SeqCst);
                            if let Some(ref mut agent) = election_agent {
                                agent.on_victory(
                                    &victory.leader_id,
                                    victory.term,
                                    &victory.coordinator_addr,
                                );
                            }
                        } else {
                            tracing::debug!(
                                "ignoring stale Victory from '{}' (term {} <= {})",
                                victory.leader_id, victory.term, cur_term
                            );
                        }
                    }
                    Err(e) => {
                        tracing::error!("Victory deserialize error: {e}");
                    }
                }
            }

            MessageType::ClusterManifest => {
                match FramedConnection::deserialize_payload::<ClusterManifestPayload>(&payload) {
                    Ok(manifest) => {
                        let mut guard = cluster_manifest.lock().await;
                        let dominated = guard.as_ref()
                            .is_some_and(|old| old.version >= manifest.version);
                        if dominated {
                            tracing::debug!(
                                "ignoring stale manifest v{} (current v{})",
                                manifest.version,
                                guard.as_ref().unwrap().version,
                            );
                        } else {
                            tracing::info!(
                                "received cluster manifest v{} (term={}, {} nodes)",
                                manifest.version,
                                manifest.term,
                                manifest.nodes.len(),
                            );
                            if manifest.term > current_term.load(Ordering::SeqCst) {
                                current_term.store(manifest.term, Ordering::SeqCst);
                            }
                            *guard = Some(manifest);
                        }
                    }
                    Err(e) => {
                        tracing::error!("manifest deserialize error: {e}");
                    }
                }
            }

            other => {
                tracing::warn!("unexpected message type: {other:?}");
                let err = ErrorPayload {
                    error_code: ErrorCode::ProtocolViolation,
                    message: format!("unexpected message type: {other:?}"),
                };
                send_or_standby!(writer, MessageType::Error, header.seq_id, &err, state);
            }
        }
    }

    // Abort the router task -- we lost connection or are shutting down.
    router_handle.abort();

    // Reconnection: on disconnect, attempt to reconnect with backoff.
    if state == WorkerState::DisconnectedStandby {
        tracing::warn!(
            "coordinator connection lost -- entering standby \
             (GPU state retained, {} cached sequences preserved)",
            handles.len()
        );

        let mut backoff = Duration::from_secs(1);
        let max_backoff = Duration::from_secs(30);
        let backoff_factor = 2u32;
        let disconnect_time = std::time::Instant::now();
        let election_timeout = Duration::from_secs(15); // 3 missed heartbeats
        let mut election_started = false;

        loop {
            // Check if we should start an election (FT-11).
            if !election_started
                && coordinator_capable
                && disconnect_time.elapsed() >= election_timeout
                && cluster_manifest.lock().await.is_some()
                && let Some(ref mut agent) = election_agent {
                    let term = agent.start_election();
                    election_started = true;
                    tracing::info!(
                        "election timeout reached -- starting election (term={term})"
                    );

                    // For now, simulate election timeout (no peer communication yet).
                    // In a full implementation, we'd broadcast ElectionStart to peers
                    // via the cluster manifest and wait for challenges.
                    // If no challenges received, we win.
                    let action = agent.on_election_timeout();
                    if action == fracture_election::state_machine::ElectionAction::DeclareVictory {
                        tracing::info!("election won -- promoting to coordinator");
                        let coord_port = 9400u16; // TODO: make configurable
                        let http_port = 8080u16;  // TODO: make configurable
                        match promote_to_coordinator(&node_id, coord_port, http_port).await {
                            Ok((listener, addr)) => {
                                tracing::info!(
                                    "coordinator promotion successful: listening on {addr}"
                                );

                                // Reconstruct state from peer workers (FT-12b).
                                let self_caps = WorkerCapabilities {
                                    node_id: node_id.clone(),
                                    gpu_model: node.engine().backend().device_name().to_string(),
                                    gpu_memory_available: node.engine().backend().available_memory(),
                                    compute_capability: (0, 0),
                                    decode_ms_per_layer: decode_ms,
                                    prefill_ms_per_layer_128: prefill_ms,
                                };
                                let peer_count = {
                                    let guard = cluster_manifest.lock().await;
                                    guard.as_ref()
                                        .map(|m| m.nodes.iter().filter(|n| n.node_id != node_id).count())
                                        .unwrap_or(0)
                                };

                                match reconstruct_state(
                                    &listener,
                                    &node_id,
                                    self_caps,
                                    &config,
                                    ack.max_seq_len as usize,
                                    peer_count,
                                    Duration::from_secs(30),
                                )
                                .await
                                {
                                    Ok((_pipeline, _registry)) => {
                                        tracing::info!("state reconstruction complete -- coordinator ready");
                                        // Full HTTP server and scheduler loop startup
                                        // would go here. For now, the pipeline is built
                                        // and peers are connected.
                                    }
                                    Err(e) => {
                                        tracing::error!("state reconstruction failed: {e}");
                                    }
                                }
                            }
                            Err(e) => {
                                tracing::error!("coordinator promotion failed: {e}");
                            }
                        }
                    }
                }
            // Apply +/-25% jitter to prevent thundering herd
            let jitter_range = backoff.as_millis() as f64 * 0.25;
            let jitter_ms = (rand::random::<f64>() - 0.5) * 2.0 * jitter_range;
            let sleep_ms = (backoff.as_millis() as f64 + jitter_ms).max(100.0);
            tracing::info!(
                "reconnecting to coordinator at {coordinator_addr} in {:.1}s...",
                sleep_ms / 1000.0
            );
            tokio::time::sleep(Duration::from_millis(sleep_ms as u64)).await;

            match TcpStream::connect(&coordinator_addr).await {
                Ok(stream) => {
                    tracing::info!("reconnected to coordinator");
                    let mut conn = FramedConnection::new(stream);

                    // Send ReRegister with current state
                    let layer_range = node.config().layer_range.clone();
                    let reregister = ReRegisterPayload {
                        node_id: node_id.clone(),
                        gpu_model: node.engine().backend().device_name().to_string(),
                        gpu_memory_total: node.engine().backend().total_memory() as u64,
                        gpu_memory_available: node.engine().backend().available_memory() as u64,
                        compute_capability: (0, 0),
                        decode_ms_per_layer: decode_ms,
                        prefill_ms_per_layer_128: prefill_ms,
                        current_layer_start: Some(layer_range.start as u32),
                        current_layer_end: Some(layer_range.end as u32),
                        active_cache_seq_ids: handles.keys().copied().collect(),
                    };
                    if let Err(e) = conn.send(MessageType::ReRegister, 0, &reregister).await {
                        tracing::error!("failed to send ReRegister: {e}");
                        backoff = (backoff * backoff_factor).min(max_backoff);
                        continue;
                    }
                    tracing::info!("sent ReRegister, waiting for coordinator response...");

                    // Wait for response: RegisterAck (unchanged) or Reconfigure (new assignment)
                    match conn.recv().await {
                        Ok((hdr, pay)) => match hdr.msg_type {
                            MessageType::RegisterAck => {
                                tracing::info!("coordinator accepted re-registration (assignment unchanged)");
                                if let Err(e) = conn.send_empty(MessageType::WorkerReady, 0).await {
                                    tracing::error!("failed to send WorkerReady: {e}");
                                    backoff = (backoff * backoff_factor).min(max_backoff);
                                    continue;
                                }
                                // Re-split connection for the router task.
                                let (new_writer, new_reader) = conn.into_split();
                                *writer.lock().await = new_writer;
                                let (new_tx, new_rx) = tokio::sync::mpsc::unbounded_channel();
                                msg_rx = new_rx;
                                let rw = writer.clone();
                                let rs = stats.clone();
                                router_handle = tokio::spawn(async move {
                                    let mut reader = new_reader;
                                    loop {
                                        let (header, payload) = match reader.recv().await {
                                            Ok(frame) => frame,
                                            Err(e) => {
                                                status!("connection lost: {e}");
                                                tracing::error!("connection lost: {e}");
                                                break;
                                            }
                                        };
                                        if header.msg_type == MessageType::Heartbeat {
                                            let ack = match FramedConnection::deserialize_payload::<HeartbeatPayload>(&payload) {
                                                Ok(hb) => HeartbeatAckPayload {
                                                    timestamp_echo: hb.timestamp_ns,
                                                    nonce_echo: hb.nonce,
                                                    gpu_memory_used: rs.gpu_memory_used.load(Ordering::Relaxed),
                                                    active_sequences: rs.active_sequences.load(Ordering::Relaxed),
                                                    free_blocks: rs.free_blocks.load(Ordering::Relaxed),
                                                },
                                                Err(e) => {
                                                    tracing::warn!("malformed Heartbeat payload: {e}");
                                                    continue;
                                                }
                                            };
                                            let mut w = rw.lock().await;
                                            if let Err(e) = w.send(MessageType::HeartbeatAck, 0, &ack).await {
                                                tracing::error!("failed to send HeartbeatAck: {e}");
                                                break;
                                            }
                                            continue;
                                        }
                                        if new_tx.send((header, payload)).is_err() {
                                            break;
                                        }
                                    }
                                });
                                state = WorkerState::Ready;
                                break;
                            }
                            MessageType::Reconfigure => {
                                let reconf: RegisterAckPayload = match FramedConnection::deserialize_payload(&pay) {
                                    Ok(r) => r,
                                    Err(e) => {
                                        tracing::error!("reconfigure deserialize error: {e}");
                                        backoff = (backoff * backoff_factor).min(max_backoff);
                                        continue;
                                    }
                                };
                                let new_range = reconf.layer_start as usize..reconf.layer_end as usize;
                                tracing::info!("coordinator assigned new layers {:?} (was {:?})", new_range, node.config().layer_range);

                                for (_, h) in handles.drain() {
                                    let _ = cache.free(h, node.engine().backend());
                                }
                                for (_, ph) in paged_handles.drain() {
                                    let _ = paged_free!(ph);
                                }
                                destroy_paged_caches!(node.engine().backend());

                                let new_weights = WeightStore::load(
                                    std::path::Path::new(&model_path),
                                    node.engine().backend(),
                                    Some(new_range.clone()),
                                )?;
                                let new_node_config = NodeConfig::new(new_range.clone(), reconf.total_layers as usize)?;
                                let reused_backend = node.into_backend();
                                node = ComputeNodeImpl::new(
                                    Engine::new(reused_backend, new_weights, new_range.clone()),
                                    new_node_config,
                                );

                                cache = KvCacheManager::new(
                                    new_range.len(),
                                    reconf.model_config.num_kv_heads,
                                    reconf.model_config.head_dim,
                                    reconf.max_seq_len as usize,
                                );
                                let (new_qc, new_pc) = rebuild_paged_caches!(
                                    node.engine().backend(), new_range, reconf.model_config
                                );
                                quantized_cache = new_qc;
                                paged_cache = new_pc;

                                // Update stats
                                stats.active_sequences.store(0, Ordering::Relaxed);
                                stats.free_blocks.store(paged_free_blocks!() as u32, Ordering::Relaxed);
                                stats.gpu_memory_used.store(
                                    node.engine().backend().total_memory() as u64
                                        - node.engine().backend().available_memory() as u64,
                                    Ordering::Relaxed,
                                );

                                tracing::info!("reconfigured after reconnect: layers {:?}", node.config().layer_range);

                                if let Err(e) = conn.send_empty(MessageType::WorkerReady, 0).await {
                                    tracing::error!("failed to send WorkerReady: {e}");
                                    backoff = (backoff * backoff_factor).min(max_backoff);
                                    continue;
                                }
                                // Re-split connection for the router task.
                                let (new_writer, new_reader) = conn.into_split();
                                *writer.lock().await = new_writer;
                                let (new_tx, new_rx) = tokio::sync::mpsc::unbounded_channel();
                                msg_rx = new_rx;
                                let rw = writer.clone();
                                let rs = stats.clone();
                                router_handle = tokio::spawn(async move {
                                    let mut reader = new_reader;
                                    loop {
                                        let (header, payload) = match reader.recv().await {
                                            Ok(frame) => frame,
                                            Err(e) => {
                                                status!("connection lost: {e}");
                                                tracing::error!("connection lost: {e}");
                                                break;
                                            }
                                        };
                                        if header.msg_type == MessageType::Heartbeat {
                                            let ack = match FramedConnection::deserialize_payload::<HeartbeatPayload>(&payload) {
                                                Ok(hb) => HeartbeatAckPayload {
                                                    timestamp_echo: hb.timestamp_ns,
                                                    nonce_echo: hb.nonce,
                                                    gpu_memory_used: rs.gpu_memory_used.load(Ordering::Relaxed),
                                                    active_sequences: rs.active_sequences.load(Ordering::Relaxed),
                                                    free_blocks: rs.free_blocks.load(Ordering::Relaxed),
                                                },
                                                Err(e) => {
                                                    tracing::warn!("malformed Heartbeat payload: {e}");
                                                    continue;
                                                }
                                            };
                                            let mut w = rw.lock().await;
                                            if let Err(e) = w.send(MessageType::HeartbeatAck, 0, &ack).await {
                                                tracing::error!("failed to send HeartbeatAck: {e}");
                                                break;
                                            }
                                            continue;
                                        }
                                        if new_tx.send((header, payload)).is_err() {
                                            break;
                                        }
                                    }
                                });
                                state = WorkerState::Ready;
                                break;
                            }
                            MessageType::Shutdown => {
                                tracing::info!("received Shutdown during reconnection -- exiting");
                                state = WorkerState::Exited;
                                break;
                            }
                            other => {
                                tracing::error!("unexpected response to ReRegister: {other:?}");
                                backoff = (backoff * backoff_factor).min(max_backoff);
                                continue;
                            }
                        },
                        Err(e) => {
                            tracing::error!("failed to receive response to ReRegister: {e}");
                            backoff = (backoff * backoff_factor).min(max_backoff);
                            continue;
                        }
                    }
                }
                Err(e) => {
                    tracing::debug!("reconnection to {coordinator_addr} failed: {e}");

                    // Seed discovery fallback: if we have seeds, try to find the
                    // new coordinator (may have changed due to election).
                    if !seed_addrs.is_empty() {
                        tracing::debug!("trying seed discovery fallback...");
                        for seed in &seed_addrs {
                            if let Ok(Ok(stream)) = tokio::time::timeout(
                                Duration::from_secs(2),
                                TcpStream::connect(seed),
                            ).await {
                                let mut seed_conn = FramedConnection::new(stream);
                                let req = WhoIsCoordinatorPayload {
                                    node_id: node_id.clone(),
                                };
                                if seed_conn.send(MessageType::WhoIsCoordinator, 0, &req).await.is_ok()
                                    && let Ok(Ok((hdr, pay))) = tokio::time::timeout(
                                        Duration::from_secs(2),
                                        seed_conn.recv(),
                                    ).await
                                        && hdr.msg_type == MessageType::WhoIsCoordinatorResponse
                                            && let Ok(resp) = FramedConnection::deserialize_payload::<WhoIsCoordinatorResponsePayload>(&pay)
                                                && let Some(addr) = resp.coordinator_addr {
                                                    if addr != coordinator_addr {
                                                        tracing::info!(
                                                            "seed discovery: coordinator moved to {addr} (was {coordinator_addr})"
                                                        );
                                                        coordinator_addr = addr;
                                                    }
                                                    break;
                                                }
                            }
                        }
                    }

                    backoff = (backoff * backoff_factor).min(max_backoff);
                }
            }
        }

        if state == WorkerState::Ready {
            tracing::info!("re-entering serve loop after reconnection");
        }
    }

    } // end outer while state != Exited

    router_handle.abort();
    tracing::info!("worker shut down");
    Ok(())
}

/// Handle a Forward message: deserialize input, run forward pass, serialize output.
fn handle_forward(
    header: &FrameHeader,
    payload: &[u8],
    node: &ComputeNodeImpl<CudaBackend>,
    cache: &mut KvCacheManager,
    handles: &mut HashMap<u64, CacheHandle>,
) -> fracture_core::Result<ForwardResultPayload> {
    let req: ForwardPayload = FramedConnection::deserialize_payload(payload)?;
    let seq_id = header.seq_id;

    // Get cache handle -- require prior CacheAlloc
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

/// Handle a BatchedForward message: deserialize input, run batched forward through
/// this node's layers with paged KV cache, serialize output.
fn handle_batched_forward<C: fracture_engine::PagedCache>(
    _header: &FrameHeader,
    payload: &[u8],
    node: &ComputeNodeImpl<CudaBackend>,
    paged_cache: &mut C,
    paged_handles: &HashMap<u64, CacheHandle>,
) -> fracture_core::Result<BatchedForwardResultPayload> {
    let req: BatchedForwardPayload = FramedConnection::deserialize_payload(payload)?;

    // Build SequenceSlice for each sequence in the batch.
    let mut sequences = Vec::with_capacity(req.sequences.len());
    for meta in &req.sequences {
        let handle = paged_handles.get(&meta.seq_id).ok_or_else(|| {
            fracture_core::FractureError::Pipeline(format!(
                "BatchedForward: seq {} has no paged cache (missing CacheAlloc)",
                meta.seq_id
            ))
        })?;
        sequences.push(SequenceSlice {
            handle: *handle,
            token_ids: Vec::new(), // Filled below for head nodes
            positions: meta.positions.clone(),
        });
    }

    // Resolve input: head gets token IDs, middle/tail get activations.
    let input_hidden_states = match req.input {
        ForwardInputWire::TokenIds { ids } => {
            // Head node: distribute token IDs to sequences.
            let mut offset = 0;
            for (i, meta) in req.sequences.iter().enumerate() {
                let n = meta.num_tokens;
                sequences[i].token_ids = ids[offset..offset + n].to_vec();
                offset += n;
            }
            None
        }
        ForwardInputWire::Activations {
            tensor_header,
            tensor_data,
        } => {
            // Middle/tail node: upload activations to GPU.
            let dtype = wire_to_dtype(tensor_header.dtype)?;
            let shape: Vec<usize> = tensor_header.shape.iter().map(|&d| d as usize).collect();
            let tensor = node.engine().backend().alloc(&shape, dtype)?;
            node.engine()
                .backend()
                .copy_to_device(&tensor, &tensor_data)?;
            Some(tensor)
        }
    };

    // Run batched forward through this node's layers.
    let output = batched_forward_node(
        node.engine().backend(),
        node.engine().weights(),
        node.config(),
        paged_cache,
        &sequences,
        input_hidden_states,
    )?;

    // Serialize output.
    match output {
        NodeOutput::Logits(flat_logits) => {
            let data: Vec<u8> = flat_logits
                .iter()
                .flat_map(|f: &f32| f.to_le_bytes())
                .collect();
            let vocab_size = node.engine().weights().config.vocab_size;
            let logit_offsets: Vec<usize> = (0..req.sequences.len())
                .map(|i| i * vocab_size * 4)
                .collect();
            Ok(BatchedForwardResultPayload {
                output: ForwardOutputWire::Logits { data },
                num_sequences: req.sequences.len(),
                logit_offsets,
            })
        }
        NodeOutput::Activations(tensor) => {
            let mut host_buf = vec![0u8; tensor.size_bytes()];
            node.engine().backend().copy_to_host(&tensor, &mut host_buf)?;
            node.engine().backend().synchronize()?;
            let th = make_header(&tensor.shape, tensor.dtype, host_buf.len());
            node.engine().backend().free(&tensor)?;
            Ok(BatchedForwardResultPayload {
                output: ForwardOutputWire::Activations {
                    tensor_header: th,
                    tensor_data: host_buf,
                },
                num_sequences: req.sequences.len(),
                logit_offsets: Vec::new(),
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
    // Use a temporary backend for calibration -- it will be dropped when done
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

// ---------------------------------------------------------------------------
// Seed node discovery
// ---------------------------------------------------------------------------

/// Discover the coordinator address by querying seed nodes.
///
/// Tries each seed in order, sending WhoIsCoordinator and waiting for a response.
/// Retries with exponential backoff if all seeds fail.
/// Returns (coordinator_addr, term, optional_manifest).
async fn discover_coordinator(
    seeds: &[String],
    node_id: &str,
) -> Result<(String, u64, Option<ClusterManifestPayload>)> {
    let mut backoff = Duration::from_secs(1);
    let max_backoff = Duration::from_secs(30);

    loop {
        for seed in seeds {
            tracing::debug!("querying seed {seed} for coordinator...");
            match tokio::time::timeout(Duration::from_secs(3), TcpStream::connect(seed)).await {
                Ok(Ok(stream)) => {
                    let mut conn = FramedConnection::new(stream);
                    let req = WhoIsCoordinatorPayload {
                        node_id: node_id.to_string(),
                    };
                    if let Err(e) = conn.send(MessageType::WhoIsCoordinator, 0, &req).await {
                        tracing::debug!("failed to send WhoIsCoordinator to {seed}: {e}");
                        continue;
                    }
                    match tokio::time::timeout(Duration::from_secs(3), conn.recv()).await {
                        Ok(Ok((header, payload)))
                            if header.msg_type == MessageType::WhoIsCoordinatorResponse =>
                        {
                            let resp: WhoIsCoordinatorResponsePayload =
                                FramedConnection::deserialize_payload(&payload)?;
                            if let Some(addr) = resp.coordinator_addr {
                                return Ok((addr, resp.term, resp.manifest));
                            }
                            tracing::debug!("seed {seed} doesn't know the coordinator (mid-election?)");
                        }
                        Ok(Ok((header, _))) => {
                            tracing::debug!(
                                "unexpected response from seed {seed}: {:?}",
                                header.msg_type
                            );
                        }
                        Ok(Err(e)) => {
                            tracing::debug!("recv from seed {seed} failed: {e}");
                        }
                        Err(_) => {
                            tracing::debug!("seed {seed} response timed out");
                        }
                    }
                }
                Ok(Err(e)) => {
                    tracing::debug!("failed to connect to seed {seed}: {e}");
                }
                Err(_) => {
                    tracing::debug!("seed {seed} connection timed out");
                }
            }
        }

        // All seeds failed -- backoff and retry
        let jitter_range = backoff.as_millis() as f64 * 0.25;
        let jitter_ms = (rand::random::<f64>() - 0.5) * 2.0 * jitter_range;
        let sleep_ms = (backoff.as_millis() as f64 + jitter_ms).max(100.0);
        tracing::info!(
            "all seeds unreachable or no coordinator known -- retrying in {:.1}s",
            sleep_ms / 1000.0
        );
        tokio::time::sleep(Duration::from_millis(sleep_ms as u64)).await;
        backoff = (backoff * 2).min(max_backoff);
    }
}

/// Spawn a background task that listens for peer connections (seed queries, election).
///
/// Returns the bound address (for advertising in manifests).
async fn spawn_peer_listener(
    peer_port: u16,
    cluster_manifest: Arc<tokio::sync::Mutex<Option<ClusterManifestPayload>>>,
    current_term: Arc<AtomicU64>,
) -> Result<std::net::SocketAddr> {
    let listener = tokio::net::TcpListener::bind(format!("0.0.0.0:{peer_port}")).await?;
    let addr = listener.local_addr()?;
    tracing::info!("peer listener started on {addr}");

    tokio::spawn(async move {
        loop {
            let (stream, peer_addr) = match listener.accept().await {
                Ok(pair) => pair,
                Err(e) => {
                    tracing::error!("peer listener accept error: {e}");
                    continue;
                }
            };

            let manifest = Arc::clone(&cluster_manifest);
            let term = Arc::clone(&current_term);
            tokio::spawn(async move {
                let mut conn = FramedConnection::new(stream);
                let recv = tokio::time::timeout(Duration::from_secs(3), conn.recv()).await;
                match recv {
                    Ok(Ok((header, _))) if header.msg_type == MessageType::WhoIsCoordinator => {
                        let manifest_guard = manifest.lock().await;
                        let coordinator_addr = manifest_guard.as_ref().and_then(|m| {
                            m.nodes
                                .iter()
                                .find(|n| n.role == fracture_protocol::messages::NodeRole::Coordinator)
                                .map(|n| n.address.clone())
                        });
                        let resp = WhoIsCoordinatorResponsePayload {
                            coordinator_addr,
                            term: term.load(Ordering::SeqCst),
                            manifest: manifest_guard.clone(),
                        };
                        let _ = conn
                            .send(MessageType::WhoIsCoordinatorResponse, 0, &resp)
                            .await;
                    }
                    Ok(Ok((header, _))) => {
                        tracing::debug!(
                            "peer {peer_addr}: unexpected message {:?}",
                            header.msg_type
                        );
                    }
                    Ok(Err(e)) => {
                        tracing::debug!("peer {peer_addr}: recv error: {e}");
                    }
                    Err(_) => {
                        tracing::debug!("peer {peer_addr}: recv timeout");
                    }
                }
            });
        }
    });

    Ok(addr)
}

use fracture_coordinator::pipeline::DistributedPipeline;
use fracture_coordinator::registry::PeerRegistry;
use fracture_coordinator::scheduler::{self, WorkerCapabilities, SchedulerInput, SchedulingMode};

/// Promote this worker to coordinator (FT-12).
///
/// Spawns coordinator tasks alongside existing worker tasks:
/// - TCP listener for worker (re)registration
/// - HTTP server for client requests
///
/// State reconstruction (FT-12b) happens after promotion: the new coordinator
/// collects ReRegister from all workers and rebuilds the pipeline.
///
/// Returns the TCP listener address and HTTP port for Victory broadcast.
async fn promote_to_coordinator(
    node_id: &str,
    coordinator_port: u16,
    http_port: u16,
) -> Result<(tokio::net::TcpListener, std::net::SocketAddr)> {
    // Bind TCP listener for worker connections.
    let tcp_addr = format!("0.0.0.0:{coordinator_port}");
    let listener = tokio::net::TcpListener::bind(&tcp_addr).await?;
    let local_addr = listener.local_addr()?;
    tracing::info!(
        "promoted to coordinator: TCP listener on {local_addr}, HTTP on port {http_port}"
    );
    tracing::info!(
        "node '{}' is now acting as both worker and coordinator",
        node_id
    );

    // HTTP server setup is deferred to FT-12b (state reconstruction)
    // because we need the pipeline to be rebuilt before we can serve requests.

    Ok((listener, local_addr))
}

/// Reconstruct coordinator state from worker registrations (FT-12b).
///
/// The newly promoted coordinator:
/// 1. Registers itself in a fresh PeerRegistry
/// 2. Accepts ReRegister from other workers (with timeout)
/// 3. Runs the scheduler to assign layers
/// 4. Builds a DistributedPipeline
///
/// Returns the pipeline and registry for the scheduler loop.
async fn reconstruct_state(
    listener: &tokio::net::TcpListener,
    self_node_id: &str,
    self_caps: WorkerCapabilities,
    model_config: &fracture_core::ModelConfig,
    max_seq_len: usize,
    expected_peers: usize,
    timeout: Duration,
) -> Result<(Arc<DistributedPipeline>, Arc<tokio::sync::Mutex<PeerRegistry>>)> {
    let registry = Arc::new(tokio::sync::Mutex::new(PeerRegistry::new()));

    // Register self as a worker in the registry (we're both coordinator and worker).
    // We don't have a FramedConnection to ourselves, so we can't use the normal
    // register() path. Instead we'll handle self-assignment separately after scheduling.
    tracing::info!("state reconstruction: expecting {} peer workers", expected_peers);

    let mut peer_count = 0;
    let deadline = tokio::time::Instant::now() + timeout;

    while peer_count < expected_peers {
        let remaining = deadline - tokio::time::Instant::now();
        if remaining.is_zero() {
            tracing::warn!(
                "state reconstruction timeout: got {}/{} peers",
                peer_count, expected_peers
            );
            break;
        }

        let accept = tokio::time::timeout(remaining, listener.accept()).await;
        match accept {
            Ok(Ok((stream, addr))) => {
                let mut conn = FramedConnection::new(stream);
                match conn.recv().await {
                    Ok((header, payload)) if header.msg_type == MessageType::ReRegister => {
                        let rereg: ReRegisterPayload =
                            FramedConnection::deserialize_payload(&payload)?;
                        tracing::info!(
                            "peer '{}' re-registered from {addr}: layers {:?}",
                            rereg.node_id,
                            rereg.current_layer_start.zip(rereg.current_layer_end),
                        );
                        let caps = WorkerCapabilities {
                            node_id: rereg.node_id.clone(),
                            gpu_model: rereg.gpu_model,
                            gpu_memory_available: rereg.gpu_memory_available as usize,
                            compute_capability: rereg.compute_capability,
                            decode_ms_per_layer: rereg.decode_ms_per_layer,
                            prefill_ms_per_layer_128: rereg.prefill_ms_per_layer_128,
                        };
                        let mut reg = registry.lock().await;
                        reg.register(caps, conn)?;
                        peer_count += 1;
                    }
                    Ok((header, _)) => {
                        tracing::warn!(
                            "expected ReRegister from {addr}, got {:?}",
                            header.msg_type
                        );
                    }
                    Err(e) => {
                        tracing::warn!("failed to read from {addr}: {e}");
                    }
                }
            }
            Ok(Err(e)) => {
                tracing::error!("accept error during reconstruction: {e}");
            }
            Err(_) => {
                tracing::warn!(
                    "state reconstruction timeout: got {}/{} peers",
                    peer_count, expected_peers
                );
                break;
            }
        }
    }

    tracing::info!(
        "state reconstruction: {} peers registered, running scheduler",
        peer_count
    );

    // Run scheduler with all workers (self + peers).
    let mut all_caps = {
        let reg = registry.lock().await;
        reg.all_capabilities()
    };
    all_caps.push(self_caps);

    let input = SchedulerInput {
        model_config: model_config.clone(),
        workers: all_caps,
        coordinator_compute: None,
        mode: SchedulingMode::Auto,
        max_seq_len,
        hop_latency_ms: 2.0,
    };
    let result = scheduler::schedule(&input)?;

    tracing::info!("state reconstruction: schedule computed");
    for a in &result.assignments {
        tracing::info!("  {} -> layers {:?} ({:?})", a.node_id, a.layer_range, a.role);
    }

    // Send RegisterAck to peers (assignment unchanged -> skip weight reload,
    // assignment changed -> Reconfigure).
    {
        let mut reg = registry.lock().await;
        for assignment in &result.assignments {
            if assignment.node_id == self_node_id {
                continue; // Self -- handled separately
            }
            let ack = RegisterAckPayload {
                layer_start: assignment.layer_range.start as u32,
                layer_end: assignment.layer_range.end as u32,
                total_layers: model_config.num_layers as u32,
                max_seq_len: max_seq_len as u32,
                model_config: model_config.clone(),
            };
            reg.assign(&assignment.node_id, assignment.clone()).ok();
            if let Some(entry) = reg.get_mut(&assignment.node_id) {
                // Use RegisterAck for now -- workers will determine if they need
                // to reconfigure based on whether the assignment matches.
                if let Err(e) = entry.writer.send(MessageType::RegisterAck, 0, &ack).await {
                    tracing::error!("failed to send to '{}': {e}", assignment.node_id);
                }
            }
        }
    }

    // Wait for WorkerReady from peers.
    {
        let reg = registry.lock().await;
        for assignment in &result.assignments {
            if assignment.node_id == self_node_id {
                continue;
            }
            if let Some(entry) = reg.get(&assignment.node_id) {
                match entry.reader.lock().await.recv().await {
                    Ok((hdr, _)) if hdr.msg_type == MessageType::WorkerReady => {
                        tracing::info!("peer '{}' ready", assignment.node_id);
                    }
                    Ok((hdr, _)) => {
                        tracing::error!(
                            "expected WorkerReady from '{}', got {:?}",
                            assignment.node_id, hdr.msg_type
                        );
                    }
                    Err(e) => {
                        tracing::error!("recv from '{}' failed: {e}", assignment.node_id);
                    }
                }
            }
        }
    }

    // Build pipeline.
    let pipeline = Arc::new(DistributedPipeline::new(
        &result.assignments,
        model_config.hidden_size,
    )?);
    tracing::info!(
        "state reconstruction complete: {} pipeline stages",
        pipeline.pipeline_order().len()
    );

    Ok((pipeline, registry))
}
