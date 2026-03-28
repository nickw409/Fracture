//! IPC integration tests: validate split equivalence over Unix domain sockets.
//!
//! These tests spawn a tail node as an IpcNodeServer in a background thread,
//! connect to it via IpcNodeClient, and verify that the IPC pipeline produces
//! the same output as an in-process pipeline.

use fracture_core::{Backend, DType, DeviceTensor, DeviceTimer, ModelConfig, Result, TensorId};
use fracture_engine::ipc::{IpcNodeClient, IpcNodeServer};
use fracture_engine::{
    ComputeNodeImpl, Engine, KvCacheManager, NodeConfig, PipelineCoordinator,
};
use fracture_gguf::{LayerWeights, WeightStore};
use std::collections::HashMap;
use std::os::unix::net::UnixListener;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

// ── MemBackend: stores tensor data in memory for IPC roundtrip testing ───

struct MemBackend {
    next_id: AtomicU64,
    store: Mutex<HashMap<u64, Vec<u8>>>,
}

impl MemBackend {
    fn new() -> Self {
        Self {
            next_id: AtomicU64::new(1000),
            store: Mutex::new(HashMap::new()),
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

// ── Test helpers ─────────────────────────────────────────────────────────

fn test_config() -> ModelConfig {
    ModelConfig {
        hidden_size: 8,
        num_layers: 2,
        num_q_heads: 2,
        num_kv_heads: 2,
        head_dim: 4,
        intermediate_size: 16,
        vocab_size: 32,
        rope_theta: 10000.0,
        rms_norm_eps: 1e-5,
        max_seq_len: 512,
    }
}

fn mock_tensor(id: &AtomicU64, shape: Vec<usize>) -> DeviceTensor {
    let i = id.fetch_add(1, Ordering::Relaxed);
    DeviceTensor::new(TensorId(i), shape, DType::FP16)
}

fn mock_weights(cfg: &ModelConfig, num_layers: usize) -> WeightStore {
    let id = AtomicU64::new(1);
    let h = cfg.hidden_size;
    let kv = cfg.num_kv_heads * cfg.head_dim;
    let inter = cfg.intermediate_size;
    let layers = (0..num_layers)
        .map(|_| LayerWeights {
            q_proj: mock_tensor(&id, vec![h, h]),
            k_proj: mock_tensor(&id, vec![kv, h]),
            v_proj: mock_tensor(&id, vec![kv, h]),
            o_proj: mock_tensor(&id, vec![h, h]),
            gate_proj: mock_tensor(&id, vec![inter, h]),
            up_proj: mock_tensor(&id, vec![inter, h]),
            down_proj: mock_tensor(&id, vec![h, inter]),
            attn_norm: mock_tensor(&id, vec![h]),
            ffn_norm: mock_tensor(&id, vec![h]),
        })
        .collect();
    WeightStore {
        config: cfg.clone(),
        token_embedding: mock_tensor(&id, vec![cfg.vocab_size, h]),
        layers,
        output_norm: mock_tensor(&id, vec![h]),
        lm_head: mock_tensor(&id, vec![cfg.vocab_size, h]),
    }
}

fn build_node(cfg: &ModelConfig, range: std::ops::Range<usize>) -> ComputeNodeImpl<MemBackend> {
    let node_config = NodeConfig::new(range.clone(), cfg.num_layers).unwrap();
    let weights = mock_weights(cfg, range.len());
    let engine = Engine::new(MemBackend::new(), weights, range);
    ComputeNodeImpl::new(engine, node_config)
}

// ── Tests ────────────────────────────────────────────────────────────────

/// IPC 2-node split: head runs in-process, tail runs over a Unix socket.
/// Output should match an in-process pipeline.
#[test]
fn test_ipc_split_matches_in_process() {
    let cfg = test_config();

    // 1. In-process pipeline for reference
    let head_inproc = build_node(&cfg, 0..1);
    let tail_inproc = build_node(&cfg, 1..2);
    let coord = PipelineCoordinator::new(vec![
        Box::new(head_inproc),
        Box::new(tail_inproc),
    ])
    .unwrap();

    let backend_ref = MemBackend::new();
    let mut ch_ref = KvCacheManager::new(1, cfg.num_kv_heads, cfg.head_dim, cfg.max_seq_len);
    let mut ct_ref = KvCacheManager::new(1, cfg.num_kv_heads, cfg.head_dim, cfg.max_seq_len);
    let hh_ref = ch_ref.alloc(&backend_ref).unwrap();
    let ht_ref = ct_ref.alloc(&backend_ref).unwrap();

    let logits_inproc = coord
        .forward(
            &[1, 2, 3],
            &[0, 1, 2],
            &mut [&mut ch_ref, &mut ct_ref],
            &[hh_ref, ht_ref],
        )
        .unwrap();

    // 2. IPC pipeline: tail node served over Unix socket
    let socket_dir = tempfile::tempdir().unwrap();
    let socket_path = socket_dir.path().join("tail.sock");
    let socket_path_str = socket_path.to_str().unwrap().to_string();

    let listener = UnixListener::bind(&socket_path).unwrap();

    // Spawn tail server in a thread
    let server_handle = std::thread::spawn(move || {
        let tail_node = build_node(&test_config(), 1..2);
        let cache = KvCacheManager::new(1, 2, 4, 512);
        let backend = MemBackend::new();
        let mut server = IpcNodeServer::new(tail_node, cache);
        // serve() blocks until connection closes
        let _ = server.serve_one(&listener, &backend);
    });

    // Give server a moment to start listening
    std::thread::sleep(std::time::Duration::from_millis(50));

    // Head runs in-process, tail via IPC
    let head_node = build_node(&cfg, 0..1);
    let tail_client = IpcNodeClient::connect(
        &socket_path_str,
        MemBackend::new(),
        NodeConfig::new(1..2, cfg.num_layers).unwrap(),
    )
    .unwrap();

    let coord_ipc = PipelineCoordinator::new(vec![
        Box::new(head_node),
        Box::new(tail_client),
    ])
    .unwrap();

    let backend_ipc = MemBackend::new();
    let mut ch_ipc = KvCacheManager::new(1, cfg.num_kv_heads, cfg.head_dim, cfg.max_seq_len);
    // IPC client doesn't use its own cache (server manages it), but coordinator requires one
    let mut ct_ipc = KvCacheManager::new(1, cfg.num_kv_heads, cfg.head_dim, cfg.max_seq_len);
    let hh_ipc = ch_ipc.alloc(&backend_ipc).unwrap();
    let ht_ipc = ct_ipc.alloc(&backend_ipc).unwrap();

    let logits_ipc = coord_ipc
        .forward(
            &[1, 2, 3],
            &[0, 1, 2],
            &mut [&mut ch_ipc, &mut ct_ipc],
            &[hh_ipc, ht_ipc],
        )
        .unwrap();

    // Drop coordinator to close the Unix socket, allowing server thread to exit
    drop(coord_ipc);
    let _ = server_handle.join();

    // 3. Compare
    assert_eq!(logits_inproc.len(), logits_ipc.len());
    assert_eq!(logits_inproc, logits_ipc, "IPC split should match in-process split");
}

/// IPC multi-step decode: prefill + 3 decode steps over Unix socket.
#[test]
fn test_ipc_multistep_decode() {
    let cfg = test_config();

    let socket_dir = tempfile::tempdir().unwrap();
    let socket_path = socket_dir.path().join("tail_decode.sock");
    let socket_path_str = socket_path.to_str().unwrap().to_string();

    let listener = UnixListener::bind(&socket_path).unwrap();

    let server_handle = std::thread::spawn(move || {
        let tail_node = build_node(&test_config(), 1..2);
        let cache = KvCacheManager::new(1, 2, 4, 512);
        let backend = MemBackend::new();
        let mut server = IpcNodeServer::new(tail_node, cache);
        let _ = server.serve_one(&listener, &backend);
    });

    std::thread::sleep(std::time::Duration::from_millis(50));

    let head_node = build_node(&cfg, 0..1);
    let tail_client = IpcNodeClient::connect(
        &socket_path_str,
        MemBackend::new(),
        NodeConfig::new(1..2, cfg.num_layers).unwrap(),
    )
    .unwrap();

    let coord = PipelineCoordinator::new(vec![
        Box::new(head_node),
        Box::new(tail_client),
    ])
    .unwrap();

    let backend = MemBackend::new();
    let mut ch = KvCacheManager::new(1, cfg.num_kv_heads, cfg.head_dim, cfg.max_seq_len);
    let mut ct = KvCacheManager::new(1, cfg.num_kv_heads, cfg.head_dim, cfg.max_seq_len);
    let hh = ch.alloc(&backend).unwrap();
    let ht = ct.alloc(&backend).unwrap();

    // Prefill
    let logits = coord
        .forward(&[1, 2, 3], &[0, 1, 2], &mut [&mut ch, &mut ct], &[hh, ht])
        .unwrap();
    assert_eq!(logits.len(), cfg.vocab_size);

    // 3 decode steps
    for step in 0..3u32 {
        let logits = coord
            .forward(&[42], &[3 + step], &mut [&mut ch, &mut ct], &[hh, ht])
            .unwrap();
        assert_eq!(logits.len(), cfg.vocab_size, "decode step {step}");
    }

    drop(coord);
    let _ = server_handle.join();
}
