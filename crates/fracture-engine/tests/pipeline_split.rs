//! Integration tests for Phase 2 pipeline splitting.
//!
//! These tests validate that splitting a model across multiple ComputeNodes
//! and chaining them through PipelineCoordinator produces correct results.

use fracture_core::{Backend, DType, DeviceTensor, DeviceTimer, ModelConfig, Result, TensorId};
use fracture_engine::{
    ComputeNodeImpl, Engine, KvCacheManager, NodeConfig, NodeInput, PipelineCoordinator,
};
use fracture_gguf::{LayerWeights, WeightStore};
use std::sync::atomic::{AtomicU64, Ordering};

// ── Mock Backend ─────────────────────────────────────────────────────────

struct MockBackend {
    next_id: AtomicU64,
}

impl MockBackend {
    fn new() -> Self {
        Self {
            next_id: AtomicU64::new(1000),
        }
    }
}

impl Backend for MockBackend {
    fn alloc(&self, shape: &[usize], dtype: DType) -> Result<DeviceTensor> {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        Ok(DeviceTensor::new(TensorId(id), shape.to_vec(), dtype))
    }
    fn free(&self, _tensor: &DeviceTensor) -> Result<()> {
        Ok(())
    }
    fn copy_to_device(&self, _dst: &DeviceTensor, _src: &[u8]) -> Result<()> {
        Ok(())
    }
    fn copy_to_host(&self, _src: &DeviceTensor, dst: &mut [u8]) -> Result<()> {
        dst.fill(0);
        Ok(())
    }
    fn matmul(&self, _a: &DeviceTensor, _b: &DeviceTensor, _out: &DeviceTensor) -> Result<()> {
        Ok(())
    }
    fn rmsnorm(
        &self,
        _input: &DeviceTensor,
        _weight: &DeviceTensor,
        _eps: f64,
        _out: &DeviceTensor,
    ) -> Result<()> {
        Ok(())
    }
    fn rope(
        &self,
        _q: &DeviceTensor,
        _k: &DeviceTensor,
        _positions: &[u32],
        _theta: f64,
        _head_dim: usize,
    ) -> Result<()> {
        Ok(())
    }
    fn attention(
        &self,
        _q: &DeviceTensor,
        _k_cache: &DeviceTensor,
        _v_cache: &DeviceTensor,
        _num_kv_heads: usize,
        _start_pos: usize,
        _out: &DeviceTensor,
    ) -> Result<()> {
        Ok(())
    }
    fn silu_mul(
        &self,
        _gate: &DeviceTensor,
        _up: &DeviceTensor,
        _out: &DeviceTensor,
    ) -> Result<()> {
        Ok(())
    }
    fn embedding(
        &self,
        _token_ids: &[u32],
        _embedding_table: &DeviceTensor,
        _out: &DeviceTensor,
    ) -> Result<()> {
        Ok(())
    }
    fn add(&self, _a: &DeviceTensor, _b: &DeviceTensor, _out: &DeviceTensor) -> Result<()> {
        Ok(())
    }
    fn copy_rows(
        &self,
        _src: &DeviceTensor,
        _dst: &DeviceTensor,
        _src_offset: usize,
        _dst_offset: usize,
        _count: usize,
    ) -> Result<()> {
        Ok(())
    }
    fn device_name(&self) -> &str {
        "mock"
    }
    fn total_memory(&self) -> usize {
        1 << 30
    }
    fn available_memory(&self) -> usize {
        1 << 30
    }
    fn synchronize(&self) -> Result<()> {
        Ok(())
    }
    fn create_timer(&self) -> Result<DeviceTimer> {
        Ok(DeviceTimer(0))
    }
    fn start_timer(&self, _timer: &DeviceTimer) -> Result<()> {
        Ok(())
    }
    fn stop_timer(&self, _timer: &DeviceTimer) -> Result<f32> {
        Ok(0.0)
    }
    fn destroy_timer(&self, _timer: &DeviceTimer) -> Result<()> {
        Ok(())
    }
}

// ── Test helpers ─────────────────────────────────────────────────────────

fn test_config(num_layers: usize) -> ModelConfig {
    ModelConfig {
        hidden_size: 8,
        num_layers,
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

fn mock_layer_weights(id: &AtomicU64, cfg: &ModelConfig) -> LayerWeights {
    let h = cfg.hidden_size;
    let kv = cfg.num_kv_heads * cfg.head_dim;
    let inter = cfg.intermediate_size;
    LayerWeights {
        q_proj: mock_tensor(id, vec![h, h]),
        k_proj: mock_tensor(id, vec![kv, h]),
        v_proj: mock_tensor(id, vec![kv, h]),
        o_proj: mock_tensor(id, vec![h, h]),
        gate_proj: mock_tensor(id, vec![inter, h]),
        up_proj: mock_tensor(id, vec![inter, h]),
        down_proj: mock_tensor(id, vec![h, inter]),
        attn_norm: mock_tensor(id, vec![h]),
        ffn_norm: mock_tensor(id, vec![h]),
    }
}

fn mock_weights(cfg: &ModelConfig, num_weight_layers: usize) -> WeightStore {
    let id = AtomicU64::new(1);
    let h = cfg.hidden_size;
    let layers = (0..num_weight_layers)
        .map(|_| mock_layer_weights(&id, cfg))
        .collect();
    WeightStore {
        config: cfg.clone(),
        token_embedding: mock_tensor(&id, vec![cfg.vocab_size, h]),
        layers,
        output_norm: mock_tensor(&id, vec![h]),
        lm_head: mock_tensor(&id, vec![cfg.vocab_size, h]),
    }
}

/// Build a ComputeNodeImpl for a given layer range.
fn build_node(
    cfg: &ModelConfig,
    layer_range: std::ops::Range<usize>,
) -> ComputeNodeImpl<MockBackend> {
    let node_config = NodeConfig::new(layer_range.clone(), cfg.num_layers).unwrap();
    let weights = mock_weights(cfg, layer_range.len());
    let engine = Engine::new(MockBackend::new(), weights, layer_range);
    ComputeNodeImpl::new(engine, node_config)
}

// ── Tests ────────────────────────────────────────────────────────────────

/// Single-node pipeline coordinator matches direct Engine::forward().
#[test]
fn test_single_node_pipeline_matches_direct() {
    let cfg = test_config(4);

    // Direct engine path
    let weights = mock_weights(&cfg, cfg.num_layers);
    let engine = Engine::new(MockBackend::new(), weights, 0..cfg.num_layers);
    let mut cache = KvCacheManager::new(cfg.num_layers, cfg.num_kv_heads, cfg.head_dim, cfg.max_seq_len);
    let handle = cache.alloc(engine.backend()).unwrap();
    let direct_logits = engine.forward(&[1, 2, 3], &[0, 1, 2], &mut cache, handle, None).unwrap();

    // Pipeline path (single node covering all layers)
    let node = build_node(&cfg, 0..cfg.num_layers);
    let coordinator = PipelineCoordinator::new(vec![Box::new(node)]).unwrap();
    let mut pipeline_cache = KvCacheManager::new(cfg.num_layers, cfg.num_kv_heads, cfg.head_dim, cfg.max_seq_len);
    let pipeline_handle = pipeline_cache.alloc(&MockBackend::new()).unwrap();

    let pipeline_logits = coordinator
        .forward(
            &[1, 2, 3],
            &[0, 1, 2],
            &mut [&mut pipeline_cache],
            &[pipeline_handle],
        )
        .unwrap();

    assert_eq!(direct_logits.len(), pipeline_logits.len());
    // Both should be zero-filled since mock backend zero-fills copy_to_host
    assert_eq!(direct_logits, pipeline_logits);
}

/// 2-node split: [0, 2) + [2, 4) on a 4-layer model.
#[test]
fn test_two_node_split() {
    let cfg = test_config(4);

    let head = build_node(&cfg, 0..2);
    let tail = build_node(&cfg, 2..4);
    let coordinator = PipelineCoordinator::new(vec![Box::new(head), Box::new(tail)]).unwrap();

    let backend = MockBackend::new();
    let mut cache_head = KvCacheManager::new(2, cfg.num_kv_heads, cfg.head_dim, cfg.max_seq_len);
    let mut cache_tail = KvCacheManager::new(2, cfg.num_kv_heads, cfg.head_dim, cfg.max_seq_len);
    let handle_head = cache_head.alloc(&backend).unwrap();
    let handle_tail = cache_tail.alloc(&backend).unwrap();

    let logits = coordinator
        .forward(
            &[1, 2, 3],
            &[0, 1, 2],
            &mut [&mut cache_head, &mut cache_tail],
            &[handle_head, handle_tail],
        )
        .unwrap();

    assert_eq!(logits.len(), cfg.vocab_size);
}

/// 3-node split: [0, 1) + [1, 3) + [3, 4) on a 4-layer model.
#[test]
fn test_three_node_split() {
    let cfg = test_config(4);

    let head = build_node(&cfg, 0..1);
    let middle = build_node(&cfg, 1..3);
    let tail = build_node(&cfg, 3..4);
    let coordinator =
        PipelineCoordinator::new(vec![Box::new(head), Box::new(middle), Box::new(tail)]).unwrap();

    let backend = MockBackend::new();
    let mut c0 = KvCacheManager::new(1, cfg.num_kv_heads, cfg.head_dim, cfg.max_seq_len);
    let mut c1 = KvCacheManager::new(2, cfg.num_kv_heads, cfg.head_dim, cfg.max_seq_len);
    let mut c2 = KvCacheManager::new(1, cfg.num_kv_heads, cfg.head_dim, cfg.max_seq_len);
    let h0 = c0.alloc(&backend).unwrap();
    let h1 = c1.alloc(&backend).unwrap();
    let h2 = c2.alloc(&backend).unwrap();

    let logits = coordinator
        .forward(
            &[5],
            &[0],
            &mut [&mut c0, &mut c1, &mut c2],
            &[h0, h1, h2],
        )
        .unwrap();

    assert_eq!(logits.len(), cfg.vocab_size);
}

/// Asymmetric split: [0, 1) + [1, 4) on a 4-layer model.
#[test]
fn test_asymmetric_split() {
    let cfg = test_config(4);

    let head = build_node(&cfg, 0..1);
    let tail = build_node(&cfg, 1..4);
    let coordinator = PipelineCoordinator::new(vec![Box::new(head), Box::new(tail)]).unwrap();

    let backend = MockBackend::new();
    let mut c0 = KvCacheManager::new(1, cfg.num_kv_heads, cfg.head_dim, cfg.max_seq_len);
    let mut c1 = KvCacheManager::new(3, cfg.num_kv_heads, cfg.head_dim, cfg.max_seq_len);
    let h0 = c0.alloc(&backend).unwrap();
    let h1 = c1.alloc(&backend).unwrap();

    let logits = coordinator
        .forward(
            &[10, 20],
            &[0, 1],
            &mut [&mut c0, &mut c1],
            &[h0, h1],
        )
        .unwrap();

    assert_eq!(logits.len(), cfg.vocab_size);
}

/// Multi-step decode: run 50 decode iterations through a 2-node split.
#[test]
fn test_multistep_decode_through_split() {
    let cfg = test_config(4);

    let head = build_node(&cfg, 0..2);
    let tail = build_node(&cfg, 2..4);
    let coordinator = PipelineCoordinator::new(vec![Box::new(head), Box::new(tail)]).unwrap();

    let backend = MockBackend::new();
    let mut c0 = KvCacheManager::new(2, cfg.num_kv_heads, cfg.head_dim, cfg.max_seq_len);
    let mut c1 = KvCacheManager::new(2, cfg.num_kv_heads, cfg.head_dim, cfg.max_seq_len);
    let h0 = c0.alloc(&backend).unwrap();
    let h1 = c1.alloc(&backend).unwrap();

    // Prefill with 3 tokens
    let logits = coordinator
        .forward(
            &[1, 2, 3],
            &[0, 1, 2],
            &mut [&mut c0, &mut c1],
            &[h0, h1],
        )
        .unwrap();
    assert_eq!(logits.len(), cfg.vocab_size);

    // 50 decode steps
    for step in 0..50 {
        let pos = 3 + step as u32;
        let logits = coordinator
            .forward(
                &[42],
                &[pos],
                &mut [&mut c0, &mut c1],
                &[h0, h1],
            )
            .unwrap();
        assert_eq!(logits.len(), cfg.vocab_size, "decode step {step} failed");
    }
}

/// Error: non-head node should not start the pipeline.
#[test]
fn test_pipeline_rejects_non_head_first() {
    let cfg = test_config(4);
    let tail = build_node(&cfg, 2..4);
    let result = PipelineCoordinator::new(vec![Box::new(tail)]);
    assert!(result.is_err());
    assert!(result
        .err()
        .unwrap()
        .to_string()
        .contains("head"));
}

/// Error: last node must be tail.
#[test]
fn test_pipeline_rejects_non_tail_last() {
    let cfg = test_config(4);
    let head = build_node(&cfg, 0..2);
    let result = PipelineCoordinator::new(vec![Box::new(head)]);
    assert!(result.is_err());
    assert!(result
        .err()
        .unwrap()
        .to_string()
        .contains("tail"));
}

/// Error: gap between nodes.
#[test]
fn test_pipeline_rejects_gap() {
    let cfg = test_config(4);
    let head = build_node(&cfg, 0..1);
    let tail = build_node(&cfg, 3..4); // gap: layers 1-2 missing
    let result = PipelineCoordinator::new(vec![Box::new(head), Box::new(tail)]);
    assert!(result.is_err());
    assert!(result
        .err()
        .unwrap()
        .to_string()
        .contains("gap"));
}

/// Error: wrong cache/handle count.
#[test]
fn test_pipeline_rejects_mismatched_caches() {
    let cfg = test_config(4);
    let node = build_node(&cfg, 0..4);
    let coordinator = PipelineCoordinator::new(vec![Box::new(node)]).unwrap();

    // Pass 0 caches instead of 1
    let result = coordinator.forward(&[1], &[0], &mut [], &[]);
    assert!(result.is_err());
    assert!(result
        .err()
        .unwrap()
        .to_string()
        .contains("expected 1"));
}

/// Error: head node receiving activations.
#[test]
fn test_head_node_rejects_activations_input() {
    let cfg = test_config(4);
    let node = build_node(&cfg, 0..2);
    let node_config = NodeConfig::new(0..2, 4).unwrap();

    let mut cache = KvCacheManager::new(2, cfg.num_kv_heads, cfg.head_dim, cfg.max_seq_len);
    let handle = cache.alloc(&MockBackend::new()).unwrap();

    let fake = MockBackend::new()
        .alloc(&[1, cfg.hidden_size], DType::FP16)
        .unwrap();
    let input = NodeInput::Activations {
        hidden_states: fake,
        positions: vec![0],
    };

    let result = node.engine().forward_node(input, &node_config, &mut cache, handle, None);
    assert!(result.is_err());
    assert!(result
        .err()
        .unwrap()
        .to_string()
        .contains("head node received Activations"));
}

// ── Fake nodes for error path testing ────────────────────────────────

use fracture_core::ForwardProfile;
use fracture_engine::{CacheHandle, ComputeNode, NodeOutput};

/// A fake node that always returns Logits regardless of position in pipeline.
struct AlwaysLogitsNode {
    node_config: NodeConfig,
}

impl ComputeNode for AlwaysLogitsNode {
    fn forward(
        &self,
        _input: NodeInput,
        _cache: &mut KvCacheManager,
        _handle: CacheHandle,
        _profile: Option<&mut ForwardProfile>,
    ) -> Result<NodeOutput> {
        Ok(NodeOutput::Logits(vec![0.0; 32]))
    }
    fn config(&self) -> &NodeConfig {
        &self.node_config
    }
}

/// A fake node that always returns Activations regardless of position.
struct AlwaysActivationsNode {
    node_config: NodeConfig,
}

impl ComputeNode for AlwaysActivationsNode {
    fn forward(
        &self,
        _input: NodeInput,
        _cache: &mut KvCacheManager,
        _handle: CacheHandle,
        _profile: Option<&mut ForwardProfile>,
    ) -> Result<NodeOutput> {
        let tensor = MockBackend::new()
            .alloc(&[1, 8], DType::FP16)
            .expect("alloc");
        Ok(NodeOutput::Activations(tensor))
    }
    fn config(&self) -> &NodeConfig {
        &self.node_config
    }
}

/// Error: intermediate node returning Logits instead of Activations.
#[test]
fn test_pipeline_rejects_intermediate_logits() {
    // Head returns Logits (wrong), tail never gets called
    let head = AlwaysLogitsNode {
        node_config: NodeConfig::new(0..2, 4).unwrap(),
    };
    let tail = build_node(&test_config(4), 2..4);
    let coordinator = PipelineCoordinator::new(vec![Box::new(head), Box::new(tail)]).unwrap();

    let backend = MockBackend::new();
    let mut c0 = KvCacheManager::new(2, 2, 4, 512);
    let mut c1 = KvCacheManager::new(2, 2, 4, 512);
    let h0 = c0.alloc(&backend).unwrap();
    let h1 = c1.alloc(&backend).unwrap();

    let result = coordinator.forward(&[1], &[0], &mut [&mut c0, &mut c1], &[h0, h1]);
    assert!(result.is_err());
    assert!(result
        .err()
        .unwrap()
        .to_string()
        .contains("not the last node"));
}

/// Error: final node returning Activations instead of Logits.
#[test]
fn test_pipeline_rejects_final_activations() {
    // Head returns Activations (correct), tail returns Activations (wrong)
    let head = AlwaysActivationsNode {
        node_config: NodeConfig::new(0..2, 4).unwrap(),
    };
    let tail = AlwaysActivationsNode {
        node_config: NodeConfig::new(2..4, 4).unwrap(),
    };
    let coordinator = PipelineCoordinator::new(vec![Box::new(head), Box::new(tail)]).unwrap();

    let backend = MockBackend::new();
    let mut c0 = KvCacheManager::new(2, 2, 4, 512);
    let mut c1 = KvCacheManager::new(2, 2, 4, 512);
    let h0 = c0.alloc(&backend).unwrap();
    let h1 = c1.alloc(&backend).unwrap();

    let result = coordinator.forward(&[1], &[0], &mut [&mut c0, &mut c1], &[h0, h1]);
    assert!(result.is_err());
    assert!(result
        .err()
        .unwrap()
        .to_string()
        .contains("Activations instead of Logits"));
}

/// Error: non-head node receiving token IDs.
#[test]
fn test_nonhead_node_rejects_token_ids_input() {
    let cfg = test_config(4);
    let node = build_node(&cfg, 2..4);
    let node_config = NodeConfig::new(2..4, 4).unwrap();

    let mut cache = KvCacheManager::new(2, cfg.num_kv_heads, cfg.head_dim, cfg.max_seq_len);
    let handle = cache.alloc(&MockBackend::new()).unwrap();

    let input = NodeInput::TokenIds {
        ids: vec![1],
        positions: vec![0],
    };

    let result = node.engine().forward_node(input, &node_config, &mut cache, handle, None);
    assert!(result.is_err());
    assert!(result
        .err()
        .unwrap()
        .to_string()
        .contains("non-head node received TokenIds"));
}
