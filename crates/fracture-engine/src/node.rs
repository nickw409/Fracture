use crate::kv_cache::{CacheHandle, KvCacheManager};
use fracture_core::{Backend, DeviceTensor, ForwardProfile, FractureError, Result};
use std::ops::Range;

use crate::Engine;

/// Describes what a node is responsible for in a pipeline.
///
/// `is_head` and `is_tail` are derived from the layer range and total layer count,
/// preventing invariant violations.
#[derive(Debug, Clone)]
pub struct NodeConfig {
    pub layer_range: Range<usize>,
    pub total_layers: usize,
}

impl NodeConfig {
    pub fn new(layer_range: Range<usize>, total_layers: usize) -> Result<Self> {
        if layer_range.is_empty() {
            return Err(FractureError::Pipeline("layer_range must not be empty".into()));
        }
        if layer_range.end > total_layers {
            return Err(FractureError::Pipeline(format!(
                "layer_range end {} exceeds total_layers {}",
                layer_range.end, total_layers
            )));
        }
        Ok(Self {
            layer_range,
            total_layers,
        })
    }

    /// True if this node owns the embedding lookup (first layers of the model).
    pub fn is_head(&self) -> bool {
        self.layer_range.start == 0
    }

    /// True if this node owns the final norm + LM head (last layers of the model).
    pub fn is_tail(&self) -> bool {
        self.layer_range.end == self.total_layers
    }

    /// True if this node covers the entire model.
    pub fn is_full(&self) -> bool {
        self.is_head() && self.is_tail()
    }

    /// Number of layers this node processes.
    pub fn num_layers(&self) -> usize {
        self.layer_range.len()
    }
}

/// Input to a compute node's forward pass.
pub enum NodeInput {
    /// Head node: receives token IDs from the generation loop.
    TokenIds {
        ids: Vec<u32>,
        positions: Vec<u32>,
    },
    /// Middle/tail node: receives activation tensor from previous node.
    Activations {
        hidden_states: DeviceTensor,
        positions: Vec<u32>,
    },
}

/// Output from a compute node's forward pass.
pub enum NodeOutput {
    /// Tail node: returns logits for sampling (host-side f32).
    Logits(Vec<f32>),
    /// Head/middle node: returns activation tensor for next node (stays on GPU).
    Activations(DeviceTensor),
}

/// Trait for a compute node in a pipeline. Object-safe so PipelineCoordinator
/// can hold `Box<dyn ComputeNode>`.
pub trait ComputeNode {
    fn forward(
        &self,
        input: NodeInput,
        cache: &mut KvCacheManager,
        cache_handle: CacheHandle,
        profile: Option<&mut ForwardProfile>,
    ) -> Result<NodeOutput>;

    fn config(&self) -> &NodeConfig;
}

/// Concrete implementation wrapping an `Engine<B>`. The `B: Backend` generic
/// is erased at this level, making the `ComputeNode` trait object-safe.
pub struct ComputeNodeImpl<B: Backend> {
    engine: Engine<B>,
    node_config: NodeConfig,
}

impl<B: Backend> ComputeNodeImpl<B> {
    pub fn new(engine: Engine<B>, node_config: NodeConfig) -> Self {
        Self {
            engine,
            node_config,
        }
    }

    pub fn engine(&self) -> &Engine<B> {
        &self.engine
    }
}

impl<B: Backend> ComputeNode for ComputeNodeImpl<B> {
    fn forward(
        &self,
        input: NodeInput,
        cache: &mut KvCacheManager,
        cache_handle: CacheHandle,
        profile: Option<&mut ForwardProfile>,
    ) -> Result<NodeOutput> {
        self.engine
            .forward_node(input, &self.node_config, cache, cache_handle, profile)
    }

    fn config(&self) -> &NodeConfig {
        &self.node_config
    }
}

// ── Node Service API (Phase 3 interface contract) ────────────────────────

/// Request to a node service for a forward pass.
pub struct ForwardRequest {
    pub seq_id: u64,
    pub input: NodeInput,
    pub is_prefill: bool,
}

/// Response from a node service after a forward pass.
pub struct ForwardResponse {
    pub seq_id: u64,
    pub output: NodeOutput,
}

/// Static information about a node's capabilities.
pub struct NodeInfo {
    pub node_id: String,
    pub layer_range: Range<usize>,
    pub is_head: bool,
    pub is_tail: bool,
    pub gpu_memory_total: usize,
    pub gpu_memory_used: usize,
}

/// The API surface that Phase 3's network layer will wrap.
///
/// In Phase 2, this is implemented locally (in-process or over Unix sockets).
/// In Phase 3, a network transport wraps this trait with TCP/QUIC.
pub trait NodeService {
    /// Process a forward pass and return the result.
    fn forward(&self, request: ForwardRequest) -> Result<ForwardResponse>;

    /// Report node capabilities.
    fn info(&self) -> NodeInfo;
}

/// In-process NodeService backed by a ComputeNode + KvCacheManager.
pub struct LocalNodeService<B: Backend> {
    node: ComputeNodeImpl<B>,
    cache: std::sync::Mutex<KvCacheManager>,
    node_id: String,
}

impl<B: Backend> LocalNodeService<B> {
    pub fn new(node: ComputeNodeImpl<B>, cache: KvCacheManager, node_id: String) -> Self {
        Self {
            node,
            cache: std::sync::Mutex::new(cache),
            node_id,
        }
    }
}

impl<B: Backend> NodeService for LocalNodeService<B> {
    fn forward(&self, request: ForwardRequest) -> Result<ForwardResponse> {
        let mut cache = self.cache.lock().map_err(|e| {
            FractureError::Pipeline(format!("cache lock poisoned: {e}"))
        })?;

        // Allocate cache on first use for this sequence, or reuse existing
        // For now, we require the caller to manage cache handles externally
        // via the seq_id. Simple approach: use seq_id as the cache handle.
        let handle = CacheHandle(request.seq_id);

        let output = self.node.forward(request.input, &mut cache, handle, None)?;

        Ok(ForwardResponse {
            seq_id: request.seq_id,
            output,
        })
    }

    fn info(&self) -> NodeInfo {
        let cfg = self.node.config();
        let engine = self.node.engine();
        NodeInfo {
            node_id: self.node_id.clone(),
            layer_range: cfg.layer_range.clone(),
            is_head: cfg.is_head(),
            is_tail: cfg.is_tail(),
            gpu_memory_total: engine.backend().total_memory(),
            gpu_memory_used: engine.backend().total_memory() - engine.backend().available_memory(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_node_config_head_tail() {
        let full = NodeConfig::new(0..32, 32).unwrap();
        assert!(full.is_head());
        assert!(full.is_tail());
        assert!(full.is_full());
        assert_eq!(full.num_layers(), 32);

        let head = NodeConfig::new(0..16, 32).unwrap();
        assert!(head.is_head());
        assert!(!head.is_tail());
        assert!(!head.is_full());

        let tail = NodeConfig::new(16..32, 32).unwrap();
        assert!(!tail.is_head());
        assert!(tail.is_tail());
        assert!(!tail.is_full());

        let middle = NodeConfig::new(8..24, 32).unwrap();
        assert!(!middle.is_head());
        assert!(!middle.is_tail());
        assert!(!middle.is_full());
        assert_eq!(middle.num_layers(), 16);
    }

    #[test]
    fn test_node_config_validation() {
        // Empty range
        let err = NodeConfig::new(5..5, 32).unwrap_err();
        assert!(matches!(err, FractureError::Pipeline(_)));
        assert!(err.to_string().contains("empty"));

        // Range exceeds total
        let err = NodeConfig::new(0..33, 32).unwrap_err();
        assert!(matches!(err, FractureError::Pipeline(_)));
        assert!(err.to_string().contains("exceeds"));
    }
}
