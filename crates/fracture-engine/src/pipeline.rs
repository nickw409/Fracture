use crate::kv_cache::{CacheHandle, KvCacheManager};
use crate::node::{ComputeNode, NodeInput, NodeOutput};
use fracture_core::{FractureError, Result};

/// Orchestrates a pipeline of compute nodes, chaining activations through
/// head -> middle(s) -> tail to produce logits from token IDs.
pub struct PipelineCoordinator {
    nodes: Vec<Box<dyn ComputeNode>>,
}

impl std::fmt::Debug for PipelineCoordinator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PipelineCoordinator")
            .field("num_nodes", &self.nodes.len())
            .finish()
    }
}

impl PipelineCoordinator {
    /// Create a new coordinator with validated node ordering.
    ///
    /// Validates that:
    /// - At least one node exists
    /// - First node is head, last node is tail
    /// - Layer ranges are contiguous with no gaps or overlaps
    pub fn new(nodes: Vec<Box<dyn ComputeNode>>) -> Result<Self> {
        if nodes.is_empty() {
            return Err(FractureError::Pipeline("pipeline must have at least one node".into()));
        }

        // Safety: nodes is guaranteed non-empty by the check above.
        let first = match nodes.first() {
            Some(n) => n,
            None => return Err(FractureError::Pipeline("no nodes".into())),
        };
        if !first.config().is_head() {
            return Err(FractureError::Pipeline(
                "first node must be head (layer_range must start at 0)".into(),
            ));
        }

        let last = match nodes.last() {
            Some(n) => n,
            None => return Err(FractureError::Pipeline("no nodes".into())),
        };
        if !last.config().is_tail() {
            return Err(FractureError::Pipeline(
                "last node must be tail (layer_range must end at total_layers)".into(),
            ));
        }

        // Validate contiguous ranges
        for i in 1..nodes.len() {
            let prev_end = nodes[i - 1].config().layer_range.end;
            let curr_start = nodes[i].config().layer_range.start;
            if prev_end != curr_start {
                return Err(FractureError::Pipeline(format!(
                    "gap or overlap between node {} (ends at {}) and node {} (starts at {})",
                    i - 1,
                    prev_end,
                    i,
                    curr_start
                )));
            }
        }

        Ok(Self { nodes })
    }

    /// Number of nodes in the pipeline.
    pub fn num_nodes(&self) -> usize {
        self.nodes.len()
    }

    /// Run a forward pass through the entire pipeline.
    ///
    /// Each node has its own `KvCacheManager` (sized for its layer range).
    /// `caches` and `cache_handles` must have the same length as the number of nodes.
    pub fn forward(
        &self,
        token_ids: &[u32],
        positions: &[u32],
        caches: &mut [&mut KvCacheManager],
        cache_handles: &[CacheHandle],
    ) -> Result<Vec<f32>> {
        if caches.len() != self.nodes.len() || cache_handles.len() != self.nodes.len() {
            return Err(FractureError::Pipeline(format!(
                "expected {} caches and handles, got {} and {}",
                self.nodes.len(),
                caches.len(),
                cache_handles.len()
            )));
        }

        let mut current_input = NodeInput::TokenIds {
            ids: token_ids.to_vec(),
            positions: positions.to_vec(),
        };

        for (i, node) in self.nodes.iter().enumerate() {
            let output = node.forward(
                current_input,
                caches[i],
                cache_handles[i],
                None,
            )?;

            match output {
                NodeOutput::Logits(logits) => {
                    if i == self.nodes.len() - 1 {
                        return Ok(logits);
                    } else {
                        return Err(FractureError::Pipeline(format!(
                            "node {} returned Logits but is not the last node",
                            i
                        )));
                    }
                }
                NodeOutput::Activations(tensor) => {
                    if i == self.nodes.len() - 1 {
                        return Err(FractureError::Pipeline(
                            "last node returned Activations instead of Logits".into(),
                        ));
                    }
                    current_input = NodeInput::Activations {
                        hidden_states: tensor,
                        positions: positions.to_vec(),
                    };
                }
            }
        }

        Err(FractureError::Pipeline("no nodes in pipeline".into()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_pipeline_rejected() {
        let result = PipelineCoordinator::new(vec![]);
        assert!(result.is_err());
        let err = match result {
            Err(e) => e,
            Ok(_) => panic!("expected error"),
        };
        assert!(err.to_string().contains("at least one node"));
    }

    // Mock node for error path testing
    use crate::node::NodeConfig;
    use fracture_core::{DeviceTensor, DType, TensorId};

    struct MockNode {
        config: NodeConfig,
        returns_logits: bool,
    }

    impl MockNode {
        fn new(start: usize, end: usize, total: usize, returns_logits: bool) -> Self {
            Self {
                config: NodeConfig::new(start..end, total).unwrap(),
                returns_logits,
            }
        }
    }

    impl ComputeNode for MockNode {
        fn forward(
            &self,
            _input: NodeInput,
            _cache: &mut KvCacheManager,
            _cache_handle: CacheHandle,
            _profile: Option<&mut fracture_core::ForwardProfile>,
        ) -> Result<NodeOutput> {
            if self.returns_logits {
                Ok(NodeOutput::Logits(vec![0.0; 10]))
            } else {
                Ok(NodeOutput::Activations(DeviceTensor::new(
                    TensorId(999),
                    vec![1, 64],
                    DType::FP16,
                )))
            }
        }

        fn config(&self) -> &NodeConfig {
            &self.config
        }
    }

    #[test]
    fn test_pipeline_cache_count_mismatch() {
        let nodes: Vec<Box<dyn ComputeNode>> = vec![
            Box::new(MockNode::new(0, 4, 4, true)),
        ];
        let pipeline = PipelineCoordinator::new(nodes).unwrap();

        // 0 caches for 1 node
        let result = pipeline.forward(&[1], &[0], &mut [], &[]);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("expected 1"));
    }

    #[test]
    fn test_pipeline_non_last_returns_logits() {
        let nodes: Vec<Box<dyn ComputeNode>> = vec![
            Box::new(MockNode::new(0, 2, 4, true)), // returns logits (wrong!)
            Box::new(MockNode::new(2, 4, 4, true)),
        ];
        let pipeline = PipelineCoordinator::new(nodes).unwrap();

        let mut cache1 = KvCacheManager::new(2, 8, 128, 128);
        let mut cache2 = KvCacheManager::new(2, 8, 128, 128);
        let h1 = CacheHandle(1);
        let h2 = CacheHandle(2);

        let result = pipeline.forward(
            &[1], &[0],
            &mut [&mut cache1, &mut cache2],
            &[h1, h2],
        );
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("not the last node"));
    }

    #[test]
    fn test_pipeline_last_returns_activations() {
        // Single node that returns activations instead of logits
        let nodes: Vec<Box<dyn ComputeNode>> = vec![
            Box::new(MockNode::new(0, 4, 4, false)), // returns activations (wrong!)
        ];
        let pipeline = PipelineCoordinator::new(nodes).unwrap();

        let mut cache = KvCacheManager::new(4, 8, 128, 128);
        let h = CacheHandle(1);

        let result = pipeline.forward(&[1], &[0], &mut [&mut cache], &[h]);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Activations instead of Logits"));
    }

    #[test]
    fn test_pipeline_overlap_rejected() {
        let nodes: Vec<Box<dyn ComputeNode>> = vec![
            Box::new(MockNode::new(0, 3, 4, false)),
            Box::new(MockNode::new(2, 4, 4, true)), // overlaps at layer 2
        ];
        let result = PipelineCoordinator::new(nodes);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("gap or overlap"));
    }

    #[test]
    fn test_pipeline_three_node_middle_passes_activations() {
        // 3-node pipeline: head(0..2) -> middle(2..6) -> tail(6..8)
        // Middle node accepts Activations and returns Activations.
        let nodes: Vec<Box<dyn ComputeNode>> = vec![
            Box::new(MockNode::new(0, 2, 8, false)),  // head: returns activations
            Box::new(MockNode::new(2, 6, 8, false)),  // middle: returns activations
            Box::new(MockNode::new(6, 8, 8, true)),   // tail: returns logits
        ];
        let pipeline = PipelineCoordinator::new(nodes).unwrap();
        assert_eq!(pipeline.num_nodes(), 3);

        let mut cache1 = KvCacheManager::new(2, 8, 128, 128);
        let mut cache2 = KvCacheManager::new(4, 8, 128, 128);
        let mut cache3 = KvCacheManager::new(2, 8, 128, 128);
        let h1 = CacheHandle(1);
        let h2 = CacheHandle(2);
        let h3 = CacheHandle(3);

        let result = pipeline.forward(
            &[1, 2, 3],
            &[0, 1, 2],
            &mut [&mut cache1, &mut cache2, &mut cache3],
            &[h1, h2, h3],
        );
        assert!(result.is_ok(), "3-node pipeline should succeed: {:?}", result.err());
        let logits = result.unwrap();
        assert_eq!(logits.len(), 10, "logits should come from tail node");
    }

    #[test]
    fn test_pipeline_non_head_first_rejected() {
        let nodes: Vec<Box<dyn ComputeNode>> = vec![
            Box::new(MockNode::new(1, 4, 4, true)), // doesn't start at 0
        ];
        let result = PipelineCoordinator::new(nodes);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("head"));
    }

    #[test]
    fn test_pipeline_non_tail_last_rejected() {
        let nodes: Vec<Box<dyn ComputeNode>> = vec![
            Box::new(MockNode::new(0, 2, 4, true)), // doesn't end at total_layers
        ];
        let result = PipelineCoordinator::new(nodes);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("tail"));
    }
}
