use crate::kv_cache::{CacheHandle, KvCacheManager};
use crate::node::{ComputeNode, NodeInput, NodeOutput};
use fracture_core::{FractureError, Result};

/// Orchestrates a pipeline of compute nodes, chaining activations through
/// head -> middle(s) -> tail to produce logits from token IDs.
pub struct PipelineCoordinator {
    nodes: Vec<Box<dyn ComputeNode>>,
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
}
