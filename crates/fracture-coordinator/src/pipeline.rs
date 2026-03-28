//! Distributed pipeline orchestrator.
//!
//! Sends Forward messages to workers in pipeline order, passing activation
//! tensors from one stage to the next. The coordinator acts as the central
//! hub: activations flow through it (worker A → coordinator → worker B).
//!
//! This replaces Phase 2's `PipelineCoordinator` for network calls while
//! preserving the same semantics: token_ids + positions in, logits out.

use crate::registry::PeerRegistry;
use crate::scheduler::LayerAssignment;
use fracture_core::{FractureError, Result};
use fracture_protocol::{frame::MessageType, messages::*, FramedConnection};

/// Orchestrates forward passes across distributed workers.
///
/// The pipeline sends Forward messages sequentially through workers in
/// layer-range order. Activation tensors are serialized and transferred
/// through the coordinator between stages.
pub struct DistributedPipeline {
    /// Node IDs in pipeline order (head first, tail last).
    pipeline_order: Vec<String>,
    /// Model hidden size for activation shape validation.
    hidden_size: usize,
}

impl DistributedPipeline {
    /// Create a new distributed pipeline from scheduler assignments.
    pub fn new(assignments: &[LayerAssignment], hidden_size: usize) -> Result<Self> {
        if assignments.is_empty() {
            return Err(FractureError::Pipeline(
                "no assignments for distributed pipeline".into(),
            ));
        }

        // Validate contiguous ranges
        let mut expected_start = 0;
        for a in assignments {
            if a.layer_range.start != expected_start {
                return Err(FractureError::Pipeline(format!(
                    "non-contiguous layer ranges: expected start {expected_start}, got {}",
                    a.layer_range.start
                )));
            }
            expected_start = a.layer_range.end;
        }

        let pipeline_order = assignments.iter().map(|a| a.node_id.clone()).collect();

        Ok(Self {
            pipeline_order,
            hidden_size,
        })
    }

    /// Send CacheAlloc to all workers for a new sequence.
    pub async fn alloc_cache(
        &self,
        registry: &mut PeerRegistry,
        seq_id: u64,
        max_seq_len: u32,
    ) -> Result<()> {
        let payload = CacheAllocPayload { max_seq_len };
        for node_id in &self.pipeline_order {
            let entry = registry.get_mut(node_id).ok_or_else(|| {
                FractureError::Pipeline(format!("worker '{node_id}' not found in registry"))
            })?;
            entry.connection.send(MessageType::CacheAlloc, seq_id, &payload).await?;
        }
        Ok(())
    }

    /// Send CacheFree to all workers for a completed sequence.
    pub async fn free_cache(
        &self,
        registry: &mut PeerRegistry,
        seq_id: u64,
    ) -> Result<()> {
        for node_id in &self.pipeline_order {
            let entry = registry.get_mut(node_id).ok_or_else(|| {
                FractureError::Pipeline(format!("worker '{node_id}' not found in registry"))
            })?;
            entry.connection.send_empty(MessageType::CacheFree, seq_id).await?;
        }
        Ok(())
    }

    /// Run a forward pass through the entire pipeline.
    ///
    /// Sends token IDs to the head worker, chains activations through
    /// intermediate workers, and returns logits from the tail worker.
    pub async fn forward(
        &self,
        registry: &mut PeerRegistry,
        seq_id: u64,
        token_ids: &[u32],
        positions: &[u32],
        is_prefill: bool,
    ) -> Result<Vec<f32>> {
        let n = self.pipeline_order.len();

        // First worker gets token IDs
        let mut current_input = ForwardInputWire::TokenIds {
            ids: token_ids.to_vec(),
        };

        for (i, node_id) in self.pipeline_order.iter().enumerate() {
            let is_last = i == n - 1;

            // Build and send Forward request
            let forward_payload = ForwardPayload {
                is_prefill,
                positions: positions.to_vec(),
                input: current_input,
            };

            let entry = registry.get_mut(node_id).ok_or_else(|| {
                FractureError::Pipeline(format!("worker '{node_id}' not found in registry"))
            })?;
            entry
                .connection
                .send(MessageType::Forward, seq_id, &forward_payload)
                .await?;

            // Receive ForwardResult
            let (header, payload) = entry.connection.recv().await?;

            if header.msg_type == MessageType::Error {
                let err: ErrorPayload = FramedConnection::deserialize_payload(&payload)?;
                return Err(FractureError::Pipeline(format!(
                    "worker '{}' returned error: {} (code {:?})",
                    node_id, err.message, err.error_code
                )));
            }

            if header.msg_type != MessageType::ForwardResult {
                return Err(FractureError::Protocol(format!(
                    "expected ForwardResult from '{}', got {:?}",
                    node_id, header.msg_type
                )));
            }

            let result: ForwardResultPayload =
                FramedConnection::deserialize_payload(&payload)?;

            match result.output {
                ForwardOutputWire::Logits { data } => {
                    if !is_last {
                        return Err(FractureError::Pipeline(format!(
                            "non-tail worker '{}' returned logits (expected activations)",
                            node_id
                        )));
                    }
                    // Convert raw f32 LE bytes to Vec<f32>
                    let logits: Vec<f32> = data
                        .chunks_exact(4)
                        .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
                        .collect();
                    return Ok(logits);
                }
                ForwardOutputWire::Activations {
                    tensor_header,
                    tensor_data,
                } => {
                    if is_last {
                        return Err(FractureError::Pipeline(format!(
                            "tail worker '{}' returned activations (expected logits)",
                            node_id
                        )));
                    }
                    // Validate activation shape: last dimension must be hidden_size
                    if let Some(&last_dim) = tensor_header.shape.last() {
                        if last_dim as usize != self.hidden_size {
                            return Err(FractureError::Pipeline(format!(
                                "worker '{}' returned activation with last dim {}, expected hidden_size {}",
                                node_id, last_dim, self.hidden_size
                            )));
                        }
                    }
                    // Pass activations as input to the next worker
                    current_input = ForwardInputWire::Activations {
                        tensor_header,
                        tensor_data,
                    };
                }
            }
        }

        Err(FractureError::Pipeline(
            "pipeline completed without producing logits".into(),
        ))
    }

    /// Get the pipeline order (node IDs).
    pub fn pipeline_order(&self) -> &[String] {
        &self.pipeline_order
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scheduler::NodeRole;

    fn assignment(id: &str, start: usize, end: usize, role: NodeRole) -> LayerAssignment {
        LayerAssignment {
            node_id: id.into(),
            layer_range: start..end,
            role,
            expected_decode_ms: 10.0,
            weight_memory_gb: 5.0,
            cache_memory_gb: 1.0,
        }
    }

    #[test]
    fn test_pipeline_creation() {
        let assignments = vec![
            assignment("head", 0, 16, NodeRole::Head),
            assignment("tail", 16, 32, NodeRole::Tail),
        ];
        let pipeline = DistributedPipeline::new(&assignments, 4096).unwrap();
        assert_eq!(pipeline.pipeline_order(), &["head", "tail"]);
    }

    #[test]
    fn test_pipeline_three_nodes() {
        let assignments = vec![
            assignment("a", 0, 10, NodeRole::Head),
            assignment("b", 10, 20, NodeRole::Middle),
            assignment("c", 20, 32, NodeRole::Tail),
        ];
        let pipeline = DistributedPipeline::new(&assignments, 4096).unwrap();
        assert_eq!(pipeline.pipeline_order(), &["a", "b", "c"]);
    }

    #[test]
    fn test_pipeline_empty() {
        assert!(DistributedPipeline::new(&[], 4096).is_err());
    }

    #[test]
    fn test_pipeline_non_contiguous() {
        let assignments = vec![
            assignment("a", 0, 10, NodeRole::Head),
            assignment("b", 15, 32, NodeRole::Tail), // gap at 10..15
        ];
        assert!(DistributedPipeline::new(&assignments, 4096).is_err());
    }
}
