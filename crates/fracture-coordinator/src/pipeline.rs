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
use std::collections::HashSet;
use std::sync::Mutex;

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
    /// Total model layers the pipeline must cover.
    total_layers: usize,
    /// Sequence IDs that currently have allocated caches.
    allocated_seqs: Mutex<HashSet<u64>>,
}

impl std::fmt::Debug for DistributedPipeline {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DistributedPipeline")
            .field("pipeline_order", &self.pipeline_order)
            .field("hidden_size", &self.hidden_size)
            .field("total_layers", &self.total_layers)
            .finish()
    }
}

impl DistributedPipeline {
    /// Create a new distributed pipeline from scheduler assignments.
    ///
    /// Validates that layer ranges are contiguous starting from 0 and cover
    /// all layers without gaps or overlaps.
    pub fn new(assignments: &[LayerAssignment], hidden_size: usize) -> Result<Self> {
        if assignments.is_empty() {
            return Err(FractureError::Pipeline(
                "no assignments for distributed pipeline".into(),
            ));
        }

        // Validate contiguous ranges starting from layer 0
        let mut expected_start = 0;
        for a in assignments {
            if a.layer_range.start != expected_start {
                return Err(FractureError::Pipeline(format!(
                    "non-contiguous layer ranges: expected start {expected_start}, got {}",
                    a.layer_range.start
                )));
            }
            if a.layer_range.end <= a.layer_range.start {
                return Err(FractureError::Pipeline(format!(
                    "empty layer range for '{}': {:?}",
                    a.node_id, a.layer_range
                )));
            }
            expected_start = a.layer_range.end;
        }

        let total_layers = expected_start;
        let pipeline_order = assignments.iter().map(|a| a.node_id.clone()).collect();

        Ok(Self {
            pipeline_order,
            hidden_size,
            total_layers,
            allocated_seqs: Mutex::new(HashSet::new()),
        })
    }

    /// Send CacheAlloc to all workers for a new sequence.
    ///
    /// Returns an error if the sequence already has an active cache allocation.
    /// If allocation fails on any worker, successfully-allocated caches on other
    /// workers are freed (rollback) to prevent memory leaks.
    pub async fn alloc_cache(
        &self,
        registry: &mut PeerRegistry,
        seq_id: u64,
        max_seq_len: u32,
    ) -> Result<()> {
        {
            let seqs = self.allocated_seqs.lock().unwrap();
            if seqs.contains(&seq_id) {
                return Err(FractureError::Pipeline(format!(
                    "duplicate alloc_cache for seq {seq_id}: cache already allocated"
                )));
            }
        }

        let payload = CacheAllocPayload { max_seq_len };
        let mut succeeded: Vec<String> = Vec::new();

        let result = self
            .try_alloc_all(registry, seq_id, &payload, &mut succeeded)
            .await;

        if result.is_err() {
            // Rollback: free caches on workers that already succeeded
            for succeeded_id in &succeeded {
                if let Some(ok_entry) = registry.get_mut(succeeded_id) {
                    let _ = ok_entry
                        .connection
                        .send_empty(MessageType::CacheFree, seq_id)
                        .await;
                }
            }
            return result;
        }

        self.allocated_seqs.lock().unwrap().insert(seq_id);
        Ok(())
    }

    /// Attempt to send CacheAlloc to all workers and collect acks.
    /// On any failure, returns the error (caller handles rollback).
    async fn try_alloc_all(
        &self,
        registry: &mut PeerRegistry,
        seq_id: u64,
        payload: &CacheAllocPayload,
        succeeded: &mut Vec<String>,
    ) -> Result<()> {
        for node_id in &self.pipeline_order {
            let entry = registry.get_mut(node_id).ok_or_else(|| {
                FractureError::Pipeline(format!("worker '{node_id}' not found in registry"))
            })?;
            entry
                .connection
                .send(MessageType::CacheAlloc, seq_id, payload)
                .await?;

            // Wait for CacheAllocAck or Error from the worker
            let (header, resp_payload) = entry.connection.recv().await?;

            if header.msg_type == MessageType::Error {
                let err: ErrorPayload = FramedConnection::deserialize_payload(&resp_payload)?;
                return Err(FractureError::Pipeline(format!(
                    "worker '{}' failed CacheAlloc: {} (code {:?})",
                    node_id, err.message, err.error_code
                )));
            }

            if header.msg_type != MessageType::CacheAllocAck {
                return Err(FractureError::Protocol(format!(
                    "expected CacheAllocAck from '{}', got {:?}",
                    node_id, header.msg_type
                )));
            }

            succeeded.push(node_id.clone());
        }
        Ok(())
    }

    /// Send CacheFree to all workers for a completed sequence.
    ///
    /// Returns an error if the sequence was never allocated or was already freed.
    pub async fn free_cache(
        &self,
        registry: &mut PeerRegistry,
        seq_id: u64,
    ) -> Result<()> {
        {
            let mut seqs = self.allocated_seqs.lock().unwrap();
            if !seqs.remove(&seq_id) {
                return Err(FractureError::Pipeline(format!(
                    "free_cache for seq {seq_id}: not allocated or already freed"
                )));
            }
        }

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
        if !self.allocated_seqs.lock().unwrap().contains(&seq_id) {
            return Err(FractureError::Pipeline(format!(
                "forward for seq {seq_id}: cache not allocated (call alloc_cache first)"
            )));
        }

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

    /// Get the total number of model layers this pipeline covers.
    pub fn total_layers(&self) -> usize {
        self.total_layers
    }

    /// Check whether a sequence ID has an active cache allocation.
    pub fn is_allocated(&self, seq_id: u64) -> bool {
        self.allocated_seqs.lock().unwrap().contains(&seq_id)
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

    #[test]
    fn test_pipeline_single_worker() {
        let assignments = vec![assignment("solo", 0, 32, NodeRole::Head)];
        let pipeline = DistributedPipeline::new(&assignments, 4096).unwrap();
        assert_eq!(pipeline.pipeline_order(), &["solo"]);
        assert_eq!(pipeline.total_layers(), 32);
    }

    #[test]
    fn test_pipeline_layer_coverage_validation() {
        // Valid: full contiguous coverage
        let assignments = vec![
            assignment("a", 0, 16, NodeRole::Head),
            assignment("b", 16, 32, NodeRole::Tail),
        ];
        let pipeline = DistributedPipeline::new(&assignments, 4096).unwrap();
        assert_eq!(pipeline.total_layers(), 32);

        // Invalid: gap between ranges
        let gap = vec![
            assignment("a", 0, 10, NodeRole::Head),
            assignment("b", 12, 32, NodeRole::Tail),
        ];
        let err = DistributedPipeline::new(&gap, 4096).unwrap_err().to_string();
        assert!(err.contains("non-contiguous"), "expected contiguity error: {err}");

        // Invalid: doesn't start at 0
        let no_zero = vec![assignment("a", 5, 32, NodeRole::Head)];
        let err = DistributedPipeline::new(&no_zero, 4096).unwrap_err().to_string();
        assert!(err.contains("non-contiguous") || err.contains("expected start 0"), "expected start-at-0 error: {err}");

        // Invalid: empty range
        let empty = vec![assignment("a", 0, 0, NodeRole::Head)];
        let err = DistributedPipeline::new(&empty, 4096).unwrap_err().to_string();
        assert!(err.contains("empty layer range"), "expected empty range error: {err}");
    }
}
