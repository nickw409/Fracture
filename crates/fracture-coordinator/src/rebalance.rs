//! Rebalance orchestrator: shared infrastructure for redistributing layers
//! across workers. Used by crash recovery, graceful leave, and dynamic join.

use crate::{
    pipeline::DistributedPipeline,
    registry::PeerRegistry,
    scheduler::{self, LayerAssignment, SchedulerInput, SchedulingMode},
};
use fracture_core::{FractureError, ModelConfig, Result};
use fracture_protocol::{frame::MessageType, messages::RegisterAckPayload};
use std::sync::Arc;
use tokio::sync::Mutex;

/// Result of a forced rebalance operation.
pub struct RebalanceResult {
    /// The new pipeline after rebalance.
    pub pipeline: Arc<DistributedPipeline>,
    /// New assignments applied to workers.
    pub assignments: Vec<LayerAssignment>,
}

/// Execute a forced rebalance: abort active sequences, reconfigure all workers,
/// and rebuild the pipeline.
///
/// This is the "nuclear option" — all active sequences are aborted. Use graceful
/// rebalance (FT-4b) when you can afford to wait for sequences to drain.
///
/// # Arguments
/// * `registry` — Peer registry with current worker connections
/// * `pipeline` — Current pipeline (used to abort sequences)
/// * `model_config` — Model configuration for scheduler and RegisterAck
/// * `scheduling_mode` — How to assign layers
/// * `max_seq_len` — Maximum sequence length for cache sizing
/// * `skip_node_ids` — Node IDs to exclude from the new schedule (e.g., dead workers)
pub async fn forced_rebalance(
    registry: &Mutex<PeerRegistry>,
    pipeline: &DistributedPipeline,
    model_config: &ModelConfig,
    scheduling_mode: &SchedulingMode,
    max_seq_len: usize,
    skip_node_ids: &[String],
) -> Result<RebalanceResult> {
    // Step 1: Abort all active sequences and free caches.
    tracing::info!("forced rebalance: aborting active sequences");
    {
        let mut reg = registry.lock().await;
        pipeline.abort_all_sequences(&mut reg).await;
    }

    // Step 2: Run scheduler with available workers (excluding skipped nodes).
    let result = {
        let reg = registry.lock().await;
        let mut caps = reg.all_capabilities();
        caps.retain(|c| !skip_node_ids.contains(&c.node_id));
        if caps.is_empty() {
            return Err(FractureError::Pipeline(
                "forced rebalance: no workers available after exclusions".into(),
            ));
        }
        let input = SchedulerInput {
            model_config: model_config.clone(),
            workers: caps,
            coordinator_compute: None,
            mode: scheduling_mode.clone(),
            max_seq_len,
            hop_latency_ms: 2.0,
        };
        scheduler::schedule(&input)?
    };

    tracing::info!("forced rebalance: new schedule computed");
    for a in &result.assignments {
        tracing::info!(
            "  {} → layers {:?} ({:?})",
            a.node_id, a.layer_range, a.role
        );
    }

    // Step 3: Send Reconfigure to all workers with new assignments.
    {
        let mut reg = registry.lock().await;
        for assignment in &result.assignments {
            let ack = RegisterAckPayload {
                layer_start: assignment.layer_range.start as u32,
                layer_end: assignment.layer_range.end as u32,
                total_layers: model_config.num_layers as u32,
                max_seq_len: max_seq_len as u32,
                model_config: model_config.clone(),
            };
            reg.assign(&assignment.node_id, assignment.clone())
                .map_err(|e| {
                    FractureError::Pipeline(format!(
                        "forced rebalance: assign '{}' failed: {e}",
                        assignment.node_id
                    ))
                })?;
            if let Some(entry) = reg.get_mut(&assignment.node_id)
                && let Err(e) = entry
                    .writer
                    .send(MessageType::Reconfigure, 0, &ack)
                    .await
                {
                    tracing::error!(
                        "forced rebalance: failed to send Reconfigure to '{}': {e}",
                        assignment.node_id
                    );
                    // Mark worker as dead — it can't participate.
                    reg.mark_dead(&assignment.node_id);
                }
        }
    }

    // Step 4: Wait for WorkerReady from all reconfigured workers.
    tracing::info!("forced rebalance: waiting for workers to reload weights...");
    {
        let reg = registry.lock().await;
        for assignment in &result.assignments {
            if let Some(entry) = reg.get(&assignment.node_id) {
                match entry.reader.lock().await.recv().await {
                    Ok((hdr, _)) if hdr.msg_type == MessageType::WorkerReady => {
                        tracing::info!(
                            "worker '{}' ready after rebalance",
                            assignment.node_id
                        );
                    }
                    Ok((hdr, _)) => {
                        tracing::error!(
                            "expected WorkerReady from '{}', got {:?}",
                            assignment.node_id, hdr.msg_type
                        );
                    }
                    Err(e) => {
                        tracing::error!(
                            "recv from '{}' failed during rebalance: {e}",
                            assignment.node_id
                        );
                    }
                }
            }
        }
    }

    // Step 5: Build new pipeline.
    let new_pipeline = Arc::new(DistributedPipeline::new(
        &result.assignments,
        model_config.hidden_size,
    )?);
    tracing::info!(
        "forced rebalance complete: {} stages",
        new_pipeline.pipeline_order().len()
    );

    Ok(RebalanceResult {
        pipeline: new_pipeline,
        assignments: result.assignments,
    })
}

// ---------------------------------------------------------------------------
// Graceful rebalance (FT-4b)
// ---------------------------------------------------------------------------

/// Handle to a pending graceful rebalance. The caller polls `is_ready()` each
/// iteration of their event loop; when active sequences drain to zero the
/// rebalance triggers automatically.
pub struct PendingGracefulRebalance {
    pub model_config: ModelConfig,
    pub scheduling_mode: SchedulingMode,
    pub max_seq_len: usize,
    /// Node IDs to exclude from the new schedule.
    pub skip_node_ids: Vec<String>,
    /// Set to true if the pending rebalance should be cancelled.
    cancelled: bool,
}

impl PendingGracefulRebalance {
    pub fn new(
        model_config: ModelConfig,
        scheduling_mode: SchedulingMode,
        max_seq_len: usize,
        skip_node_ids: Vec<String>,
    ) -> Self {
        Self {
            model_config,
            scheduling_mode,
            max_seq_len,
            skip_node_ids,
            cancelled: false,
        }
    }

    /// Check if the graceful rebalance should trigger now.
    /// Returns true when there are no active sequences.
    pub fn is_ready(&self, active_sequence_count: usize) -> bool {
        !self.cancelled && active_sequence_count == 0
    }

    /// Cancel this pending rebalance. Used when a forced rebalance supersedes it
    /// (e.g., a worker crashes while we're waiting for sequences to drain).
    pub fn cancel(&mut self) {
        self.cancelled = true;
    }

    pub fn is_cancelled(&self) -> bool {
        self.cancelled
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rebalance_result_has_pipeline_and_assignments() {
        let pipeline = Arc::new(DistributedPipeline::empty(4096));
        let result = RebalanceResult {
            pipeline,
            assignments: vec![],
        };
        assert!(result.assignments.is_empty());
        assert!(result.pipeline.pipeline_order().is_empty());
    }

    fn test_config() -> ModelConfig {
        ModelConfig {
            hidden_size: 4096,
            num_layers: 32,
            num_q_heads: 32,
            num_kv_heads: 8,
            head_dim: 128,
            intermediate_size: 14336,
            vocab_size: 128256,
            rope_theta: 500000.0,
            rms_norm_eps: 1e-5,
            max_seq_len: 8192,
        }
    }

    #[test]
    fn test_graceful_rebalance_ready_when_no_active_sequences() {
        let pending = PendingGracefulRebalance::new(
            test_config(),
            SchedulingMode::EqualSplit,
            4096,
            vec![],
        );
        assert!(!pending.is_ready(5));
        assert!(!pending.is_ready(1));
        assert!(pending.is_ready(0));
    }

    #[test]
    fn test_graceful_rebalance_cancel() {
        let mut pending = PendingGracefulRebalance::new(
            test_config(),
            SchedulingMode::EqualSplit,
            4096,
            vec![],
        );
        assert!(pending.is_ready(0));
        pending.cancel();
        assert!(pending.is_cancelled());
        assert!(!pending.is_ready(0));
    }
}
