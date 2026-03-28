//! Heartbeat protocol for worker health monitoring.
//!
//! The coordinator sends Heartbeat messages to all workers every `interval`
//! and expects HeartbeatAck responses. Workers that miss `max_missed`
//! consecutive heartbeats are marked as dead.

use crate::registry::PeerRegistry;
use fracture_protocol::{frame::MessageType, messages::*};
use std::time::{Duration, Instant};

/// Default heartbeat interval.
pub const DEFAULT_INTERVAL: Duration = Duration::from_secs(5);

/// Default number of missed heartbeats before marking a worker dead.
pub const DEFAULT_MAX_MISSED: usize = 3;

/// Send a heartbeat to all ready workers. Returns the node IDs of workers
/// that have exceeded the heartbeat timeout.
///
/// This should be called periodically by the coordinator (e.g., in a
/// tokio::interval loop).
pub async fn send_heartbeats(
    registry: &mut PeerRegistry,
    timeout: Duration,
) -> Vec<String> {
    let now_ns = Instant::now().elapsed().as_nanos() as u64;
    let nonce = rand::random::<u64>();

    let payload = HeartbeatPayload {
        timestamp_ns: now_ns,
        nonce,
    };

    // Send to all workers (best-effort — connection errors are logged)
    let node_ids: Vec<String> = registry
        .pipeline_order()
        .into_iter()
        .collect();

    for node_id in &node_ids {
        if let Some(entry) = registry.get_mut(node_id)
            && let Err(e) = entry.connection.send(MessageType::Heartbeat, 0, &payload).await
        {
            tracing::warn!("failed to send heartbeat to '{}': {e}", node_id);
        }
    }

    // Check for timed-out workers
    registry.check_heartbeats(timeout)
}

/// Mark timed-out workers as dead in the registry and log the failures.
pub fn mark_dead_workers(registry: &mut PeerRegistry, timed_out: &[String]) {
    for node_id in timed_out {
        tracing::error!("worker '{}' missed heartbeat deadline — marking as dead", node_id);
        registry.mark_dead(node_id);
    }
}
