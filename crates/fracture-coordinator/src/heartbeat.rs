//! Heartbeat protocol for worker health monitoring.
//!
//! The coordinator sends Heartbeat messages to all workers every `interval`
//! and expects HeartbeatAck responses. Workers that miss `max_missed`
//! consecutive heartbeats are marked as dead.

use crate::registry::{PeerRegistry, WorkerStatus};
use fracture_protocol::messages::*;
use std::collections::HashMap;
use std::time::Duration;

/// Default heartbeat interval.
pub const DEFAULT_INTERVAL: Duration = Duration::from_secs(5);

/// Default number of missed heartbeats before marking a worker dead.
pub const DEFAULT_MAX_MISSED: usize = 3;

/// Tracks pending nonces for heartbeat ack validation.
///
/// Each heartbeat round sends a unique nonce to all workers. When
/// processing acks, only a matching nonce resets the heartbeat timer.
pub struct HeartbeatTracker {
    /// The nonce sent in the most recent heartbeat round.
    pending_nonce: Option<u64>,
    /// Per-worker missed-heartbeat counters.
    missed_counts: HashMap<String, usize>,
}

impl Default for HeartbeatTracker {
    fn default() -> Self {
        Self::new()
    }
}

impl HeartbeatTracker {
    pub fn new() -> Self {
        Self {
            pending_nonce: None,
            missed_counts: HashMap::new(),
        }
    }

    /// Get the current pending nonce (if any).
    pub fn pending_nonce(&self) -> Option<u64> {
        self.pending_nonce
    }

    /// Get the missed-heartbeat count for a worker.
    pub fn missed_count(&self, node_id: &str) -> usize {
        self.missed_counts.get(node_id).copied().unwrap_or(0)
    }

    /// Record the nonce sent in a heartbeat round.
    pub fn set_pending_nonce(&mut self, nonce: u64) {
        self.pending_nonce = Some(nonce);
    }

    /// Process a HeartbeatAck from a worker.
    ///
    /// Returns `true` if the ack was valid (matching nonce and worker is
    /// not Dead), `false` otherwise.
    ///
    /// A valid ack resets the worker's missed-heartbeat counter to zero
    /// and updates the heartbeat timestamp in the registry.
    ///
    /// Invalid cases (returns false):
    /// - Worker is Dead
    /// - Nonce does not match the pending nonce
    /// - No pending nonce (no heartbeat was sent)
    /// - Worker not found in registry
    pub fn process_ack(
        &mut self,
        registry: &mut PeerRegistry,
        node_id: &str,
        ack: &HeartbeatAckPayload,
    ) -> bool {
        // Check worker exists
        let entry = match registry.get(node_id) {
            Some(e) => e,
            None => return false,
        };

        // Dead workers are ignored
        if entry.status == WorkerStatus::Dead {
            return false;
        }

        // Validate nonce
        let expected = match self.pending_nonce {
            Some(n) => n,
            None => return false,
        };

        if ack.nonce_echo != expected {
            return false;
        }

        // Valid ack: reset missed counter and update heartbeat timestamp + stats
        self.missed_counts.insert(node_id.to_string(), 0);
        registry.record_heartbeat(node_id, ack.free_blocks, ack.gpu_memory_used);
        true
    }

    /// Increment missed counters for workers that did not ack.
    /// Returns node IDs of workers that have exceeded `max_missed`.
    pub fn increment_missed(&mut self, ready_node_ids: &[String], max_missed: usize) -> Vec<String> {
        let mut timed_out = Vec::new();
        for node_id in ready_node_ids {
            let count = self.missed_counts.entry(node_id.clone()).or_insert(0);
            *count += 1;
            if *count >= max_missed {
                timed_out.push(node_id.clone());
            }
        }
        timed_out
    }
}

/// Mark timed-out workers as dead in the registry and log the failures.
pub fn mark_dead_workers(registry: &mut PeerRegistry, timed_out: &[String]) {
    for node_id in timed_out {
        tracing::error!("worker '{}' missed heartbeat deadline — marking as dead", node_id);
        registry.mark_dead(node_id);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::registry::PeerRegistry;
    use crate::scheduler::{LayerAssignment, NodeRole, WorkerCapabilities};
    use tokio::net::TcpListener;
    use fracture_protocol::FramedConnection;

    fn test_caps(id: &str) -> WorkerCapabilities {
        WorkerCapabilities {
            node_id: id.into(),
            gpu_model: "Test".into(),
            gpu_memory_available: 24_000_000_000,
            compute_capability: (8, 0),
            decode_ms_per_layer: 1.0,
            prefill_ms_per_layer_128: 3.0,
        }
    }

    fn test_assignment(id: &str) -> LayerAssignment {
        LayerAssignment {
            node_id: id.into(),
            layer_range: 0..32,
            role: NodeRole::Head,
            expected_decode_ms: 32.0,
            weight_memory_gb: 12.0,
            cache_memory_gb: 2.0,
        }
    }

    async fn dummy_connection() -> FramedConnection {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let (client, _server) =
            tokio::join!(tokio::net::TcpStream::connect(addr), listener.accept());
        FramedConnection::new(client.unwrap())
    }

    fn make_ack(nonce: u64) -> HeartbeatAckPayload {
        HeartbeatAckPayload {
            timestamp_echo: 0,
            nonce_echo: nonce,
            gpu_memory_used: 0,
            active_sequences: 0,
            free_blocks: 0,
        }
    }

    async fn setup_registry_with_worker(id: &str) -> PeerRegistry {
        let mut reg = PeerRegistry::new();
        reg.register(test_caps(id), dummy_connection().await).unwrap();
        reg.assign(id, test_assignment(id)).unwrap();
        reg
    }

    #[tokio::test]
    async fn test_valid_ack_resets_counter() {
        let mut reg = setup_registry_with_worker("w1").await;
        let mut tracker = HeartbeatTracker::new();

        // Simulate a heartbeat round
        tracker.set_pending_nonce(42);

        // Increment missed (simulating one missed round)
        tracker.increment_missed(&["w1".into()], 3);
        assert_eq!(tracker.missed_count("w1"), 1);

        // Valid ack resets counter
        let ack = make_ack(42);
        assert!(tracker.process_ack(&mut reg, "w1", &ack));
        assert_eq!(tracker.missed_count("w1"), 0);
    }

    #[tokio::test]
    async fn test_mismatched_nonce_rejected() {
        let mut reg = setup_registry_with_worker("w1").await;
        let mut tracker = HeartbeatTracker::new();

        tracker.set_pending_nonce(42);
        tracker.increment_missed(&["w1".into()], 3);
        assert_eq!(tracker.missed_count("w1"), 1);

        // Wrong nonce — should NOT reset counter
        let ack = make_ack(999);
        assert!(!tracker.process_ack(&mut reg, "w1", &ack));
        // Counter should still be 1 (not reset)
        assert_eq!(tracker.missed_count("w1"), 1);
    }

    #[tokio::test]
    async fn test_no_pending_nonce_rejected() {
        let mut reg = setup_registry_with_worker("w1").await;
        let mut tracker = HeartbeatTracker::new();

        // No heartbeat was sent (no pending nonce)
        let ack = make_ack(42);
        assert!(!tracker.process_ack(&mut reg, "w1", &ack));
    }

    #[tokio::test]
    async fn test_dead_worker_ack_ignored() {
        let mut reg = setup_registry_with_worker("w1").await;
        let mut tracker = HeartbeatTracker::new();

        tracker.set_pending_nonce(42);

        // Mark worker as dead
        reg.mark_dead("w1");

        // Even with matching nonce, dead worker is ignored
        let ack = make_ack(42);
        assert!(!tracker.process_ack(&mut reg, "w1", &ack));
    }

    #[tokio::test]
    async fn test_dead_worker_does_not_revive() {
        let mut reg = setup_registry_with_worker("w1").await;
        let mut tracker = HeartbeatTracker::new();

        tracker.set_pending_nonce(42);
        reg.mark_dead("w1");

        // Attempt ack
        let ack = make_ack(42);
        tracker.process_ack(&mut reg, "w1", &ack);

        // Worker should still be Dead
        assert_eq!(reg.get("w1").unwrap().status, WorkerStatus::Dead);
    }

    #[tokio::test]
    async fn test_unknown_worker_ack_rejected() {
        let mut reg = PeerRegistry::new();
        let mut tracker = HeartbeatTracker::new();

        tracker.set_pending_nonce(42);
        let ack = make_ack(42);
        assert!(!tracker.process_ack(&mut reg, "nonexistent", &ack));
    }

    #[tokio::test]
    async fn test_increment_missed_tracks_correctly() {
        let mut tracker = HeartbeatTracker::new();

        // Two rounds of missed heartbeats
        let timed_out = tracker.increment_missed(&["w1".into()], 3);
        assert!(timed_out.is_empty());
        assert_eq!(tracker.missed_count("w1"), 1);

        let timed_out = tracker.increment_missed(&["w1".into()], 3);
        assert!(timed_out.is_empty());
        assert_eq!(tracker.missed_count("w1"), 2);

        // Third miss exceeds threshold
        let timed_out = tracker.increment_missed(&["w1".into()], 3);
        assert_eq!(timed_out, vec!["w1"]);
        assert_eq!(tracker.missed_count("w1"), 3);
    }

    #[tokio::test]
    async fn test_valid_ack_then_miss_then_ack() {
        let mut reg = setup_registry_with_worker("w1").await;
        let mut tracker = HeartbeatTracker::new();

        // Round 1: heartbeat sent, valid ack
        tracker.set_pending_nonce(100);
        assert!(tracker.process_ack(&mut reg, "w1", &make_ack(100)));
        assert_eq!(tracker.missed_count("w1"), 0);

        // Round 2: heartbeat sent, no ack (miss)
        tracker.set_pending_nonce(200);
        tracker.increment_missed(&["w1".into()], 3);
        assert_eq!(tracker.missed_count("w1"), 1);

        // Round 3: heartbeat sent, valid ack — counter resets
        tracker.set_pending_nonce(300);
        assert!(tracker.process_ack(&mut reg, "w1", &make_ack(300)));
        assert_eq!(tracker.missed_count("w1"), 0);
    }

    /// Worker misses 2 rounds (one short of death), then acks — counter
    /// resets and the worker survives. Verifies near-death recovery.
    #[tokio::test]
    async fn test_recovery_at_threshold_boundary() {
        let mut reg = setup_registry_with_worker("w1").await;
        let mut tracker = HeartbeatTracker::new();

        // Round 1: miss
        tracker.set_pending_nonce(10);
        assert!(tracker.increment_missed(&["w1".into()], 3).is_empty());
        assert_eq!(tracker.missed_count("w1"), 1);

        // Round 2: miss
        tracker.set_pending_nonce(20);
        assert!(tracker.increment_missed(&["w1".into()], 3).is_empty());
        assert_eq!(tracker.missed_count("w1"), 2);

        // Round 3: ack arrives just in time — counter resets before increment
        tracker.set_pending_nonce(30);
        assert!(tracker.process_ack(&mut reg, "w1", &make_ack(30)));
        assert_eq!(tracker.missed_count("w1"), 0);

        // Subsequent increment starts from 0 again
        let dead = tracker.increment_missed(&["w1".into()], 3);
        assert!(dead.is_empty());
        assert_eq!(tracker.missed_count("w1"), 1);
    }

    /// Two workers in pipeline — one acks, one doesn't. Only the
    /// non-responding worker accumulates missed counts.
    #[tokio::test]
    async fn test_multi_worker_selective_ack() {
        let mut reg = PeerRegistry::new();
        reg.register(test_caps("healthy"), dummy_connection().await).unwrap();
        reg.register(test_caps("flaky"), dummy_connection().await).unwrap();
        reg.assign("healthy", test_assignment("healthy")).unwrap();
        reg.assign("flaky", test_assignment("flaky")).unwrap();

        let mut tracker = HeartbeatTracker::new();
        let workers: Vec<String> = vec!["healthy".into(), "flaky".into()];

        // Round 1: both get heartbeat, only healthy acks
        tracker.set_pending_nonce(100);
        assert!(tracker.process_ack(&mut reg, "healthy", &make_ack(100)));
        // Don't ack for flaky

        // Increment: healthy was reset to 0, goes to 1; flaky starts at 0, goes to 1
        tracker.increment_missed(&workers, 3);
        assert_eq!(tracker.missed_count("healthy"), 1);
        assert_eq!(tracker.missed_count("flaky"), 1);

        // Round 2: healthy acks again, flaky still silent
        tracker.set_pending_nonce(200);
        assert!(tracker.process_ack(&mut reg, "healthy", &make_ack(200)));

        tracker.increment_missed(&workers, 3);
        assert_eq!(tracker.missed_count("healthy"), 1); // reset then +1
        assert_eq!(tracker.missed_count("flaky"), 2);   // no reset, +1

        // Round 3: flaky hits threshold
        tracker.set_pending_nonce(300);
        assert!(tracker.process_ack(&mut reg, "healthy", &make_ack(300)));

        let dead = tracker.increment_missed(&workers, 3);
        assert_eq!(dead, vec!["flaky"]);
        assert_eq!(tracker.missed_count("healthy"), 1); // still fine
        assert_eq!(tracker.missed_count("flaky"), 3);
    }

    /// Ack with a nonce from two rounds ago is rejected, even if it was
    /// once valid. Only the current pending_nonce is accepted.
    #[tokio::test]
    async fn test_stale_nonce_from_earlier_round_rejected() {
        let mut reg = setup_registry_with_worker("w1").await;
        let mut tracker = HeartbeatTracker::new();

        // Round 1: nonce 100
        tracker.set_pending_nonce(100);
        tracker.increment_missed(&["w1".into()], 3);

        // Round 2: nonce 200 (overwrites 100)
        tracker.set_pending_nonce(200);
        tracker.increment_missed(&["w1".into()], 3);

        // Worker finally responds with round-1 nonce — rejected
        assert!(!tracker.process_ack(&mut reg, "w1", &make_ack(100)));
        assert_eq!(tracker.missed_count("w1"), 2); // not reset
    }

    /// After a worker is marked dead and removed from the ready list,
    /// increment_missed should not count it (it's no longer in the
    /// ready_node_ids passed by the caller).
    #[tokio::test]
    async fn test_dead_worker_excluded_from_increment() {
        let mut tracker = HeartbeatTracker::new();

        // Worker alive for 2 rounds
        tracker.set_pending_nonce(1);
        tracker.increment_missed(&["w1".into()], 3);
        tracker.set_pending_nonce(2);
        tracker.increment_missed(&["w1".into()], 3);
        assert_eq!(tracker.missed_count("w1"), 2);

        // Worker marked dead — caller stops including it in ready list
        // (simulating registry.pipeline_order() excluding dead workers)
        tracker.set_pending_nonce(3);
        let dead = tracker.increment_missed(&[], 3); // empty list
        assert!(dead.is_empty());

        // w1's counter stays at 2 — not incremented further
        assert_eq!(tracker.missed_count("w1"), 2);
    }

    /// Nonce changes every round. A valid ack for nonce N, received
    /// after set_pending_nonce(N+1), is rejected.
    #[tokio::test]
    async fn test_ack_after_nonce_rotation_rejected() {
        let mut reg = setup_registry_with_worker("w1").await;
        let mut tracker = HeartbeatTracker::new();

        tracker.set_pending_nonce(50);
        // Nonce rotates before the ack arrives
        tracker.set_pending_nonce(51);

        // Ack for old nonce 50 — rejected
        assert!(!tracker.process_ack(&mut reg, "w1", &make_ack(50)));
        // Ack for current nonce 51 — accepted
        assert!(tracker.process_ack(&mut reg, "w1", &make_ack(51)));
    }

    /// process_ack updates free_blocks and gpu_memory_used in the registry.
    #[tokio::test]
    async fn test_process_ack_updates_registry_stats() {
        let mut reg = setup_registry_with_worker("w1").await;
        let mut tracker = HeartbeatTracker::new();

        tracker.set_pending_nonce(42);
        let ack = HeartbeatAckPayload {
            timestamp_echo: 0,
            nonce_echo: 42,
            gpu_memory_used: 12_000_000_000,
            active_sequences: 5,
            free_blocks: 200,
        };
        assert!(tracker.process_ack(&mut reg, "w1", &ack));

        let entry = reg.get("w1").unwrap();
        assert_eq!(entry.free_blocks, 200);
        assert_eq!(entry.gpu_memory_used, 12_000_000_000);
    }

    /// Simulates the full coordinator heartbeat cycle order:
    /// poll acks → increment missed → send new heartbeat.
    /// A healthy worker should never accumulate missed counts.
    #[tokio::test]
    async fn test_full_cycle_order_healthy_worker() {
        let mut reg = setup_registry_with_worker("w1").await;
        let mut tracker = HeartbeatTracker::new();

        // --- Cycle 1: first heartbeat, no prior nonce to poll ---
        // (pending_nonce is None, so poll+increment are skipped)
        tracker.set_pending_nonce(100);

        // --- Cycle 2: poll acks for nonce 100, then increment, then send ---
        // Worker acked nonce 100
        assert!(tracker.process_ack(&mut reg, "w1", &make_ack(100)));
        let dead = tracker.increment_missed(&["w1".into()], 3);
        assert!(dead.is_empty());
        assert_eq!(tracker.missed_count("w1"), 1); // 0 (reset) + 1 (increment)
        tracker.set_pending_nonce(200);

        // --- Cycle 3: same pattern ---
        assert!(tracker.process_ack(&mut reg, "w1", &make_ack(200)));
        let dead = tracker.increment_missed(&["w1".into()], 3);
        assert!(dead.is_empty());
        assert_eq!(tracker.missed_count("w1"), 1); // always oscillates 0→1
        tracker.set_pending_nonce(300);

        // --- Cycle 4: same ---
        assert!(tracker.process_ack(&mut reg, "w1", &make_ack(300)));
        let dead = tracker.increment_missed(&["w1".into()], 3);
        assert!(dead.is_empty());
        assert_eq!(tracker.missed_count("w1"), 1);
    }

    /// A worker that is added to the pipeline mid-operation (new worker
    /// appears in ready_node_ids) gets a missed counter initialized on
    /// first increment. It should not be falsely flagged as dead.
    #[tokio::test]
    async fn test_new_worker_joins_mid_operation() {
        let mut tracker = HeartbeatTracker::new();

        // Round 1: only w1 exists
        tracker.set_pending_nonce(10);
        tracker.increment_missed(&["w1".into()], 3);
        assert_eq!(tracker.missed_count("w1"), 1);
        assert_eq!(tracker.missed_count("w2"), 0); // not yet tracked

        // Round 2: w2 joins the pipeline
        tracker.set_pending_nonce(20);
        let dead = tracker.increment_missed(&["w1".into(), "w2".into()], 3);
        assert!(dead.is_empty());
        assert_eq!(tracker.missed_count("w1"), 2);
        assert_eq!(tracker.missed_count("w2"), 1); // first miss, not 3

        // w2 needs 2 more misses to be flagged, not instant death
        tracker.set_pending_nonce(30);
        let dead = tracker.increment_missed(&["w1".into(), "w2".into()], 3);
        assert_eq!(dead, vec!["w1"]); // w1 hits 3
        assert_eq!(tracker.missed_count("w2"), 2); // w2 still alive
    }

    #[test]
    fn test_default_constants() {
        assert_eq!(DEFAULT_INTERVAL, Duration::from_secs(5));
        assert_eq!(DEFAULT_MAX_MISSED, 3);
    }

    #[test]
    fn test_tracker_default() {
        let tracker = HeartbeatTracker::default();
        assert_eq!(tracker.pending_nonce(), None);
        assert_eq!(tracker.missed_count("any"), 0);
    }
}
