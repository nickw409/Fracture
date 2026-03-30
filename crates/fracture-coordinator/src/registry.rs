//! Peer registry: tracks connected workers, their capabilities,
//! layer assignments, health state, and connection handles.

use crate::scheduler::{LayerAssignment, WorkerCapabilities};
use fracture_core::{FractureError, Result};
use fracture_protocol::{FramedConnection, FramedReader, FramedWriter};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

/// Health status of a worker.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkerStatus {
    /// Connected but not yet assigned layers.
    Connected,
    /// Assigned layers and ready to serve.
    Ready,
    /// Missed too many heartbeats.
    Dead,
}

/// A registered worker and its associated state.
pub struct WorkerEntry {
    pub capabilities: WorkerCapabilities,
    pub writer: FramedWriter,
    pub reader: Arc<tokio::sync::Mutex<FramedReader>>,
    pub assignment: Option<LayerAssignment>,
    pub last_heartbeat: Instant,
    pub status: WorkerStatus,
    /// Free blocks in this worker's paged KV cache pool (from heartbeat ack).
    pub free_blocks: u32,
    /// GPU memory used in bytes (from heartbeat ack).
    pub gpu_memory_used: u64,
}

impl std::fmt::Debug for WorkerEntry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WorkerEntry")
            .field("node_id", &self.capabilities.node_id)
            .field("status", &self.status)
            .field("assignment", &self.assignment)
            .finish()
    }
}

/// Tracks all connected workers.
pub struct PeerRegistry {
    workers: HashMap<String, WorkerEntry>,
}

impl Default for PeerRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl PeerRegistry {
    pub fn new() -> Self {
        Self {
            workers: HashMap::new(),
        }
    }

    /// Register a new worker. Returns error if a worker with the same
    /// node_id is already registered and not dead.
    ///
    /// The connection is split into independent writer and reader halves.
    /// The writer stays in `WorkerEntry` for sending commands; the reader
    /// is wrapped in `Arc<Mutex>` so it can be polled concurrently for
    /// heartbeat acks without holding the registry lock.
    pub fn register(
        &mut self,
        capabilities: WorkerCapabilities,
        connection: FramedConnection,
    ) -> Result<()> {
        let node_id = capabilities.node_id.clone();

        if let Some(existing) = self.workers.get(&node_id)
            && existing.status != WorkerStatus::Dead
        {
            return Err(FractureError::Protocol(format!(
                "worker '{node_id}' is already registered"
            )));
        }

        let (writer, reader) = connection.into_split();

        self.workers.insert(
            node_id,
            WorkerEntry {
                capabilities,
                writer,
                reader: Arc::new(tokio::sync::Mutex::new(reader)),
                assignment: None,
                last_heartbeat: Instant::now(),
                status: WorkerStatus::Connected,
                free_blocks: 0,
                gpu_memory_used: 0,
            },
        );
        Ok(())
    }

    /// Get a reference to a worker entry.
    pub fn get(&self, node_id: &str) -> Option<&WorkerEntry> {
        self.workers.get(node_id)
    }

    /// Get a mutable reference to a worker entry.
    pub fn get_mut(&mut self, node_id: &str) -> Option<&mut WorkerEntry> {
        self.workers.get_mut(node_id)
    }

    /// Assign layers to a worker and mark it Ready.
    ///
    /// Returns an error if the worker is not found or is Dead.
    /// If the worker already has an assignment (is Ready), the
    /// assignment is overwritten (layer reassignment).
    pub fn assign(&mut self, node_id: &str, assignment: LayerAssignment) -> Result<()> {
        let entry = self.workers.get_mut(node_id).ok_or_else(|| {
            FractureError::Protocol(format!("worker '{node_id}' not found"))
        })?;
        if entry.status == WorkerStatus::Dead {
            return Err(FractureError::Protocol(format!(
                "cannot assign layers to dead worker '{node_id}'"
            )));
        }
        entry.assignment = Some(assignment);
        entry.status = WorkerStatus::Ready;
        Ok(())
    }

    /// Look up a worker entry, returning an error if not found.
    pub fn lookup(&self, node_id: &str) -> Result<&WorkerEntry> {
        self.workers.get(node_id).ok_or_else(|| {
            FractureError::Protocol(format!("worker '{node_id}' not found"))
        })
    }

    /// Mark a worker as dead.
    pub fn mark_dead(&mut self, node_id: &str) {
        if let Some(entry) = self.workers.get_mut(node_id) {
            entry.status = WorkerStatus::Dead;
        }
    }

    /// Update a worker's last heartbeat time and block pool stats.
    pub fn record_heartbeat(&mut self, node_id: &str, free_blocks: u32, gpu_memory_used: u64) {
        if let Some(entry) = self.workers.get_mut(node_id) {
            entry.last_heartbeat = Instant::now();
            entry.free_blocks = free_blocks;
            entry.gpu_memory_used = gpu_memory_used;
        }
    }

    /// Minimum free blocks across all ready pipeline workers.
    ///
    /// Returns 0 if no workers are ready. The distributed scheduler uses
    /// this as the effective memory constraint: the bottleneck worker
    /// determines how many new sequences can be admitted.
    pub fn min_free_blocks(&self) -> u32 {
        self.workers
            .values()
            .filter(|e| e.status == WorkerStatus::Ready && e.assignment.is_some())
            .map(|e| e.free_blocks)
            .min()
            .unwrap_or(0)
    }

    /// Return all worker capabilities (for scheduler input).
    pub fn all_capabilities(&self) -> Vec<WorkerCapabilities> {
        self.workers
            .values()
            .filter(|e| e.status != WorkerStatus::Dead)
            .map(|e| e.capabilities.clone())
            .collect()
    }

    /// Return node IDs of all ready workers in pipeline order
    /// (sorted by layer_range.start).
    pub fn pipeline_order(&self) -> Vec<String> {
        let mut ready: Vec<_> = self
            .workers
            .values()
            .filter(|e| e.status == WorkerStatus::Ready && e.assignment.is_some())
            .collect();
        ready.sort_by_key(|e| e.assignment.as_ref().unwrap().layer_range.start);
        ready
            .into_iter()
            .map(|e| e.capabilities.node_id.clone())
            .collect()
    }

    /// Number of registered (non-dead) workers.
    pub fn active_count(&self) -> usize {
        self.workers
            .values()
            .filter(|e| e.status != WorkerStatus::Dead)
            .count()
    }

    /// Collect reader handles for all ready workers.
    ///
    /// Returns `(node_id, Arc<Mutex<FramedReader>>)` pairs. The caller can
    /// release the registry lock and poll all readers concurrently.
    pub fn reader_handles(&self) -> Vec<(String, Arc<tokio::sync::Mutex<FramedReader>>)> {
        self.workers
            .values()
            .filter(|e| e.status == WorkerStatus::Ready && e.assignment.is_some())
            .map(|e| (e.capabilities.node_id.clone(), e.reader.clone()))
            .collect()
    }

    /// Iterate over immutable entries.
    pub fn iter(&self) -> impl Iterator<Item = (&String, &WorkerEntry)> {
        self.workers.iter()
    }

    /// Iterate over mutable entries (for sending messages to all workers).
    pub fn iter_mut(&mut self) -> impl Iterator<Item = (&String, &mut WorkerEntry)> {
        self.workers.iter_mut()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scheduler::NodeRole;
    use tokio::net::TcpListener;

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

    async fn dummy_connection() -> FramedConnection {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let (client, _server) =
            tokio::join!(tokio::net::TcpStream::connect(addr), listener.accept());
        FramedConnection::new(client.unwrap())
    }

    #[tokio::test]
    async fn test_register_and_get() {
        let mut reg = PeerRegistry::new();
        let conn = dummy_connection().await;
        reg.register(test_caps("w1"), conn).unwrap();

        assert_eq!(reg.active_count(), 1);
        let entry = reg.get("w1").unwrap();
        assert_eq!(entry.status, WorkerStatus::Connected);
    }

    #[tokio::test]
    async fn test_duplicate_register_fails() {
        let mut reg = PeerRegistry::new();
        reg.register(test_caps("w1"), dummy_connection().await).unwrap();
        assert!(reg.register(test_caps("w1"), dummy_connection().await).is_err());
    }

    #[tokio::test]
    async fn test_dead_worker_can_reregister() {
        let mut reg = PeerRegistry::new();
        reg.register(test_caps("w1"), dummy_connection().await).unwrap();
        reg.mark_dead("w1");
        assert!(reg.register(test_caps("w1"), dummy_connection().await).is_ok());
    }

    #[tokio::test]
    async fn test_assign_and_pipeline_order() {
        let mut reg = PeerRegistry::new();
        reg.register(test_caps("b"), dummy_connection().await).unwrap();
        reg.register(test_caps("a"), dummy_connection().await).unwrap();

        reg.assign(
            "a",
            LayerAssignment {
                node_id: "a".into(),
                layer_range: 0..16,
                role: NodeRole::Head,
                expected_decode_ms: 16.0,
                weight_memory_gb: 6.0,
                cache_memory_gb: 1.0,
            },
        )
        .unwrap();
        reg.assign(
            "b",
            LayerAssignment {
                node_id: "b".into(),
                layer_range: 16..32,
                role: NodeRole::Tail,
                expected_decode_ms: 16.0,
                weight_memory_gb: 6.0,
                cache_memory_gb: 1.0,
            },
        )
        .unwrap();

        let order = reg.pipeline_order();
        assert_eq!(order, vec!["a", "b"]);
    }

    #[tokio::test]
    async fn test_reader_handles_returns_ready_workers() {
        let mut reg = PeerRegistry::new();
        reg.register(test_caps("w1"), dummy_connection().await).unwrap();
        reg.register(test_caps("w2"), dummy_connection().await).unwrap();

        // No workers are Ready yet (only Connected), so no reader handles.
        assert!(reg.reader_handles().is_empty());

        reg.assign(
            "w1",
            LayerAssignment {
                node_id: "w1".into(),
                layer_range: 0..16,
                role: NodeRole::Head,
                expected_decode_ms: 16.0,
                weight_memory_gb: 6.0,
                cache_memory_gb: 1.0,
            },
        )
        .unwrap();

        let handles = reg.reader_handles();
        assert_eq!(handles.len(), 1);
        assert_eq!(handles[0].0, "w1");
    }

    #[tokio::test]
    async fn test_mark_dead_sets_status() {
        let mut reg = PeerRegistry::new();
        reg.register(test_caps("w1"), dummy_connection().await).unwrap();
        reg.mark_dead("w1");
        assert_eq!(reg.get("w1").unwrap().status, WorkerStatus::Dead);
    }

    #[tokio::test]
    async fn test_mark_dead_excludes_from_pipeline_order() {
        let mut reg = PeerRegistry::new();
        reg.register(test_caps("w1"), dummy_connection().await).unwrap();
        reg.register(test_caps("w2"), dummy_connection().await).unwrap();

        let assign = |id: &str, start: usize, end: usize| LayerAssignment {
            node_id: id.into(),
            layer_range: start..end,
            role: if start == 0 { NodeRole::Head } else { NodeRole::Tail },
            expected_decode_ms: 16.0,
            weight_memory_gb: 6.0,
            cache_memory_gb: 1.0,
        };
        reg.assign("w1", assign("w1", 0, 16)).unwrap();
        reg.assign("w2", assign("w2", 16, 32)).unwrap();

        assert_eq!(reg.pipeline_order().len(), 2);

        reg.mark_dead("w1");
        let order = reg.pipeline_order();
        assert_eq!(order.len(), 1);
        assert_eq!(order[0], "w2");
    }

    #[tokio::test]
    async fn test_mark_dead_excludes_from_active_count() {
        let mut reg = PeerRegistry::new();
        reg.register(test_caps("w1"), dummy_connection().await).unwrap();
        assert_eq!(reg.active_count(), 1);

        reg.mark_dead("w1");
        assert_eq!(reg.active_count(), 0);
    }

    #[tokio::test]
    async fn test_assign_dead_worker_rejected() {
        let mut reg = PeerRegistry::new();
        reg.register(test_caps("w1"), dummy_connection().await).unwrap();
        reg.mark_dead("w1");

        let result = reg.assign(
            "w1",
            LayerAssignment {
                node_id: "w1".into(),
                layer_range: 0..32,
                role: NodeRole::Head,
                expected_decode_ms: 32.0,
                weight_memory_gb: 12.0,
                cache_memory_gb: 2.0,
            },
        );
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("dead"));
    }

    #[tokio::test]
    async fn test_lookup_returns_error_for_unknown() {
        let reg = PeerRegistry::new();
        assert!(reg.lookup("nonexistent").is_err());
    }

    #[tokio::test]
    async fn test_lookup_returns_entry() {
        let mut reg = PeerRegistry::new();
        reg.register(test_caps("w1"), dummy_connection().await).unwrap();
        let entry = reg.lookup("w1").unwrap();
        assert_eq!(entry.status, WorkerStatus::Connected);
    }

    /// Verify that lookup() returns an entry whose capabilities match what was
    /// passed to register().
    #[tokio::test]
    async fn test_lookup_returns_capabilities() {
        let mut reg = PeerRegistry::new();
        let caps = WorkerCapabilities {
            node_id: "gpu-node-7".into(),
            gpu_model: "NVIDIA A100".into(),
            gpu_memory_available: 80_000_000_000,
            compute_capability: (8, 0),
            decode_ms_per_layer: 0.5,
            prefill_ms_per_layer_128: 2.0,
        };
        reg.register(caps.clone(), dummy_connection().await).unwrap();

        let entry = reg.lookup("gpu-node-7").unwrap();
        assert_eq!(entry.capabilities.node_id, "gpu-node-7");
        assert_eq!(entry.capabilities.gpu_model, "NVIDIA A100");
        assert_eq!(entry.capabilities.gpu_memory_available, 80_000_000_000);
        assert_eq!(entry.capabilities.compute_capability, (8, 0));
        assert!((entry.capabilities.decode_ms_per_layer - 0.5).abs() < 1e-6);
        assert!((entry.capabilities.prefill_ms_per_layer_128 - 2.0).abs() < 1e-6);
    }

    /// Verify that all_capabilities() excludes workers marked dead.
    #[tokio::test]
    async fn test_all_capabilities_filters_dead() {
        let mut reg = PeerRegistry::new();
        reg.register(test_caps("w1"), dummy_connection().await).unwrap();
        reg.register(test_caps("w2"), dummy_connection().await).unwrap();
        reg.register(test_caps("w3"), dummy_connection().await).unwrap();

        // Mark w2 as dead — it should be excluded from all_capabilities().
        reg.mark_dead("w2");

        let caps = reg.all_capabilities();
        assert_eq!(caps.len(), 2, "dead worker w2 should be excluded");

        let ids: Vec<&str> = caps.iter().map(|c| c.node_id.as_str()).collect();
        assert!(ids.contains(&"w1"), "w1 should be present");
        assert!(ids.contains(&"w3"), "w3 should be present");
        assert!(!ids.contains(&"w2"), "dead w2 should be absent");
    }

    /// Verify that the duplicate-register error message contains the node_id so
    /// operators can identify which worker caused the conflict.
    #[tokio::test]
    async fn test_duplicate_register_error_context() {
        let mut reg = PeerRegistry::new();
        let node_id = "conflicting-worker-42";
        reg.register(test_caps(node_id), dummy_connection().await).unwrap();

        let err = reg
            .register(test_caps(node_id), dummy_connection().await)
            .unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains(node_id),
            "duplicate-register error should mention the node_id '{node_id}', got: {msg}"
        );
    }

    #[tokio::test]
    async fn test_layer_reassignment_overwrites() {
        let mut reg = PeerRegistry::new();
        reg.register(test_caps("w1"), dummy_connection().await).unwrap();

        // First assignment
        reg.assign(
            "w1",
            LayerAssignment {
                node_id: "w1".into(),
                layer_range: 0..16,
                role: NodeRole::Head,
                expected_decode_ms: 16.0,
                weight_memory_gb: 6.0,
                cache_memory_gb: 1.0,
            },
        )
        .unwrap();
        assert_eq!(reg.get("w1").unwrap().assignment.as_ref().unwrap().layer_range, 0..16);

        // Reassignment overwrites
        reg.assign(
            "w1",
            LayerAssignment {
                node_id: "w1".into(),
                layer_range: 0..32,
                role: NodeRole::Head,
                expected_decode_ms: 32.0,
                weight_memory_gb: 12.0,
                cache_memory_gb: 2.0,
            },
        )
        .unwrap();
        assert_eq!(reg.get("w1").unwrap().assignment.as_ref().unwrap().layer_range, 0..32);
        assert_eq!(reg.get("w1").unwrap().status, WorkerStatus::Ready);
    }

    #[tokio::test]
    async fn test_record_heartbeat_updates_stats() {
        let mut reg = PeerRegistry::new();
        reg.register(test_caps("w1"), dummy_connection().await).unwrap();
        reg.assign(
            "w1",
            LayerAssignment {
                node_id: "w1".into(), layer_range: 0..32, role: NodeRole::Head,
                expected_decode_ms: 32.0, weight_memory_gb: 12.0, cache_memory_gb: 2.0,
            },
        ).unwrap();

        // Record heartbeat and verify stats are stored
        reg.record_heartbeat("w1", 100, 4_000_000_000);
        assert_eq!(reg.get("w1").unwrap().free_blocks, 100);
        assert_eq!(reg.get("w1").unwrap().gpu_memory_used, 4_000_000_000);
    }

    #[tokio::test]
    async fn test_min_free_blocks_empty() {
        let reg = PeerRegistry::new();
        assert_eq!(reg.min_free_blocks(), 0);
    }

    #[tokio::test]
    async fn test_min_free_blocks_single_worker() {
        let mut reg = PeerRegistry::new();
        reg.register(test_caps("w1"), dummy_connection().await).unwrap();
        let assign = |id: &str, start: usize, end: usize| LayerAssignment {
            node_id: id.into(),
            layer_range: start..end,
            role: if start == 0 { NodeRole::Head } else { NodeRole::Tail },
            expected_decode_ms: 16.0,
            weight_memory_gb: 6.0,
            cache_memory_gb: 1.0,
        };
        reg.assign("w1", assign("w1", 0, 32)).unwrap();
        reg.record_heartbeat("w1", 50, 0);
        assert_eq!(reg.min_free_blocks(), 50);
    }

    #[tokio::test]
    async fn test_min_free_blocks_bottleneck() {
        let mut reg = PeerRegistry::new();
        reg.register(test_caps("w1"), dummy_connection().await).unwrap();
        reg.register(test_caps("w2"), dummy_connection().await).unwrap();

        let assign = |id: &str, start: usize, end: usize| LayerAssignment {
            node_id: id.into(),
            layer_range: start..end,
            role: if start == 0 { NodeRole::Head } else { NodeRole::Tail },
            expected_decode_ms: 16.0,
            weight_memory_gb: 6.0,
            cache_memory_gb: 1.0,
        };
        reg.assign("w1", assign("w1", 0, 16)).unwrap();
        reg.assign("w2", assign("w2", 16, 32)).unwrap();

        reg.record_heartbeat("w1", 200, 0);
        reg.record_heartbeat("w2", 75, 0);

        // Bottleneck is w2 with 75 free blocks
        assert_eq!(reg.min_free_blocks(), 75);
    }

    #[tokio::test]
    async fn test_min_free_blocks_excludes_dead_workers() {
        let mut reg = PeerRegistry::new();
        reg.register(test_caps("w1"), dummy_connection().await).unwrap();
        reg.register(test_caps("w2"), dummy_connection().await).unwrap();

        let assign = |id: &str, start: usize, end: usize| LayerAssignment {
            node_id: id.into(),
            layer_range: start..end,
            role: if start == 0 { NodeRole::Head } else { NodeRole::Tail },
            expected_decode_ms: 16.0,
            weight_memory_gb: 6.0,
            cache_memory_gb: 1.0,
        };
        reg.assign("w1", assign("w1", 0, 16)).unwrap();
        reg.assign("w2", assign("w2", 16, 32)).unwrap();

        reg.record_heartbeat("w1", 200, 0);
        reg.record_heartbeat("w2", 10, 0);

        // Mark the bottleneck as dead — should be excluded
        reg.mark_dead("w2");
        assert_eq!(reg.min_free_blocks(), 200);
    }
}
