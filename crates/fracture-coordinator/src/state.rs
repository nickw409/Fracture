//! Sequence state management for the coordinator.
//!
//! Tracks the lifecycle of every active inference sequence across the
//! distributed pipeline: status, position, generated tokens, and which
//! workers have active KV cache for the sequence.

use fracture_core::{FractureError, Result};
use std::collections::HashMap;

/// Status of an inference sequence.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SequenceStatus {
    Prefilling,
    Decoding,
    Complete,
    Error,
}

/// Per-sequence state tracked by the coordinator.
#[derive(Debug, Clone)]
pub struct SequenceState {
    pub seq_id: u64,
    pub status: SequenceStatus,
    /// Next position to generate.
    pub current_pos: usize,
    /// Maximum tokens to generate.
    pub max_tokens: usize,
    /// Token IDs generated so far.
    pub generated_tokens: Vec<u32>,
    /// Node IDs that have active KV cache for this sequence.
    pub cache_allocated_on: Vec<String>,
}

/// Manages all active sequences.
pub struct SequenceStateManager {
    sequences: HashMap<u64, SequenceState>,
    next_id: u64,
}

impl Default for SequenceStateManager {
    fn default() -> Self {
        Self::new()
    }
}

impl SequenceStateManager {
    pub fn new() -> Self {
        Self {
            sequences: HashMap::new(),
            next_id: 1,
        }
    }

    /// Create a new sequence. Returns the assigned seq_id.
    pub fn create(
        &mut self,
        prompt_len: usize,
        max_tokens: usize,
        node_ids: Vec<String>,
    ) -> u64 {
        let seq_id = self.next_id;
        self.next_id += 1;

        self.sequences.insert(
            seq_id,
            SequenceState {
                seq_id,
                status: SequenceStatus::Prefilling,
                current_pos: prompt_len,
                max_tokens,
                generated_tokens: Vec::new(),
                cache_allocated_on: node_ids,
            },
        );

        seq_id
    }

    /// Transition from Prefilling to Decoding.
    pub fn begin_decoding(&mut self, seq_id: u64) -> Result<()> {
        let seq = self.get_mut(seq_id)?;
        if seq.status != SequenceStatus::Prefilling {
            return Err(FractureError::Pipeline(format!(
                "seq {seq_id}: cannot transition to Decoding from {:?}",
                seq.status
            )));
        }
        seq.status = SequenceStatus::Decoding;
        Ok(())
    }

    /// Record a generated token and advance position.
    pub fn record_token(&mut self, seq_id: u64, token_id: u32) -> Result<()> {
        let seq = self.get_mut(seq_id)?;
        if seq.status != SequenceStatus::Decoding {
            return Err(FractureError::Pipeline(format!(
                "seq {seq_id}: cannot record token in {:?} state",
                seq.status
            )));
        }
        seq.generated_tokens.push(token_id);
        seq.current_pos += 1;
        Ok(())
    }

    /// Mark sequence as complete.
    pub fn complete(&mut self, seq_id: u64) -> Result<()> {
        let seq = self.get_mut(seq_id)?;
        seq.status = SequenceStatus::Complete;
        Ok(())
    }

    /// Mark sequence as errored.
    pub fn mark_error(&mut self, seq_id: u64) -> Result<()> {
        let seq = self.get_mut(seq_id)?;
        seq.status = SequenceStatus::Error;
        Ok(())
    }

    /// Remove a sequence (after cache is freed on all workers).
    pub fn remove(&mut self, seq_id: u64) -> Option<SequenceState> {
        self.sequences.remove(&seq_id)
    }

    /// Get sequence state.
    pub fn get(&self, seq_id: u64) -> Result<&SequenceState> {
        self.sequences.get(&seq_id).ok_or_else(|| {
            FractureError::Pipeline(format!("unknown sequence: {seq_id}"))
        })
    }

    /// Check if a sequence has reached its token limit.
    pub fn is_at_limit(&self, seq_id: u64) -> Result<bool> {
        let seq = self.get(seq_id)?;
        Ok(seq.generated_tokens.len() >= seq.max_tokens)
    }

    /// Number of active (non-complete, non-error) sequences.
    pub fn active_count(&self) -> usize {
        self.sequences
            .values()
            .filter(|s| s.status == SequenceStatus::Prefilling || s.status == SequenceStatus::Decoding)
            .count()
    }

    fn get_mut(&mut self, seq_id: u64) -> Result<&mut SequenceState> {
        self.sequences.get_mut(&seq_id).ok_or_else(|| {
            FractureError::Pipeline(format!("unknown sequence: {seq_id}"))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_sequence() {
        let mut mgr = SequenceStateManager::new();
        let id = mgr.create(128, 256, vec!["a".into(), "b".into()]);

        let seq = mgr.get(id).unwrap();
        assert_eq!(seq.status, SequenceStatus::Prefilling);
        assert_eq!(seq.current_pos, 128);
        assert_eq!(seq.max_tokens, 256);
        assert_eq!(seq.cache_allocated_on, vec!["a", "b"]);
    }

    #[test]
    fn test_sequence_lifecycle() {
        let mut mgr = SequenceStateManager::new();
        let id = mgr.create(10, 100, vec!["w1".into()]);

        // Prefilling -> Decoding
        mgr.begin_decoding(id).unwrap();
        assert_eq!(mgr.get(id).unwrap().status, SequenceStatus::Decoding);

        // Record tokens
        mgr.record_token(id, 42).unwrap();
        mgr.record_token(id, 43).unwrap();
        let seq = mgr.get(id).unwrap();
        assert_eq!(seq.generated_tokens, vec![42, 43]);
        assert_eq!(seq.current_pos, 12); // 10 + 2

        // Complete
        mgr.complete(id).unwrap();
        assert_eq!(mgr.get(id).unwrap().status, SequenceStatus::Complete);
    }

    #[test]
    fn test_invalid_transition_decode_from_complete() {
        let mut mgr = SequenceStateManager::new();
        let id = mgr.create(10, 100, vec![]);
        mgr.begin_decoding(id).unwrap();
        mgr.complete(id).unwrap();

        // Cannot go back to Decoding from Complete
        assert!(mgr.begin_decoding(id).is_err());
    }

    #[test]
    fn test_cannot_record_token_while_prefilling() {
        let mut mgr = SequenceStateManager::new();
        let id = mgr.create(10, 100, vec![]);
        assert!(mgr.record_token(id, 42).is_err());
    }

    #[test]
    fn test_is_at_limit() {
        let mut mgr = SequenceStateManager::new();
        let id = mgr.create(10, 3, vec![]);
        mgr.begin_decoding(id).unwrap();

        assert!(!mgr.is_at_limit(id).unwrap());
        mgr.record_token(id, 1).unwrap();
        mgr.record_token(id, 2).unwrap();
        assert!(!mgr.is_at_limit(id).unwrap());
        mgr.record_token(id, 3).unwrap();
        assert!(mgr.is_at_limit(id).unwrap());
    }

    #[test]
    fn test_active_count() {
        let mut mgr = SequenceStateManager::new();
        let id1 = mgr.create(10, 100, vec![]);
        let id2 = mgr.create(10, 100, vec![]);
        assert_eq!(mgr.active_count(), 2);

        mgr.begin_decoding(id1).unwrap();
        mgr.complete(id1).unwrap();
        assert_eq!(mgr.active_count(), 1);

        mgr.mark_error(id2).unwrap();
        assert_eq!(mgr.active_count(), 0);
    }

    #[test]
    fn test_remove() {
        let mut mgr = SequenceStateManager::new();
        let id = mgr.create(10, 100, vec![]);
        assert!(mgr.remove(id).is_some());
        assert!(mgr.get(id).is_err());
    }

    #[test]
    fn test_unknown_sequence() {
        let mgr = SequenceStateManager::new();
        assert!(mgr.get(999).is_err());
    }

    #[test]
    fn test_sequential_ids() {
        let mut mgr = SequenceStateManager::new();
        let id1 = mgr.create(10, 100, vec![]);
        let id2 = mgr.create(10, 100, vec![]);
        assert_eq!(id2, id1 + 1);
    }

    #[test]
    fn test_cannot_record_token_while_complete() {
        let mut mgr = SequenceStateManager::new();
        let id = mgr.create(10, 100, vec![]);
        mgr.begin_decoding(id).unwrap();
        mgr.complete(id).unwrap();
        assert!(mgr.record_token(id, 42).is_err());
    }

    #[test]
    fn test_cannot_record_token_while_error() {
        let mut mgr = SequenceStateManager::new();
        let id = mgr.create(10, 100, vec![]);
        mgr.mark_error(id).unwrap();
        assert!(mgr.record_token(id, 42).is_err());
    }

    #[test]
    fn test_invalid_transition_decode_from_error() {
        let mut mgr = SequenceStateManager::new();
        let id = mgr.create(10, 100, vec![]);
        mgr.mark_error(id).unwrap();
        assert!(mgr.begin_decoding(id).is_err());
    }

    #[test]
    fn test_invalid_transition_decode_from_decoding() {
        let mut mgr = SequenceStateManager::new();
        let id = mgr.create(10, 100, vec![]);
        mgr.begin_decoding(id).unwrap();
        assert!(mgr.begin_decoding(id).is_err());
    }

    #[test]
    fn test_cache_allocated_on_preserved() {
        let mut mgr = SequenceStateManager::new();
        let id = mgr.create(10, 100, vec!["node-a".into(), "node-b".into()]);
        mgr.begin_decoding(id).unwrap();
        mgr.complete(id).unwrap();
        // cache_allocated_on should still be accessible after completion
        let seq = mgr.get(id).unwrap();
        assert_eq!(seq.cache_allocated_on, vec!["node-a", "node-b"]);
    }

    #[test]
    fn test_cache_allocated_on_drives_cleanup() {
        // Simulates the coordinator pattern: create sequence with node IDs,
        // complete it, then use cache_allocated_on to determine which workers
        // need CacheFree messages.
        let mut mgr = SequenceStateManager::new();
        let id = mgr.create(10, 50, vec!["worker-a".into(), "worker-b".into(), "worker-c".into()]);
        mgr.begin_decoding(id).unwrap();
        mgr.record_token(id, 1).unwrap();
        mgr.complete(id).unwrap();

        // The coordinator reads cache_allocated_on to know where to send CacheFree
        let seq = mgr.get(id).unwrap();
        let nodes_to_free: Vec<String> = seq.cache_allocated_on.clone();
        assert_eq!(nodes_to_free, vec!["worker-a", "worker-b", "worker-c"]);

        // After freeing, remove the sequence
        let removed = mgr.remove(id).unwrap();
        assert_eq!(removed.cache_allocated_on.len(), 3);
        assert!(mgr.get(id).is_err());
    }
}
