//! Election term tracking.
//!
//! Terms are monotonically increasing counters that prevent stale elections.
//! Nodes reject election messages from older terms.

use std::sync::atomic::{AtomicU64, Ordering};

/// Thread-safe monotonic term counter.
pub struct TermTracker {
    current: AtomicU64,
}

impl TermTracker {
    pub fn new(initial_term: u64) -> Self {
        Self {
            current: AtomicU64::new(initial_term),
        }
    }

    /// Get the current term.
    pub fn current(&self) -> u64 {
        self.current.load(Ordering::SeqCst)
    }

    /// Advance to a new term. Returns the new term.
    /// Panics if `new_term <= current` (terms must be strictly increasing).
    pub fn advance_to(&self, new_term: u64) -> u64 {
        let old = self.current.load(Ordering::SeqCst);
        assert!(
            new_term > old,
            "term must advance: new={new_term} <= current={old}"
        );
        self.current.store(new_term, Ordering::SeqCst);
        new_term
    }

    /// Increment the term by 1. Returns the new term.
    pub fn increment(&self) -> u64 {
        self.current.fetch_add(1, Ordering::SeqCst) + 1
    }

    /// Check if a received term is acceptable (>= current).
    pub fn is_acceptable(&self, received_term: u64) -> bool {
        received_term >= self.current.load(Ordering::SeqCst)
    }

    /// Check if a received term is strictly newer (> current).
    pub fn is_newer(&self, received_term: u64) -> bool {
        received_term > self.current.load(Ordering::SeqCst)
    }

    /// Update to the received term if it's newer. Returns true if updated.
    pub fn update_if_newer(&self, received_term: u64) -> bool {
        let current = self.current.load(Ordering::SeqCst);
        if received_term > current {
            self.current.store(received_term, Ordering::SeqCst);
            true
        } else {
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_initial_term() {
        let tracker = TermTracker::new(0);
        assert_eq!(tracker.current(), 0);
    }

    #[test]
    fn test_increment() {
        let tracker = TermTracker::new(0);
        assert_eq!(tracker.increment(), 1);
        assert_eq!(tracker.increment(), 2);
        assert_eq!(tracker.current(), 2);
    }

    #[test]
    fn test_advance_to() {
        let tracker = TermTracker::new(0);
        tracker.advance_to(5);
        assert_eq!(tracker.current(), 5);
    }

    #[test]
    #[should_panic(expected = "term must advance")]
    fn test_advance_to_same_panics() {
        let tracker = TermTracker::new(5);
        tracker.advance_to(5);
    }

    #[test]
    fn test_is_acceptable() {
        let tracker = TermTracker::new(3);
        assert!(!tracker.is_acceptable(2));
        assert!(tracker.is_acceptable(3));
        assert!(tracker.is_acceptable(4));
    }

    #[test]
    fn test_is_newer() {
        let tracker = TermTracker::new(3);
        assert!(!tracker.is_newer(2));
        assert!(!tracker.is_newer(3));
        assert!(tracker.is_newer(4));
    }

    #[test]
    fn test_update_if_newer() {
        let tracker = TermTracker::new(3);
        assert!(!tracker.update_if_newer(2));
        assert_eq!(tracker.current(), 3);
        assert!(tracker.update_if_newer(5));
        assert_eq!(tracker.current(), 5);
    }
}
