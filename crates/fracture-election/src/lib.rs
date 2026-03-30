//! Leader election protocol for Fracture distributed inference.
//!
//! Implements a priority-based bully algorithm for coordinator failover.
//! When the coordinator dies, coordinator-capable nodes run an election
//! to choose a new leader.

pub mod state_machine;
pub mod term;
