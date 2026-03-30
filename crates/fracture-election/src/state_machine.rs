//! Election state machine: Follower → Candidate → Leader.
//!
//! Implements the priority-based bully algorithm:
//! 1. Candidate broadcasts ElectionStart with its priority
//! 2. Higher-priority nodes respond with ElectionChallenge
//! 3. If no challenge received within election_window: candidate wins
//! 4. Winner broadcasts Victory to all peers

use crate::term::TermTracker;
use std::time::Duration;

/// Election state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ElectionState {
    /// Normal operation — following an active coordinator.
    Follower,
    /// Running for coordinator — waiting for challenges or victory timeout.
    Candidate,
    /// Won the election — this node is the coordinator.
    Leader,
}

/// Configuration for the election agent.
#[derive(Debug, Clone)]
pub struct ElectionConfig {
    /// This node's ID.
    pub node_id: String,
    /// This node's election priority (lower = higher priority).
    pub priority: u32,
    /// Time to wait for challenges before declaring victory.
    pub election_window: Duration,
}

/// Result of processing an election message.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ElectionAction {
    /// No action needed.
    None,
    /// Send ElectionChallenge to the candidate (we have higher priority).
    Challenge { candidate_id: String },
    /// Broadcast Victory — we won the election.
    DeclareVictory,
    /// Stand down — a higher-priority candidate or leader exists.
    StandDown { leader_id: String },
    /// Accept the new leader.
    AcceptLeader { leader_id: String, coordinator_addr: String },
}

/// Election agent — drives the state machine for a single node.
pub struct ElectionAgent {
    pub config: ElectionConfig,
    pub term: TermTracker,
    state: ElectionState,
}

impl ElectionAgent {
    pub fn new(config: ElectionConfig, initial_term: u64) -> Self {
        Self {
            config,
            term: TermTracker::new(initial_term),
            state: ElectionState::Follower,
        }
    }

    pub fn state(&self) -> ElectionState {
        self.state
    }

    /// Start an election: increment term, transition to Candidate.
    /// Returns the new term for the ElectionStart message.
    pub fn start_election(&mut self) -> u64 {
        let new_term = self.term.increment();
        self.state = ElectionState::Candidate;
        tracing::info!(
            "starting election: node={}, priority={}, term={}",
            self.config.node_id, self.config.priority, new_term
        );
        new_term
    }

    /// Process a received ElectionStart message.
    pub fn on_election_start(
        &mut self,
        candidate_id: &str,
        candidate_priority: u32,
        candidate_term: u64,
    ) -> ElectionAction {
        if !self.term.is_acceptable(candidate_term) {
            tracing::debug!(
                "ignoring stale ElectionStart from {candidate_id} (term {candidate_term} < {})",
                self.term.current()
            );
            return ElectionAction::None;
        }

        self.term.update_if_newer(candidate_term);

        // If we have higher priority (lower number), challenge the candidate.
        if self.config.priority < candidate_priority
            || (self.config.priority == candidate_priority
                && self.config.node_id.as_str() < candidate_id)
        {
            tracing::info!(
                "challenging {candidate_id} (our priority {} < their {})",
                self.config.priority, candidate_priority
            );
            ElectionAction::Challenge {
                candidate_id: candidate_id.to_string(),
            }
        } else {
            // Lower priority — stand down if we were also a candidate.
            if self.state == ElectionState::Candidate {
                tracing::info!(
                    "standing down: {candidate_id} has higher priority ({})",
                    candidate_priority
                );
                self.state = ElectionState::Follower;
            }
            ElectionAction::None
        }
    }

    /// Process a received ElectionChallenge message.
    pub fn on_election_challenge(
        &mut self,
        challenger_id: &str,
        challenger_priority: u32,
        challenger_term: u64,
    ) -> ElectionAction {
        if !self.term.is_acceptable(challenger_term) {
            return ElectionAction::None;
        }
        self.term.update_if_newer(challenger_term);

        if self.state == ElectionState::Candidate {
            // A higher-priority node challenged us — stand down.
            tracing::info!(
                "received challenge from {challenger_id} (priority {}) — standing down",
                challenger_priority
            );
            self.state = ElectionState::Follower;
            ElectionAction::StandDown {
                leader_id: challenger_id.to_string(),
            }
        } else {
            ElectionAction::None
        }
    }

    /// Process a received Victory message.
    pub fn on_victory(
        &mut self,
        leader_id: &str,
        leader_term: u64,
        coordinator_addr: &str,
    ) -> ElectionAction {
        if !self.term.is_acceptable(leader_term) {
            tracing::debug!(
                "ignoring stale Victory from {leader_id} (term {leader_term} < {})",
                self.term.current()
            );
            return ElectionAction::None;
        }

        self.term.update_if_newer(leader_term);
        self.state = ElectionState::Follower;
        tracing::info!("accepting leader: {leader_id} at term {leader_term}");

        ElectionAction::AcceptLeader {
            leader_id: leader_id.to_string(),
            coordinator_addr: coordinator_addr.to_string(),
        }
    }

    /// Called when the election window expires without receiving any challenges.
    /// If still a Candidate, declare victory.
    pub fn on_election_timeout(&mut self) -> ElectionAction {
        if self.state == ElectionState::Candidate {
            self.state = ElectionState::Leader;
            tracing::info!(
                "election won: node={}, term={}",
                self.config.node_id, self.term.current()
            );
            ElectionAction::DeclareVictory
        } else {
            ElectionAction::None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn agent(id: &str, priority: u32) -> ElectionAgent {
        ElectionAgent::new(
            ElectionConfig {
                node_id: id.to_string(),
                priority,
                election_window: Duration::from_secs(5),
            },
            0,
        )
    }

    #[test]
    fn test_start_election() {
        let mut a = agent("node-a", 1);
        assert_eq!(a.state(), ElectionState::Follower);
        let term = a.start_election();
        assert_eq!(term, 1);
        assert_eq!(a.state(), ElectionState::Candidate);
    }

    #[test]
    fn test_higher_priority_challenges() {
        let mut a = agent("node-a", 0); // higher priority
        let action = a.on_election_start("node-b", 1, 1);
        assert_eq!(
            action,
            ElectionAction::Challenge {
                candidate_id: "node-b".into()
            }
        );
    }

    #[test]
    fn test_lower_priority_does_not_challenge() {
        let mut a = agent("node-a", 2); // lower priority
        let action = a.on_election_start("node-b", 1, 1);
        assert_eq!(action, ElectionAction::None);
    }

    #[test]
    fn test_candidate_stands_down_on_challenge() {
        let mut a = agent("node-a", 1);
        a.start_election();
        assert_eq!(a.state(), ElectionState::Candidate);

        let action = a.on_election_challenge("node-b", 0, 1);
        assert_eq!(
            action,
            ElectionAction::StandDown {
                leader_id: "node-b".into()
            }
        );
        assert_eq!(a.state(), ElectionState::Follower);
    }

    #[test]
    fn test_victory_accepted() {
        let mut a = agent("node-a", 1);
        let action = a.on_victory("node-b", 1, "192.168.1.10:9400");
        assert_eq!(
            action,
            ElectionAction::AcceptLeader {
                leader_id: "node-b".into(),
                coordinator_addr: "192.168.1.10:9400".into()
            }
        );
        assert_eq!(a.state(), ElectionState::Follower);
    }

    #[test]
    fn test_stale_victory_rejected() {
        let mut a = agent("node-a", 1);
        a.term.advance_to(5);
        let action = a.on_victory("node-b", 3, "addr");
        assert_eq!(action, ElectionAction::None);
    }

    #[test]
    fn test_election_timeout_wins() {
        let mut a = agent("node-a", 0);
        a.start_election();
        let action = a.on_election_timeout();
        assert_eq!(action, ElectionAction::DeclareVictory);
        assert_eq!(a.state(), ElectionState::Leader);
    }

    #[test]
    fn test_election_timeout_no_op_if_not_candidate() {
        let mut a = agent("node-a", 0);
        let action = a.on_election_timeout();
        assert_eq!(action, ElectionAction::None);
        assert_eq!(a.state(), ElectionState::Follower);
    }

    #[test]
    fn test_same_priority_tiebreak_by_node_id() {
        let mut a = agent("node-a", 1);
        // node-a < node-b lexicographically, so node-a has higher priority (challenges)
        let action = a.on_election_start("node-b", 1, 1);
        assert_eq!(
            action,
            ElectionAction::Challenge {
                candidate_id: "node-b".into()
            }
        );

        let mut b = agent("node-b", 1);
        // node-b > node-a, so node-b does NOT challenge node-a
        let action = b.on_election_start("node-a", 1, 1);
        assert_eq!(action, ElectionAction::None);
    }

    #[test]
    fn test_three_node_election_highest_priority_wins() {
        let mut a = agent("node-a", 0); // highest priority
        let mut b = agent("node-b", 1);
        let mut c = agent("node-c", 2); // lowest priority

        // C starts election
        let term = c.start_election();

        // A and B both see C's ElectionStart
        let a_action = a.on_election_start("node-c", 2, term);
        let b_action = b.on_election_start("node-c", 2, term);

        // A challenges (priority 0 < 2)
        assert_eq!(
            a_action,
            ElectionAction::Challenge {
                candidate_id: "node-c".into()
            }
        );
        // B challenges (priority 1 < 2)
        assert_eq!(
            b_action,
            ElectionAction::Challenge {
                candidate_id: "node-c".into()
            }
        );

        // C receives challenges — stands down
        let c_action = c.on_election_challenge("node-a", 0, term);
        assert_eq!(
            c_action,
            ElectionAction::StandDown {
                leader_id: "node-a".into()
            }
        );

        // A also starts election (it challenged, so it should run)
        let a_term = a.start_election();
        // B sees A's ElectionStart — A has higher priority, B does not challenge
        let b_action2 = b.on_election_start("node-a", 0, a_term);
        assert_eq!(b_action2, ElectionAction::None);

        // A times out with no challenges — wins
        let win = a.on_election_timeout();
        assert_eq!(win, ElectionAction::DeclareVictory);
        assert_eq!(a.state(), ElectionState::Leader);
    }
}
