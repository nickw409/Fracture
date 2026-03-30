# Fracture: Fault Tolerance Architecture Document
## Network Hardening Against Node Death

**Depends on:** Phase 4 Step 4 complete (distributed batching with heartbeat monitoring)
**Goal:** Make the distributed inference network resilient to node failure — workers survive coordinator death, a new coordinator is elected automatically, and nodes can join or leave mid-inference with graceful recovery.

---

## What Changes from Phase 4

Phase 4 built a working distributed batched inference pipeline with heartbeat monitoring and basic failure detection. But the system has a single point of failure: the coordinator. If the coordinator dies, every worker exits. If a worker dies, all in-flight sequences are aborted and the pipeline stalls until the worker restarts. There is no mechanism for new nodes to join a running cluster or for nodes to leave without disrupting active work.

| Component | Phase 4 | Fault Tolerance |
|---|---|---|
| Coordinator failure | All workers exit (connection lost) | Workers enter standby, new coordinator elected |
| Worker failure | All sequences aborted, pipeline degraded until reconnect | Affected sequences aborted, pipeline rebalances with remaining workers |
| Node join | Only at startup (initial acceptance phase) | Mid-inference join with pipeline rebalancing |
| Node departure | Crash = heartbeat timeout + full abort | Graceful leave (drain + depart) or crash recovery |
| Coordinator redundancy | None (SPOF) | Leader election among coordinator-capable nodes |
| Worker connection loss | Immediate exit | Reconnection attempts with exponential backoff |

---

## Problem 1: Coordinator is a Single Point of Failure

### Current Behavior

When the coordinator process dies:

```
Coordinator dies (crash, OOM, host failure)
    ↓
TCP connections to all workers drop
    ↓
Each worker's recv() returns Err(connection lost)
    ↓
Workers exit gracefully (break from serve loop)
    ↓
Entire cluster is dead — GPU resources freed, KV caches lost
    ↓
Manual restart of coordinator + all workers required
```

The coordinator holds all orchestration state: the peer registry, pipeline ordering, sequence state, heartbeat tracking, and the HTTP server for client requests. Losing it means losing everything.

### Solution: Leader Election

Instead of a single coordinator, multiple nodes in the network are **coordinator-capable**. One is the active leader; the others are standby. When the leader dies, a new leader is elected from the standby pool.

#### Coordinator-Capable Nodes

Any node in the cluster can be coordinator-capable. This includes:

- **Dedicated coordinator nodes** — Machines running only the coordinator binary (no GPU work). These are the simplest case: they exist solely to coordinate, and election just picks a new one.
- **Worker nodes with coordinator capability** — A worker that can promote itself to also run coordination logic. This is useful in small clusters where dedicating a machine to coordination is wasteful.

The coordinator role is lightweight (no GPU, no large memory). A worker that takes on coordinator duties continues serving forward requests while also managing the pipeline. The coordinator logic runs in separate tokio tasks and does not block the forward-serving path.

#### Network Topology

Phase 4's topology is a star: every worker connects to the coordinator, and workers don't know about each other.

```
Phase 4 (star):

    Worker 0 ←──→ Coordinator ←──→ Worker 2
                      ↕
                  Worker 1
```

Fault tolerance requires workers to discover each other for election. The coordinator broadcasts a **cluster manifest** — the list of all nodes, their addresses, roles, and election priority — to every worker as part of registration and on every topology change.

```
Fault tolerant (star + peer awareness):

    Worker 0 ←──→ Coordinator ←──→ Worker 2
        ↕              ↕              ↕
        └──── peer awareness ─────────┘
              (manifest only, no
               direct data path)
```

Workers don't open persistent connections to each other during normal operation. They only use peer addresses during election. The data path remains star-shaped through the coordinator — changing this would require GPU-direct transfer and is out of scope.

#### Election Protocol

The election uses a **priority-based bully algorithm**, chosen for simplicity and determinism in small clusters (2-6 nodes typical for Fracture). Raft would be more robust for large clusters but adds significant complexity (log replication, term tracking, split-brain resolution) that isn't justified for this scale.

**Priority assignment:**
- Each coordinator-capable node has a `priority: u32` (configured at startup via `--election-priority`)
- Lower number = higher priority (priority 0 wins over priority 1)
- Ties broken by lexicographic node ID
- Non-coordinator-capable nodes (e.g., GPU-only workers started with `--no-coordinator`) do not participate in election

**Election trigger:**
A node initiates election when it detects the coordinator is unreachable:
- Workers: coordinator heartbeat missed for `election_timeout` (default: 3 missed heartbeats = 15 seconds)
- Standby coordinators: direct health check to active coordinator fails for `election_timeout`

**Election flow:**

```
1. Node detects coordinator death
   ↓
2. Node broadcasts ElectionStart { candidate_id, priority, term }
   to all peers in the cluster manifest
   ↓
3. Higher-priority nodes respond with ElectionChallenge { challenger_id, priority, term }
   (meaning: "I outrank you, I'll take over")
   ↓
4. If candidate receives any ElectionChallenge within election_window (default: 5 seconds):
   → Stand down, wait for the challenger's Victory message
   ↓
5. If candidate receives no challenges within election_window:
   → Candidate wins, broadcasts Victory { leader_id, term }
   ↓
6. All nodes accept the new leader and establish connections
```

**Term numbers** prevent stale elections. Each election increments the term. Nodes reject election messages from older terms. If a previously-dead coordinator comes back online and sees a higher term, it yields rather than causing a split-brain.

**Split-brain prevention:**
- Term monotonicity: nodes only accept leaders with term >= their current term
- Priority determinism: given the same set of candidates, the same winner is chosen
- Victory broadcast: all nodes converge on the same leader
- If two nodes simultaneously start elections at the same term, the higher-priority one wins via challenge responses

#### State Reconstruction After Election

The new coordinator doesn't inherit state from the old one — it reconstructs it from the workers. This is possible because workers retain their GPU state (weights, KV caches, assignments) across coordinator transitions.

**Reconstruction sequence:**

```
New coordinator elected
    ↓
1. Accept re-registration from all workers
   Workers send Register with:
   - Standard capabilities (GPU model, memory, compute)
   - Current assignment (layer range, role) — if they have one
   - Active cache entries (seq_id list + positions)
    ↓
2. Rebuild PeerRegistry from registrations
   - Workers already have weights loaded for their assigned layers
   - No need to re-run scheduler or reload weights (unless the pipeline shape changed)
    ↓
3. Rebuild DistributedPipeline from reported assignments
   - Validate layer ranges are contiguous and complete
   - If gaps exist (dead worker's layers), trigger rebalancing (see Problem 3)
    ↓
4. Rebuild sequence state from worker reports
   - Workers report active cache allocations (seq_id + current position)
   - New coordinator creates SequenceState entries for each
   - Sequences that were mid-forward when coordinator died are in an unknown state
   → Policy: abort mid-forward sequences (free caches), preserve idle cached sequences
    ↓
5. Start heartbeat, scheduler loop, and HTTP server
   - Resume accepting new requests
   - Existing cached sequences can continue generation
```

**What's lost:**
- In-flight forward passes (the coordinator was relaying activations) — these sequences are aborted
- HTTP connections to the old coordinator — clients must reconnect and retry
- Pending request queue — clients must resubmit

**What's preserved:**
- Worker GPU state (weights, KV caches for completed decode steps)
- Pipeline assignments (no weight reloading)
- Sequences that had completed their current decode step and were waiting for the next

#### Wire Protocol Additions

New message types for election:

| Type | Direction | Payload |
|---|---|---|
| `0x10 ElectionStart` | Node → Peers | `{ candidate_id, priority, term }` |
| `0x11 ElectionChallenge` | Node → Candidate | `{ challenger_id, priority, term }` |
| `0x12 Victory` | Leader → Peers | `{ leader_id, term, coordinator_addr }` |
| `0x13 ClusterManifest` | Coordinator → Workers | `{ term, nodes: Vec<NodeInfo> }` |
| `0x14 ReRegister` | Worker → New Coordinator | `{ capabilities, current_assignment, active_caches }` |

`NodeInfo` in the manifest:
```rust
struct NodeInfo {
    node_id: String,
    address: SocketAddr,        // peer address for election
    election_priority: u32,     // 0 = highest
    coordinator_capable: bool,
    role: NodeRole,             // Coordinator | Worker | Standby
}
```

---

## Problem 2: Workers Die on Coordinator Loss

### Current Behavior

The worker serve loop exits immediately on any connection error:

```rust
// Current (bins/fracture-worker-cuda/src/main.rs)
loop {
    match conn.recv().await {
        Ok((header, payload)) => { /* handle message */ }
        Err(e) => {
            tracing::error!("connection lost: {e}");
            break;  // ← exits, GPU state freed
        }
    }
}
```

This means temporary network glitches, coordinator restarts, or coordinator crashes all cause the worker to terminate and release its GPU resources (weights, KV caches).

### Solution: Worker Resilience with Reconnection

Workers enter a **disconnected standby** state instead of exiting when the coordinator connection drops. They retain all GPU state and wait for either:
1. Reconnection to the original coordinator (if it restarts)
2. A new coordinator to emerge (via election)
3. An explicit `Shutdown` command (the only thing that causes worker exit)

#### Worker State Machine

```
                    ┌──────────────────────────────────┐
                    │                                  │
                    ▼                                  │
    ┌──────────┐  Register  ┌──────────┐  Assignment  ┌──────────┐
    │          │ ─────────→ │          │ ──────────→ │          │
    │ Starting │            │Connected │              │  Ready   │◄──┐
    │          │            │          │              │          │   │
    └──────────┘            └────┬─────┘              └──┬───┬──┘   │
                                 │                       │   │      │
                          conn   │                 conn  │   │ Reconfigure
                          lost   │                 lost  │   │      │
                                 │                       │   │      │
                                 ▼                       ▼   │      │
                            ┌────────────────────────────┐   │      │
                            │                            │   │      │
                            │    Disconnected Standby    │───┘      │
                            │                            │──────────┘
                            │  • GPU state retained      │  reconnect
                            │  • KV caches preserved     │
                            │  • Election participation  │
                            │  • Reconnection attempts   │
                            └────────────┬───────────────┘
                                         │
                                    Shutdown
                                    (explicit)
                                         │
                                         ▼
                                   ┌──────────┐
                                   │  Exited  │
                                   └──────────┘
```

#### Reconnection Strategy

When a worker enters Disconnected Standby:

1. **Start election timer** — if `coordinator_capable`, participate in election after `election_timeout`
2. **Attempt reconnection** — try to connect to the coordinator address with exponential backoff:
   - Initial delay: 1 second
   - Max delay: 30 seconds
   - Backoff factor: 2x
   - Jitter: +/- 25% (prevents thundering herd when all workers reconnect simultaneously)
3. **Listen for Victory messages** — if a new coordinator is elected, connect to it instead
4. **On successful reconnection** — send `ReRegister` (not `Register`) with current state:
   - Capabilities (same as initial registration)
   - Current layer assignment (the layers whose weights are already loaded)
   - Active cache entries (seq_ids with allocated KV caches + positions)

The new coordinator can then decide whether to keep the current assignment or send a `Reconfigure`.

#### Coordinator Address Discovery

Workers need to find the new coordinator after election. Three mechanisms, tried in order:

1. **Victory message** — Workers participating in election receive the winner's `Victory` broadcast with `coordinator_addr`. This is the fast path.
2. **Manifest fallback** — Workers that missed the Victory message (temporary network partition) iterate through coordinator-capable nodes in the last-known cluster manifest and attempt connection. The new coordinator accepts `ReRegister` messages.
3. **Original address retry** — If the original coordinator simply restarted (same address), the exponential backoff reconnection will find it.

---

## Problem 3: Dynamic Node Join and Leave

### Current Behavior

- **Join:** Workers can only join during the initial acceptance phase (`accept_and_setup_pipeline`). The coordinator waits for exactly N workers, then builds the pipeline. Late arrivals are handled by `reconnection_listener`, but only as replacements for dead workers — not as new additions.
- **Leave:** A worker crash is detected by heartbeat timeout. All active sequences are aborted. The pipeline is marked degraded until the worker reconnects.

### Solution: Live Rebalancing

#### Joining Mid-Inference

A new worker can join a running cluster at any time. The coordinator handles this without disrupting active sequences where possible.

**Join flow:**

```
New Worker                           Coordinator
    │                                     │
    │  TCP connect                        │
    │────────────────────────────────────→│
    │  Register (capabilities)            │
    │────────────────────────────────────→│
    │                                     │ (worker added to registry as Connected)
    │                                     │ (scheduler re-runs with N+1 workers)
    │                                     │
    │                          ◄── Decision point ──►
    │                          │                     │
    │                    Deferred join          Immediate rebalance
    │                    (active sequences      (no active sequences
    │                     exist, rebalancing     OR user-triggered
    │                     would disrupt them)    via API)
    │                          │                     │
    │                          ▼                     ▼
    │                    Worker marked as        Rebalance now:
    │                    "Pending" — waits       Reconfigure all workers
    │                    until active seqs       with new assignments
    │                    drain, then rebalance
    │                                     │
    │  RegisterAck (assignment)           │
    │←────────────────────────────────────│
    │  (load weights)                     │
    │  WorkerReady                        │
    │────────────────────────────────────→│
    │                                     │ (rebuild pipeline, resume)
```

**Deferred vs. Immediate join:**

The coordinator uses a **drain-then-rebalance** strategy by default:
1. New worker registered but held in `Pending` state
2. Active sequences continue on the current pipeline (no disruption)
3. Once all active sequences complete (or the pending queue drains to zero), the coordinator triggers rebalance
4. All workers are reconfigured with new layer assignments
5. The new worker loads its assigned weights and joins the pipeline

An **immediate rebalance** can be triggered via an HTTP API endpoint (`POST /admin/rebalance`) or automatically if there are no active sequences. This aborts any in-flight sequences, reconfigures all workers, and rebuilds the pipeline.

**Scheduler changes:**

The existing scheduler already handles variable worker counts — it computes layer assignments from capabilities. No scheduler changes are needed for adding a worker. The scheduler simply runs with N+1 workers instead of N.

#### Leaving Mid-Inference

Two kinds of departure: **graceful** (planned) and **crash** (unplanned).

**Graceful leave:**

A worker announces its intent to leave before disconnecting. This allows the coordinator to drain work from it before removing it from the pipeline.

```
Worker                                Coordinator
    │                                     │
    │  LeaveIntent                        │
    │────────────────────────────────────→│
    │                                     │ (stop scheduling new prefills to this worker)
    │                                     │ (wait for active sequences to complete)
    │                                     │ (... sequences drain ...)
    │                                     │
    │  Shutdown (ack: you may leave)      │
    │←────────────────────────────────────│
    │                                     │
    │  (disconnect, free GPU resources)   │
    │                                     │ (re-run scheduler with N-1 workers)
    │                                     │ (Reconfigure remaining workers)
    │                                     │ (rebuild pipeline)
```

New wire protocol message:

| Type | Direction | Payload |
|---|---|---|
| `0x15 LeaveIntent` | Worker → Coordinator | `{ reason: String }` |

The coordinator responds with `Shutdown` once the worker is drained. The worker treats `Shutdown` as the permission to exit (same as today's behavior).

**Crash leave (unplanned):**

When a worker crashes, the coordinator detects it via heartbeat timeout (3 missed heartbeats = 15 seconds). The current behavior aborts all sequences and waits for the worker to return. The new behavior is:

```
Worker crashes
    ↓
Heartbeat timeout (15s) → mark Dead
    ↓
Abort sequences that used the dead worker's layers
(only sequences whose pipeline included the dead worker — not all sequences)
    ↓
Decision: Can the remaining workers cover all layers?
    ├── Yes: Re-run scheduler with N-1 workers, Reconfigure, rebuild pipeline
    │        Resume serving immediately (no waiting for dead worker)
    │
    └── No:  Pipeline is broken (e.g., the only worker for layers 0-10 died)
             Mark pipeline degraded, reject new requests
             Wait for: replacement worker to join, or operator intervention
```

**Key improvement over Phase 4:** The coordinator doesn't wait for the dead worker to return. If the remaining workers have enough GPU memory and compute to cover all layers, the scheduler redistributes and the pipeline recovers automatically.

#### Pipeline Rebalancing

Rebalancing redistributes layers across workers. It requires all workers to free their KV caches, receive new layer assignments, reload weights for the new range, and rebuild their caches.

**Rebalance sequence:**

```
1. Coordinator decides to rebalance (new join, crash recovery, or manual trigger)
    ↓
2. Drain or abort active sequences
   • Graceful: wait for completion (preferred for join)
   • Forced: abort (required for crash recovery)
    ↓
3. Send CacheFree for all remaining cached sequences to all workers
    ↓
4. Run scheduler with current worker set
    ↓
5. Send Reconfigure to all existing workers (new assignments)
   Send RegisterAck to new workers (if joining)
    ↓
6. Workers free all caches, reload weights for new layer range, rebuild caches
    ↓
7. Wait for WorkerReady from all workers
    ↓
8. Build new DistributedPipeline with new assignments
    ↓
9. Broadcast pipeline to scheduler loop
    ↓
10. Resume accepting requests
```

**Rebalance cost:**
- Weight reloading: ~1-3 seconds per worker (reading from disk + GPU transfer)
- Cache loss: all KV caches are freed (sequences must re-prefill)
- Downtime: duration of the slowest worker's weight reload

This cost is acceptable because rebalancing is rare (node join/leave events, not per-request). The alternative — migrating KV cache blocks between workers over the network — would be significantly more complex and is deferred.

---

## Implementation Plan

### Step FT-1: Worker State Machine

**Scope:** Add worker state tracking with the `DisconnectedStandby` state. Workers no longer exit on connection loss.

| Sub-step | Description |
|---|---|
| FT-1a | Add `WorkerState` enum (`Starting`, `Connected`, `Ready`, `DisconnectedStandby`, `Exited`) to worker binary |
| FT-1b | Replace `break` on connection loss with transition to `DisconnectedStandby` — GPU state retained, serve loop paused |
| FT-1c | Only `Shutdown` message (explicit from coordinator) causes transition to `Exited` |
| FT-1d | Unit tests: state transitions, verify worker process stays alive after simulated connection drop |

**Validation:** Worker process stays alive after coordinator connection drops. GPU resources (weights, KV caches) are not freed. Worker exits only on explicit `Shutdown`.

### Step FT-2: Worker Reconnection

**Scope:** Workers in `DisconnectedStandby` attempt to reconnect to the coordinator automatically.

| Sub-step | Description |
|---|---|
| FT-2a | Implement exponential backoff reconnection loop with jitter (1s initial, 30s max, 2x factor, +/-25% jitter) |
| FT-2b | Add `ReRegister` message type (0x14) to wire protocol — carries capabilities, current layer assignment, and active cache entries (seq_id + position) |
| FT-2c | On successful reconnection, send `ReRegister` instead of `Register` and re-enter serve loop |
| FT-2d | Unit tests: backoff timing, jitter bounds, `ReRegister` payload serialization round-trip |

**Validation:** Worker in `DisconnectedStandby` reconnects within backoff window when coordinator becomes reachable. `ReRegister` payload correctly encodes current assignment and cache state.

### Step FT-3: Coordinator Handles ReRegister

**Scope:** Coordinator accepts `ReRegister` from reconnecting workers and restores them without full re-setup.

| Sub-step | Description |
|---|---|
| FT-3a | Coordinator recognizes `ReRegister` message type and distinguishes it from fresh `Register` |
| FT-3b | If worker's reported assignment matches current scheduler output: restore worker entry as `Ready`, skip weight reload |
| FT-3c | If assignment doesn't match (topology changed while worker was disconnected): send `Reconfigure` with new assignment |
| FT-3d | Rebuild `DistributedPipeline` after re-registration completes |
| FT-3e | E2e test: kill coordinator, restart it, verify workers reconnect via `ReRegister` and pipeline resumes without weight reloading |

**Validation:** Workers reconnect to restarted coordinator within 30 seconds. Pipeline resumes serving requests. No weight reloading when assignment is unchanged.

### Step FT-4: Forced Rebalance

**Scope:** Build the core rebalance engine for the immediate/forced path (abort active sequences, reconfigure, rebuild). This is the foundation that crash recovery, graceful leave, and dynamic join all build on.

| Sub-step | Description |
|---|---|
| FT-4a | New module `crates/fracture-coordinator/src/rebalance.rs` with `RebalanceOrchestrator` |
| FT-4b | Implement forced rebalance sequence: abort active sequences → free caches → run scheduler → send Reconfigure/RegisterAck → wait WorkerReady → rebuild pipeline → broadcast |
| FT-4c | Unit tests: forced rebalance orchestration with mock workers (verify correct message ordering, all caches freed before reconfigure sent) |

**Validation:** Forced rebalance correctly sequences abort → reconfigure → rebuild. Pipeline is functional after rebalance with new layer assignments.

### Step FT-4b: Graceful Rebalance

**Scope:** Add the drain-then-rebalance path on top of the forced rebalance engine. Used by graceful leave and deferred join.

| Sub-step | Description |
|---|---|
| FT-4b-a | Add `RebalanceMode` enum (`Forced`, `Graceful`) to `RebalanceOrchestrator` |
| FT-4b-b | Graceful mode: monitor active sequence count, trigger forced rebalance only when count reaches zero |
| FT-4b-c | Cancellation: if a new rebalance is requested while a graceful drain is pending, the new request supersedes (e.g., crash during drain escalates to forced) |
| FT-4b-d | Unit tests: graceful drain waits for sequences, cancellation escalates correctly |

**Validation:** Graceful rebalance waits for active sequences to complete before reconfiguring. Pending graceful rebalance can be cancelled or escalated to forced.

### Step FT-5: Crash Recovery

**Scope:** When a worker crashes, the coordinator rebalances with remaining workers instead of waiting.

| Sub-step | Description |
|---|---|
| FT-5a | On worker death, run `scheduler.try_schedule()` with N-1 workers to check if remaining workers can cover all layers |
| FT-5b | If coverable: trigger forced rebalance via `RebalanceOrchestrator`. If not coverable: current behavior (pipeline degraded, wait for replacement) |
| FT-5c | Track which sequences were routed through the dead worker — abort only those, not all |
| FT-5d | E2e test: 3-worker pipeline, kill middle worker, verify pipeline recovers with 2 workers and resumes serving |

**Validation:** With 3 workers, killing one causes the other two to absorb its layers and resume serving within weight-reload time (~3 seconds). Sequences not involving the dead worker are unaffected.

### Step FT-6: Graceful Leave

**Scope:** Workers can announce departure and be drained before removal.

| Sub-step | Description |
|---|---|
| FT-6a | Add `LeaveIntent` message type (0x15) to wire protocol |
| FT-6b | Coordinator handles `LeaveIntent`: mark worker as `Draining`, stop scheduling new work to it |
| FT-6c | When all active sequences on the draining worker complete, trigger graceful rebalance via `RebalanceOrchestrator`, send `Shutdown` to departing worker |
| FT-6d | Worker CLI: SIGTERM handler sends `LeaveIntent` instead of hard exit, waits for `Shutdown` response |
| FT-6e | E2e test: start generation, send SIGTERM to one worker, verify in-flight request completes and pipeline rebalances |

**Validation:** Active request completes successfully. Pipeline rebalances to N-1 workers. No sequence aborts.

### Step FT-7: Dynamic Join

**Scope:** New workers can join a running cluster and be incorporated into the pipeline.

| Sub-step | Description |
|---|---|
| FT-7a | Modify `reconnection_listener` to accept truly new workers (not just replacements for dead ones) — add `Pending` worker status |
| FT-7b | Implement deferred join: new worker held in `Pending` until active sequences drain, then trigger graceful rebalance |
| FT-7c | Auto-rebalance when pending workers exist and active sequence count hits zero |
| FT-7d | E2e test: start 2-worker pipeline, send requests, add 3rd worker, verify it joins after drain |

**Validation:** New worker joins without aborting active sequences (deferred mode). After rebalance, pipeline uses all workers with correct layer distribution.

### Step FT-8: Admin API

**Scope:** HTTP endpoints for operator control over rebalancing and cluster management.

| Sub-step | Description |
|---|---|
| FT-8a | `POST /admin/rebalance` — trigger immediate forced rebalance (aborts active sequences) |
| FT-8b | `GET /admin/cluster` — return cluster state: workers, assignments, status, pending joins |
| FT-8c | `POST /admin/drain` — trigger graceful rebalance (waits for active sequences to complete) |
| FT-8d | E2e test: add worker, trigger `/admin/rebalance`, verify immediate incorporation |

**Validation:** Admin endpoints correctly trigger rebalance and report cluster state.

### Step FT-9: Cluster Manifest and Peer Discovery

**Scope:** Workers maintain awareness of all peers for election and reconnection.

| Sub-step | Description |
|---|---|
| FT-9a | Define `ClusterManifest` payload (0x13): list of `NodeInfo` (id, address, priority, coordinator_capable, role) with monotonic version number |
| FT-9b | Coordinator broadcasts manifest on: worker join, worker leave, rebalance, election victory |
| FT-9c | Workers store manifest locally, use for coordinator fallback (iterate manifest to find new coordinator) |
| FT-9d | Workers reject manifests with version <= their current version (stale protection) |
| FT-9e | Unit tests: manifest serialization, version rejection, peer list accuracy after topology changes |

**Validation:** Workers receive updated manifest after topology changes. Manifest version monotonically increases. Workers can enumerate coordinator-capable peers from manifest.

### Step FT-10: Election Protocol

**Scope:** New `fracture-election` crate implementing the priority-based bully algorithm.

| Sub-step | Description |
|---|---|
| FT-10a | Create `crates/fracture-election/` crate with `ElectionAgent` public API |
| FT-10b | Add election message types to wire protocol: `ElectionStart` (0x10), `ElectionChallenge` (0x11), `Victory` (0x12) |
| FT-10c | Implement term tracking (`term.rs`): monotonic term counter, term comparison, reject-older-term logic |
| FT-10d | Implement election state machine (`state_machine.rs`): `Follower` → `Candidate` → `Leader` transitions |
| FT-10e | Election flow: broadcast `ElectionStart`, wait for challenges, declare victory or stand down |
| FT-10f | Unit tests: election with 1/2/3 candidates, priority ordering, term advancement, simultaneous election resolution |

**Validation:** Given a set of coordinator-capable nodes, the highest-priority node wins election deterministically. Two simultaneous elections at the same term resolve to the same winner. Stale-term messages are rejected.

### Step FT-11: Election Integration

**Scope:** Wire the election crate into workers so they detect coordinator death and run election.

| Sub-step | Description |
|---|---|
| FT-11a | Add `--election-priority` and `--no-coordinator` CLI flags to worker binary |
| FT-11b | When worker enters `DisconnectedStandby` and is coordinator-capable: start election timer (default: 15 seconds = 3 missed heartbeats) |
| FT-11c | On election timer expiry: open peer connections from manifest, run `ElectionAgent` |
| FT-11d | On receiving `Victory` from another node: connect to new coordinator, send `ReRegister` |
| FT-11e | E2e test: 3 coordinator-capable workers + coordinator, kill coordinator, verify election completes and one worker becomes leader |

**Validation:** Election triggers within `election_timeout` of coordinator death. Exactly one node wins. Non-winners connect to the winner.

### Step FT-12: Coordinator Promotion

**Scope:** The election winner spawns coordinator tasks alongside its existing worker tasks.

| Sub-step | Description |
|---|---|
| FT-12a | Implement `promote_to_coordinator()`: spawn coordinator tokio tasks (TCP listener, heartbeat, HTTP server) alongside existing worker tasks |
| FT-12b | Bind HTTP server on a configurable port (`--coordinator-http-port`, default: 8080) and start accepting client connections |
| FT-12c | Broadcast `Victory` with the new coordinator address so other workers know where to connect |
| FT-12d | E2e test: kill coordinator, verify winning worker starts listening on HTTP port and TCP port |

**Validation:** Promoted worker runs both worker forward-serving and coordinator networking simultaneously. HTTP and TCP listeners are reachable.

### Step FT-12b: State Reconstruction

**Scope:** The newly promoted coordinator collects state from all workers and rebuilds the pipeline.

| Sub-step | Description |
|---|---|
| FT-12b-a | Accept `ReRegister` from all other workers, rebuild `PeerRegistry` from their reported capabilities and assignments |
| FT-12b-b | Register self as a worker in the `PeerRegistry` (the promoted node is both coordinator and worker) |
| FT-12b-c | Determine sequence state from worker cache reports: abort mid-forward sequences (unknown state), preserve idle cached sequences |
| FT-12b-d | Rebuild `DistributedPipeline`, broadcast `ClusterManifest` with new topology, start scheduler loop |
| FT-12b-e | E2e test: kill coordinator, verify new coordinator reconstructs pipeline and serves correct inference output |

**Validation:** Pipeline is rebuilt from worker reports without weight reloading. New coordinator produces correct inference output. Mid-forward sequences are cleanly aborted.

### Step FT-13: Stale Coordinator Handling

**Scope:** When the old coordinator comes back online, it yields to the current leader instead of causing split-brain.

| Sub-step | Description |
|---|---|
| FT-13a | Old coordinator on startup checks if a higher-term leader exists (attempts connection to manifest peers) |
| FT-13b | If higher-term leader found: yield — either join as a worker or shut down |
| FT-13c | If no higher-term leader found: resume as coordinator (normal restart case) |
| FT-13d | Workers reject connections from coordinators with term < their current term |
| FT-13e | E2e test: kill coordinator, election happens, old coordinator restarts, verify it yields to new leader (no split-brain) |

**Validation:** Old coordinator does not compete with current leader. Workers maintain exactly one coordinator connection. No split-brain under any restart ordering.

---

## Implementation Order and Dependencies

```
FT-1  (Worker State Machine)       ← no dependencies, start here
  ↓
FT-2  (Worker Reconnection)        ← requires FT-1 (state machine)
  ↓
FT-3  (Coordinator ReRegister)     ← requires FT-2 (ReRegister message)
  ↓
FT-4  (Forced Rebalance)           ← requires FT-3 (pipeline rebuild infrastructure)
  ↓
FT-4b (Graceful Rebalance)         ← requires FT-4 (forced rebalance as foundation)
  ↓
FT-5  (Crash Recovery)             ← requires FT-4 (forced rebalance)
  ↓
FT-6  (Graceful Leave)             ← requires FT-4b (graceful rebalance)
  ↓
FT-7  (Dynamic Join)               ← requires FT-4b (graceful rebalance)
  ↓
FT-8  (Admin API)                  ← requires FT-4, FT-4b (both rebalance modes)
  ↓
FT-9  (Cluster Manifest)           ← requires FT-1 (worker state machine)
  ↓
FT-10 (Election Protocol)          ← standalone crate, can start after FT-2
  ↓
FT-11 (Election Integration)       ← requires FT-9, FT-10 (manifest + election crate)
  ↓
FT-12  (Coordinator Promotion)     ← requires FT-11 (election integration)
  ↓
FT-12b (State Reconstruction)      ← requires FT-3, FT-12 (ReRegister + promotion)
  ↓
FT-13 (Stale Coordinator)          ← requires FT-12b (reconstruction must work first)
```

**Parallelizable work:**
- FT-5 needs only FT-4 (forced); FT-6, FT-7 need FT-4b (graceful) — so FT-5 can start before FT-4b is done
- FT-6, FT-7, FT-8 can be done in any order after FT-4b
- FT-9 and FT-10 can be done in parallel with FT-5 through FT-8 (independent concerns)

FT-4/FT-4b is the linchpin: crash recovery, graceful leave, dynamic join, and admin API all depend on the rebalancing infrastructure. FT-12/FT-12b is the integration point where election meets coordinator functionality.

---

## Crate Changes

### New Crate: `fracture-election`

Contains the election state machine, term tracking, and peer discovery logic. Depends only on `fracture-protocol` (for message types) and `fracture-core` (for error types). Does not depend on any backend crate.

```
crates/fracture-election/
├── Cargo.toml
└── src/
    ├── lib.rs           # Public API: ElectionAgent
    ├── state_machine.rs # Election states: Follower, Candidate, Leader
    ├── term.rs          # Term tracking, monotonic comparison
    └── manifest.rs      # ClusterManifest storage and versioning
```

### Modified Crates

| Crate | Changes |
|---|---|
| `fracture-protocol` | New message types: ElectionStart, ElectionChallenge, Victory, ClusterManifest, ReRegister, LeaveIntent |
| `fracture-coordinator` | PeerRegistry: `Pending` worker status, `ReRegister` handling, rebalance orchestration |
| `fracture-coordinator` | New: `rebalance.rs` — shared rebalancing sequence (drain/abort → reconfigure → rebuild) |
| `bins/fracture-worker-cuda` | Worker state machine, disconnected standby, reconnection loop, election participation |
| `bins/fracture-coordinator-cuda` | Manifest broadcasting, deferred join logic, `/admin/rebalance` endpoint |

### Wire Protocol Message Type Summary

```
Existing (Phase 4):
  0x01-0x0F  Register through Reconfigure

New (Fault Tolerance):
  0x10  ElectionStart        Node → Peers
  0x11  ElectionChallenge    Node → Candidate
  0x12  Victory              Leader → Peers
  0x13  ClusterManifest      Coordinator → Workers
  0x14  ReRegister           Worker → New Coordinator
  0x15  LeaveIntent          Worker → Coordinator
```

---

## What This Does NOT Include (Deferred)

- **KV cache migration** — On rebalance, caches are freed and sequences must re-prefill. Migrating cache blocks over the network would reduce rebalance cost but adds significant complexity (block serialization, cross-worker transfer, block table remapping).
- **Multi-coordinator active-active** — Only one coordinator is active at a time. Active-active coordination (multiple coordinators sharing load) would require distributed consensus for sequence state.
- **Persistent sequence state** — Sequence state lives in memory. If the entire cluster goes down, all state is lost. Persistent checkpointing to disk would enable full cluster restart recovery.
- **Speculative execution** — Running the same forward pass on multiple workers for redundancy. This would reduce latency impact of worker failure but doubles GPU utilization.
- **Network-level redundancy** — One TCP connection per worker. Redundant connections or alternative transports (QUIC, UDP) are not included.
- **Automatic scale-up/down** — The system handles join/leave but doesn't decide when to add or remove nodes. Integration with orchestrators (Kubernetes, Nomad) for auto-scaling is out of scope.

---

## Success Criteria

1. Worker processes survive coordinator death — no exit, no GPU resource release
2. Workers reconnect to restarted coordinator within 30 seconds and resume serving without weight reload
3. Worker crash with 3+ node cluster: pipeline rebalances to remaining workers within weight-reload time
4. Graceful worker departure: in-flight sequences complete, then pipeline rebalances
5. New worker joins running cluster without aborting active sequences (deferred join)
6. Leader election completes within 20 seconds of coordinator death
7. New elected coordinator reconstructs pipeline from worker state and resumes serving
8. Old coordinator yields to new leader on rejoin (no split-brain)
9. All fault tolerance features validated with e2e tests
