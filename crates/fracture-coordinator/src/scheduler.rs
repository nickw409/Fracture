//! Layer assignment scheduler for distributed pipeline-parallel inference.
//!
//! Pure function: takes worker capabilities and model config, returns
//! optimal layer assignments. No networking or side effects.
//!
//! Three scheduling modes:
//! - **Auto**: Compute-balanced assignment with slow-node pruning
//! - **EqualSplit**: Even layer count per node (for correctness testing)
//! - **Manual**: Explicit layer range overrides

use fracture_core::{FractureError, ModelConfig, Result};
use std::ops::Range;

// ── Input types ─────────────────────────────────────────────────────────

/// Input to the scheduler.
#[derive(Debug, Clone)]
pub struct SchedulerInput {
    pub model_config: ModelConfig,
    pub workers: Vec<WorkerCapabilities>,
    pub coordinator_compute: Option<WorkerCapabilities>,
    pub mode: SchedulingMode,
    pub max_seq_len: usize,
    /// Estimated network hop latency in ms. Default 2.0 for cross-machine,
    /// 0.1 for localhost.
    pub hop_latency_ms: f32,
}

/// Measured capabilities of a worker node.
#[derive(Debug, Clone)]
pub struct WorkerCapabilities {
    pub node_id: String,
    pub gpu_model: String,
    pub gpu_memory_available: usize,
    pub compute_capability: (u32, u32),
    /// Measured: ms per single-layer forward pass (N=1, decode).
    pub decode_ms_per_layer: f32,
    /// Measured: ms per single-layer forward pass (N=128, prefill).
    pub prefill_ms_per_layer_128: f32,
}

/// Maximum plausible ms/layer for calibration. Anything above this
/// indicates a broken benchmark or severely misconfigured GPU.
const MAX_PLAUSIBLE_MS_PER_LAYER: f32 = 500.0;

impl WorkerCapabilities {
    /// Validate that calibration results are plausible.
    /// Returns an error if values are zero, negative, or absurdly large.
    pub fn validate_calibration(&self) -> Result<()> {
        if self.decode_ms_per_layer.is_nan() || self.decode_ms_per_layer <= 0.0 {
            return Err(FractureError::Pipeline(format!(
                "worker '{}': implausible decode_ms_per_layer={} (must be > 0)",
                self.node_id, self.decode_ms_per_layer
            )));
        }
        if self.prefill_ms_per_layer_128.is_nan() || self.prefill_ms_per_layer_128 <= 0.0 {
            return Err(FractureError::Pipeline(format!(
                "worker '{}': implausible prefill_ms_per_layer_128={} (must be > 0)",
                self.node_id, self.prefill_ms_per_layer_128
            )));
        }
        if self.decode_ms_per_layer > MAX_PLAUSIBLE_MS_PER_LAYER {
            return Err(FractureError::Pipeline(format!(
                "worker '{}': decode_ms_per_layer={:.1} exceeds maximum plausible value ({:.0}ms)",
                self.node_id, self.decode_ms_per_layer, MAX_PLAUSIBLE_MS_PER_LAYER
            )));
        }
        if self.prefill_ms_per_layer_128 > MAX_PLAUSIBLE_MS_PER_LAYER {
            return Err(FractureError::Pipeline(format!(
                "worker '{}': prefill_ms_per_layer_128={:.1} exceeds maximum plausible value ({:.0}ms)",
                self.node_id, self.prefill_ms_per_layer_128, MAX_PLAUSIBLE_MS_PER_LAYER
            )));
        }
        Ok(())
    }
}

/// Scheduling mode.
#[derive(Debug, Clone)]
pub enum SchedulingMode {
    /// Compute-balanced assignment with slow-node pruning.
    Auto,
    /// Force equal layer count per node (for correctness testing).
    EqualSplit,
    /// Explicit manual assignment.
    Manual(Vec<ManualAssignment>),
}

/// Explicit layer range assignment for Manual mode.
#[derive(Debug, Clone)]
pub struct ManualAssignment {
    pub node_id: String,
    pub layer_range: Range<usize>,
}

// ── Output types ────────────────────────────────────────────────────────

/// Pipeline role for a node.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NodeRole {
    /// First node: handles embedding.
    Head,
    /// Interior node: activations in, activations out.
    Middle,
    /// Last node: handles final norm + LM head.
    Tail,
}

/// A single node's layer assignment.
#[derive(Debug, Clone)]
pub struct LayerAssignment {
    pub node_id: String,
    pub layer_range: Range<usize>,
    pub role: NodeRole,
    /// Predicted per-token decode time for this node (ms).
    pub expected_decode_ms: f32,
    /// Estimated GPU memory for weights (GB).
    pub weight_memory_gb: f32,
    /// Estimated GPU memory for KV cache at max_seq_len (GB).
    pub cache_memory_gb: f32,
}

/// Reason a node was excluded from the pipeline.
#[derive(Debug, Clone)]
pub enum ExclusionReason {
    /// Removing this node made the pipeline faster.
    SlowsDownPipeline {
        latency_with_ms: f32,
        latency_without_ms: f32,
    },
}

/// A node excluded from the pipeline.
#[derive(Debug, Clone)]
pub struct ExcludedNode {
    pub node_id: String,
    pub reason: ExclusionReason,
}

/// Scheduler output.
#[derive(Debug, Clone)]
pub struct SchedulerResult {
    pub assignments: Vec<LayerAssignment>,
    pub excluded_nodes: Vec<ExcludedNode>,
    /// Predicted total per-token decode latency (ms), including network hops.
    pub pipeline_decode_ms: f32,
    /// max(per_node_time) / min(per_node_time). 1.0 = perfectly balanced.
    pub imbalance_ratio: f32,
    /// Node ID of the bottleneck (slowest) node.
    pub bottleneck_node: String,
}

// ── Memory estimation ───────────────────────────────────────────────────

/// Estimate per-layer weight memory in bytes (FP16).
fn weight_memory_per_layer(config: &ModelConfig) -> usize {
    let h = config.hidden_size;
    let kv_dim = config.num_kv_heads * config.head_dim;
    let inter = config.intermediate_size;
    // q_proj + k_proj + v_proj + o_proj + gate_proj + up_proj + down_proj (all FP16)
    let proj_bytes = (h * h + h * kv_dim + h * kv_dim + h * h + h * inter + h * inter + inter * h) * 2;
    // attn_norm + ffn_norm (FP16)
    let norm_bytes = h * 2 * 2;
    proj_bytes + norm_bytes
}

/// Estimate per-layer KV cache memory in bytes (FP16) at a given max_seq_len.
fn cache_memory_per_layer(config: &ModelConfig, max_seq_len: usize) -> usize {
    let kv_dim = config.num_kv_heads * config.head_dim;
    // K + V: [max_seq_len, kv_dim] * 2 bytes each
    max_seq_len * kv_dim * 2 * 2
}

/// Head node overhead: token embedding table.
fn head_overhead(config: &ModelConfig) -> usize {
    config.vocab_size * config.hidden_size * 2
}

/// Tail node overhead: output norm + LM head.
fn tail_overhead(config: &ModelConfig) -> usize {
    config.hidden_size * 2 + config.vocab_size * config.hidden_size * 2
}

/// Maximum number of layers a node can hold given its available memory and role.
fn max_layers_for_node(
    available_memory: usize,
    per_layer_total: usize,
    role_overhead: usize,
) -> usize {
    available_memory
        .saturating_sub(role_overhead)
        .checked_div(per_layer_total)
        .unwrap_or(0)
}

// ── Core algorithm ──────────────────────────────────────────────────────

/// Run the scheduler.
pub fn schedule(input: &SchedulerInput) -> Result<SchedulerResult> {
    // Merge coordinator compute (if any) as the first node
    let mut candidates: Vec<WorkerCapabilities> = Vec::new();
    if let Some(ref coord) = input.coordinator_compute {
        candidates.push(coord.clone());
    }
    candidates.extend(input.workers.iter().cloned());

    if candidates.is_empty() {
        return Err(FractureError::Pipeline(
            "no workers available for scheduling".into(),
        ));
    }

    // Validate calibration plausibility for all candidates
    for c in &candidates {
        c.validate_calibration()?;
    }

    match &input.mode {
        SchedulingMode::Auto => schedule_auto(input, candidates),
        SchedulingMode::EqualSplit => schedule_equal(input, candidates),
        SchedulingMode::Manual(assignments) => schedule_manual(input, candidates, assignments),
    }
}

fn schedule_auto(
    input: &SchedulerInput,
    mut candidates: Vec<WorkerCapabilities>,
) -> Result<SchedulerResult> {
    let num_layers = input.model_config.num_layers;
    let per_layer_weight = weight_memory_per_layer(&input.model_config);
    let per_layer_cache = cache_memory_per_layer(&input.model_config, input.max_seq_len);
    let per_layer_total = per_layer_weight + per_layer_cache;

    let mut excluded = Vec::new();

    // Step 0: Prune slow nodes
    loop {
        if candidates.len() <= 1 {
            break;
        }

        // Find slowest node
        let slowest_idx = candidates
            .iter()
            .enumerate()
            .max_by(|a, b| {
                a.1.decode_ms_per_layer
                    .partial_cmp(&b.1.decode_ms_per_layer)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(i, _)| i)
            .unwrap();

        let latency_with = estimate_pipeline_latency(
            &candidates,
            num_layers,
            per_layer_total,
            &input.model_config,
            input.hop_latency_ms,
        );

        let mut without = candidates.clone();
        let removed = without.remove(slowest_idx);

        // Check if model fits without this node
        if !model_fits(&without, num_layers, per_layer_total, &input.model_config) {
            break; // Keep the node — memory is needed
        }

        let latency_without = estimate_pipeline_latency(
            &without,
            num_layers,
            per_layer_total,
            &input.model_config,
            input.hop_latency_ms,
        );

        if latency_without < latency_with {
            excluded.push(ExcludedNode {
                node_id: removed.node_id.clone(),
                reason: ExclusionReason::SlowsDownPipeline {
                    latency_with_ms: latency_with,
                    latency_without_ms: latency_without,
                },
            });
            candidates = without;
        } else {
            break;
        }
    }

    // Steps 1-6: Compute-balanced assignment
    let total_speed: f32 = candidates.iter().map(|w| 1.0 / w.decode_ms_per_layer).sum();
    let mut layer_counts: Vec<usize> = candidates
        .iter()
        .map(|w| {
            let speed = 1.0 / w.decode_ms_per_layer;
            (speed / total_speed * num_layers as f32).round() as usize
        })
        .collect();

    // Clamp to memory ceilings
    for (i, w) in candidates.iter().enumerate() {
        let overhead = role_overhead(i, candidates.len(), &input.model_config);
        let max = max_layers_for_node(w.gpu_memory_available, per_layer_total, overhead);
        layer_counts[i] = layer_counts[i].min(max);
    }

    // Redistribute to match total
    redistribute_layers(&mut layer_counts, &candidates, num_layers, per_layer_total, &input.model_config)?;

    build_result(input, &candidates, &layer_counts, excluded)
}

fn schedule_equal(
    input: &SchedulerInput,
    candidates: Vec<WorkerCapabilities>,
) -> Result<SchedulerResult> {
    let num_layers = input.model_config.num_layers;
    let n = candidates.len();
    let base = num_layers / n;
    let remainder = num_layers % n;

    let per_layer_total =
        weight_memory_per_layer(&input.model_config) + cache_memory_per_layer(&input.model_config, input.max_seq_len);

    // Give remainder layers to nodes with most available memory
    let mut memory_order: Vec<usize> = (0..n).collect();
    memory_order.sort_by(|&a, &b| {
        candidates[b]
            .gpu_memory_available
            .cmp(&candidates[a].gpu_memory_available)
    });

    let mut layer_counts = vec![base; n];
    for &idx in memory_order.iter().take(remainder) {
        layer_counts[idx] += 1;
    }

    // Validate memory fits
    for (i, w) in candidates.iter().enumerate() {
        let overhead = role_overhead(i, n, &input.model_config);
        let max = max_layers_for_node(w.gpu_memory_available, per_layer_total, overhead);
        if layer_counts[i] > max {
            return Err(FractureError::Pipeline(format!(
                "node '{}' cannot hold {} layers (max {}): {:.1} GB available, needs {:.1} GB",
                w.node_id,
                layer_counts[i],
                max,
                w.gpu_memory_available as f64 / 1e9,
                (layer_counts[i] * per_layer_total + overhead) as f64 / 1e9,
            )));
        }
    }

    build_result(input, &candidates, &layer_counts, Vec::new())
}

fn schedule_manual(
    input: &SchedulerInput,
    candidates: Vec<WorkerCapabilities>,
    assignments: &[ManualAssignment],
) -> Result<SchedulerResult> {
    let num_layers = input.model_config.num_layers;
    let per_layer_total =
        weight_memory_per_layer(&input.model_config) + cache_memory_per_layer(&input.model_config, input.max_seq_len);

    // Validate no duplicate node IDs
    {
        let mut seen = std::collections::HashSet::new();
        for a in assignments {
            if !seen.insert(&a.node_id) {
                return Err(FractureError::Pipeline(format!(
                    "manual assignment has duplicate node_id '{}'",
                    a.node_id
                )));
            }
        }
    }

    // Validate coverage: assignments must cover [0, num_layers) exactly
    let mut covered = vec![false; num_layers];
    for a in assignments {
        if a.layer_range.end > num_layers {
            return Err(FractureError::Pipeline(format!(
                "manual assignment for '{}' exceeds model layers: {:?} > {}",
                a.node_id, a.layer_range, num_layers
            )));
        }
        for l in a.layer_range.clone() {
            if covered[l] {
                return Err(FractureError::Pipeline(format!(
                    "manual assignment overlap at layer {l}"
                )));
            }
            covered[l] = true;
        }
    }
    if covered.iter().any(|&c| !c) {
        let missing: Vec<usize> = covered
            .iter()
            .enumerate()
            .filter(|&(_, c)| !c)
            .map(|(i, _)| i)
            .collect();
        return Err(FractureError::Pipeline(format!(
            "manual assignments have gaps at layers: {missing:?}"
        )));
    }

    // Build layer_counts in assignment order, matching to candidates by node_id
    let mut ordered_candidates = Vec::new();
    let mut layer_counts = Vec::new();

    for a in assignments {
        let worker = candidates
            .iter()
            .find(|w| w.node_id == a.node_id)
            .ok_or_else(|| {
                FractureError::Pipeline(format!(
                    "manual assignment references unknown node '{}'",
                    a.node_id
                ))
            })?;

        let i = ordered_candidates.len();
        let n = assignments.len();
        let overhead = role_overhead(i, n, &input.model_config);
        let max = max_layers_for_node(worker.gpu_memory_available, per_layer_total, overhead);
        let count = a.layer_range.len();
        if count > max {
            return Err(FractureError::Pipeline(format!(
                "node '{}' cannot hold {} layers (max {})",
                a.node_id, count, max
            )));
        }

        ordered_candidates.push(worker.clone());
        layer_counts.push(count);
    }

    build_result(input, &ordered_candidates, &layer_counts, Vec::new())
}

// ── Helpers ─────────────────────────────────────────────────────────────

/// Estimate the pipeline latency for a set of candidates with compute-balanced assignment.
fn estimate_pipeline_latency(
    candidates: &[WorkerCapabilities],
    num_layers: usize,
    per_layer_total: usize,
    config: &ModelConfig,
    hop_latency_ms: f32,
) -> f32 {
    let total_speed: f32 = candidates.iter().map(|w| 1.0 / w.decode_ms_per_layer).sum();
    let mut max_node_time: f32 = 0.0;

    for (i, w) in candidates.iter().enumerate() {
        let speed = 1.0 / w.decode_ms_per_layer;
        let ideal = (speed / total_speed * num_layers as f32).round() as usize;
        let overhead = role_overhead(i, candidates.len(), config);
        let max = max_layers_for_node(w.gpu_memory_available, per_layer_total, overhead);
        let layers = ideal.min(max).min(num_layers);
        let node_time = layers as f32 * w.decode_ms_per_layer;
        max_node_time = max_node_time.max(node_time);
    }

    let num_hops = if candidates.len() > 1 {
        (candidates.len() - 1) as f32
    } else {
        0.0
    };

    max_node_time + num_hops * hop_latency_ms
}

/// Check if the model fits across the given set of nodes.
fn model_fits(
    candidates: &[WorkerCapabilities],
    num_layers: usize,
    per_layer_total: usize,
    config: &ModelConfig,
) -> bool {
    let total_available: usize = candidates
        .iter()
        .enumerate()
        .map(|(i, w)| {
            let overhead = role_overhead(i, candidates.len(), config);
            w.gpu_memory_available.saturating_sub(overhead)
        })
        .sum();
    total_available >= num_layers * per_layer_total
}

/// Role overhead for the i-th node in a pipeline of n nodes.
fn role_overhead(i: usize, n: usize, config: &ModelConfig) -> usize {
    match (i == 0, i == n - 1) {
        (true, true) => head_overhead(config) + tail_overhead(config), // single node: both
        (true, false) => head_overhead(config),                        // head
        (false, true) => tail_overhead(config),                        // tail
        (false, false) => 0,                                           // middle
    }
}

/// Redistribute layers so the total matches num_layers exactly.
fn redistribute_layers(
    layer_counts: &mut [usize],
    candidates: &[WorkerCapabilities],
    num_layers: usize,
    per_layer_total: usize,
    config: &ModelConfig,
) -> Result<()> {
    let mut total: usize = layer_counts.iter().sum();

    // Add layers to fastest nodes with capacity
    while total < num_layers {
        let mut best = None;
        let mut best_speed = f32::MIN;
        for (i, w) in candidates.iter().enumerate() {
            let overhead = role_overhead(i, candidates.len(), config);
            let max = max_layers_for_node(w.gpu_memory_available, per_layer_total, overhead);
            if layer_counts[i] < max {
                let speed = 1.0 / w.decode_ms_per_layer;
                if speed > best_speed {
                    best_speed = speed;
                    best = Some(i);
                }
            }
        }
        match best {
            Some(i) => {
                layer_counts[i] += 1;
                total += 1;
            }
            None => {
                return Err(FractureError::Pipeline(format!(
                    "insufficient total capacity: need {num_layers} layers, can fit {total}"
                )));
            }
        }
    }

    // Remove layers from slowest nodes if over-assigned
    while total > num_layers {
        let mut worst = None;
        let mut worst_speed = f32::MAX;
        for (i, w) in candidates.iter().enumerate() {
            if layer_counts[i] > 0 {
                let speed = 1.0 / w.decode_ms_per_layer;
                if speed < worst_speed {
                    worst_speed = speed;
                    worst = Some(i);
                }
            }
        }
        match worst {
            Some(i) => {
                layer_counts[i] -= 1;
                total -= 1;
            }
            None => break,
        }
    }

    if total != num_layers {
        return Err(FractureError::Pipeline(format!(
            "failed to assign all layers: assigned {total}, need {num_layers}"
        )));
    }

    Ok(())
}

/// Build the final SchedulerResult from layer counts.
fn build_result(
    input: &SchedulerInput,
    candidates: &[WorkerCapabilities],
    layer_counts: &[usize],
    excluded: Vec<ExcludedNode>,
) -> Result<SchedulerResult> {
    let per_layer_weight = weight_memory_per_layer(&input.model_config);
    let per_layer_cache = cache_memory_per_layer(&input.model_config, input.max_seq_len);
    let n = candidates.len();

    let mut assignments = Vec::with_capacity(n);
    let mut start = 0;
    let mut per_node_times = Vec::with_capacity(n);

    for (i, (w, &count)) in candidates.iter().zip(layer_counts.iter()).enumerate() {
        let end = start + count;
        let role = if i == 0 {
            NodeRole::Head
        } else if i == n - 1 {
            NodeRole::Tail
        } else {
            NodeRole::Middle
        };

        let decode_ms = count as f32 * w.decode_ms_per_layer;
        per_node_times.push(decode_ms);

        assignments.push(LayerAssignment {
            node_id: w.node_id.clone(),
            layer_range: start..end,
            role,
            expected_decode_ms: decode_ms,
            weight_memory_gb: (count * per_layer_weight) as f32 / 1e9,
            cache_memory_gb: (count * per_layer_cache) as f32 / 1e9,
        });

        start = end;
    }

    let max_time = per_node_times.iter().cloned().fold(f32::MIN, f32::max);
    let min_time = per_node_times
        .iter()
        .cloned()
        .filter(|&t| t > 0.0)
        .fold(f32::MAX, f32::min);
    let imbalance_ratio = if min_time > 0.0 {
        max_time / min_time
    } else {
        1.0
    };

    let bottleneck_idx = per_node_times
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0);

    let num_hops = if n > 1 { (n - 1) as f32 } else { 0.0 };
    let pipeline_decode_ms = max_time + num_hops * input.hop_latency_ms;

    Ok(SchedulerResult {
        assignments,
        excluded_nodes: excluded,
        pipeline_decode_ms,
        imbalance_ratio,
        bottleneck_node: candidates[bottleneck_idx].node_id.clone(),
    })
}

#[cfg(test)]
mod scheduler_tests;
