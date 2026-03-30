use super::*;

fn llama_8b_config() -> ModelConfig {
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

fn worker(id: &str, memory_gb: f64, decode_ms: f32) -> WorkerCapabilities {
    WorkerCapabilities {
        node_id: id.into(),
        gpu_model: "Test GPU".into(),
        gpu_memory_available: (memory_gb * 1e9) as usize,
        compute_capability: (8, 0),
        decode_ms_per_layer: decode_ms,
        prefill_ms_per_layer_128: decode_ms * 3.0,
    }
}

#[test]
fn test_auto_homogeneous_two_nodes() {
    let input = SchedulerInput {
        model_config: llama_8b_config(),
        workers: vec![worker("a", 22.0, 1.0), worker("b", 22.0, 1.0)],
        coordinator_compute: None,
        mode: SchedulingMode::Auto,
        max_seq_len: 4096,
        hop_latency_ms: 2.0,
    };
    let result = schedule(&input).unwrap();

    assert_eq!(result.assignments.len(), 2);
    assert_eq!(result.assignments[0].layer_range, 0..16);
    assert_eq!(result.assignments[1].layer_range, 16..32);
    assert_eq!(result.assignments[0].role, NodeRole::Head);
    assert_eq!(result.assignments[1].role, NodeRole::Tail);
    assert!((result.imbalance_ratio - 1.0).abs() < 0.01);
}

#[test]
fn test_auto_heterogeneous_faster_gets_more() {
    let input = SchedulerInput {
        model_config: llama_8b_config(),
        workers: vec![
            worker("fast", 30.0, 0.5),
            worker("slow", 22.0, 1.1),
        ],
        coordinator_compute: None,
        mode: SchedulingMode::Auto,
        max_seq_len: 4096,
        hop_latency_ms: 2.0,
    };
    let result = schedule(&input).unwrap();

    // Fast node should get more layers
    let fast = &result.assignments[0];
    let slow = &result.assignments[1];
    assert!(
        fast.layer_range.len() > slow.layer_range.len(),
        "fast node should get more layers: {} vs {}",
        fast.layer_range.len(),
        slow.layer_range.len()
    );

    // Total should be 32
    assert_eq!(
        fast.layer_range.len() + slow.layer_range.len(),
        32
    );

    // Imbalance should be close to 1.0
    assert!(
        result.imbalance_ratio < 1.5,
        "imbalance ratio {} should be < 1.5",
        result.imbalance_ratio
    );
}

#[test]
fn test_auto_prunes_very_slow_node() {
    let input = SchedulerInput {
        model_config: llama_8b_config(),
        workers: vec![
            worker("fast1", 30.0, 0.5),
            worker("fast2", 30.0, 0.5),
            worker("turtle", 22.0, 10.0), // 20x slower
        ],
        coordinator_compute: None,
        mode: SchedulingMode::Auto,
        max_seq_len: 4096,
        hop_latency_ms: 2.0,
    };
    let result = schedule(&input).unwrap();

    // Turtle should be excluded (2 fast nodes can hold the model)
    assert_eq!(result.assignments.len(), 2);
    assert_eq!(result.excluded_nodes.len(), 1);
    assert_eq!(result.excluded_nodes[0].node_id, "turtle");
}

#[test]
fn test_auto_keeps_slow_node_when_memory_needed() {
    // Give fast nodes barely enough memory for ~10 layers each.
    // Model needs 32 layers, so the slow node is required.
    let per_layer = weight_memory_per_layer(&llama_8b_config())
        + cache_memory_per_layer(&llama_8b_config(), 4096);
    let small_mem = (per_layer * 11 + head_overhead(&llama_8b_config())) as f64 / 1e9;

    let input = SchedulerInput {
        model_config: llama_8b_config(),
        workers: vec![
            worker("fast1", small_mem, 0.5),
            worker("fast2", small_mem, 0.5),
            worker("slow", 30.0, 5.0),
        ],
        coordinator_compute: None,
        mode: SchedulingMode::Auto,
        max_seq_len: 4096,
        hop_latency_ms: 2.0,
    };
    let result = schedule(&input).unwrap();

    // All 3 nodes should be assigned (slow node kept for memory)
    assert_eq!(result.assignments.len(), 3);
    assert!(result.excluded_nodes.is_empty());
}

#[test]
fn test_auto_memory_ceiling_clamps() {
    // Fast node has limited memory, slow node has plenty
    let per_layer = weight_memory_per_layer(&llama_8b_config())
        + cache_memory_per_layer(&llama_8b_config(), 4096);
    let small_mem = (per_layer * 10 + head_overhead(&llama_8b_config())) as f64 / 1e9;

    let input = SchedulerInput {
        model_config: llama_8b_config(),
        workers: vec![
            worker("fast_small", small_mem, 0.5),
            worker("slow_big", 30.0, 1.0),
        ],
        coordinator_compute: None,
        mode: SchedulingMode::Auto,
        max_seq_len: 4096,
        hop_latency_ms: 2.0,
    };
    let result = schedule(&input).unwrap();

    // Fast node should be clamped to ~10 layers despite wanting more
    let fast = &result.assignments[0];
    assert!(
        fast.layer_range.len() <= 10,
        "fast node should be clamped: got {}",
        fast.layer_range.len()
    );
    assert_eq!(
        result.assignments[0].layer_range.len() + result.assignments[1].layer_range.len(),
        32
    );
}

#[test]
fn test_auto_insufficient_memory() {
    let per_layer = weight_memory_per_layer(&llama_8b_config())
        + cache_memory_per_layer(&llama_8b_config(), 4096);
    let tiny_mem = (per_layer * 5) as f64 / 1e9;

    let input = SchedulerInput {
        model_config: llama_8b_config(),
        workers: vec![worker("small1", tiny_mem, 1.0), worker("small2", tiny_mem, 1.0)],
        coordinator_compute: None,
        mode: SchedulingMode::Auto,
        max_seq_len: 4096,
        hop_latency_ms: 2.0,
    };
    let result = schedule(&input);
    assert!(result.is_err(), "should fail: insufficient total memory");
}

#[test]
fn test_equal_split_two_nodes() {
    let input = SchedulerInput {
        model_config: llama_8b_config(),
        workers: vec![worker("a", 22.0, 0.5), worker("b", 22.0, 1.1)],
        coordinator_compute: None,
        mode: SchedulingMode::EqualSplit,
        max_seq_len: 4096,
        hop_latency_ms: 2.0,
    };
    let result = schedule(&input).unwrap();

    assert_eq!(result.assignments.len(), 2);
    assert_eq!(result.assignments[0].layer_range.len(), 16);
    assert_eq!(result.assignments[1].layer_range.len(), 16);
}

#[test]
fn test_equal_split_three_nodes_remainder() {
    // 32 layers / 3 = 10 each + 2 remainder
    let input = SchedulerInput {
        model_config: llama_8b_config(),
        workers: vec![
            worker("a", 22.0, 1.0),
            worker("b", 30.0, 1.0),
            worker("c", 25.0, 1.0),
        ],
        coordinator_compute: None,
        mode: SchedulingMode::EqualSplit,
        max_seq_len: 4096,
        hop_latency_ms: 2.0,
    };
    let result = schedule(&input).unwrap();

    let total: usize = result.assignments.iter().map(|a| a.layer_range.len()).sum();
    assert_eq!(total, 32);

    // Each node gets 10 or 11 layers
    for a in &result.assignments {
        assert!(
            a.layer_range.len() == 10 || a.layer_range.len() == 11,
            "expected 10 or 11, got {}",
            a.layer_range.len()
        );
    }
}

#[test]
fn test_manual_valid() {
    let input = SchedulerInput {
        model_config: llama_8b_config(),
        workers: vec![worker("a", 22.0, 1.0), worker("b", 22.0, 1.0)],
        coordinator_compute: None,
        mode: SchedulingMode::Manual(vec![
            ManualAssignment {
                node_id: "a".into(),
                layer_range: 0..20,
            },
            ManualAssignment {
                node_id: "b".into(),
                layer_range: 20..32,
            },
        ]),
        max_seq_len: 4096,
        hop_latency_ms: 2.0,
    };
    let result = schedule(&input).unwrap();

    assert_eq!(result.assignments.len(), 2);
    assert_eq!(result.assignments[0].layer_range, 0..20);
    assert_eq!(result.assignments[1].layer_range, 20..32);
}

#[test]
fn test_manual_gap() {
    let input = SchedulerInput {
        model_config: llama_8b_config(),
        workers: vec![worker("a", 22.0, 1.0), worker("b", 22.0, 1.0)],
        coordinator_compute: None,
        mode: SchedulingMode::Manual(vec![
            ManualAssignment {
                node_id: "a".into(),
                layer_range: 0..10,
            },
            ManualAssignment {
                node_id: "b".into(),
                layer_range: 20..32, // gap at 10..20
            },
        ]),
        max_seq_len: 4096,
        hop_latency_ms: 2.0,
    };
    assert!(schedule(&input).is_err());
}

#[test]
fn test_manual_overlap() {
    let input = SchedulerInput {
        model_config: llama_8b_config(),
        workers: vec![worker("a", 22.0, 1.0), worker("b", 22.0, 1.0)],
        coordinator_compute: None,
        mode: SchedulingMode::Manual(vec![
            ManualAssignment {
                node_id: "a".into(),
                layer_range: 0..20,
            },
            ManualAssignment {
                node_id: "b".into(),
                layer_range: 15..32, // overlap at 15..20
            },
        ]),
        max_seq_len: 4096,
        hop_latency_ms: 2.0,
    };
    assert!(schedule(&input).is_err());
}

#[test]
fn test_manual_unknown_node() {
    let input = SchedulerInput {
        model_config: llama_8b_config(),
        workers: vec![worker("a", 22.0, 1.0)],
        coordinator_compute: None,
        mode: SchedulingMode::Manual(vec![
            ManualAssignment {
                node_id: "a".into(),
                layer_range: 0..16,
            },
            ManualAssignment {
                node_id: "nonexistent".into(),
                layer_range: 16..32,
            },
        ]),
        max_seq_len: 4096,
        hop_latency_ms: 2.0,
    };
    assert!(schedule(&input).is_err());
}

#[test]
fn test_no_workers() {
    let input = SchedulerInput {
        model_config: llama_8b_config(),
        workers: vec![],
        coordinator_compute: None,
        mode: SchedulingMode::Auto,
        max_seq_len: 4096,
        hop_latency_ms: 2.0,
    };
    assert!(schedule(&input).is_err());
}

#[test]
fn test_single_node() {
    let input = SchedulerInput {
        model_config: llama_8b_config(),
        workers: vec![worker("solo", 30.0, 1.0)],
        coordinator_compute: None,
        mode: SchedulingMode::Auto,
        max_seq_len: 4096,
        hop_latency_ms: 2.0,
    };
    let result = schedule(&input).unwrap();

    assert_eq!(result.assignments.len(), 1);
    assert_eq!(result.assignments[0].layer_range, 0..32);
    assert_eq!(result.pipeline_decode_ms, 32.0); // no hops
}

#[test]
fn test_coordinator_as_worker() {
    let input = SchedulerInput {
        model_config: llama_8b_config(),
        workers: vec![worker("remote", 22.0, 1.0)],
        coordinator_compute: Some(worker("coord", 30.0, 0.5)),
        mode: SchedulingMode::Auto,
        max_seq_len: 4096,
        hop_latency_ms: 2.0,
    };
    let result = schedule(&input).unwrap();

    assert_eq!(result.assignments.len(), 2);
    // Coordinator (faster) should be first (head) and get more layers
    assert_eq!(result.assignments[0].node_id, "coord");
    assert!(result.assignments[0].layer_range.len() > result.assignments[1].layer_range.len());
}

#[test]
fn test_imbalance_reporting() {
    let input = SchedulerInput {
        model_config: llama_8b_config(),
        workers: vec![worker("a", 22.0, 0.5), worker("b", 22.0, 1.0)],
        coordinator_compute: None,
        mode: SchedulingMode::EqualSplit, // forces imbalance
        max_seq_len: 4096,
        hop_latency_ms: 2.0,
    };
    let result = schedule(&input).unwrap();

    // Node b is 2x slower, with equal layers imbalance should be ~2.0
    assert!(
        result.imbalance_ratio > 1.5,
        "equal split with 2x speed diff should show imbalance: {}",
        result.imbalance_ratio
    );
    assert_eq!(result.bottleneck_node, "b");
}

#[test]
fn test_contiguous_ranges() {
    let input = SchedulerInput {
        model_config: llama_8b_config(),
        workers: vec![
            worker("a", 22.0, 1.0),
            worker("b", 22.0, 1.0),
            worker("c", 22.0, 1.0),
        ],
        coordinator_compute: None,
        mode: SchedulingMode::EqualSplit,
        max_seq_len: 4096,
        hop_latency_ms: 2.0,
    };
    let result = schedule(&input).unwrap();

    // Verify ranges are contiguous
    let mut expected_start = 0;
    for a in &result.assignments {
        assert_eq!(a.layer_range.start, expected_start);
        expected_start = a.layer_range.end;
    }
    assert_eq!(expected_start, 32);
}

#[test]
fn test_roles_assigned_correctly() {
    let input = SchedulerInput {
        model_config: llama_8b_config(),
        workers: vec![
            worker("a", 22.0, 1.0),
            worker("b", 22.0, 1.0),
            worker("c", 22.0, 1.0),
        ],
        coordinator_compute: None,
        mode: SchedulingMode::EqualSplit,
        max_seq_len: 4096,
        hop_latency_ms: 2.0,
    };
    let result = schedule(&input).unwrap();

    assert_eq!(result.assignments[0].role, NodeRole::Head);
    assert_eq!(result.assignments[1].role, NodeRole::Middle);
    assert_eq!(result.assignments[2].role, NodeRole::Tail);
}

#[test]
fn test_pruning_exclusion_reason_has_latency_values() {
    let input = SchedulerInput {
        model_config: llama_8b_config(),
        workers: vec![
            worker("fast1", 30.0, 0.5),
            worker("fast2", 30.0, 0.5),
            worker("turtle", 22.0, 10.0),
        ],
        coordinator_compute: None,
        mode: SchedulingMode::Auto,
        max_seq_len: 4096,
        hop_latency_ms: 2.0,
    };
    let result = schedule(&input).unwrap();

    assert_eq!(result.excluded_nodes.len(), 1);
    match &result.excluded_nodes[0].reason {
        ExclusionReason::SlowsDownPipeline {
            latency_with_ms,
            latency_without_ms,
        } => {
            assert!(*latency_with_ms > 0.0);
            assert!(*latency_without_ms > 0.0);
            assert!(latency_without_ms < latency_with_ms);
        }
    }
}

#[test]
fn test_equal_split_remainder_goes_to_largest_memory() {
    // 32 layers / 3 = 10 each + 2 remainder
    // Node "big" has most memory, should get a remainder layer
    let input = SchedulerInput {
        model_config: llama_8b_config(),
        workers: vec![
            worker("small", 20.0, 1.0),
            worker("big", 30.0, 1.0),
            worker("medium", 25.0, 1.0),
        ],
        coordinator_compute: None,
        mode: SchedulingMode::EqualSplit,
        max_seq_len: 4096,
        hop_latency_ms: 2.0,
    };
    let result = schedule(&input).unwrap();

    // "big" (30 GB) should have 11 layers (got a remainder)
    let big = result.assignments.iter().find(|a| a.node_id == "big").unwrap();
    assert_eq!(big.layer_range.len(), 11);
}

#[test]
fn test_equal_split_memory_too_small_fails() {
    let per_layer = weight_memory_per_layer(&llama_8b_config())
        + cache_memory_per_layer(&llama_8b_config(), 4096);
    // Node can hold only 5 layers but equal split wants 16
    let tiny_mem = (per_layer * 5 + head_overhead(&llama_8b_config())) as f64 / 1e9;

    let input = SchedulerInput {
        model_config: llama_8b_config(),
        workers: vec![worker("small", tiny_mem, 1.0), worker("big", 30.0, 1.0)],
        coordinator_compute: None,
        mode: SchedulingMode::EqualSplit,
        max_seq_len: 4096,
        hop_latency_ms: 2.0,
    };
    let result = schedule(&input);
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("cannot hold"));
}

#[test]
fn test_manual_mode_memory_too_small_fails() {
    let per_layer = weight_memory_per_layer(&llama_8b_config())
        + cache_memory_per_layer(&llama_8b_config(), 4096);
    let tiny_mem = (per_layer * 5 + head_overhead(&llama_8b_config())) as f64 / 1e9;

    let input = SchedulerInput {
        model_config: llama_8b_config(),
        workers: vec![worker("small", tiny_mem, 1.0), worker("big", 30.0, 1.0)],
        coordinator_compute: None,
        mode: SchedulingMode::Manual(vec![
            ManualAssignment { node_id: "small".into(), layer_range: 0..20 },
            ManualAssignment { node_id: "big".into(), layer_range: 20..32 },
        ]),
        max_seq_len: 4096,
        hop_latency_ms: 2.0,
    };
    let result = schedule(&input);
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("cannot hold"));
}

#[test]
fn test_memory_estimation_sanity() {
    let config = llama_8b_config();
    let w_per_layer = weight_memory_per_layer(&config);
    let c_per_layer = cache_memory_per_layer(&config, 4096);
    let h_overhead = head_overhead(&config);
    let t_overhead = tail_overhead(&config);

    // Llama 8B: ~440 MB per layer weights (FP16)
    assert!(w_per_layer > 400_000_000, "weight/layer too small: {w_per_layer}");
    assert!(w_per_layer < 500_000_000, "weight/layer too large: {w_per_layer}");

    // KV cache at 4096 seq len: ~8 MB per layer
    assert!(c_per_layer > 4_000_000, "cache/layer too small: {c_per_layer}");
    assert!(c_per_layer < 20_000_000, "cache/layer too large: {c_per_layer}");

    // Head overhead: embedding table ~1 GB
    assert!(h_overhead > 500_000_000, "head overhead too small: {h_overhead}");
    assert!(h_overhead < 2_000_000_000, "head overhead too large: {h_overhead}");

    // Tail overhead: LM head ~1 GB + norm
    assert!(t_overhead > 500_000_000, "tail overhead too small: {t_overhead}");
    assert!(t_overhead < 2_000_000_000, "tail overhead too large: {t_overhead}");
}

#[test]
fn test_insufficient_capacity_error_message() {
    let per_layer = weight_memory_per_layer(&llama_8b_config())
        + cache_memory_per_layer(&llama_8b_config(), 4096);
    let tiny_mem = (per_layer * 5) as f64 / 1e9;

    let input = SchedulerInput {
        model_config: llama_8b_config(),
        workers: vec![worker("a", tiny_mem, 1.0), worker("b", tiny_mem, 1.0)],
        coordinator_compute: None,
        mode: SchedulingMode::Auto,
        max_seq_len: 4096,
        hop_latency_ms: 2.0,
    };
    let err = schedule(&input).unwrap_err().to_string();
    assert!(
        err.contains("capacity") || err.contains("layers"),
        "error should mention capacity: {err}"
    );
}

#[test]
fn test_weight_and_cache_memory_in_result() {
    let input = SchedulerInput {
        model_config: llama_8b_config(),
        workers: vec![worker("a", 22.0, 1.0), worker("b", 22.0, 1.0)],
        coordinator_compute: None,
        mode: SchedulingMode::EqualSplit,
        max_seq_len: 4096,
        hop_latency_ms: 2.0,
    };
    let result = schedule(&input).unwrap();

    for a in &result.assignments {
        assert!(a.weight_memory_gb > 0.0, "weight_memory_gb should be positive");
        assert!(a.cache_memory_gb > 0.0, "cache_memory_gb should be positive");
    }
}

#[test]
fn test_auto_iterative_multi_node_pruning() {
    // 2 fast nodes + 2 very slow nodes. Both slow nodes should be pruned
    // since the fast nodes have enough memory for the full model.
    let input = SchedulerInput {
        model_config: llama_8b_config(),
        workers: vec![
            worker("fast1", 30.0, 0.5),
            worker("fast2", 30.0, 0.5),
            worker("turtle1", 22.0, 10.0),
            worker("turtle2", 22.0, 15.0),
        ],
        coordinator_compute: None,
        mode: SchedulingMode::Auto,
        max_seq_len: 4096,
        hop_latency_ms: 2.0,
    };
    let result = schedule(&input).unwrap();

    assert_eq!(result.assignments.len(), 2, "both slow nodes should be pruned");
    assert_eq!(result.excluded_nodes.len(), 2);
    let excluded_ids: Vec<&str> = result.excluded_nodes.iter().map(|e| e.node_id.as_str()).collect();
    assert!(excluded_ids.contains(&"turtle1"), "turtle1 should be excluded");
    assert!(excluded_ids.contains(&"turtle2"), "turtle2 should be excluded");
}

#[test]
fn test_auto_bottleneck_node_correct() {
    let input = SchedulerInput {
        model_config: llama_8b_config(),
        workers: vec![
            worker("fast", 30.0, 0.5),
            worker("slow", 22.0, 1.5),
        ],
        coordinator_compute: None,
        mode: SchedulingMode::Auto,
        max_seq_len: 4096,
        hop_latency_ms: 2.0,
    };
    let result = schedule(&input).unwrap();

    // The slower node should be the bottleneck
    // (even with fewer layers, it takes longer per layer)
    let slow_assignment = result.assignments.iter().find(|a| a.node_id == "slow").unwrap();
    let fast_assignment = result.assignments.iter().find(|a| a.node_id == "fast").unwrap();

    // bottleneck is whichever has higher expected_decode_ms
    if slow_assignment.expected_decode_ms >= fast_assignment.expected_decode_ms {
        assert_eq!(result.bottleneck_node, "slow");
    } else {
        assert_eq!(result.bottleneck_node, "fast");
    }
}

#[test]
fn test_auto_redistribution_assigns_all_layers() {
    // 3 workers with different speeds; rounding may not sum to 32
    // Redistribution must ensure exactly 32 layers assigned.
    let input = SchedulerInput {
        model_config: llama_8b_config(),
        workers: vec![
            worker("a", 30.0, 0.7),
            worker("b", 22.0, 1.3),
            worker("c", 25.0, 1.0),
        ],
        coordinator_compute: None,
        mode: SchedulingMode::Auto,
        max_seq_len: 4096,
        hop_latency_ms: 2.0,
    };
    let result = schedule(&input).unwrap();

    let total: usize = result.assignments.iter().map(|a| a.layer_range.len()).sum();
    assert_eq!(total, 32, "redistribution must produce exactly 32 layers");

    // Verify contiguous
    let mut start = 0;
    for a in &result.assignments {
        assert_eq!(a.layer_range.start, start);
        start = a.layer_range.end;
    }
    assert_eq!(start, 32);
}

#[test]
fn test_calibration_plausibility_valid() {
    let w = worker("ok", 22.0, 1.0);
    assert!(w.validate_calibration().is_ok());
}

#[test]
fn test_calibration_plausibility_zero_decode() {
    let mut w = worker("bad", 22.0, 0.0);
    w.decode_ms_per_layer = 0.0;
    assert!(w.validate_calibration().is_err());
    assert!(w.validate_calibration().unwrap_err().to_string().contains("decode"));
}

#[test]
fn test_calibration_plausibility_negative_prefill() {
    let mut w = worker("bad", 22.0, 1.0);
    w.prefill_ms_per_layer_128 = -1.0;
    assert!(w.validate_calibration().is_err());
    assert!(w.validate_calibration().unwrap_err().to_string().contains("prefill"));
}

#[test]
fn test_calibration_plausibility_absurdly_large() {
    let mut w = worker("bad", 22.0, 1.0);
    w.decode_ms_per_layer = 999.0;
    assert!(w.validate_calibration().is_err());
    assert!(w.validate_calibration().unwrap_err().to_string().contains("exceeds"));
}

#[test]
fn test_scheduler_rejects_implausible_calibration() {
    let mut w = worker("bad", 22.0, 0.0);
    w.decode_ms_per_layer = 0.0;
    let input = SchedulerInput {
        model_config: llama_8b_config(),
        workers: vec![w],
        coordinator_compute: None,
        mode: SchedulingMode::Auto,
        max_seq_len: 4096,
        hop_latency_ms: 2.0,
    };
    assert!(schedule(&input).is_err());
}

#[test]
fn test_scheduler_nan_decode_ms_rejected() {
    let mut w = worker("nan", 22.0, 1.0);
    w.decode_ms_per_layer = f32::NAN;
    // NaN fails the <= 0.0 check (NaN comparisons are false)
    // and also fails the > MAX check, so it should be caught
    assert!(w.validate_calibration().is_err());
}

#[test]
fn test_scheduler_inf_decode_ms_rejected() {
    let mut w = worker("inf", 22.0, 1.0);
    w.decode_ms_per_layer = f32::INFINITY;
    assert!(w.validate_calibration().is_err());
}

#[test]
fn test_scheduler_negative_decode_ms_rejected() {
    let mut w = worker("neg", 22.0, 1.0);
    w.decode_ms_per_layer = -0.5;
    assert!(w.validate_calibration().is_err());
}

#[test]
fn test_scheduler_zero_max_seq_len() {
    // Zero max_seq_len means no KV cache needed — should still work
    let input = SchedulerInput {
        model_config: llama_8b_config(),
        workers: vec![worker("a", 22.0, 1.0)],
        coordinator_compute: None,
        mode: SchedulingMode::Auto,
        max_seq_len: 0,
        hop_latency_ms: 2.0,
    };
    let result = schedule(&input).unwrap();
    assert_eq!(result.assignments[0].layer_range, 0..32);
}

#[test]
fn test_manual_duplicate_node_ids() {
    let input = SchedulerInput {
        model_config: llama_8b_config(),
        workers: vec![worker("a", 22.0, 1.0)],
        coordinator_compute: None,
        mode: SchedulingMode::Manual(vec![
            ManualAssignment {
                node_id: "a".into(),
                layer_range: 0..16,
            },
            ManualAssignment {
                node_id: "a".into(),
                layer_range: 16..32,
            },
        ]),
        max_seq_len: 4096,
        hop_latency_ms: 2.0,
    };
    let err = schedule(&input).unwrap_err().to_string();
    assert!(
        err.contains("duplicate"),
        "error should mention duplicate: {err}"
    );
}

#[test]
fn test_zero_workers_all_modes() {
    // Auto mode
    let input = SchedulerInput {
        model_config: llama_8b_config(),
        workers: vec![],
        coordinator_compute: None,
        mode: SchedulingMode::Auto,
        max_seq_len: 4096,
        hop_latency_ms: 2.0,
    };
    assert!(schedule(&input).is_err());

    // EqualSplit mode
    let input = SchedulerInput {
        model_config: llama_8b_config(),
        workers: vec![],
        coordinator_compute: None,
        mode: SchedulingMode::EqualSplit,
        max_seq_len: 4096,
        hop_latency_ms: 2.0,
    };
    assert!(schedule(&input).is_err());

    // Manual mode
    let input = SchedulerInput {
        model_config: llama_8b_config(),
        workers: vec![],
        coordinator_compute: None,
        mode: SchedulingMode::Manual(vec![]),
        max_seq_len: 4096,
        hop_latency_ms: 2.0,
    };
    assert!(schedule(&input).is_err());
}

#[test]
fn test_auto_mode_invalid_metrics_rejected_by_scheduler() {
    // Negative decode_ms_per_layer
    let mut w = worker("bad", 22.0, 1.0);
    w.decode_ms_per_layer = -1.0;
    let input = SchedulerInput {
        model_config: llama_8b_config(),
        workers: vec![w],
        coordinator_compute: None,
        mode: SchedulingMode::Auto,
        max_seq_len: 4096,
        hop_latency_ms: 2.0,
    };
    assert!(schedule(&input).is_err());

    // Zero decode_ms_per_layer
    let mut w = worker("bad", 22.0, 1.0);
    w.decode_ms_per_layer = 0.0;
    let input = SchedulerInput {
        model_config: llama_8b_config(),
        workers: vec![worker("good", 22.0, 1.0), w],
        coordinator_compute: None,
        mode: SchedulingMode::Auto,
        max_seq_len: 4096,
        hop_latency_ms: 2.0,
    };
    assert!(schedule(&input).is_err());
}

#[test]
fn test_scheduler_single_node_all_modes() {
    // Auto mode: single node gets all layers, no pruning
    let input = SchedulerInput {
        model_config: llama_8b_config(),
        workers: vec![worker("solo", 30.0, 1.0)],
        coordinator_compute: None,
        mode: SchedulingMode::Auto,
        max_seq_len: 4096,
        hop_latency_ms: 2.0,
    };
    let result = schedule(&input).unwrap();
    assert_eq!(result.assignments.len(), 1);
    assert_eq!(result.assignments[0].layer_range, 0..32);
    assert!(result.excluded_nodes.is_empty(), "Auto should not prune the only node");

    // EqualSplit mode: single node gets all layers
    let input = SchedulerInput {
        model_config: llama_8b_config(),
        workers: vec![worker("solo", 30.0, 1.0)],
        coordinator_compute: None,
        mode: SchedulingMode::EqualSplit,
        max_seq_len: 4096,
        hop_latency_ms: 2.0,
    };
    let result = schedule(&input).unwrap();
    assert_eq!(result.assignments.len(), 1);
    assert_eq!(result.assignments[0].layer_range, 0..32);

    // Manual mode: single node gets all layers
    let input = SchedulerInput {
        model_config: llama_8b_config(),
        workers: vec![worker("solo", 30.0, 1.0)],
        coordinator_compute: None,
        mode: SchedulingMode::Manual(vec![ManualAssignment {
            node_id: "solo".into(),
            layer_range: 0..32,
        }]),
        max_seq_len: 4096,
        hop_latency_ms: 2.0,
    };
    let result = schedule(&input).unwrap();
    assert_eq!(result.assignments.len(), 1);
    assert_eq!(result.assignments[0].layer_range, 0..32);
}

#[test]
fn test_scheduler_contiguous_layer_ranges_auto_and_equal() {
    // Auto mode: 3 heterogeneous workers
    let input = SchedulerInput {
        model_config: llama_8b_config(),
        workers: vec![
            worker("a", 30.0, 0.7),
            worker("b", 22.0, 1.3),
            worker("c", 25.0, 1.0),
        ],
        coordinator_compute: None,
        mode: SchedulingMode::Auto,
        max_seq_len: 4096,
        hop_latency_ms: 2.0,
    };
    let result = schedule(&input).unwrap();

    // Verify sorted, adjacent, no gaps
    let mut expected_start = 0;
    for a in &result.assignments {
        assert_eq!(
            a.layer_range.start, expected_start,
            "Auto: gap detected at layer {} (expected {})",
            a.layer_range.start, expected_start
        );
        assert!(
            a.layer_range.start < a.layer_range.end,
            "Auto: empty range for node {}",
            a.node_id
        );
        expected_start = a.layer_range.end;
    }
    assert_eq!(expected_start, 32, "Auto: ranges must cover all 32 layers");

    // EqualSplit mode: 3 equal workers
    let input = SchedulerInput {
        model_config: llama_8b_config(),
        workers: vec![
            worker("a", 22.0, 1.0),
            worker("b", 22.0, 1.0),
            worker("c", 22.0, 1.0),
        ],
        coordinator_compute: None,
        mode: SchedulingMode::EqualSplit,
        max_seq_len: 4096,
        hop_latency_ms: 2.0,
    };
    let result = schedule(&input).unwrap();

    let mut expected_start = 0;
    for a in &result.assignments {
        assert_eq!(
            a.layer_range.start, expected_start,
            "EqualSplit: gap detected at layer {} (expected {})",
            a.layer_range.start, expected_start
        );
        assert!(
            a.layer_range.start < a.layer_range.end,
            "EqualSplit: empty range for node {}",
            a.node_id
        );
        expected_start = a.layer_range.end;
    }
    assert_eq!(expected_start, 32, "EqualSplit: ranges must cover all 32 layers");
}

#[test]
fn test_scheduler_equal_split_node_overflow_returns_error() {
    let per_layer = weight_memory_per_layer(&llama_8b_config())
        + cache_memory_per_layer(&llama_8b_config(), 4096);
    // Node "small" can hold only 5 layers but equal split of 32/2 = 16 per node
    let tiny_mem = (per_layer * 5 + head_overhead(&llama_8b_config())) as f64 / 1e9;

    let input = SchedulerInput {
        model_config: llama_8b_config(),
        workers: vec![worker("small", tiny_mem, 1.0), worker("big", 30.0, 1.0)],
        coordinator_compute: None,
        mode: SchedulingMode::EqualSplit,
        max_seq_len: 4096,
        hop_latency_ms: 2.0,
    };
    let result = schedule(&input);
    assert!(result.is_err(), "EqualSplit should fail when a node cannot hold its share");
    let err_msg = result.unwrap_err().to_string();
    assert!(
        err_msg.contains("cannot hold"),
        "error should mention 'cannot hold': {err_msg}"
    );
}

#[test]
fn test_scheduler_very_large_hop_latency_prunes_to_single_node() {
    // If hop latency is extremely high, adding a second node is never worth it
    let input = SchedulerInput {
        model_config: llama_8b_config(),
        workers: vec![worker("a", 30.0, 1.0), worker("b", 30.0, 1.0)],
        coordinator_compute: None,
        mode: SchedulingMode::Auto,
        max_seq_len: 4096,
        hop_latency_ms: 1000.0, // 1 second per hop!
    };
    let result = schedule(&input).unwrap();
    // Should prune to 1 node since the hop latency makes 2 nodes worse
    assert_eq!(result.assignments.len(), 1);
}
