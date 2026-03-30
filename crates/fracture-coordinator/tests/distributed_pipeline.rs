//! Integration tests for the distributed pipeline.
//!
//! These tests spawn mock worker tasks on localhost that respond to wire
//! protocol messages, then exercise the coordinator's distributed pipeline
//! against them. No GPU required — workers return deterministic dummy data.

use fracture_coordinator::{
    pipeline::DistributedPipeline,
    registry::PeerRegistry,
    scheduler::{LayerAssignment, NodeRole},
};
use fracture_protocol::{
    connection::FramedConnection,
    frame::MessageType,
    messages::*,
    tensor::TensorWireHeader,
};
use tokio::net::TcpListener;

/// Spawn a mock worker that:
/// - Responds to CacheAlloc (no-op)
/// - Responds to CacheFree (no-op)
/// - Responds to Forward by returning either:
///   - Activations (if `is_tail` is false): echo back a dummy [1, hidden_size] FP16 tensor
///   - Logits (if `is_tail` is true): return deterministic f32 logits
/// - Responds to BatchedForward with batched results:
///   - Activations (non-tail): dummy [total_tokens, hidden_size] FP16 tensor
///   - Logits (tail): deterministic per-sequence f32 logits
/// - Responds to Shutdown by exiting
async fn spawn_mock_worker(
    listener: TcpListener,
    is_tail: bool,
    hidden_size: usize,
    vocab_size: usize,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        let (stream, _) = listener.accept().await.unwrap();
        let mut conn = FramedConnection::new(stream);

        loop {
            let (header, payload) = match conn.recv().await {
                Ok(frame) => frame,
                Err(_) => break,
            };

            match header.msg_type {
                MessageType::CacheAlloc => {
                    // Respond with CacheAllocAck
                    conn.send_empty(MessageType::CacheAllocAck, header.seq_id)
                        .await
                        .unwrap();
                }
                MessageType::CacheFree => {
                    // No response needed
                }
                MessageType::Forward => {
                    let _req: ForwardPayload =
                        FramedConnection::deserialize_payload(&payload).unwrap();

                    let result = if is_tail {
                        // Return deterministic logits: logit[i] = i as f32
                        let data: Vec<u8> = (0..vocab_size)
                            .flat_map(|i| (i as f32).to_le_bytes())
                            .collect();
                        ForwardResultPayload {
                            output: ForwardOutputWire::Logits { data },
                        }
                    } else {
                        // Return dummy activations: [1, hidden_size] FP16 zeros
                        let data_len = hidden_size * 2; // FP16
                        ForwardResultPayload {
                            output: ForwardOutputWire::Activations {
                                tensor_header: TensorWireHeader {
                                    ndim: 2,
                                    shape: vec![1, hidden_size as u32],
                                    dtype: 0, // FP16
                                    compression: 0,
                                    data_len: data_len as u32,
                                },
                                tensor_data: vec![0u8; data_len],
                            },
                        }
                    };

                    conn.send(MessageType::ForwardResult, header.seq_id, &result)
                        .await
                        .unwrap();
                }
                MessageType::BatchedForward => {
                    let req: BatchedForwardPayload =
                        FramedConnection::deserialize_payload(&payload).unwrap();
                    let num_seqs = req.sequences.len();
                    let total_tokens: usize =
                        req.sequences.iter().map(|s| s.num_tokens).sum();

                    let result = if is_tail {
                        // Return deterministic per-sequence logits:
                        // seq i, logit j → (i * vocab_size + j) as f32
                        let data: Vec<u8> = (0..num_seqs)
                            .flat_map(|si| {
                                (0..vocab_size)
                                    .flat_map(move |j| {
                                        ((si * vocab_size + j) as f32).to_le_bytes()
                                    })
                            })
                            .collect();
                        let logit_offsets: Vec<usize> =
                            (0..num_seqs).map(|i| i * vocab_size * 4).collect();
                        BatchedForwardResultPayload {
                            output: ForwardOutputWire::Logits { data },
                            num_sequences: num_seqs,
                            logit_offsets,
                        }
                    } else {
                        // Return dummy activations: [total_tokens, hidden_size] FP16 zeros
                        let data_len = total_tokens * hidden_size * 2;
                        BatchedForwardResultPayload {
                            output: ForwardOutputWire::Activations {
                                tensor_header: TensorWireHeader {
                                    ndim: 2,
                                    shape: vec![total_tokens as u32, hidden_size as u32],
                                    dtype: 0, // FP16
                                    compression: 0,
                                    data_len: data_len as u32,
                                },
                                tensor_data: vec![0u8; data_len],
                            },
                            num_sequences: num_seqs,
                            logit_offsets: Vec::new(),
                        }
                    };

                    conn.send(MessageType::BatchedForwardResult, header.seq_id, &result)
                        .await
                        .unwrap();
                }
                MessageType::Shutdown => break,
                _ => {}
            }
        }
    })
}

/// Helper: set up a 2-node pipeline with mock workers on localhost.
/// Returns (pipeline, registry) ready for forward passes.
async fn setup_two_node_pipeline() -> (DistributedPipeline, PeerRegistry) {
    let hidden_size = 4096;
    let vocab_size = 128256;

    // Bind listeners first to get addresses
    let head_listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let tail_listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let head_addr = head_listener.local_addr().unwrap();
    let tail_addr = tail_listener.local_addr().unwrap();

    // Spawn mock workers
    let _head_task = spawn_mock_worker(head_listener, false, hidden_size, vocab_size).await;
    let _tail_task = spawn_mock_worker(tail_listener, true, hidden_size, vocab_size).await;

    // Connect coordinator to workers
    let head_stream = tokio::net::TcpStream::connect(head_addr).await.unwrap();
    let tail_stream = tokio::net::TcpStream::connect(tail_addr).await.unwrap();

    let head_conn = FramedConnection::new(head_stream);
    let tail_conn = FramedConnection::new(tail_stream);

    // Build registry
    let mut registry = PeerRegistry::new();
    use fracture_coordinator::scheduler::WorkerCapabilities;

    let head_caps = WorkerCapabilities {
        node_id: "head".into(),
        gpu_model: "Mock".into(),
        gpu_memory_available: 24_000_000_000,
        compute_capability: (8, 0),
        decode_ms_per_layer: 1.0,
        prefill_ms_per_layer_128: 3.0,
    };
    let tail_caps = WorkerCapabilities {
        node_id: "tail".into(),
        gpu_model: "Mock".into(),
        gpu_memory_available: 24_000_000_000,
        compute_capability: (8, 0),
        decode_ms_per_layer: 1.0,
        prefill_ms_per_layer_128: 3.0,
    };

    registry.register(head_caps, head_conn).unwrap();
    registry.register(tail_caps, tail_conn).unwrap();

    let head_assignment = LayerAssignment {
        node_id: "head".into(),
        layer_range: 0..16,
        role: NodeRole::Head,
        expected_decode_ms: 16.0,
        weight_memory_gb: 6.0,
        cache_memory_gb: 1.0,
    };
    let tail_assignment = LayerAssignment {
        node_id: "tail".into(),
        layer_range: 16..32,
        role: NodeRole::Tail,
        expected_decode_ms: 16.0,
        weight_memory_gb: 6.0,
        cache_memory_gb: 1.0,
    };

    registry
        .assign("head", head_assignment.clone())
        .unwrap();
    registry
        .assign("tail", tail_assignment.clone())
        .unwrap();

    let pipeline =
        DistributedPipeline::new(&[head_assignment, tail_assignment], 4096).unwrap();

    (pipeline, registry)
}

#[tokio::test]
async fn test_two_node_forward_returns_logits() {
    let (pipeline, mut registry) = setup_two_node_pipeline().await;

    let token_ids = vec![128000, 791, 1401]; // dummy prompt
    let positions = vec![0, 1, 2];
    let seq_id = 1;

    // Allocate cache
    pipeline
        .alloc_cache(&mut registry, seq_id, 4096)
        .await
        .unwrap();

    // Forward (prefill)
    let logits = pipeline
        .forward(&mut registry, seq_id, &token_ids, &positions, true)
        .await
        .unwrap();

    // Mock tail returns logits[i] = i as f32
    assert_eq!(logits.len(), 128256);
    assert!((logits[0] - 0.0).abs() < 1e-6);
    assert!((logits[1] - 1.0).abs() < 1e-6);
    assert!((logits[100] - 100.0).abs() < 1e-6);

    // Free cache
    pipeline
        .free_cache(&mut registry, seq_id)
        .await
        .unwrap();
}

#[tokio::test]
async fn test_two_node_multi_step_decode() {
    let (pipeline, mut registry) = setup_two_node_pipeline().await;
    let seq_id = 42;

    pipeline
        .alloc_cache(&mut registry, seq_id, 4096)
        .await
        .unwrap();

    // Prefill
    let logits = pipeline
        .forward(&mut registry, seq_id, &[1, 2, 3], &[0, 1, 2], true)
        .await
        .unwrap();
    assert_eq!(logits.len(), 128256);

    // 10 decode steps
    for step in 0..10u32 {
        let pos = 3 + step;
        let logits = pipeline
            .forward(&mut registry, seq_id, &[42], &[pos], false)
            .await
            .unwrap();
        assert_eq!(logits.len(), 128256);
        // Verify logits are deterministic (same mock returns same data)
        assert!((logits[42] - 42.0).abs() < 1e-6);
    }

    pipeline
        .free_cache(&mut registry, seq_id)
        .await
        .unwrap();
}

#[tokio::test]
async fn test_cache_alloc_and_free() {
    let (pipeline, mut registry) = setup_two_node_pipeline().await;

    // Allocate and free multiple sequences
    for seq_id in 1..=5 {
        pipeline
            .alloc_cache(&mut registry, seq_id, 4096)
            .await
            .unwrap();
    }
    for seq_id in 1..=5 {
        pipeline
            .free_cache(&mut registry, seq_id)
            .await
            .unwrap();
    }
}

#[tokio::test]
async fn test_shutdown_propagation() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();

    // Track whether worker received shutdown
    let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel::<()>();

    let _worker = tokio::spawn(async move {
        let (stream, _) = listener.accept().await.unwrap();
        let mut conn = FramedConnection::new(stream);

        loop {
            let (header, _payload) = match conn.recv().await {
                Ok(frame) => frame,
                Err(_) => break,
            };
            if header.msg_type == MessageType::Shutdown {
                let _ = shutdown_tx.send(());
                break;
            }
        }
    });

    let stream = tokio::net::TcpStream::connect(addr).await.unwrap();
    let conn = FramedConnection::new(stream);

    let mut registry = PeerRegistry::new();
    use fracture_coordinator::scheduler::WorkerCapabilities;
    registry
        .register(
            WorkerCapabilities {
                node_id: "w".into(),
                gpu_model: "Mock".into(),
                gpu_memory_available: 24_000_000_000,
                compute_capability: (8, 0),
                decode_ms_per_layer: 1.0,
                prefill_ms_per_layer_128: 3.0,
            },
            conn,
        )
        .unwrap();

    let assignment = LayerAssignment {
        node_id: "w".into(),
        layer_range: 0..32,
        role: NodeRole::Head,
        expected_decode_ms: 32.0,
        weight_memory_gb: 12.0,
        cache_memory_gb: 2.0,
    };
    registry.assign("w", assignment).unwrap();

    // Send shutdown
    let entry = registry.get_mut("w").unwrap();
    entry
        .writer
        .send_empty(MessageType::Shutdown, 0)
        .await
        .unwrap();

    // Verify worker received it
    tokio::time::timeout(std::time::Duration::from_secs(2), shutdown_rx)
        .await
        .expect("timeout waiting for shutdown")
        .expect("shutdown channel closed without signal");
}

#[tokio::test]
async fn test_three_node_pipeline() {
    let hidden_size = 4096;
    let vocab_size = 128256;

    // Three workers: head, middle, tail
    let head_listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let mid_listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let tail_listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let head_addr = head_listener.local_addr().unwrap();
    let mid_addr = mid_listener.local_addr().unwrap();
    let tail_addr = tail_listener.local_addr().unwrap();

    let _h = spawn_mock_worker(head_listener, false, hidden_size, vocab_size).await;
    let _m = spawn_mock_worker(mid_listener, false, hidden_size, vocab_size).await;
    let _t = spawn_mock_worker(tail_listener, true, hidden_size, vocab_size).await;

    let head_conn = FramedConnection::new(
        tokio::net::TcpStream::connect(head_addr).await.unwrap(),
    );
    let mid_conn = FramedConnection::new(
        tokio::net::TcpStream::connect(mid_addr).await.unwrap(),
    );
    let tail_conn = FramedConnection::new(
        tokio::net::TcpStream::connect(tail_addr).await.unwrap(),
    );

    let mut registry = PeerRegistry::new();
    use fracture_coordinator::scheduler::WorkerCapabilities;

    for (id, conn) in [("head", head_conn), ("mid", mid_conn), ("tail", tail_conn)] {
        registry
            .register(
                WorkerCapabilities {
                    node_id: id.into(),
                    gpu_model: "Mock".into(),
                    gpu_memory_available: 24_000_000_000,
                    compute_capability: (8, 0),
                    decode_ms_per_layer: 1.0,
                    prefill_ms_per_layer_128: 3.0,
                },
                conn,
            )
            .unwrap();
    }

    let assignments = vec![
        LayerAssignment {
            node_id: "head".into(),
            layer_range: 0..10,
            role: NodeRole::Head,
            expected_decode_ms: 10.0,
            weight_memory_gb: 4.0,
            cache_memory_gb: 0.5,
        },
        LayerAssignment {
            node_id: "mid".into(),
            layer_range: 10..20,
            role: NodeRole::Middle,
            expected_decode_ms: 10.0,
            weight_memory_gb: 4.0,
            cache_memory_gb: 0.5,
        },
        LayerAssignment {
            node_id: "tail".into(),
            layer_range: 20..32,
            role: NodeRole::Tail,
            expected_decode_ms: 12.0,
            weight_memory_gb: 5.0,
            cache_memory_gb: 0.6,
        },
    ];

    for a in &assignments {
        registry.assign(&a.node_id, a.clone()).unwrap();
    }

    let pipeline = DistributedPipeline::new(&assignments, 4096).unwrap();
    let seq_id = 99;

    pipeline
        .alloc_cache(&mut registry, seq_id, 4096)
        .await
        .unwrap();

    // Prefill through 3 stages
    let logits = pipeline
        .forward(&mut registry, seq_id, &[1, 2, 3, 4], &[0, 1, 2, 3], true)
        .await
        .unwrap();

    assert_eq!(logits.len(), 128256);
    assert!((logits[0] - 0.0).abs() < 1e-6);
    assert!((logits[1000] - 1000.0).abs() < 1e-6);

    // Decode step
    let logits = pipeline
        .forward(&mut registry, seq_id, &[42], &[4], false)
        .await
        .unwrap();
    assert_eq!(logits.len(), 128256);

    pipeline
        .free_cache(&mut registry, seq_id)
        .await
        .unwrap();
}

// ── Error path tests ────────────────────────────────────────────────────

/// Spawn a mock worker that returns an Error message on Forward.
async fn spawn_error_worker(listener: TcpListener) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        let (stream, _) = listener.accept().await.unwrap();
        let mut conn = FramedConnection::new(stream);

        loop {
            let (header, _payload) = match conn.recv().await {
                Ok(frame) => frame,
                Err(_) => break,
            };
            match header.msg_type {
                MessageType::CacheAlloc => {
                    conn.send_empty(MessageType::CacheAllocAck, header.seq_id)
                        .await
                        .unwrap();
                }
                MessageType::CacheFree => {}
                MessageType::Forward => {
                    let err = ErrorPayload {
                        error_code: ErrorCode::OutOfMemory,
                        message: "GPU OOM during forward pass".into(),
                    };
                    conn.send(MessageType::Error, header.seq_id, &err)
                        .await
                        .unwrap();
                }
                MessageType::Shutdown => break,
                _ => {}
            }
        }
    })
}

/// Spawn a mock worker that returns the wrong output type.
async fn spawn_wrong_output_worker(
    listener: TcpListener,
    return_logits: bool,
    hidden_size: usize,
    vocab_size: usize,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        let (stream, _) = listener.accept().await.unwrap();
        let mut conn = FramedConnection::new(stream);

        loop {
            let (header, _payload) = match conn.recv().await {
                Ok(frame) => frame,
                Err(_) => break,
            };
            match header.msg_type {
                MessageType::CacheAlloc => {
                    conn.send_empty(MessageType::CacheAllocAck, header.seq_id)
                        .await
                        .unwrap();
                }
                MessageType::CacheFree => {}
                MessageType::Forward => {
                    let result = if return_logits {
                        let data: Vec<u8> = (0..vocab_size)
                            .flat_map(|i| (i as f32).to_le_bytes())
                            .collect();
                        ForwardResultPayload {
                            output: ForwardOutputWire::Logits { data },
                        }
                    } else {
                        let data_len = hidden_size * 2;
                        ForwardResultPayload {
                            output: ForwardOutputWire::Activations {
                                tensor_header: TensorWireHeader {
                                    ndim: 2,
                                    shape: vec![1, hidden_size as u32],
                                    dtype: 0,
                                    compression: 0,
                                    data_len: data_len as u32,
                                },
                                tensor_data: vec![0u8; data_len],
                            },
                        }
                    };
                    conn.send(MessageType::ForwardResult, header.seq_id, &result)
                        .await
                        .unwrap();
                }
                MessageType::Shutdown => break,
                _ => {}
            }
        }
    })
}

fn make_caps(id: &str) -> fracture_coordinator::scheduler::WorkerCapabilities {
    fracture_coordinator::scheduler::WorkerCapabilities {
        node_id: id.into(),
        gpu_model: "Mock".into(),
        gpu_memory_available: 24_000_000_000,
        compute_capability: (8, 0),
        decode_ms_per_layer: 1.0,
        prefill_ms_per_layer_128: 3.0,
    }
}

#[tokio::test]
async fn test_worker_error_propagates() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let _task = spawn_error_worker(listener).await;

    let stream = tokio::net::TcpStream::connect(addr).await.unwrap();
    let conn = FramedConnection::new(stream);

    let mut registry = PeerRegistry::new();
    registry.register(make_caps("err-worker"), conn).unwrap();

    let assignment = LayerAssignment {
        node_id: "err-worker".into(),
        layer_range: 0..32,
        role: NodeRole::Head,
        expected_decode_ms: 32.0,
        weight_memory_gb: 12.0,
        cache_memory_gb: 2.0,
    };
    registry.assign("err-worker", assignment.clone()).unwrap();

    let pipeline = DistributedPipeline::new(&[assignment], 4096).unwrap();
    pipeline.alloc_cache(&mut registry, 1, 4096).await.unwrap();

    let result = pipeline.forward(&mut registry, 1, &[1, 2], &[0, 1], true).await;
    assert!(result.is_err());
    let err_msg = result.unwrap_err().to_string();
    assert!(err_msg.contains("GPU OOM"), "error should contain OOM: {err_msg}");
}

#[tokio::test]
async fn test_non_tail_returning_logits_is_error() {
    let hidden_size = 4096;
    let vocab_size = 128256;

    let head_listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let tail_listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let head_addr = head_listener.local_addr().unwrap();
    let tail_addr = tail_listener.local_addr().unwrap();

    // Head returns logits (wrong)
    let _h = spawn_wrong_output_worker(head_listener, true, hidden_size, vocab_size).await;
    let _t = spawn_mock_worker(tail_listener, true, hidden_size, vocab_size).await;

    let head_conn = FramedConnection::new(tokio::net::TcpStream::connect(head_addr).await.unwrap());
    let tail_conn = FramedConnection::new(tokio::net::TcpStream::connect(tail_addr).await.unwrap());

    let mut registry = PeerRegistry::new();
    registry.register(make_caps("head"), head_conn).unwrap();
    registry.register(make_caps("tail"), tail_conn).unwrap();

    let assignments = vec![
        LayerAssignment { node_id: "head".into(), layer_range: 0..16, role: NodeRole::Head, expected_decode_ms: 16.0, weight_memory_gb: 6.0, cache_memory_gb: 1.0 },
        LayerAssignment { node_id: "tail".into(), layer_range: 16..32, role: NodeRole::Tail, expected_decode_ms: 16.0, weight_memory_gb: 6.0, cache_memory_gb: 1.0 },
    ];
    for a in &assignments { registry.assign(&a.node_id, a.clone()).unwrap(); }

    let pipeline = DistributedPipeline::new(&assignments, 4096).unwrap();
    pipeline.alloc_cache(&mut registry, 1, 4096).await.unwrap();

    let result = pipeline.forward(&mut registry, 1, &[1], &[0], true).await;
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("non-tail"));
}

#[tokio::test]
async fn test_tail_returning_activations_is_error() {
    let hidden_size = 4096;
    let vocab_size = 128256;

    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    // Tail returns activations (wrong)
    let _t = spawn_wrong_output_worker(listener, false, hidden_size, vocab_size).await;

    let conn = FramedConnection::new(tokio::net::TcpStream::connect(addr).await.unwrap());
    let mut registry = PeerRegistry::new();
    registry.register(make_caps("tail"), conn).unwrap();

    let assignment = LayerAssignment { node_id: "tail".into(), layer_range: 0..32, role: NodeRole::Tail, expected_decode_ms: 32.0, weight_memory_gb: 12.0, cache_memory_gb: 2.0 };
    registry.assign("tail", assignment.clone()).unwrap();

    let pipeline = DistributedPipeline::new(&[assignment], 4096).unwrap();
    pipeline.alloc_cache(&mut registry, 1, 4096).await.unwrap();

    let result = pipeline.forward(&mut registry, 1, &[1], &[0], true).await;
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("tail"));
}

// ── Heartbeat integration test ──────────────────────────────────────────

#[tokio::test]
async fn test_heartbeat_round_trip() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();

    let worker = tokio::spawn(async move {
        let (stream, _) = listener.accept().await.unwrap();
        let mut conn = FramedConnection::new(stream);

        let (header, payload) = conn.recv().await.unwrap();
        assert_eq!(header.msg_type, MessageType::Heartbeat);

        let hb: HeartbeatPayload = FramedConnection::deserialize_payload(&payload).unwrap();
        let ack = HeartbeatAckPayload {
            timestamp_echo: hb.timestamp_ns,
            nonce_echo: hb.nonce,
            gpu_memory_used: 8_000_000_000,
            active_sequences: 2,
            free_blocks: 0,
        };
        conn.send(MessageType::HeartbeatAck, 0, &ack).await.unwrap();
    });

    let stream = tokio::net::TcpStream::connect(addr).await.unwrap();
    let conn = FramedConnection::new(stream);

    let mut registry = PeerRegistry::new();
    registry.register(make_caps("hb-worker"), conn).unwrap();
    let assignment = LayerAssignment { node_id: "hb-worker".into(), layer_range: 0..32, role: NodeRole::Head, expected_decode_ms: 32.0, weight_memory_gb: 12.0, cache_memory_gb: 2.0 };
    registry.assign("hb-worker", assignment).unwrap();

    let hb = HeartbeatPayload { timestamp_ns: 1234567890, nonce: 42 };
    let entry = registry.get_mut("hb-worker").unwrap();
    entry.writer.send(MessageType::Heartbeat, 0, &hb).await.unwrap();

    let (header, payload) = entry.reader.lock().await.recv().await.unwrap();
    assert_eq!(header.msg_type, MessageType::HeartbeatAck);
    let ack: HeartbeatAckPayload = FramedConnection::deserialize_payload(&payload).unwrap();
    assert_eq!(ack.timestamp_echo, 1234567890);
    assert_eq!(ack.nonce_echo, 42);
    assert_eq!(ack.gpu_memory_used, 8_000_000_000);
    assert_eq!(ack.active_sequences, 2);

    worker.await.unwrap();
}

// ── Worker serve loop protocol tests ────────────────────────────────────

async fn spawn_protocol_worker(listener: TcpListener) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        let (stream, _) = listener.accept().await.unwrap();
        let mut conn = FramedConnection::new(stream);
        let mut cache_count: u32 = 0;

        loop {
            let (header, payload) = match conn.recv().await {
                Ok(frame) => frame,
                Err(_) => break,
            };

            match header.msg_type {
                MessageType::CacheAlloc => {
                    cache_count += 1;
                    conn.send_empty(MessageType::CacheAllocAck, header.seq_id)
                        .await
                        .unwrap();
                }
                MessageType::CacheFree => { cache_count = cache_count.saturating_sub(1); }
                MessageType::Heartbeat => {
                    let hb: HeartbeatPayload = FramedConnection::deserialize_payload(&payload).unwrap();
                    let ack = HeartbeatAckPayload {
                        timestamp_echo: hb.timestamp_ns, nonce_echo: hb.nonce,
                        gpu_memory_used: 0, active_sequences: cache_count,
                        free_blocks: 0,
                    };
                    conn.send(MessageType::HeartbeatAck, 0, &ack).await.unwrap();
                }
                MessageType::Shutdown => break,
                _ => {
                    let err = ErrorPayload {
                        error_code: ErrorCode::ProtocolViolation,
                        message: format!("unexpected: {:?}", header.msg_type),
                    };
                    conn.send(MessageType::Error, header.seq_id, &err).await.unwrap();
                }
            }
        }
    })
}

#[tokio::test]
async fn test_worker_cache_alloc_free_via_protocol() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let _task = spawn_protocol_worker(listener).await;

    let stream = tokio::net::TcpStream::connect(addr).await.unwrap();
    let mut conn = FramedConnection::new(stream);

    // Alloc 3 caches
    for seq_id in 1..=3u64 {
        conn.send(MessageType::CacheAlloc, seq_id, &CacheAllocPayload { max_seq_len: 4096 }).await.unwrap();
        let (ack_header, _) = conn.recv().await.unwrap();
        assert_eq!(ack_header.msg_type, MessageType::CacheAllocAck);
    }

    // Heartbeat to verify count
    conn.send(MessageType::Heartbeat, 0, &HeartbeatPayload { timestamp_ns: 0, nonce: 1 }).await.unwrap();
    let (_, payload) = conn.recv().await.unwrap();
    let ack: HeartbeatAckPayload = FramedConnection::deserialize_payload(&payload).unwrap();
    assert_eq!(ack.active_sequences, 3);

    // Free 2
    conn.send_empty(MessageType::CacheFree, 1).await.unwrap();
    conn.send_empty(MessageType::CacheFree, 2).await.unwrap();

    // Verify count again
    conn.send(MessageType::Heartbeat, 0, &HeartbeatPayload { timestamp_ns: 0, nonce: 2 }).await.unwrap();
    let (_, payload) = conn.recv().await.unwrap();
    let ack: HeartbeatAckPayload = FramedConnection::deserialize_payload(&payload).unwrap();
    assert_eq!(ack.active_sequences, 1);

    conn.send_empty(MessageType::Shutdown, 0).await.unwrap();
}

#[tokio::test]
async fn test_worker_unknown_message_returns_error() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let _task = spawn_protocol_worker(listener).await;

    let stream = tokio::net::TcpStream::connect(addr).await.unwrap();
    let mut conn = FramedConnection::new(stream);

    // Send RegisterAck (unexpected for a worker)
    let ack_payload = RegisterAckPayload {
        layer_start: 0, layer_end: 32, total_layers: 32, max_seq_len: 4096,
        model_config: fracture_core::ModelConfig {
            hidden_size: 4096, num_layers: 32, num_q_heads: 32, num_kv_heads: 8,
            head_dim: 128, intermediate_size: 14336, vocab_size: 128256,
            rope_theta: 500000.0, rms_norm_eps: 1e-5, max_seq_len: 8192,
        },
    };
    conn.send(MessageType::RegisterAck, 0, &ack_payload).await.unwrap();

    let (header, payload) = conn.recv().await.unwrap();
    assert_eq!(header.msg_type, MessageType::Error);
    let err: ErrorPayload = FramedConnection::deserialize_payload(&payload).unwrap();
    assert_eq!(err.error_code, ErrorCode::ProtocolViolation);

    conn.send_empty(MessageType::Shutdown, 0).await.unwrap();
}

// ── Heartbeat module tests ──────────────────────────────────────────────

#[tokio::test]
async fn test_heartbeat_tracker_valid_ack_over_wire() {
    use fracture_coordinator::heartbeat::HeartbeatTracker;

    // Spawn a mock worker that echoes heartbeat nonces.
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();

    let _worker = tokio::spawn(async move {
        let (stream, _) = listener.accept().await.unwrap();
        let mut conn = FramedConnection::new(stream);
        let (header, payload) = conn.recv().await.unwrap();
        assert_eq!(header.msg_type, MessageType::Heartbeat);
        let hb: HeartbeatPayload = FramedConnection::deserialize_payload(&payload).unwrap();
        let ack = HeartbeatAckPayload {
            timestamp_echo: hb.timestamp_ns,
            nonce_echo: hb.nonce,
            gpu_memory_used: 0,
            active_sequences: 0,
            free_blocks: 50,
        };
        conn.send(MessageType::HeartbeatAck, 0, &ack).await.unwrap();
    });

    let stream = tokio::net::TcpStream::connect(addr).await.unwrap();
    let conn = FramedConnection::new(stream);

    let mut registry = PeerRegistry::new();
    registry.register(make_caps("hb-test"), conn).unwrap();
    let assignment = LayerAssignment {
        node_id: "hb-test".into(), layer_range: 0..32, role: NodeRole::Head,
        expected_decode_ms: 32.0, weight_memory_gb: 12.0, cache_memory_gb: 2.0,
    };
    registry.assign("hb-test", assignment).unwrap();

    let mut tracker = HeartbeatTracker::new();

    // Send heartbeat with nonce 42.
    let nonce = 42u64;
    let hb = HeartbeatPayload { timestamp_ns: 1234, nonce };
    registry.get_mut("hb-test").unwrap().writer
        .send(MessageType::Heartbeat, 0, &hb).await.unwrap();
    tracker.set_pending_nonce(nonce);

    // Receive ack from worker.
    let (header, payload) = registry.get("hb-test").unwrap()
        .reader.lock().await.recv().await.unwrap();
    assert_eq!(header.msg_type, MessageType::HeartbeatAck);
    let ack: HeartbeatAckPayload = FramedConnection::deserialize_payload(&payload).unwrap();

    // Process through tracker — should succeed and update registry.
    assert!(tracker.process_ack(&mut registry, "hb-test", &ack));
    assert_eq!(tracker.missed_count("hb-test"), 0);
    assert_eq!(registry.get("hb-test").unwrap().free_blocks, 50);
}

#[tokio::test]
async fn test_heartbeat_tracker_missed_count_detects_dead() {
    use fracture_coordinator::heartbeat::HeartbeatTracker;

    // Worker that accepts but never responds.
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let _worker = tokio::spawn(async move {
        let (stream, _) = listener.accept().await.unwrap();
        let mut conn = FramedConnection::new(stream);
        let _ = conn.recv().await;
        tokio::time::sleep(std::time::Duration::from_secs(10)).await;
    });

    let stream = tokio::net::TcpStream::connect(addr).await.unwrap();
    let conn = FramedConnection::new(stream);

    let mut registry = PeerRegistry::new();
    registry.register(make_caps("slow"), conn).unwrap();
    let assignment = LayerAssignment {
        node_id: "slow".into(), layer_range: 0..32, role: NodeRole::Head,
        expected_decode_ms: 32.0, weight_memory_gb: 12.0, cache_memory_gb: 2.0,
    };
    registry.assign("slow", assignment).unwrap();

    let mut tracker = HeartbeatTracker::new();
    tracker.set_pending_nonce(1);

    // Simulate 3 missed rounds — worker should be flagged.
    assert!(tracker.increment_missed(&["slow".into()], 3).is_empty());
    tracker.set_pending_nonce(2);
    assert!(tracker.increment_missed(&["slow".into()], 3).is_empty());
    tracker.set_pending_nonce(3);
    let dead = tracker.increment_missed(&["slow".into()], 3);
    assert_eq!(dead, vec!["slow"]);
}

#[tokio::test]
async fn test_mark_dead_workers_transitions_status() {
    use fracture_coordinator::heartbeat;
    use fracture_coordinator::registry::WorkerStatus;

    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let _worker = tokio::spawn(async move {
        let _ = listener.accept().await;
    });

    let stream = tokio::net::TcpStream::connect(addr).await.unwrap();
    let conn = FramedConnection::new(stream);

    let mut registry = PeerRegistry::new();
    registry.register(make_caps("doomed"), conn).unwrap();
    let assignment = LayerAssignment {
        node_id: "doomed".into(), layer_range: 0..32, role: NodeRole::Head,
        expected_decode_ms: 32.0, weight_memory_gb: 12.0, cache_memory_gb: 2.0,
    };
    registry.assign("doomed", assignment).unwrap();

    assert_eq!(registry.get("doomed").unwrap().status, WorkerStatus::Ready);

    heartbeat::mark_dead_workers(&mut registry, &["doomed".to_string()]);

    assert_eq!(registry.get("doomed").unwrap().status, WorkerStatus::Dead);
    assert_eq!(registry.active_count(), 0);
    assert!(registry.pipeline_order().is_empty());
}

/// Two workers over real TCP — one responds, one hangs. Verify that
/// reader_handles() returns both and concurrent polling doesn't block
/// the healthy worker on the slow one.
#[tokio::test]
async fn test_concurrent_poll_mixed_workers() {
    use fracture_coordinator::heartbeat::HeartbeatTracker;

    // Worker A: responds immediately
    let listener_a = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr_a = listener_a.local_addr().unwrap();
    let _worker_a = tokio::spawn(async move {
        let (stream, _) = listener_a.accept().await.unwrap();
        let mut conn = FramedConnection::new(stream);
        let (_, payload) = conn.recv().await.unwrap();
        let hb: HeartbeatPayload = FramedConnection::deserialize_payload(&payload).unwrap();
        let ack = HeartbeatAckPayload {
            timestamp_echo: hb.timestamp_ns, nonce_echo: hb.nonce,
            gpu_memory_used: 4_000_000_000, active_sequences: 1, free_blocks: 100,
        };
        conn.send(MessageType::HeartbeatAck, 0, &ack).await.unwrap();
    });

    // Worker B: accepts but never responds
    let listener_b = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr_b = listener_b.local_addr().unwrap();
    let _worker_b = tokio::spawn(async move {
        let (stream, _) = listener_b.accept().await.unwrap();
        let mut conn = FramedConnection::new(stream);
        let _ = conn.recv().await;
        tokio::time::sleep(std::time::Duration::from_secs(30)).await;
    });

    let conn_a = FramedConnection::new(tokio::net::TcpStream::connect(addr_a).await.unwrap());
    let conn_b = FramedConnection::new(tokio::net::TcpStream::connect(addr_b).await.unwrap());

    let mut registry = PeerRegistry::new();
    registry.register(make_caps("fast"), conn_a).unwrap();
    registry.register(make_caps("slow"), conn_b).unwrap();
    registry.assign("fast", LayerAssignment {
        node_id: "fast".into(), layer_range: 0..16, role: NodeRole::Head,
        expected_decode_ms: 16.0, weight_memory_gb: 6.0, cache_memory_gb: 1.0,
    }).unwrap();
    registry.assign("slow", LayerAssignment {
        node_id: "slow".into(), layer_range: 16..32, role: NodeRole::Tail,
        expected_decode_ms: 16.0, weight_memory_gb: 6.0, cache_memory_gb: 1.0,
    }).unwrap();

    let mut tracker = HeartbeatTracker::new();
    let nonce: u64 = 777;

    // Send heartbeats to both
    let hb = HeartbeatPayload { timestamp_ns: 0, nonce };
    registry.get_mut("fast").unwrap().writer
        .send(MessageType::Heartbeat, 0, &hb).await.unwrap();
    registry.get_mut("slow").unwrap().writer
        .send(MessageType::Heartbeat, 0, &hb).await.unwrap();
    tracker.set_pending_nonce(nonce);

    // Concurrent poll — should complete quickly (not blocked by slow worker)
    let reader_handles = registry.reader_handles();
    assert_eq!(reader_handles.len(), 2);

    let start = std::time::Instant::now();
    let mut set = tokio::task::JoinSet::new();
    for (node_id, reader) in &reader_handles {
        let node_id = node_id.clone();
        let reader = reader.clone();
        set.spawn(async move {
            let result = tokio::time::timeout(
                std::time::Duration::from_millis(200),
                async { reader.lock().await.recv().await },
            ).await;
            let ack = match result {
                Ok(Ok((header, payload))) if header.msg_type == MessageType::HeartbeatAck => {
                    FramedConnection::deserialize_payload::<HeartbeatAckPayload>(&payload).ok()
                }
                _ => None,
            };
            (node_id, ack)
        });
    }
    let mut results = Vec::new();
    while let Some(Ok(r)) = set.join_next().await {
        results.push(r);
    }
    let elapsed = start.elapsed();

    // Should complete in ~200ms (the timeout), not 200ms × 2 = 400ms
    assert!(elapsed < std::time::Duration::from_millis(350),
        "concurrent poll took {elapsed:?}, expected < 350ms");

    // Process results
    for (node_id, ack) in &results {
        if let Some(ack) = ack {
            tracker.process_ack(&mut registry, node_id, ack);
        }
    }

    // fast worker: ack processed, counter reset, stats updated
    assert_eq!(tracker.missed_count("fast"), 0);
    assert_eq!(registry.get("fast").unwrap().free_blocks, 100);
    assert_eq!(registry.get("fast").unwrap().gpu_memory_used, 4_000_000_000);

    // slow worker: no ack, counter still 0 (hasn't been incremented yet)
    // After increment_missed, it goes to 1
    let dead = tracker.increment_missed(&["fast".into(), "slow".into()], 3);
    assert!(dead.is_empty());
    assert_eq!(tracker.missed_count("fast"), 1); // was 0 (ack), now +1
    assert_eq!(tracker.missed_count("slow"), 1); // was 0 (never acked), now +1
}

/// Worker sends a non-HeartbeatAck message during the poll window.
/// The poll should ignore it and treat the worker as not having acked.
#[tokio::test]
async fn test_poll_ignores_non_ack_messages() {
    use fracture_coordinator::heartbeat::HeartbeatTracker;

    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();

    // Worker responds with an Error message instead of HeartbeatAck
    let _worker = tokio::spawn(async move {
        let (stream, _) = listener.accept().await.unwrap();
        let mut conn = FramedConnection::new(stream);
        let _ = conn.recv().await;
        let err = ErrorPayload {
            error_code: ErrorCode::Internal,
            message: "something went wrong".into(),
        };
        conn.send(MessageType::Error, 0, &err).await.unwrap();
    });

    let conn = FramedConnection::new(tokio::net::TcpStream::connect(addr).await.unwrap());
    let mut registry = PeerRegistry::new();
    registry.register(make_caps("confused"), conn).unwrap();
    registry.assign("confused", LayerAssignment {
        node_id: "confused".into(), layer_range: 0..32, role: NodeRole::Head,
        expected_decode_ms: 32.0, weight_memory_gb: 12.0, cache_memory_gb: 2.0,
    }).unwrap();

    let mut tracker = HeartbeatTracker::new();
    let nonce = 42u64;
    let hb = HeartbeatPayload { timestamp_ns: 0, nonce };
    registry.get_mut("confused").unwrap().writer
        .send(MessageType::Heartbeat, 0, &hb).await.unwrap();
    tracker.set_pending_nonce(nonce);

    // Poll — worker sent Error, not HeartbeatAck
    let reader_handles = registry.reader_handles();
    let mut set = tokio::task::JoinSet::new();
    for (node_id, reader) in &reader_handles {
        let node_id = node_id.clone();
        let reader = reader.clone();
        set.spawn(async move {
            let result = tokio::time::timeout(
                std::time::Duration::from_millis(200),
                async { reader.lock().await.recv().await },
            ).await;
            let ack = match result {
                Ok(Ok((header, payload))) if header.msg_type == MessageType::HeartbeatAck => {
                    FramedConnection::deserialize_payload::<HeartbeatAckPayload>(&payload).ok()
                }
                _ => None,
            };
            (node_id, ack)
        });
    }
    let mut results = Vec::new();
    while let Some(Ok(r)) = set.join_next().await {
        results.push(r);
    }

    // Should have no valid acks
    for (node_id, ack) in &results {
        if let Some(ack) = ack {
            tracker.process_ack(&mut registry, node_id, ack);
        }
    }
    assert_eq!(tracker.missed_count("confused"), 0); // not incremented yet

    // After increment, worker is counted as missed
    let dead = tracker.increment_missed(&["confused".into()], 3);
    assert!(dead.is_empty());
    assert_eq!(tracker.missed_count("confused"), 1);
}

/// Simulates the full cycle order over the wire for 4 rounds:
/// worker responds to rounds 1-2, goes silent for rounds 3-4,
/// then is flagged dead on round 5.
#[tokio::test]
async fn test_full_lifecycle_over_wire() {
    use fracture_coordinator::heartbeat::{self, HeartbeatTracker};
    use fracture_coordinator::registry::WorkerStatus;

    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();

    // Worker responds to first 2 heartbeats, then stops
    let _worker = tokio::spawn(async move {
        let (stream, _) = listener.accept().await.unwrap();
        let mut conn = FramedConnection::new(stream);

        for _ in 0..2 {
            let (_, payload) = conn.recv().await.unwrap();
            let hb: HeartbeatPayload = FramedConnection::deserialize_payload(&payload).unwrap();
            let ack = HeartbeatAckPayload {
                timestamp_echo: hb.timestamp_ns, nonce_echo: hb.nonce,
                gpu_memory_used: 0, active_sequences: 0, free_blocks: 50,
            };
            conn.send(MessageType::HeartbeatAck, 0, &ack).await.unwrap();
        }
        // Read remaining heartbeats but don't respond
        loop {
            if conn.recv().await.is_err() { break; }
        }
    });

    let conn = FramedConnection::new(tokio::net::TcpStream::connect(addr).await.unwrap());
    let mut registry = PeerRegistry::new();
    registry.register(make_caps("lifecycle"), conn).unwrap();
    registry.assign("lifecycle", LayerAssignment {
        node_id: "lifecycle".into(), layer_range: 0..32, role: NodeRole::Head,
        expected_decode_ms: 32.0, weight_memory_gb: 12.0, cache_memory_gb: 2.0,
    }).unwrap();

    let mut tracker = HeartbeatTracker::new();
    let workers = vec!["lifecycle".to_string()];

    // Helper: run one heartbeat cycle (poll, increment, send)
    async fn cycle(
        tracker: &mut HeartbeatTracker,
        registry: &mut PeerRegistry,
        workers: &[String],
        nonce: u64,
    ) -> Vec<String> {
        // 1. Poll acks for previous nonce
        let reader_handles = registry.reader_handles();
        let mut ack_result = None;
        if !reader_handles.is_empty() {
            let (_, reader) = &reader_handles[0];
            if let Ok(Ok((header, payload))) = tokio::time::timeout(
                std::time::Duration::from_millis(100),
                reader.lock().await.recv(),
            ).await {
                if header.msg_type == MessageType::HeartbeatAck {
                    ack_result = FramedConnection::deserialize_payload::<HeartbeatAckPayload>(&payload).ok();
                }
            }
        }
        if let Some(ack) = &ack_result {
            tracker.process_ack(registry, &workers[0], ack);
        }

        // 2. Increment missed
        let dead = tracker.increment_missed(workers, 3);

        // 3. Send new heartbeat
        let hb = HeartbeatPayload { timestamp_ns: 0, nonce };
        registry.get_mut(&workers[0]).unwrap().writer
            .send(MessageType::Heartbeat, 0, &hb).await.unwrap();
        tracker.set_pending_nonce(nonce);

        dead
    }

    // Cycle 1: first heartbeat (no prior nonce, skip poll/increment)
    let hb = HeartbeatPayload { timestamp_ns: 0, nonce: 100 };
    registry.get_mut("lifecycle").unwrap().writer
        .send(MessageType::Heartbeat, 0, &hb).await.unwrap();
    tracker.set_pending_nonce(100);

    // Give worker time to respond
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;

    // Cycle 2: worker acked round 1 — should be healthy
    let dead = cycle(&mut tracker, &mut registry, &workers, 200).await;
    assert!(dead.is_empty());
    assert_eq!(tracker.missed_count("lifecycle"), 1); // 0 (ack) + 1 (increment)

    tokio::time::sleep(std::time::Duration::from_millis(50)).await;

    // Cycle 3: worker acked round 2 — still healthy
    let dead = cycle(&mut tracker, &mut registry, &workers, 300).await;
    assert!(dead.is_empty());
    assert_eq!(tracker.missed_count("lifecycle"), 1);

    // Worker stops responding after this point
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;

    // Cycle 4: no ack for round 3 — missed count rises
    let dead = cycle(&mut tracker, &mut registry, &workers, 400).await;
    assert!(dead.is_empty());
    assert_eq!(tracker.missed_count("lifecycle"), 2);

    tokio::time::sleep(std::time::Duration::from_millis(50)).await;

    // Cycle 5: still no ack — missed count hits threshold
    let dead = cycle(&mut tracker, &mut registry, &workers, 500).await;
    assert_eq!(dead, vec!["lifecycle"]);
    assert_eq!(tracker.missed_count("lifecycle"), 3);

    // Mark dead
    heartbeat::mark_dead_workers(&mut registry, &dead);
    assert_eq!(registry.get("lifecycle").unwrap().status, WorkerStatus::Dead);
    assert!(registry.pipeline_order().is_empty());
}

#[test]
fn test_heartbeat_constants() {
    use fracture_coordinator::heartbeat::{DEFAULT_INTERVAL, DEFAULT_MAX_MISSED};

    assert_eq!(DEFAULT_INTERVAL, std::time::Duration::from_secs(5));
    assert_eq!(DEFAULT_MAX_MISSED, 3);
    // 3 missed × 5s = 15s failure detection, matching the arch doc
    assert_eq!(
        DEFAULT_INTERVAL * DEFAULT_MAX_MISSED as u32,
        std::time::Duration::from_secs(15)
    );
}

// ── Pipeline unexpected message type test ────────────────────────────────

/// Spawn a worker that sends a Heartbeat instead of ForwardResult after Forward.
async fn spawn_unexpected_response_worker(listener: TcpListener) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        let (stream, _) = listener.accept().await.unwrap();
        let mut conn = FramedConnection::new(stream);

        loop {
            let (header, _payload) = match conn.recv().await {
                Ok(frame) => frame,
                Err(_) => break,
            };
            match header.msg_type {
                MessageType::CacheAlloc => {
                    conn.send_empty(MessageType::CacheAllocAck, header.seq_id)
                        .await
                        .unwrap();
                }
                MessageType::CacheFree => {}
                MessageType::Forward => {
                    // Send Heartbeat instead of ForwardResult (wrong!)
                    let hb = HeartbeatPayload { timestamp_ns: 0, nonce: 0 };
                    conn.send(MessageType::Heartbeat, header.seq_id, &hb).await.unwrap();
                }
                MessageType::Shutdown => break,
                _ => {}
            }
        }
    })
}

#[tokio::test]
async fn test_pipeline_unexpected_message_type_is_error() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let _task = spawn_unexpected_response_worker(listener).await;

    let stream = tokio::net::TcpStream::connect(addr).await.unwrap();
    let conn = FramedConnection::new(stream);

    let mut registry = PeerRegistry::new();
    registry.register(make_caps("bad"), conn).unwrap();
    let assignment = LayerAssignment {
        node_id: "bad".into(), layer_range: 0..32, role: NodeRole::Head,
        expected_decode_ms: 32.0, weight_memory_gb: 12.0, cache_memory_gb: 2.0,
    };
    registry.assign("bad", assignment.clone()).unwrap();

    let pipeline = DistributedPipeline::new(&[assignment], 4096).unwrap();
    pipeline.alloc_cache(&mut registry, 1, 4096).await.unwrap();

    let result = pipeline.forward(&mut registry, 1, &[1], &[0], true).await;
    assert!(result.is_err());
    let err_msg = result.unwrap_err().to_string();
    assert!(
        err_msg.contains("expected ForwardResult") || err_msg.contains("Heartbeat"),
        "error should mention unexpected message type: {err_msg}"
    );
}

// ── Activation shape validation test ────────────────────────────────────

/// Spawn a worker that returns activations with wrong hidden_size.
async fn spawn_wrong_shape_worker(
    listener: TcpListener,
    wrong_hidden_size: usize,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        let (stream, _) = listener.accept().await.unwrap();
        let mut conn = FramedConnection::new(stream);

        loop {
            let (header, _payload) = match conn.recv().await {
                Ok(frame) => frame,
                Err(_) => break,
            };
            match header.msg_type {
                MessageType::CacheAlloc => {
                    conn.send_empty(MessageType::CacheAllocAck, header.seq_id)
                        .await
                        .unwrap();
                }
                MessageType::CacheFree => {}
                MessageType::Forward => {
                    let data_len = wrong_hidden_size * 2;
                    let result = ForwardResultPayload {
                        output: ForwardOutputWire::Activations {
                            tensor_header: TensorWireHeader {
                                ndim: 2,
                                shape: vec![1, wrong_hidden_size as u32],
                                dtype: 0,
                                compression: 0,
                                data_len: data_len as u32,
                            },
                            tensor_data: vec![0u8; data_len],
                        },
                    };
                    conn.send(MessageType::ForwardResult, header.seq_id, &result)
                        .await
                        .unwrap();
                }
                MessageType::Shutdown => break,
                _ => {}
            }
        }
    })
}

#[tokio::test]
async fn test_activation_shape_mismatch_detected() {
    // Head worker returns activations with hidden_size=8192, but pipeline expects 4096
    let head_listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let tail_listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let head_addr = head_listener.local_addr().unwrap();
    let tail_addr = tail_listener.local_addr().unwrap();

    let _h = spawn_wrong_shape_worker(head_listener, 8192).await;
    let _t = spawn_mock_worker(tail_listener, true, 4096, 128256).await;

    let head_conn = FramedConnection::new(tokio::net::TcpStream::connect(head_addr).await.unwrap());
    let tail_conn = FramedConnection::new(tokio::net::TcpStream::connect(tail_addr).await.unwrap());

    let mut registry = PeerRegistry::new();
    registry.register(make_caps("head"), head_conn).unwrap();
    registry.register(make_caps("tail"), tail_conn).unwrap();

    let assignments = vec![
        LayerAssignment { node_id: "head".into(), layer_range: 0..16, role: NodeRole::Head, expected_decode_ms: 16.0, weight_memory_gb: 6.0, cache_memory_gb: 1.0 },
        LayerAssignment { node_id: "tail".into(), layer_range: 16..32, role: NodeRole::Tail, expected_decode_ms: 16.0, weight_memory_gb: 6.0, cache_memory_gb: 1.0 },
    ];
    for a in &assignments { registry.assign(&a.node_id, a.clone()).unwrap(); }

    let pipeline = DistributedPipeline::new(&assignments, 4096).unwrap();
    pipeline.alloc_cache(&mut registry, 1, 4096).await.unwrap();

    let result = pipeline.forward(&mut registry, 1, &[1], &[0], true).await;
    assert!(result.is_err());
    let err_msg = result.unwrap_err().to_string();
    assert!(
        err_msg.contains("hidden_size") || err_msg.contains("8192"),
        "error should mention shape mismatch: {err_msg}"
    );
}

// ── Distributed pipeline error path tests ───────────────────────────────

#[tokio::test]
async fn test_alloc_cache_unknown_worker_is_error() {
    // Pipeline references "missing" which isn't in the registry
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let _task = spawn_mock_worker(listener, true, 4096, 128256).await;

    let conn = FramedConnection::new(tokio::net::TcpStream::connect(addr).await.unwrap());
    let mut registry = PeerRegistry::new();
    registry.register(make_caps("present"), conn).unwrap();
    let assignment = LayerAssignment {
        node_id: "present".into(), layer_range: 0..16, role: NodeRole::Head,
        expected_decode_ms: 16.0, weight_memory_gb: 6.0, cache_memory_gb: 1.0,
    };
    registry.assign("present", assignment.clone()).unwrap();

    // Create pipeline with a second node that doesn't exist in registry
    let missing_assignment = LayerAssignment {
        node_id: "missing".into(), layer_range: 16..32, role: NodeRole::Tail,
        expected_decode_ms: 16.0, weight_memory_gb: 6.0, cache_memory_gb: 1.0,
    };
    let pipeline = DistributedPipeline::new(&[assignment, missing_assignment], 4096).unwrap();

    let result = pipeline.alloc_cache(&mut registry, 1, 4096).await;
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("missing"));
}

#[tokio::test]
async fn test_free_cache_unknown_worker_is_error() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let _task = spawn_mock_worker(listener, true, 4096, 128256).await;

    let conn = FramedConnection::new(tokio::net::TcpStream::connect(addr).await.unwrap());
    let mut registry = PeerRegistry::new();
    registry.register(make_caps("present"), conn).unwrap();
    let assignment = LayerAssignment {
        node_id: "present".into(), layer_range: 0..16, role: NodeRole::Head,
        expected_decode_ms: 16.0, weight_memory_gb: 6.0, cache_memory_gb: 1.0,
    };
    registry.assign("present", assignment.clone()).unwrap();

    let missing_assignment = LayerAssignment {
        node_id: "ghost".into(), layer_range: 16..32, role: NodeRole::Tail,
        expected_decode_ms: 16.0, weight_memory_gb: 6.0, cache_memory_gb: 1.0,
    };
    let pipeline = DistributedPipeline::new(&[assignment, missing_assignment], 4096).unwrap();

    // free_cache for a never-allocated seq_id returns "not allocated" error
    let result = pipeline.free_cache(&mut registry, 1).await;
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("not allocated"));
}

#[tokio::test]
async fn test_forward_unknown_worker_is_error() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let _task = spawn_mock_worker(listener, true, 4096, 128256).await;

    let conn = FramedConnection::new(tokio::net::TcpStream::connect(addr).await.unwrap());
    let mut registry = PeerRegistry::new();
    registry.register(make_caps("exists"), conn).unwrap();
    let assignment = LayerAssignment {
        node_id: "exists".into(), layer_range: 0..16, role: NodeRole::Head,
        expected_decode_ms: 16.0, weight_memory_gb: 6.0, cache_memory_gb: 1.0,
    };
    registry.assign("exists", assignment.clone()).unwrap();

    let missing = LayerAssignment {
        node_id: "vanished".into(), layer_range: 16..32, role: NodeRole::Tail,
        expected_decode_ms: 16.0, weight_memory_gb: 6.0, cache_memory_gb: 1.0,
    };
    let pipeline = DistributedPipeline::new(&[assignment, missing], 4096).unwrap();

    // Forward without alloc_cache now fails with cache-not-allocated error
    let result = pipeline.forward(&mut registry, 1, &[1], &[0], true).await;
    assert!(result.is_err(), "should fail when cache is not allocated");
    assert!(result.unwrap_err().to_string().contains("cache not allocated"));
}

// ── Pipeline cache tracking and single-worker tests ────────────────────

/// Helper: set up a single-worker pipeline (head+tail) with a mock worker.
async fn setup_single_node_pipeline() -> (DistributedPipeline, PeerRegistry) {
    let hidden_size = 4096;
    let vocab_size = 128256;

    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();

    // Single worker acts as tail (returns logits)
    let _task = spawn_mock_worker(listener, true, hidden_size, vocab_size).await;

    let conn = FramedConnection::new(
        tokio::net::TcpStream::connect(addr).await.unwrap(),
    );

    let mut registry = PeerRegistry::new();
    registry.register(make_caps("solo"), conn).unwrap();

    let assignment = LayerAssignment {
        node_id: "solo".into(),
        layer_range: 0..32,
        role: NodeRole::Head,
        expected_decode_ms: 32.0,
        weight_memory_gb: 12.0,
        cache_memory_gb: 2.0,
    };
    registry.assign("solo", assignment.clone()).unwrap();

    let pipeline = DistributedPipeline::new(&[assignment], 4096).unwrap();
    (pipeline, registry)
}

#[tokio::test]
async fn test_single_worker_forward_returns_logits() {
    let (pipeline, mut registry) = setup_single_node_pipeline().await;
    let seq_id = 1;

    pipeline.alloc_cache(&mut registry, seq_id, 4096).await.unwrap();

    // Prefill
    let logits = pipeline
        .forward(&mut registry, seq_id, &[128000, 791, 1401], &[0, 1, 2], true)
        .await
        .unwrap();
    assert_eq!(logits.len(), 128256);
    assert!((logits[0] - 0.0).abs() < 1e-6);
    assert!((logits[42] - 42.0).abs() < 1e-6);

    // Decode step
    let logits = pipeline
        .forward(&mut registry, seq_id, &[42], &[3], false)
        .await
        .unwrap();
    assert_eq!(logits.len(), 128256);

    pipeline.free_cache(&mut registry, seq_id).await.unwrap();
}

#[tokio::test]
async fn test_forward_without_cache_is_error() {
    let (pipeline, mut registry) = setup_single_node_pipeline().await;

    // Forward without alloc_cache
    let result = pipeline.forward(&mut registry, 99, &[1, 2], &[0, 1], true).await;
    assert!(result.is_err());
    let err_msg = result.unwrap_err().to_string();
    assert!(
        err_msg.contains("cache not allocated"),
        "expected cache-not-allocated error: {err_msg}"
    );
}

#[tokio::test]
async fn test_duplicate_cache_alloc_is_error() {
    let (pipeline, mut registry) = setup_single_node_pipeline().await;
    let seq_id = 10;

    // First alloc succeeds
    pipeline.alloc_cache(&mut registry, seq_id, 4096).await.unwrap();

    // Second alloc for same seq_id fails
    let result = pipeline.alloc_cache(&mut registry, seq_id, 4096).await;
    assert!(result.is_err());
    let err_msg = result.unwrap_err().to_string();
    assert!(
        err_msg.contains("duplicate") || err_msg.contains("already allocated"),
        "expected duplicate-alloc error: {err_msg}"
    );

    // Cleanup
    pipeline.free_cache(&mut registry, seq_id).await.unwrap();
}

#[tokio::test]
async fn test_free_unknown_sequence_is_error() {
    let (pipeline, mut registry) = setup_single_node_pipeline().await;

    // Free a seq_id that was never allocated
    let result = pipeline.free_cache(&mut registry, 999).await;
    assert!(result.is_err());
    let err_msg = result.unwrap_err().to_string();
    assert!(
        err_msg.contains("not allocated") || err_msg.contains("already freed"),
        "expected not-allocated error: {err_msg}"
    );
}

#[tokio::test]
async fn test_double_free_is_error() {
    let (pipeline, mut registry) = setup_single_node_pipeline().await;
    let seq_id = 20;

    pipeline.alloc_cache(&mut registry, seq_id, 4096).await.unwrap();
    pipeline.free_cache(&mut registry, seq_id).await.unwrap();

    // Second free fails
    let result = pipeline.free_cache(&mut registry, seq_id).await;
    assert!(result.is_err());
    let err_msg = result.unwrap_err().to_string();
    assert!(
        err_msg.contains("not allocated") || err_msg.contains("already freed"),
        "expected already-freed error: {err_msg}"
    );
}

#[tokio::test]
async fn test_cache_reuse_after_free() {
    let (pipeline, mut registry) = setup_single_node_pipeline().await;
    let seq_id = 30;

    // Alloc, use, free
    pipeline.alloc_cache(&mut registry, seq_id, 4096).await.unwrap();
    let logits = pipeline
        .forward(&mut registry, seq_id, &[1], &[0], false)
        .await
        .unwrap();
    assert_eq!(logits.len(), 128256);
    pipeline.free_cache(&mut registry, seq_id).await.unwrap();

    // Re-alloc same seq_id succeeds
    pipeline.alloc_cache(&mut registry, seq_id, 4096).await.unwrap();
    let logits = pipeline
        .forward(&mut registry, seq_id, &[2], &[0], false)
        .await
        .unwrap();
    assert_eq!(logits.len(), 128256);
    pipeline.free_cache(&mut registry, seq_id).await.unwrap();
}

#[tokio::test]
async fn test_multiple_sequence_isolation() {
    let (pipeline, mut registry) = setup_single_node_pipeline().await;
    let seq_a = 100;
    let seq_b = 200;

    // Allocate both sequences
    pipeline.alloc_cache(&mut registry, seq_a, 4096).await.unwrap();
    pipeline.alloc_cache(&mut registry, seq_b, 4096).await.unwrap();

    // Forward sequence A
    let logits_a = pipeline
        .forward(&mut registry, seq_a, &[1, 2, 3], &[0, 1, 2], true)
        .await
        .unwrap();
    assert_eq!(logits_a.len(), 128256);

    // Forward sequence B (independent)
    let logits_b = pipeline
        .forward(&mut registry, seq_b, &[10, 20], &[0, 1], true)
        .await
        .unwrap();
    assert_eq!(logits_b.len(), 128256);

    // Interleaved decode steps
    let logits_a2 = pipeline
        .forward(&mut registry, seq_a, &[4], &[3], false)
        .await
        .unwrap();
    assert_eq!(logits_a2.len(), 128256);

    let logits_b2 = pipeline
        .forward(&mut registry, seq_b, &[30], &[2], false)
        .await
        .unwrap();
    assert_eq!(logits_b2.len(), 128256);

    // Free A, B should still work
    pipeline.free_cache(&mut registry, seq_a).await.unwrap();
    let logits_b3 = pipeline
        .forward(&mut registry, seq_b, &[40], &[3], false)
        .await
        .unwrap();
    assert_eq!(logits_b3.len(), 128256);

    // A is freed, forward should fail
    let result = pipeline.forward(&mut registry, seq_a, &[5], &[4], false).await;
    assert!(result.is_err());

    pipeline.free_cache(&mut registry, seq_b).await.unwrap();
}

// ── Partial cache alloc rollback test ──────────────────────────────────

/// Spawn a mock worker that fails CacheAlloc with an OOM error.
async fn spawn_oom_alloc_worker(listener: TcpListener) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        let (stream, _) = listener.accept().await.unwrap();
        let mut conn = FramedConnection::new(stream);

        loop {
            let (header, _payload) = match conn.recv().await {
                Ok(frame) => frame,
                Err(_) => break,
            };
            match header.msg_type {
                MessageType::CacheAlloc => {
                    // Respond with Error (OOM)
                    let err = ErrorPayload {
                        error_code: ErrorCode::OutOfMemory,
                        message: "GPU OOM during cache allocation".into(),
                    };
                    conn.send(MessageType::Error, header.seq_id, &err)
                        .await
                        .unwrap();
                }
                MessageType::CacheFree => {}
                MessageType::Shutdown => break,
                _ => {}
            }
        }
    })
}

/// Spawn a mock worker that tracks CacheAlloc/CacheFree via a channel,
/// so the test can verify rollback (CacheFree sent after partial failure).
async fn spawn_tracking_worker(
    listener: TcpListener,
    event_tx: tokio::sync::mpsc::UnboundedSender<(MessageType, u64)>,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        let (stream, _) = listener.accept().await.unwrap();
        let mut conn = FramedConnection::new(stream);

        loop {
            let (header, _payload) = match conn.recv().await {
                Ok(frame) => frame,
                Err(_) => break,
            };
            match header.msg_type {
                MessageType::CacheAlloc => {
                    let _ = event_tx.send((MessageType::CacheAlloc, header.seq_id));
                    conn.send_empty(MessageType::CacheAllocAck, header.seq_id)
                        .await
                        .unwrap();
                }
                MessageType::CacheFree => {
                    let _ = event_tx.send((MessageType::CacheFree, header.seq_id));
                }
                MessageType::Forward => {
                    // Return logits (acts as tail)
                    let data: Vec<u8> = (0..128256u32)
                        .flat_map(|i| (i as f32).to_le_bytes())
                        .collect();
                    let result = ForwardResultPayload {
                        output: ForwardOutputWire::Logits { data },
                    };
                    conn.send(MessageType::ForwardResult, header.seq_id, &result)
                        .await
                        .unwrap();
                }
                MessageType::Shutdown => break,
                _ => {}
            }
        }
    })
}

#[tokio::test]
async fn test_partial_cache_alloc_rollback() {
    // Set up two workers: head (tracking) succeeds, tail (OOM) fails.
    // The coordinator should rollback by sending CacheFree to head.
    let head_listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let tail_listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let head_addr = head_listener.local_addr().unwrap();
    let tail_addr = tail_listener.local_addr().unwrap();

    let (event_tx, mut event_rx) = tokio::sync::mpsc::unbounded_channel();

    let _head_task = spawn_tracking_worker(head_listener, event_tx).await;
    let _tail_task = spawn_oom_alloc_worker(tail_listener).await;

    let head_conn = FramedConnection::new(
        tokio::net::TcpStream::connect(head_addr).await.unwrap(),
    );
    let tail_conn = FramedConnection::new(
        tokio::net::TcpStream::connect(tail_addr).await.unwrap(),
    );

    let mut registry = PeerRegistry::new();
    registry.register(make_caps("head"), head_conn).unwrap();
    registry.register(make_caps("tail"), tail_conn).unwrap();

    let assignments = vec![
        LayerAssignment {
            node_id: "head".into(),
            layer_range: 0..16,
            role: NodeRole::Head,
            expected_decode_ms: 16.0,
            weight_memory_gb: 6.0,
            cache_memory_gb: 1.0,
        },
        LayerAssignment {
            node_id: "tail".into(),
            layer_range: 16..32,
            role: NodeRole::Tail,
            expected_decode_ms: 16.0,
            weight_memory_gb: 6.0,
            cache_memory_gb: 1.0,
        },
    ];
    for a in &assignments {
        registry.assign(&a.node_id, a.clone()).unwrap();
    }

    let pipeline = DistributedPipeline::new(&assignments, 4096).unwrap();
    let seq_id = 42;

    // alloc_cache should fail because the tail worker returns OOM
    let result = pipeline.alloc_cache(&mut registry, seq_id, 4096).await;
    assert!(result.is_err());
    let err_msg = result.unwrap_err().to_string();
    assert!(
        err_msg.contains("GPU OOM") || err_msg.contains("OutOfMemory"),
        "expected OOM error: {err_msg}"
    );

    // Verify the sequence was NOT marked as allocated
    assert!(
        !pipeline.is_allocated(seq_id),
        "seq should not be marked allocated after partial failure"
    );

    // Verify events: head received CacheAlloc, then CacheFree (rollback)
    // Allow time for the rollback CacheFree to be delivered and processed.
    let mut events = Vec::new();
    for _ in 0..10 {
        while let Ok(event) = event_rx.try_recv() {
            events.push(event);
        }
        if events.len() >= 2 {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
    }
    assert_eq!(events.len(), 2, "expected 2 events (alloc + rollback free), got: {events:?}");
    assert_eq!(events[0], (MessageType::CacheAlloc, seq_id));
    assert_eq!(events[1], (MessageType::CacheFree, seq_id));
}

#[tokio::test]
async fn test_partial_cache_alloc_rollback_three_nodes() {
    // Three workers: head and mid succeed, tail fails.
    // Rollback should free caches on head and mid.
    let head_listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let mid_listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let tail_listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let head_addr = head_listener.local_addr().unwrap();
    let mid_addr = mid_listener.local_addr().unwrap();
    let tail_addr = tail_listener.local_addr().unwrap();

    let (head_tx, mut head_rx) = tokio::sync::mpsc::unbounded_channel();
    let (mid_tx, mut mid_rx) = tokio::sync::mpsc::unbounded_channel();

    let _head = spawn_tracking_worker(head_listener, head_tx).await;
    let _mid = spawn_tracking_worker(mid_listener, mid_tx).await;
    let _tail = spawn_oom_alloc_worker(tail_listener).await;

    let head_conn = FramedConnection::new(tokio::net::TcpStream::connect(head_addr).await.unwrap());
    let mid_conn = FramedConnection::new(tokio::net::TcpStream::connect(mid_addr).await.unwrap());
    let tail_conn = FramedConnection::new(tokio::net::TcpStream::connect(tail_addr).await.unwrap());

    let mut registry = PeerRegistry::new();
    registry.register(make_caps("head"), head_conn).unwrap();
    registry.register(make_caps("mid"), mid_conn).unwrap();
    registry.register(make_caps("tail"), tail_conn).unwrap();

    let assignments = vec![
        LayerAssignment {
            node_id: "head".into(), layer_range: 0..10, role: NodeRole::Head,
            expected_decode_ms: 10.0, weight_memory_gb: 4.0, cache_memory_gb: 0.5,
        },
        LayerAssignment {
            node_id: "mid".into(), layer_range: 10..20, role: NodeRole::Middle,
            expected_decode_ms: 10.0, weight_memory_gb: 4.0, cache_memory_gb: 0.5,
        },
        LayerAssignment {
            node_id: "tail".into(), layer_range: 20..32, role: NodeRole::Tail,
            expected_decode_ms: 12.0, weight_memory_gb: 5.0, cache_memory_gb: 0.6,
        },
    ];
    for a in &assignments {
        registry.assign(&a.node_id, a.clone()).unwrap();
    }

    let pipeline = DistributedPipeline::new(&assignments, 4096).unwrap();
    let seq_id = 77;

    let result = pipeline.alloc_cache(&mut registry, seq_id, 4096).await;
    assert!(result.is_err());
    assert!(!pipeline.is_allocated(seq_id));

    // Both head and mid should have received CacheAlloc then CacheFree (rollback).
    // Allow time for the rollback CacheFree to be delivered and processed.
    let mut head_events = Vec::new();
    let mut mid_events = Vec::new();
    for _ in 0..10 {
        while let Ok(e) = head_rx.try_recv() { head_events.push(e); }
        while let Ok(e) = mid_rx.try_recv() { mid_events.push(e); }
        if head_events.len() >= 2 && mid_events.len() >= 2 {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
    }

    assert_eq!(head_events.len(), 2, "head: expected alloc+free, got: {head_events:?}");
    assert_eq!(head_events[0].0, MessageType::CacheAlloc);
    assert_eq!(head_events[1].0, MessageType::CacheFree);

    assert_eq!(mid_events.len(), 2, "mid: expected alloc+free, got: {mid_events:?}");
    assert_eq!(mid_events[0].0, MessageType::CacheAlloc);
    assert_eq!(mid_events[1].0, MessageType::CacheFree);
}

// ── Batched Forward Tests ──────────────────────────────────────────────

#[tokio::test]
async fn test_batched_forward_two_nodes_returns_per_sequence_logits() {
    let (pipeline, mut registry) = setup_two_node_pipeline().await;

    let seq1 = 100;
    let seq2 = 101;

    // Allocate caches
    pipeline.alloc_cache(&mut registry, seq1, 0).await.unwrap();
    pipeline.alloc_cache(&mut registry, seq2, 0).await.unwrap();

    // Build batch metadata
    let sequences = vec![
        SequenceMetadataWire {
            seq_id: seq1,
            num_tokens: 3,
            positions: vec![0, 1, 2],
            block_table: vec![0],
            cache_seq_len: 3,
            last_block_tokens: 3,
        },
        SequenceMetadataWire {
            seq_id: seq2,
            num_tokens: 1,
            positions: vec![5],
            block_table: vec![1, 2],
            cache_seq_len: 6,
            last_block_tokens: 6,
        },
    ];

    let all_token_ids = vec![128000, 791, 1401, 42]; // 3 tokens seq1, 1 token seq2

    let per_seq_logits = pipeline
        .batched_forward(&mut registry, &sequences, &all_token_ids, true)
        .await
        .unwrap();

    assert_eq!(per_seq_logits.len(), 2, "should get logits for 2 sequences");

    let vocab_size = 128256;
    assert_eq!(per_seq_logits[0].len(), vocab_size, "seq 0 should have vocab_size logits");
    assert_eq!(per_seq_logits[1].len(), vocab_size, "seq 1 should have vocab_size logits");

    // Verify deterministic logit values from mock worker:
    // seq 0, logit j → j as f32
    assert_eq!(per_seq_logits[0][0], 0.0);
    assert_eq!(per_seq_logits[0][1], 1.0);

    // seq 1, logit j → (vocab_size + j) as f32
    assert_eq!(per_seq_logits[1][0], vocab_size as f32);
    assert_eq!(per_seq_logits[1][1], (vocab_size + 1) as f32);

    // Cleanup
    pipeline.free_cache(&mut registry, seq1).await.unwrap();
    pipeline.free_cache(&mut registry, seq2).await.unwrap();
}

#[tokio::test]
async fn test_batched_forward_single_sequence() {
    let (pipeline, mut registry) = setup_two_node_pipeline().await;

    let seq_id = 200;
    pipeline.alloc_cache(&mut registry, seq_id, 0).await.unwrap();

    let sequences = vec![SequenceMetadataWire {
        seq_id,
        num_tokens: 5,
        positions: vec![0, 1, 2, 3, 4],
        block_table: vec![0],
        cache_seq_len: 5,
        last_block_tokens: 5,
    }];

    let per_seq_logits = pipeline
        .batched_forward(
            &mut registry,
            &sequences,
            &[1, 2, 3, 4, 5],
            true,
        )
        .await
        .unwrap();

    assert_eq!(per_seq_logits.len(), 1);
    assert_eq!(per_seq_logits[0].len(), 128256);

    pipeline.free_cache(&mut registry, seq_id).await.unwrap();
}

#[tokio::test]
async fn test_batched_forward_without_cache_is_error() {
    let (pipeline, mut registry) = setup_two_node_pipeline().await;

    let sequences = vec![SequenceMetadataWire {
        seq_id: 999, // never allocated
        num_tokens: 1,
        positions: vec![0],
        block_table: vec![],
        cache_seq_len: 0,
        last_block_tokens: 0,
    }];

    let result = pipeline
        .batched_forward(&mut registry, &sequences, &[1], false)
        .await;

    assert!(result.is_err());
    assert!(
        result.unwrap_err().to_string().contains("cache not allocated"),
        "should mention missing cache"
    );
}

#[tokio::test]
async fn test_batched_forward_single_node_pipeline() {
    // Setup single-node (head+tail) pipeline
    let hidden_size = 4096;
    let vocab_size = 128256;

    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();

    let _task = spawn_mock_worker(listener, true, hidden_size, vocab_size).await;
    let stream = tokio::net::TcpStream::connect(addr).await.unwrap();
    let conn = FramedConnection::new(stream);

    let mut registry = PeerRegistry::new();
    use fracture_coordinator::scheduler::WorkerCapabilities;
    registry
        .register(
            WorkerCapabilities {
                node_id: "solo".into(),
                gpu_model: "Mock".into(),
                gpu_memory_available: 24_000_000_000,
                compute_capability: (8, 0),
                decode_ms_per_layer: 1.0,
                prefill_ms_per_layer_128: 3.0,
            },
            conn,
        )
        .unwrap();
    registry
        .assign(
            "solo",
            LayerAssignment {
                node_id: "solo".into(),
                layer_range: 0..32,
                role: NodeRole::Head,
                expected_decode_ms: 32.0,
                weight_memory_gb: 12.0,
                cache_memory_gb: 2.0,
            },
        )
        .unwrap();

    let pipeline = DistributedPipeline::new(
        &[LayerAssignment {
            node_id: "solo".into(),
            layer_range: 0..32,
            role: NodeRole::Head,
            expected_decode_ms: 32.0,
            weight_memory_gb: 12.0,
            cache_memory_gb: 2.0,
        }],
        hidden_size,
    )
    .unwrap();

    let seq_id = 1;
    pipeline.alloc_cache(&mut registry, seq_id, 0).await.unwrap();

    let sequences = vec![SequenceMetadataWire {
        seq_id,
        num_tokens: 2,
        positions: vec![0, 1],
        block_table: vec![0],
        cache_seq_len: 2,
        last_block_tokens: 2,
    }];

    let per_seq_logits = pipeline
        .batched_forward(&mut registry, &sequences, &[1, 2], true)
        .await
        .unwrap();

    assert_eq!(per_seq_logits.len(), 1);
    assert_eq!(per_seq_logits[0].len(), vocab_size);

    pipeline.free_cache(&mut registry, seq_id).await.unwrap();
}

#[tokio::test]
async fn test_batched_forward_three_node_pipeline() {
    let hidden_size = 4096;
    let vocab_size = 128256;

    let head_listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let mid_listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let tail_listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let head_addr = head_listener.local_addr().unwrap();
    let mid_addr = mid_listener.local_addr().unwrap();
    let tail_addr = tail_listener.local_addr().unwrap();

    let _h = spawn_mock_worker(head_listener, false, hidden_size, vocab_size).await;
    let _m = spawn_mock_worker(mid_listener, false, hidden_size, vocab_size).await;
    let _t = spawn_mock_worker(tail_listener, true, hidden_size, vocab_size).await;

    let head_conn = FramedConnection::new(
        tokio::net::TcpStream::connect(head_addr).await.unwrap(),
    );
    let mid_conn = FramedConnection::new(
        tokio::net::TcpStream::connect(mid_addr).await.unwrap(),
    );
    let tail_conn = FramedConnection::new(
        tokio::net::TcpStream::connect(tail_addr).await.unwrap(),
    );

    let mut registry = PeerRegistry::new();
    use fracture_coordinator::scheduler::WorkerCapabilities;

    for (id, conn) in [("head", head_conn), ("mid", mid_conn), ("tail", tail_conn)] {
        registry
            .register(
                WorkerCapabilities {
                    node_id: id.into(),
                    gpu_model: "Mock".into(),
                    gpu_memory_available: 24_000_000_000,
                    compute_capability: (8, 0),
                    decode_ms_per_layer: 1.0,
                    prefill_ms_per_layer_128: 3.0,
                },
                conn,
            )
            .unwrap();
    }

    let assignments = vec![
        LayerAssignment {
            node_id: "head".into(),
            layer_range: 0..10,
            role: NodeRole::Head,
            expected_decode_ms: 10.0,
            weight_memory_gb: 4.0,
            cache_memory_gb: 0.5,
        },
        LayerAssignment {
            node_id: "mid".into(),
            layer_range: 10..20,
            role: NodeRole::Middle,
            expected_decode_ms: 10.0,
            weight_memory_gb: 4.0,
            cache_memory_gb: 0.5,
        },
        LayerAssignment {
            node_id: "tail".into(),
            layer_range: 20..32,
            role: NodeRole::Tail,
            expected_decode_ms: 12.0,
            weight_memory_gb: 5.0,
            cache_memory_gb: 0.6,
        },
    ];

    for a in &assignments {
        registry.assign(&a.node_id, a.clone()).unwrap();
    }

    let pipeline = DistributedPipeline::new(&assignments, hidden_size).unwrap();

    let seq1 = 10;
    let seq2 = 11;
    pipeline.alloc_cache(&mut registry, seq1, 0).await.unwrap();
    pipeline.alloc_cache(&mut registry, seq2, 0).await.unwrap();

    let sequences = vec![
        SequenceMetadataWire {
            seq_id: seq1,
            num_tokens: 2,
            positions: vec![0, 1],
            block_table: vec![0],
            cache_seq_len: 2,
            last_block_tokens: 2,
        },
        SequenceMetadataWire {
            seq_id: seq2,
            num_tokens: 1,
            positions: vec![3],
            block_table: vec![1],
            cache_seq_len: 4,
            last_block_tokens: 4,
        },
    ];

    let per_seq_logits = pipeline
        .batched_forward(&mut registry, &sequences, &[1, 2, 3], true)
        .await
        .unwrap();

    assert_eq!(per_seq_logits.len(), 2);
    assert_eq!(per_seq_logits[0].len(), vocab_size);
    assert_eq!(per_seq_logits[1].len(), vocab_size);

    pipeline.free_cache(&mut registry, seq1).await.unwrap();
    pipeline.free_cache(&mut registry, seq2).await.unwrap();
}

/// Spawn a mock worker that responds to BatchedForward with a non-batched
/// ForwardResult (testing the fallback path in the coordinator).
async fn spawn_fallback_mock_worker(
    listener: TcpListener,
    is_tail: bool,
    hidden_size: usize,
    vocab_size: usize,
    num_sequences: usize,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        let (stream, _) = listener.accept().await.unwrap();
        let mut conn = FramedConnection::new(stream);

        loop {
            let (header, payload) = match conn.recv().await {
                Ok(frame) => frame,
                Err(_) => break,
            };

            match header.msg_type {
                MessageType::CacheAlloc => {
                    conn.send_empty(MessageType::CacheAllocAck, header.seq_id)
                        .await
                        .unwrap();
                }
                MessageType::CacheFree => {}
                MessageType::BatchedForward => {
                    // Respond with non-batched ForwardResult (fallback path)
                    let result = if is_tail {
                        let data: Vec<u8> = (0..num_sequences)
                            .flat_map(|si| {
                                (0..vocab_size)
                                    .flat_map(move |j| {
                                        ((si * vocab_size + j) as f32).to_le_bytes()
                                    })
                            })
                            .collect();
                        ForwardResultPayload {
                            output: ForwardOutputWire::Logits { data },
                        }
                    } else {
                        let _req: BatchedForwardPayload =
                            FramedConnection::deserialize_payload(&payload).unwrap();
                        let total_tokens: usize =
                            _req.sequences.iter().map(|s| s.num_tokens).sum();
                        let data_len = total_tokens * hidden_size * 2;
                        ForwardResultPayload {
                            output: ForwardOutputWire::Activations {
                                tensor_header: TensorWireHeader {
                                    ndim: 2,
                                    shape: vec![total_tokens as u32, hidden_size as u32],
                                    dtype: 0,
                                    compression: 0,
                                    data_len: data_len as u32,
                                },
                                tensor_data: vec![0u8; data_len],
                            },
                        }
                    };
                    // Send ForwardResult instead of BatchedForwardResult
                    conn.send(MessageType::ForwardResult, header.seq_id, &result)
                        .await
                        .unwrap();
                }
                MessageType::Shutdown => break,
                _ => {}
            }
        }
    })
}

#[tokio::test]
async fn test_batched_forward_fallback_to_forward_result() {
    let hidden_size = 4096;
    let vocab_size = 128256;
    let num_seqs = 2;

    let head_listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let tail_listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let head_addr = head_listener.local_addr().unwrap();
    let tail_addr = tail_listener.local_addr().unwrap();

    let _h = spawn_fallback_mock_worker(head_listener, false, hidden_size, vocab_size, num_seqs).await;
    let _t = spawn_fallback_mock_worker(tail_listener, true, hidden_size, vocab_size, num_seqs).await;

    let head_conn = FramedConnection::new(
        tokio::net::TcpStream::connect(head_addr).await.unwrap(),
    );
    let tail_conn = FramedConnection::new(
        tokio::net::TcpStream::connect(tail_addr).await.unwrap(),
    );

    let mut registry = PeerRegistry::new();
    use fracture_coordinator::scheduler::WorkerCapabilities;

    for (id, conn) in [("head", head_conn), ("tail", tail_conn)] {
        registry
            .register(
                WorkerCapabilities {
                    node_id: id.into(),
                    gpu_model: "Mock".into(),
                    gpu_memory_available: 24_000_000_000,
                    compute_capability: (8, 0),
                    decode_ms_per_layer: 1.0,
                    prefill_ms_per_layer_128: 3.0,
                },
                conn,
            )
            .unwrap();
    }

    let head_assignment = LayerAssignment {
        node_id: "head".into(),
        layer_range: 0..16,
        role: NodeRole::Head,
        expected_decode_ms: 16.0,
        weight_memory_gb: 6.0,
        cache_memory_gb: 1.0,
    };
    let tail_assignment = LayerAssignment {
        node_id: "tail".into(),
        layer_range: 16..32,
        role: NodeRole::Tail,
        expected_decode_ms: 16.0,
        weight_memory_gb: 6.0,
        cache_memory_gb: 1.0,
    };

    registry.assign("head", head_assignment.clone()).unwrap();
    registry.assign("tail", tail_assignment.clone()).unwrap();

    let pipeline = DistributedPipeline::new(&[head_assignment, tail_assignment], hidden_size).unwrap();

    let seq1 = 50;
    let seq2 = 51;
    pipeline.alloc_cache(&mut registry, seq1, 0).await.unwrap();
    pipeline.alloc_cache(&mut registry, seq2, 0).await.unwrap();

    let sequences = vec![
        SequenceMetadataWire {
            seq_id: seq1,
            num_tokens: 2,
            positions: vec![0, 1],
            block_table: vec![0],
            cache_seq_len: 2,
            last_block_tokens: 2,
        },
        SequenceMetadataWire {
            seq_id: seq2,
            num_tokens: 1,
            positions: vec![0],
            block_table: vec![1],
            cache_seq_len: 1,
            last_block_tokens: 1,
        },
    ];

    // Workers respond with ForwardResult (not BatchedForwardResult) —
    // coordinator should handle this fallback gracefully.
    let per_seq_logits = pipeline
        .batched_forward(&mut registry, &sequences, &[1, 2, 3], true)
        .await
        .unwrap();

    assert_eq!(per_seq_logits.len(), 2);
    assert_eq!(per_seq_logits[0].len(), vocab_size);
    assert_eq!(per_seq_logits[1].len(), vocab_size);

    pipeline.free_cache(&mut registry, seq1).await.unwrap();
    pipeline.free_cache(&mut registry, seq2).await.unwrap();
}

/// Spawn a mock worker that returns an error for BatchedForward.
async fn spawn_error_batched_worker(
    listener: TcpListener,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        let (stream, _) = listener.accept().await.unwrap();
        let mut conn = FramedConnection::new(stream);

        loop {
            let (header, _payload) = match conn.recv().await {
                Ok(frame) => frame,
                Err(_) => break,
            };

            match header.msg_type {
                MessageType::CacheAlloc => {
                    conn.send_empty(MessageType::CacheAllocAck, header.seq_id)
                        .await
                        .unwrap();
                }
                MessageType::CacheFree => {}
                MessageType::BatchedForward => {
                    let err = ErrorPayload {
                        error_code: ErrorCode::Internal,
                        message: "batched forward failed: GPU error".into(),
                    };
                    conn.send(MessageType::Error, header.seq_id, &err)
                        .await
                        .unwrap();
                }
                MessageType::Shutdown => break,
                _ => {}
            }
        }
    })
}

#[tokio::test]
async fn test_batched_forward_worker_error_propagates() {
    let hidden_size = 4096;

    let head_listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let head_addr = head_listener.local_addr().unwrap();

    // Single-node pipeline that errors on BatchedForward
    let _h = spawn_error_batched_worker(head_listener).await;

    let head_conn = FramedConnection::new(
        tokio::net::TcpStream::connect(head_addr).await.unwrap(),
    );

    let mut registry = PeerRegistry::new();
    use fracture_coordinator::scheduler::WorkerCapabilities;

    registry
        .register(
            WorkerCapabilities {
                node_id: "head".into(),
                gpu_model: "Mock".into(),
                gpu_memory_available: 24_000_000_000,
                compute_capability: (8, 0),
                decode_ms_per_layer: 1.0,
                prefill_ms_per_layer_128: 3.0,
            },
            head_conn,
        )
        .unwrap();

    registry
        .assign(
            "head",
            LayerAssignment {
                node_id: "head".into(),
                layer_range: 0..32,
                role: NodeRole::Head,
                expected_decode_ms: 32.0,
                weight_memory_gb: 12.0,
                cache_memory_gb: 2.0,
            },
        )
        .unwrap();

    let pipeline = DistributedPipeline::new(
        &[LayerAssignment {
            node_id: "head".into(),
            layer_range: 0..32,
            role: NodeRole::Head,
            expected_decode_ms: 32.0,
            weight_memory_gb: 12.0,
            cache_memory_gb: 2.0,
        }],
        hidden_size,
    )
    .unwrap();

    let seq_id = 1;
    pipeline.alloc_cache(&mut registry, seq_id, 0).await.unwrap();

    let sequences = vec![SequenceMetadataWire {
        seq_id,
        num_tokens: 1,
        positions: vec![0],
        block_table: vec![],
        cache_seq_len: 0,
        last_block_tokens: 0,
    }];

    let result = pipeline
        .batched_forward(&mut registry, &sequences, &[1], false)
        .await;

    assert!(result.is_err());
    let err_msg = result.unwrap_err().to_string();
    assert!(
        err_msg.contains("GPU error"),
        "error should propagate worker's message: {err_msg}"
    );

    pipeline.free_cache(&mut registry, seq_id).await.unwrap();
}

// =========================================================================
// Fault Tolerance Integration Tests
// =========================================================================

/// Spawn a mock worker that handles Reconfigure (for rebalance tests):
/// receives Reconfigure, responds with WorkerReady.
async fn spawn_reconfigurable_worker(
    listener: TcpListener,
    is_tail: bool,
    hidden_size: usize,
    vocab_size: usize,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        let (stream, _) = listener.accept().await.unwrap();
        let mut conn = FramedConnection::new(stream);

        loop {
            let (header, payload) = match conn.recv().await {
                Ok(frame) => frame,
                Err(_) => break,
            };

            match header.msg_type {
                MessageType::CacheAlloc => {
                    conn.send_empty(MessageType::CacheAllocAck, header.seq_id)
                        .await
                        .unwrap();
                }
                MessageType::CacheFree => {}
                MessageType::Forward => {
                    let _req: ForwardPayload =
                        FramedConnection::deserialize_payload(&payload).unwrap();
                    let result = if is_tail {
                        let data: Vec<u8> = (0..vocab_size)
                            .flat_map(|i| (i as f32).to_le_bytes())
                            .collect();
                        ForwardResultPayload {
                            output: ForwardOutputWire::Logits { data },
                        }
                    } else {
                        let data_len = hidden_size * 2;
                        ForwardResultPayload {
                            output: ForwardOutputWire::Activations {
                                tensor_header: TensorWireHeader {
                                    ndim: 2,
                                    shape: vec![1, hidden_size as u32],
                                    dtype: 0,
                                    compression: 0,
                                    data_len: data_len as u32,
                                },
                                tensor_data: vec![0u8; data_len],
                            },
                        }
                    };
                    conn.send(MessageType::ForwardResult, header.seq_id, &result)
                        .await
                        .unwrap();
                }
                MessageType::Reconfigure => {
                    // Handle reconfiguration: respond with WorkerReady
                    conn.send_empty(MessageType::WorkerReady, 0)
                        .await
                        .unwrap();
                }
                MessageType::Shutdown => break,
                _ => {}
            }
        }
    })
}

/// Helper: set up a 3-node pipeline with reconfigurable mock workers.
async fn setup_three_node_reconfigurable() -> (
    DistributedPipeline,
    PeerRegistry,
    Vec<tokio::task::JoinHandle<()>>,
) {
    use fracture_coordinator::scheduler::WorkerCapabilities;

    let hidden_size = 4096;
    let vocab_size = 128256;

    let l0 = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let l1 = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let l2 = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let a0 = l0.local_addr().unwrap();
    let a1 = l1.local_addr().unwrap();
    let a2 = l2.local_addr().unwrap();

    let t0 = spawn_reconfigurable_worker(l0, false, hidden_size, vocab_size).await;
    let t1 = spawn_reconfigurable_worker(l1, false, hidden_size, vocab_size).await;
    let t2 = spawn_reconfigurable_worker(l2, true, hidden_size, vocab_size).await;

    let c0 = FramedConnection::new(tokio::net::TcpStream::connect(a0).await.unwrap());
    let c1 = FramedConnection::new(tokio::net::TcpStream::connect(a1).await.unwrap());
    let c2 = FramedConnection::new(tokio::net::TcpStream::connect(a2).await.unwrap());

    let mut registry = PeerRegistry::new();
    let caps = |name: &str| WorkerCapabilities {
        node_id: name.into(),
        gpu_model: "Mock".into(),
        gpu_memory_available: 24_000_000_000,
        compute_capability: (8, 0),
        decode_ms_per_layer: 1.0,
        prefill_ms_per_layer_128: 3.0,
    };

    registry.register(caps("w0"), c0).unwrap();
    registry.register(caps("w1"), c1).unwrap();
    registry.register(caps("w2"), c2).unwrap();

    let a = |name: &str, start: usize, end: usize, role: NodeRole| LayerAssignment {
        node_id: name.into(),
        layer_range: start..end,
        role,
        expected_decode_ms: 10.0,
        weight_memory_gb: 5.0,
        cache_memory_gb: 1.0,
    };

    let assignments = vec![
        a("w0", 0, 11, NodeRole::Head),
        a("w1", 11, 22, NodeRole::Middle),
        a("w2", 22, 32, NodeRole::Tail),
    ];
    for ass in &assignments {
        registry.assign(&ass.node_id, ass.clone()).unwrap();
    }

    let pipeline = DistributedPipeline::new(&assignments, hidden_size).unwrap();
    (pipeline, registry, vec![t0, t1, t2])
}

#[tokio::test]
async fn test_reregister_payload_over_wire() {
    // Test ReRegister message round-trip over TCP
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();

    let send_task = tokio::spawn(async move {
        let stream = tokio::net::TcpStream::connect(addr).await.unwrap();
        let mut conn = FramedConnection::new(stream);
        let payload = ReRegisterPayload {
            node_id: "worker-gpu0".into(),
            gpu_model: "RTX 3090".into(),
            gpu_memory_total: 24_000_000_000,
            gpu_memory_available: 8_000_000_000,
            compute_capability: (8, 6),
            decode_ms_per_layer: 1.1,
            prefill_ms_per_layer_128: 3.5,
            current_layer_start: Some(0),
            current_layer_end: Some(16),
            active_cache_seq_ids: vec![1, 5, 42],
        };
        conn.send(MessageType::ReRegister, 0, &payload)
            .await
            .unwrap();
    });

    let (stream, _) = listener.accept().await.unwrap();
    let mut conn = FramedConnection::new(stream);
    let (header, payload) = conn.recv().await.unwrap();

    assert_eq!(header.msg_type, MessageType::ReRegister);
    let rereg: ReRegisterPayload =
        FramedConnection::deserialize_payload(&payload).unwrap();
    assert_eq!(rereg.node_id, "worker-gpu0");
    assert_eq!(rereg.current_layer_start, Some(0));
    assert_eq!(rereg.current_layer_end, Some(16));
    assert_eq!(rereg.active_cache_seq_ids, vec![1, 5, 42]);

    send_task.await.unwrap();
}

#[tokio::test]
async fn test_leave_intent_over_wire() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();

    let send_task = tokio::spawn(async move {
        let stream = tokio::net::TcpStream::connect(addr).await.unwrap();
        let mut conn = FramedConnection::new(stream);
        let payload = LeaveIntentPayload {
            reason: "SIGTERM received".into(),
        };
        conn.send(MessageType::LeaveIntent, 0, &payload)
            .await
            .unwrap();
    });

    let (stream, _) = listener.accept().await.unwrap();
    let mut conn = FramedConnection::new(stream);
    let (header, payload) = conn.recv().await.unwrap();

    assert_eq!(header.msg_type, MessageType::LeaveIntent);
    let leave: LeaveIntentPayload =
        FramedConnection::deserialize_payload(&payload).unwrap();
    assert_eq!(leave.reason, "SIGTERM received");

    send_task.await.unwrap();
}

#[tokio::test]
async fn test_cluster_manifest_over_wire() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();

    let send_task = tokio::spawn(async move {
        let stream = tokio::net::TcpStream::connect(addr).await.unwrap();
        let mut conn = FramedConnection::new(stream);
        let payload = ClusterManifestPayload {
            version: 3,
            term: 1,
            nodes: vec![
                NodeInfo {
                    node_id: "coordinator".into(),
                    address: "192.168.1.1:9400".into(),
                    election_priority: 0,
                    coordinator_capable: true,
                    role: fracture_protocol::messages::NodeRole::Coordinator,
                },
                NodeInfo {
                    node_id: "worker-0".into(),
                    address: "192.168.1.2:9400".into(),
                    election_priority: 1,
                    coordinator_capable: true,
                    role: fracture_protocol::messages::NodeRole::Worker,
                },
            ],
        };
        conn.send(MessageType::ClusterManifest, 0, &payload)
            .await
            .unwrap();
    });

    let (stream, _) = listener.accept().await.unwrap();
    let mut conn = FramedConnection::new(stream);
    let (header, payload) = conn.recv().await.unwrap();

    assert_eq!(header.msg_type, MessageType::ClusterManifest);
    let manifest: ClusterManifestPayload =
        FramedConnection::deserialize_payload(&payload).unwrap();
    assert_eq!(manifest.version, 3);
    assert_eq!(manifest.term, 1);
    assert_eq!(manifest.nodes.len(), 2);
    assert_eq!(manifest.nodes[0].role, fracture_protocol::messages::NodeRole::Coordinator);
    assert!(manifest.nodes[1].coordinator_capable);

    send_task.await.unwrap();
}

#[tokio::test]
async fn test_election_messages_over_wire() {
    // Test all 3 election message types over TCP
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();

    let send_task = tokio::spawn(async move {
        let stream = tokio::net::TcpStream::connect(addr).await.unwrap();
        let mut conn = FramedConnection::new(stream);

        // ElectionStart
        conn.send(
            MessageType::ElectionStart,
            0,
            &ElectionStartPayload {
                candidate_id: "node-a".into(),
                priority: 0,
                term: 5,
            },
        )
        .await
        .unwrap();

        // ElectionChallenge
        conn.send(
            MessageType::ElectionChallenge,
            0,
            &ElectionChallengePayload {
                challenger_id: "node-b".into(),
                priority: 1,
                term: 5,
            },
        )
        .await
        .unwrap();

        // Victory
        conn.send(
            MessageType::Victory,
            0,
            &VictoryPayload {
                leader_id: "node-a".into(),
                term: 5,
                coordinator_addr: "192.168.1.10:9400".into(),
            },
        )
        .await
        .unwrap();
    });

    let (stream, _) = listener.accept().await.unwrap();
    let mut conn = FramedConnection::new(stream);

    // Receive ElectionStart
    let (header, payload) = conn.recv().await.unwrap();
    assert_eq!(header.msg_type, MessageType::ElectionStart);
    let es: ElectionStartPayload =
        FramedConnection::deserialize_payload(&payload).unwrap();
    assert_eq!(es.candidate_id, "node-a");
    assert_eq!(es.priority, 0);
    assert_eq!(es.term, 5);

    // Receive ElectionChallenge
    let (header, payload) = conn.recv().await.unwrap();
    assert_eq!(header.msg_type, MessageType::ElectionChallenge);
    let ec: ElectionChallengePayload =
        FramedConnection::deserialize_payload(&payload).unwrap();
    assert_eq!(ec.challenger_id, "node-b");

    // Receive Victory
    let (header, payload) = conn.recv().await.unwrap();
    assert_eq!(header.msg_type, MessageType::Victory);
    let v: VictoryPayload =
        FramedConnection::deserialize_payload(&payload).unwrap();
    assert_eq!(v.leader_id, "node-a");
    assert_eq!(v.term, 5);
    assert_eq!(v.coordinator_addr, "192.168.1.10:9400");

    send_task.await.unwrap();
}

#[tokio::test]
async fn test_abort_all_sequences_frees_caches() {
    let (pipeline, mut registry) = setup_two_node_pipeline().await;

    // Allocate caches for 3 sequences
    pipeline.alloc_cache(&mut registry, 1, 4096).await.unwrap();
    pipeline.alloc_cache(&mut registry, 2, 4096).await.unwrap();
    pipeline.alloc_cache(&mut registry, 3, 4096).await.unwrap();

    assert!(pipeline.is_allocated(1));
    assert!(pipeline.is_allocated(2));
    assert!(pipeline.is_allocated(3));

    // Abort all
    let aborted = pipeline.abort_all_sequences(&mut registry).await;
    assert_eq!(aborted.len(), 3);
    assert!(aborted.contains(&1));
    assert!(aborted.contains(&2));
    assert!(aborted.contains(&3));

    // All freed
    assert!(!pipeline.is_allocated(1));
    assert!(!pipeline.is_allocated(2));
    assert!(!pipeline.is_allocated(3));
}

#[tokio::test]
async fn test_pipeline_order_excludes_dead_workers() {
    let (_, mut registry) = setup_two_node_pipeline().await;

    // Initially both in pipeline order
    let order = registry.pipeline_order();
    assert_eq!(order.len(), 2);

    // Kill head
    registry.mark_dead("head");
    let order = registry.pipeline_order();
    assert_eq!(order.len(), 1);
    assert_eq!(order[0], "tail");
}

#[tokio::test]
async fn test_reregister_reconnection_handshake() {
    // Simulate the full reconnection flow:
    // Worker connects → sends ReRegister → receives RegisterAck → sends WorkerReady
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();

    let worker_task = tokio::spawn(async move {
        let stream = tokio::net::TcpStream::connect(addr).await.unwrap();
        let mut conn = FramedConnection::new(stream);

        // Send ReRegister
        let rereg = ReRegisterPayload {
            node_id: "worker-0".into(),
            gpu_model: "Mock GPU".into(),
            gpu_memory_total: 24_000_000_000,
            gpu_memory_available: 20_000_000_000,
            compute_capability: (8, 0),
            decode_ms_per_layer: 1.0,
            prefill_ms_per_layer_128: 3.0,
            current_layer_start: Some(0),
            current_layer_end: Some(16),
            active_cache_seq_ids: vec![],
        };
        conn.send(MessageType::ReRegister, 0, &rereg).await.unwrap();

        // Receive RegisterAck (assignment unchanged)
        let (header, payload) = conn.recv().await.unwrap();
        assert_eq!(header.msg_type, MessageType::RegisterAck);
        let ack: RegisterAckPayload =
            FramedConnection::deserialize_payload(&payload).unwrap();
        assert_eq!(ack.layer_start, 0);
        assert_eq!(ack.layer_end, 16);

        // Send WorkerReady
        conn.send_empty(MessageType::WorkerReady, 0).await.unwrap();
    });

    // Coordinator side
    let (stream, _) = listener.accept().await.unwrap();
    let mut conn = FramedConnection::new(stream);

    // Receive ReRegister
    let (header, payload) = conn.recv().await.unwrap();
    assert_eq!(header.msg_type, MessageType::ReRegister);
    let rereg: ReRegisterPayload =
        FramedConnection::deserialize_payload(&payload).unwrap();
    assert_eq!(rereg.node_id, "worker-0");
    assert_eq!(rereg.current_layer_start, Some(0));

    // Send RegisterAck (same assignment)
    let ack = RegisterAckPayload {
        layer_start: 0,
        layer_end: 16,
        total_layers: 32,
        max_seq_len: 4096,
        model_config: fracture_core::ModelConfig {
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
        },
    };
    conn.send(MessageType::RegisterAck, 0, &ack).await.unwrap();

    // Receive WorkerReady
    let (header, _) = conn.recv().await.unwrap();
    assert_eq!(header.msg_type, MessageType::WorkerReady);

    worker_task.await.unwrap();
}

#[tokio::test]
async fn test_forced_rebalance_excludes_dead_worker() {
    use fracture_coordinator::rebalance;
    use fracture_coordinator::scheduler::SchedulingMode;
    use std::sync::Arc;

    let (pipeline, registry, _tasks) = setup_three_node_reconfigurable().await;
    let registry = Arc::new(tokio::sync::Mutex::new(registry));

    // Mark middle worker (w1) as dead
    {
        let mut reg = registry.lock().await;
        reg.mark_dead("w1");
    }

    let model_config = fracture_core::ModelConfig {
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
    };

    // Forced rebalance excluding dead worker
    let result = rebalance::forced_rebalance(
        &registry,
        &pipeline,
        &model_config,
        &SchedulingMode::EqualSplit,
        4096,
        &["w1".to_string()],
    )
    .await
    .unwrap();

    // Pipeline should have 2 stages (w0, w2), covering all 32 layers
    assert_eq!(result.pipeline.pipeline_order().len(), 2);
    assert_eq!(result.assignments.len(), 2);

    // Verify all layers are covered
    let total_layers: usize = result
        .assignments
        .iter()
        .map(|a| a.layer_range.len())
        .sum();
    assert_eq!(total_layers, 32);

    // Verify w1 is not in assignments
    assert!(result
        .assignments
        .iter()
        .all(|a| a.node_id != "w1"));
}

#[tokio::test]
async fn test_draining_worker_excluded_from_pipeline_order() {
    let (_, mut registry) = setup_two_node_pipeline().await;

    // Both workers in pipeline
    assert_eq!(registry.pipeline_order().len(), 2);

    // Mark head as draining
    registry.mark_draining("head");

    // Draining worker excluded from pipeline order
    let order = registry.pipeline_order();
    assert_eq!(order.len(), 1);
    assert_eq!(order[0], "tail");

    // But still in all_capabilities (for rescheduling)
    let caps = registry.all_capabilities();
    assert_eq!(caps.len(), 2);
}

#[tokio::test]
async fn test_pending_worker_in_capabilities_not_pipeline() {
    let (_, mut registry) = setup_two_node_pipeline().await;

    // Mark head as pending (simulating a new join before rebalance)
    registry.mark_pending("head");

    // Pending worker NOT in pipeline order
    let order = registry.pipeline_order();
    assert_eq!(order.len(), 1);
    assert_eq!(order[0], "tail");

    // But IS in all_capabilities (scheduler can assign it layers)
    let caps = registry.all_capabilities();
    assert_eq!(caps.len(), 2);

    // pending_workers returns it
    let pending = registry.pending_workers();
    assert_eq!(pending.len(), 1);
    assert_eq!(pending[0], "head");
}

// =========================================================================
// Seed Node Discovery Tests
// =========================================================================

#[tokio::test]
async fn test_who_is_coordinator_over_wire() {
    // Test WhoIsCoordinator request/response round-trip
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();

    // Simulate a seed node that knows the coordinator
    let seed_task = tokio::spawn(async move {
        let (stream, _) = listener.accept().await.unwrap();
        let mut conn = FramedConnection::new(stream);

        let (header, payload) = conn.recv().await.unwrap();
        assert_eq!(header.msg_type, MessageType::WhoIsCoordinator);
        let req: WhoIsCoordinatorPayload =
            FramedConnection::deserialize_payload(&payload).unwrap();
        assert_eq!(req.node_id, "new-worker");

        let resp = WhoIsCoordinatorResponsePayload {
            coordinator_addr: Some("192.168.1.100:9400".into()),
            term: 3,
            manifest: Some(ClusterManifestPayload {
                version: 5,
                term: 3,
                nodes: vec![
                    NodeInfo {
                        node_id: "coordinator".into(),
                        address: "192.168.1.100:9400".into(),
                        election_priority: 0,
                        coordinator_capable: true,
                        role: fracture_protocol::messages::NodeRole::Coordinator,
                    },
                ],
            }),
        };
        conn.send(MessageType::WhoIsCoordinatorResponse, 0, &resp)
            .await
            .unwrap();
    });

    // New worker queries the seed
    let stream = tokio::net::TcpStream::connect(addr).await.unwrap();
    let mut conn = FramedConnection::new(stream);

    let req = WhoIsCoordinatorPayload {
        node_id: "new-worker".into(),
    };
    conn.send(MessageType::WhoIsCoordinator, 0, &req)
        .await
        .unwrap();

    let (header, payload) = conn.recv().await.unwrap();
    assert_eq!(header.msg_type, MessageType::WhoIsCoordinatorResponse);
    let resp: WhoIsCoordinatorResponsePayload =
        FramedConnection::deserialize_payload(&payload).unwrap();
    assert_eq!(resp.coordinator_addr, Some("192.168.1.100:9400".into()));
    assert_eq!(resp.term, 3);
    assert!(resp.manifest.is_some());
    let manifest = resp.manifest.unwrap();
    assert_eq!(manifest.nodes.len(), 1);
    assert_eq!(manifest.nodes[0].node_id, "coordinator");

    seed_task.await.unwrap();
}

#[tokio::test]
async fn test_who_is_coordinator_unknown() {
    // Seed node doesn't know the coordinator (mid-election)
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();

    let seed_task = tokio::spawn(async move {
        let (stream, _) = listener.accept().await.unwrap();
        let mut conn = FramedConnection::new(stream);
        let _ = conn.recv().await.unwrap();

        let resp = WhoIsCoordinatorResponsePayload {
            coordinator_addr: None,
            term: 0,
            manifest: None,
        };
        conn.send(MessageType::WhoIsCoordinatorResponse, 0, &resp)
            .await
            .unwrap();
    });

    let stream = tokio::net::TcpStream::connect(addr).await.unwrap();
    let mut conn = FramedConnection::new(stream);
    conn.send(
        MessageType::WhoIsCoordinator,
        0,
        &WhoIsCoordinatorPayload {
            node_id: "joiner".into(),
        },
    )
    .await
    .unwrap();

    let (_, payload) = conn.recv().await.unwrap();
    let resp: WhoIsCoordinatorResponsePayload =
        FramedConnection::deserialize_payload(&payload).unwrap();
    assert!(resp.coordinator_addr.is_none());

    seed_task.await.unwrap();
}

#[tokio::test]
async fn test_seed_discovery_skips_unreachable_tries_next() {
    // First seed is unreachable, second seed responds
    let unreachable_addr = "127.0.0.1:1"; // port 1 is almost certainly refused
    let good_listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let good_addr = good_listener.local_addr().unwrap();

    let seed_task = tokio::spawn(async move {
        let (stream, _) = good_listener.accept().await.unwrap();
        let mut conn = FramedConnection::new(stream);
        let _ = conn.recv().await.unwrap();
        let resp = WhoIsCoordinatorResponsePayload {
            coordinator_addr: Some("10.0.0.1:9400".into()),
            term: 1,
            manifest: None,
        };
        conn.send(MessageType::WhoIsCoordinatorResponse, 0, &resp)
            .await
            .unwrap();
    });

    // Use discover_coordinator-like logic inline (can't call the worker's function directly)
    let seeds = vec![unreachable_addr.to_string(), good_addr.to_string()];
    let mut found = None;

    for seed in &seeds {
        match tokio::time::timeout(
            std::time::Duration::from_secs(2),
            tokio::net::TcpStream::connect(seed),
        )
        .await
        {
            Ok(Ok(stream)) => {
                let mut conn = FramedConnection::new(stream);
                conn.send(
                    MessageType::WhoIsCoordinator,
                    0,
                    &WhoIsCoordinatorPayload {
                        node_id: "test".into(),
                    },
                )
                .await
                .unwrap();
                let (_, payload) = conn.recv().await.unwrap();
                let resp: WhoIsCoordinatorResponsePayload =
                    FramedConnection::deserialize_payload(&payload).unwrap();
                if let Some(addr) = resp.coordinator_addr {
                    found = Some(addr);
                    break;
                }
            }
            _ => continue,
        }
    }

    assert_eq!(found, Some("10.0.0.1:9400".into()));
    seed_task.await.unwrap();
}
