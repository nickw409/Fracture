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
                MessageType::CacheAlloc | MessageType::CacheFree => {
                    // No response needed for these
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
        .connection
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
                MessageType::CacheAlloc | MessageType::CacheFree => {}
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
                MessageType::CacheAlloc | MessageType::CacheFree => {}
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
    entry.connection.send(MessageType::Heartbeat, 0, &hb).await.unwrap();

    let (header, payload) = entry.connection.recv().await.unwrap();
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
                MessageType::CacheAlloc => { cache_count += 1; }
                MessageType::CacheFree => { cache_count = cache_count.saturating_sub(1); }
                MessageType::Heartbeat => {
                    let hb: HeartbeatPayload = FramedConnection::deserialize_payload(&payload).unwrap();
                    let ack = HeartbeatAckPayload {
                        timestamp_echo: hb.timestamp_ns, nonce_echo: hb.nonce,
                        gpu_memory_used: 0, active_sequences: cache_count,
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
async fn test_send_heartbeats_and_check_timeout() {
    use fracture_coordinator::heartbeat;

    // Spawn a mock worker that responds to heartbeats
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();

    let _worker = tokio::spawn(async move {
        let (stream, _) = listener.accept().await.unwrap();
        let mut conn = FramedConnection::new(stream);
        // Respond to one heartbeat
        let (header, payload) = conn.recv().await.unwrap();
        assert_eq!(header.msg_type, MessageType::Heartbeat);
        let hb: HeartbeatPayload = FramedConnection::deserialize_payload(&payload).unwrap();
        let ack = HeartbeatAckPayload {
            timestamp_echo: hb.timestamp_ns,
            nonce_echo: hb.nonce,
            gpu_memory_used: 0,
            active_sequences: 0,
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

    // send_heartbeats with generous timeout — should return no timed-out workers
    let timed_out = heartbeat::send_heartbeats(
        &mut registry,
        std::time::Duration::from_secs(60),
    ).await;
    assert!(timed_out.is_empty(), "no workers should be timed out");
}

#[tokio::test]
async fn test_send_heartbeats_detects_timeout() {
    use fracture_coordinator::heartbeat;

    // Worker that does NOT respond (just accepts connection and hangs)
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();

    let _worker = tokio::spawn(async move {
        let (stream, _) = listener.accept().await.unwrap();
        let mut conn = FramedConnection::new(stream);
        // Read the heartbeat but don't respond
        let _ = conn.recv().await;
        // Keep connection alive
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

    // Manually set the last heartbeat to the past so it times out
    // We do this by calling send_heartbeats with zero timeout
    let timed_out = heartbeat::send_heartbeats(
        &mut registry,
        std::time::Duration::ZERO,
    ).await;
    assert_eq!(timed_out, vec!["slow"]);
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
                MessageType::CacheAlloc | MessageType::CacheFree => {}
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
                MessageType::CacheAlloc | MessageType::CacheFree => {}
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
