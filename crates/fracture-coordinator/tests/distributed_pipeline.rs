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
        DistributedPipeline::new(&[head_assignment, tail_assignment]).unwrap();

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

    let pipeline = DistributedPipeline::new(&assignments).unwrap();
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
