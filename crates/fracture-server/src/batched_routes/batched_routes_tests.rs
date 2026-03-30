use super::*;
use axum::http::StatusCode;
use fracture_core::StopReason;
use fracture_engine::{BatchScheduler, GenerationEvent, PendingRequest};
use http_body_util::BodyExt;
use tokio::sync::mpsc;

/// Gap 111 — async-request-enqueue: PendingRequest can be constructed
/// and submitted to a BatchScheduler via enqueue().
#[test]
fn test_pending_request_enqueue() {
    let (event_tx, _event_rx) = mpsc::unbounded_channel();
    let pending = PendingRequest {
        seq_id: 0,
        prompt_tokens: vec![1, 2, 3, 4, 5],
        max_tokens: 64,
        temperature: 0.0,
        top_k: 0,
        top_p: 1.0,
        seed: None,
        stop_tokens: vec![128001, 128008, 128009],
        event_tx,
    };

    let mut scheduler = BatchScheduler::new(64, 4096, 512, 0.1);
    assert_eq!(scheduler.num_pending(), 0);
    assert!(!scheduler.has_work());

    scheduler.enqueue(pending);

    assert_eq!(scheduler.num_pending(), 1);
    assert!(scheduler.has_work());
}

/// Gap 111 (continued) — enqueuing multiple requests queues them in order.
#[test]
fn test_multiple_requests_enqueue() {
    let mut scheduler = BatchScheduler::new(64, 4096, 512, 0.1);

    for i in 0u64..3 {
        let (tx, _rx) = mpsc::unbounded_channel();
        scheduler.enqueue(PendingRequest {
            seq_id: i,
            prompt_tokens: vec![10 + i as u32; (i as usize + 1) * 4],
            max_tokens: 32,
            temperature: 0.6,
            top_k: 50,
            top_p: 0.9,
            seed: Some(42),
            stop_tokens: vec![128001],
            event_tx: tx,
        });
    }

    assert_eq!(scheduler.num_pending(), 3);
    // Verify FIFO order: first enqueued has seq_id 0.
    assert_eq!(scheduler.prefill_queue[0].seq_id, 0);
    assert_eq!(scheduler.prefill_queue[1].seq_id, 1);
    assert_eq!(scheduler.prefill_queue[2].seq_id, 2);
}

/// Gap 113 — async-non-streaming-response: Token events followed by
/// Finished can be collected into a complete response.
#[tokio::test]
async fn test_collect_generation_events_into_response() {
    let (event_tx, mut event_rx) = mpsc::unbounded_channel();

    // Simulate the scheduler sending tokens then finishing.
    event_tx.send(GenerationEvent::Token(100)).unwrap();
    event_tx.send(GenerationEvent::Token(200)).unwrap();
    event_tx.send(GenerationEvent::Token(300)).unwrap();
    event_tx
        .send(GenerationEvent::Finished {
            stop_reason: StopReason::Stop,
            completion_tokens: 3,
        })
        .unwrap();
    drop(event_tx);

    // Collect tokens, mirroring handle_non_streaming logic.
    let mut tokens = Vec::new();
    let mut finish_reason = "length".to_string();

    while let Some(event) = event_rx.recv().await {
        match event {
            GenerationEvent::Token(t) => tokens.push(t),
            GenerationEvent::Finished { stop_reason, .. } => {
                finish_reason = match stop_reason {
                    StopReason::Stop => "stop",
                    StopReason::Length => "length",
                }
                .to_string();
                break;
            }
            GenerationEvent::Error(_) => panic!("unexpected error event"),
        }
    }

    assert_eq!(tokens, vec![100, 200, 300]);
    assert_eq!(finish_reason, "stop");
}

/// Gap 113 (continued) — Length stop reason is correctly mapped.
#[tokio::test]
async fn test_collect_events_length_finish() {
    let (event_tx, mut event_rx) = mpsc::unbounded_channel();

    event_tx.send(GenerationEvent::Token(42)).unwrap();
    event_tx
        .send(GenerationEvent::Finished {
            stop_reason: StopReason::Length,
            completion_tokens: 1,
        })
        .unwrap();
    drop(event_tx);

    let mut tokens = Vec::new();
    let mut finish_reason = String::new();

    while let Some(event) = event_rx.recv().await {
        match event {
            GenerationEvent::Token(t) => tokens.push(t),
            GenerationEvent::Finished { stop_reason, .. } => {
                finish_reason = match stop_reason {
                    StopReason::Stop => "stop",
                    StopReason::Length => "length",
                }
                .to_string();
                break;
            }
            GenerationEvent::Error(_) => panic!("unexpected error event"),
        }
    }

    assert_eq!(tokens, vec![42]);
    assert_eq!(finish_reason, "length");
}

/// Gap 114 — async-cancellation-on-disconnect: when the event receiver is
/// dropped, sending on event_tx fails, signalling client disconnect.
#[test]
fn test_event_tx_fails_when_rx_dropped() {
    let (event_tx, event_rx) = mpsc::unbounded_channel::<GenerationEvent>();

    // Receiver still alive — send should succeed.
    assert!(event_tx.send(GenerationEvent::Token(1)).is_ok());

    // Drop receiver — simulate client disconnect.
    drop(event_rx);

    // Now send should fail with a closed-channel error.
    let result = event_tx.send(GenerationEvent::Token(2));
    assert!(result.is_err(), "send should fail after receiver is dropped");
}

/// Gap 114 (continued) — the scheduler detects disconnect via is_closed().
#[test]
fn test_event_tx_is_closed_after_rx_drop() {
    let (event_tx, event_rx) = mpsc::unbounded_channel::<GenerationEvent>();
    assert!(!event_tx.is_closed());

    drop(event_rx);
    assert!(event_tx.is_closed(), "sender should report closed after receiver drop");
}

// ── Helper: minimal tokenizer for handler tests ─────────────────────────

fn make_test_tokenizer() -> tokenizers::Tokenizer {
    use tokenizers::models::bpe::BPE;
    let model = BPE::default();
    let mut tok = tokenizers::Tokenizer::new(model);
    let tokens: Vec<tokenizers::AddedToken> = (0..256u32)
        .map(|i| tokenizers::AddedToken::from(format!("{}", i as u8 as char), false))
        .collect();
    tok.add_tokens(&tokens);
    tok
}

/// Helper: build a BatchedAppState backed by a real mpsc channel.
/// Returns the state and the receiver end to observe submitted PendingRequests.
fn make_batched_state() -> (
    std::sync::Arc<BatchedAppState>,
    mpsc::UnboundedReceiver<PendingRequest>,
) {
    let (scheduler_tx, scheduler_rx) = mpsc::unbounded_channel::<PendingRequest>();
    let scheduler = crate::scheduler_loop::SchedulerHandle::from_sender(scheduler_tx);
    let tokenizer = make_test_tokenizer();
    let dashboard = {
        use crate::dashboard::dto::ModelInfo;
        use crate::dashboard::metrics::MetricsCollector;
        use crate::dashboard::request_log::RequestLog;
        use crate::dashboard::routes::{ClusterProvider, DashboardState};
        std::sync::Arc::new(DashboardState {
            metrics: std::sync::Arc::new(MetricsCollector::new()),
            request_log: std::sync::Arc::new(RequestLog::new()),
            cluster: ClusterProvider::Standalone {
                gpu_name: "mock".to_string(),
                vram_total_mb: 1024,
                vram_used_mb: 0,
                model: ModelInfo {
                    name: "test".to_string(),
                    parameters: "0".to_string(),
                    layers: 2,
                    context_length: 128,
                    dtype: "FP32".to_string(),
                },
                total_layers: 2,
            },
            scheduler: None,
        })
    };
    let state = std::sync::Arc::new(BatchedAppState::new(scheduler, tokenizer, dashboard));
    (state, scheduler_rx)
}

/// Parse raw SSE bytes into a vec of payload strings (the part after "data: ").
fn parse_sse_events(body: &[u8]) -> Vec<String> {
    let text = std::str::from_utf8(body).expect("SSE body must be valid UTF-8");
    let mut events = Vec::new();
    for chunk in text.split("\n\n") {
        let chunk = chunk.trim();
        if chunk.is_empty() {
            continue;
        }
        for line in chunk.lines() {
            if let Some(payload) = line.strip_prefix("data: ") {
                events.push(payload.to_string());
            }
        }
    }
    events
}

// ── Gap: batched-completions-handler-enqueues ────────────────────────────

/// POST /v1/completions to the batched router enqueues a PendingRequest
/// on the scheduler channel with the correct seq_id and non-empty prompt tokens.
#[tokio::test]
async fn test_batched_completions_handler_enqueues() {
    let (state, mut scheduler_rx) = make_batched_state();
    let app = create_batched_router(state);

    let body = serde_json::json!({
        "prompt": "hello world",
        "max_tokens": 10,
        "temperature": 0.0,
        "stream": false
    });

    // We spawn the request in a task because the handler awaits on the event
    // channel (which no one feeds), so we just check the scheduler received
    // the request without driving the handler to completion.
    let req = axum::http::Request::builder()
        .method("POST")
        .uri("/v1/completions")
        .header("content-type", "application/json")
        .body(axum::body::Body::from(serde_json::to_vec(&body).unwrap()))
        .unwrap();

    // Spawn the handler so it doesn't block the test.
    let handle = tokio::spawn(tower::ServiceExt::oneshot(app, req));

    // The handler submits to the scheduler before awaiting the event channel,
    // so we can receive the PendingRequest immediately.
    let pending = tokio::time::timeout(
        std::time::Duration::from_secs(2),
        scheduler_rx.recv(),
    )
    .await
    .expect("timeout waiting for PendingRequest")
    .expect("scheduler channel closed unexpectedly");

    assert_eq!(pending.max_tokens, 10);
    assert!(!pending.prompt_tokens.is_empty(), "prompt_tokens must not be empty");

    // Drop the handle to avoid leaking the task.
    handle.abort();
}

// ── Gap: batched-streaming-token-events ──────────────────────────────────

/// POST stream=true — send Token events then Finished on the event channel;
/// verify that the SSE response contains token chunks and ends with [DONE].
#[tokio::test]
async fn test_batched_streaming_token_events() {
    let (state, mut scheduler_rx) = make_batched_state();
    let app = create_batched_router(state);

    let body = serde_json::json!({
        "prompt": "hi",
        "max_tokens": 5,
        "temperature": 0.0,
        "stream": true
    });

    let req = axum::http::Request::builder()
        .method("POST")
        .uri("/v1/completions")
        .header("content-type", "application/json")
        .body(axum::body::Body::from(serde_json::to_vec(&body).unwrap()))
        .unwrap();

    let handle = tokio::spawn(tower::ServiceExt::oneshot(app, req));

    // Grab the pending request to get its event_tx.
    let pending = tokio::time::timeout(
        std::time::Duration::from_secs(2),
        scheduler_rx.recv(),
    )
    .await
    .expect("timeout")
    .expect("channel closed");

    // Feed Token events then Finished.
    pending.event_tx.send(GenerationEvent::Token(65)).unwrap(); // token 65 = 'A'
    pending.event_tx.send(GenerationEvent::Token(66)).unwrap(); // token 66 = 'B'
    pending
        .event_tx
        .send(GenerationEvent::Finished {
            stop_reason: StopReason::Stop,
            completion_tokens: 2,
        })
        .unwrap();
    drop(pending.event_tx);

    let resp = tokio::time::timeout(std::time::Duration::from_secs(5), handle)
        .await
        .expect("timeout waiting for response")
        .expect("task panicked")
        .unwrap();

    assert_eq!(resp.status(), StatusCode::OK);

    let ct = resp
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");
    assert!(ct.contains("text/event-stream"), "must be SSE, got: {ct}");

    let body_bytes = resp.into_body().collect().await.unwrap().to_bytes();
    let events = parse_sse_events(&body_bytes);

    assert!(!events.is_empty(), "must have SSE events");
    assert_eq!(events.last().unwrap(), "[DONE]", "stream must end with [DONE]");

    // There should be content events before [DONE].
    let content_events: Vec<_> = events.iter().filter(|e| *e != "[DONE]").collect();
    assert!(!content_events.is_empty(), "must have at least one content event");

    // Each content event (non-finished) must have choices.
    for event in &content_events {
        let json: serde_json::Value = serde_json::from_str(event)
            .unwrap_or_else(|_| panic!("not valid JSON: {event}"));
        assert!(json["choices"].is_array(), "content event must have choices: {event}");
    }
}

// ── Gap: batched-streaming-error-event ───────────────────────────────────

/// When a GenerationEvent::Error is sent, the SSE stream emits an error
/// event (JSON with an "error" key) before the [DONE] sentinel.
#[tokio::test]
async fn test_batched_streaming_error_event() {
    let (state, mut scheduler_rx) = make_batched_state();
    let app = create_batched_router(state);

    let body = serde_json::json!({
        "prompt": "hi",
        "max_tokens": 5,
        "temperature": 0.0,
        "stream": true
    });

    let req = axum::http::Request::builder()
        .method("POST")
        .uri("/v1/completions")
        .header("content-type", "application/json")
        .body(axum::body::Body::from(serde_json::to_vec(&body).unwrap()))
        .unwrap();

    let handle = tokio::spawn(tower::ServiceExt::oneshot(app, req));

    let pending = tokio::time::timeout(
        std::time::Duration::from_secs(2),
        scheduler_rx.recv(),
    )
    .await
    .expect("timeout")
    .expect("channel closed");

    pending
        .event_tx
        .send(GenerationEvent::Error("kernel panic".to_string()))
        .unwrap();
    drop(pending.event_tx);

    let resp = tokio::time::timeout(std::time::Duration::from_secs(5), handle)
        .await
        .expect("timeout")
        .expect("task panicked")
        .unwrap();

    assert_eq!(resp.status(), StatusCode::OK,
        "SSE responses always return 200 even on error");

    let body_bytes = resp.into_body().collect().await.unwrap().to_bytes();
    let events = parse_sse_events(&body_bytes);

    assert_eq!(events.last().unwrap(), "[DONE]", "must end with [DONE]");

    let error_event = events
        .iter()
        .filter(|e| *e != "[DONE]")
        .find(|e| {
            serde_json::from_str::<serde_json::Value>(e)
                .map(|j| j.get("error").is_some())
                .unwrap_or(false)
        })
        .expect("must have an SSE event with 'error' key");

    let json: serde_json::Value = serde_json::from_str(error_event).unwrap();
    let err = &json["error"];
    assert!(err["message"].as_str().is_some(), "error.message must be a string");
    assert_eq!(
        err["type"].as_str().unwrap_or(""),
        "server_error",
        "error.type must be 'server_error'"
    );
}

// ── Gap: batched-non-streaming-completions ───────────────────────────────

/// POST stream=false — feed Token + Finished events; assert the response JSON
/// has the correct structure (object, choices, usage).
#[tokio::test]
async fn test_batched_non_streaming_completions() {
    let (state, mut scheduler_rx) = make_batched_state();
    let app = create_batched_router(state);

    let body = serde_json::json!({
        "prompt": "hello",
        "max_tokens": 10,
        "temperature": 0.0,
        "stream": false
    });

    let req = axum::http::Request::builder()
        .method("POST")
        .uri("/v1/completions")
        .header("content-type", "application/json")
        .body(axum::body::Body::from(serde_json::to_vec(&body).unwrap()))
        .unwrap();

    let handle = tokio::spawn(tower::ServiceExt::oneshot(app, req));

    let pending = tokio::time::timeout(
        std::time::Duration::from_secs(2),
        scheduler_rx.recv(),
    )
    .await
    .expect("timeout")
    .expect("channel closed");

    pending.event_tx.send(GenerationEvent::Token(72)).unwrap(); // 'H'
    pending.event_tx.send(GenerationEvent::Token(105)).unwrap(); // 'i'
    pending
        .event_tx
        .send(GenerationEvent::Finished {
            stop_reason: StopReason::Stop,
            completion_tokens: 2,
        })
        .unwrap();
    drop(pending.event_tx);

    let resp = tokio::time::timeout(std::time::Duration::from_secs(5), handle)
        .await
        .expect("timeout")
        .expect("task panicked")
        .unwrap();

    assert_eq!(resp.status(), StatusCode::OK);

    let body_bytes = resp.into_body().collect().await.unwrap().to_bytes();
    let json: serde_json::Value = serde_json::from_slice(&body_bytes).unwrap();

    assert_eq!(json["object"], "text_completion");
    assert!(json["id"].as_str().unwrap_or("").starts_with("cmpl-"),
        "id must start with cmpl-");
    assert_eq!(json["choices"][0]["finish_reason"], "stop");
    assert!(json["usage"]["prompt_tokens"].as_u64().is_some());
    assert!(json["usage"]["completion_tokens"].as_u64().is_some());
    let pt = json["usage"]["prompt_tokens"].as_u64().unwrap();
    let ct = json["usage"]["completion_tokens"].as_u64().unwrap();
    let tt = json["usage"]["total_tokens"].as_u64().unwrap();
    assert_eq!(tt, pt + ct, "total_tokens must equal prompt_tokens + completion_tokens");
}

// ── Gap: batched-non-streaming-error ────────────────────────────────────

/// POST stream=false — when a GenerationEvent::Error is sent, the handler
/// must return HTTP 500 with an error body.
#[tokio::test]
async fn test_batched_non_streaming_error() {
    let (state, mut scheduler_rx) = make_batched_state();
    let app = create_batched_router(state);

    let body = serde_json::json!({
        "prompt": "hello",
        "max_tokens": 5,
        "temperature": 0.0,
        "stream": false
    });

    let req = axum::http::Request::builder()
        .method("POST")
        .uri("/v1/completions")
        .header("content-type", "application/json")
        .body(axum::body::Body::from(serde_json::to_vec(&body).unwrap()))
        .unwrap();

    let handle = tokio::spawn(tower::ServiceExt::oneshot(app, req));

    let pending = tokio::time::timeout(
        std::time::Duration::from_secs(2),
        scheduler_rx.recv(),
    )
    .await
    .expect("timeout")
    .expect("channel closed");

    pending
        .event_tx
        .send(GenerationEvent::Error("OOM in batched_forward".to_string()))
        .unwrap();
    drop(pending.event_tx);

    let resp = tokio::time::timeout(std::time::Duration::from_secs(5), handle)
        .await
        .expect("timeout")
        .expect("task panicked")
        .unwrap();

    assert_eq!(resp.status(), StatusCode::INTERNAL_SERVER_ERROR,
        "error event must produce HTTP 500 for non-streaming");

    let body_bytes = resp.into_body().collect().await.unwrap().to_bytes();
    let json: serde_json::Value = serde_json::from_slice(&body_bytes).unwrap();

    let err = json.get("error").expect("response must have 'error' key");
    let msg = err["message"].as_str().expect("error.message must be a string");
    assert!(msg.contains("generation failed"),
        "error.message must contain 'generation failed', got: {msg}");
}
