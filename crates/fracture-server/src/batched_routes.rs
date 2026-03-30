//! HTTP handlers for the batched (Phase 4) serving mode.
//!
//! Instead of blocking on GenerationLoop::generate(), requests are
//! enqueued in the scheduler and tokens stream back via channels.

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::sse::{Event, Sse};
use axum::response::{IntoResponse, Json};
use axum::routing::{get, post};
use axum::Router;
use fracture_engine::{BatchScheduler, GenerationEvent, PendingRequest};
use std::convert::Infallible;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Instant, SystemTime, UNIX_EPOCH};
use tokio::sync::mpsc;
use tokenizers::Tokenizer;
use tower_http::cors::CorsLayer;

use crate::api::*;
use crate::dashboard::dto::RequestRecord;
use crate::dashboard::routes::{dashboard_routes, DashboardState};
use crate::routes::{validate_chat_request, validate_completion_request, validate_model_name};
use crate::scheduler_loop::SchedulerHandle;

const LOADED_MODEL_NAME: &str = "llama-3-8b";

/// Shared state for the batched serving mode.
pub struct BatchedAppState {
    pub scheduler: SchedulerHandle,
    pub tokenizer: Tokenizer,
    pub dashboard: Arc<DashboardState>,
    next_seq_id: AtomicU64,
}

impl BatchedAppState {
    pub fn new(
        scheduler: SchedulerHandle,
        tokenizer: Tokenizer,
        dashboard: Arc<DashboardState>,
    ) -> Self {
        Self {
            scheduler,
            tokenizer,
            dashboard,
            next_seq_id: AtomicU64::new(0),
        }
    }

    fn next_seq_id(&self) -> u64 {
        self.next_seq_id.fetch_add(1, Ordering::SeqCst)
    }
}

/// Create the HTTP router for batched serving mode.
pub fn create_batched_router(state: Arc<BatchedAppState>) -> Router {
    let dashboard_state = Arc::clone(&state.dashboard);

    let api = Router::new()
        .route("/v1/completions", post(completions_handler))
        .route("/v1/chat/completions", post(chat_completions_handler))
        .route("/v1/models", get(models_handler))
        .route("/health", get(health_handler))
        .with_state(state);

    api.merge(dashboard_routes(dashboard_state))
        .layer(CorsLayer::permissive())
}

async fn models_handler() -> impl IntoResponse {
    (
        StatusCode::OK,
        Json(serde_json::json!({
            "object": "list",
            "data": [{
                "id": LOADED_MODEL_NAME,
                "object": "model",
                "created": unix_timestamp(),
                "owned_by": "fracture"
            }]
        })),
    )
}

async fn health_handler() -> impl IntoResponse {
    (
        StatusCode::OK,
        Json(serde_json::json!({"status": "ready"})),
    )
}

async fn completions_handler(
    State(state): State<Arc<BatchedAppState>>,
    Json(req): Json<CompletionRequest>,
) -> impl IntoResponse {
    if let Err(resp) = validate_model_name(req.model.as_deref()) {
        return resp;
    }
    if let Err(resp) = validate_completion_request(&req) {
        return resp;
    }

    let prompt_tokens: Vec<u32> = state
        .tokenizer
        .encode(req.prompt.as_str(), false)
        .map(|enc| enc.get_ids().to_vec())
        .unwrap_or_default();
    let prompt_len = prompt_tokens.len();

    let (event_tx, event_rx) = mpsc::unbounded_channel();
    let seq_id = state.next_seq_id();

    let pending = PendingRequest {
        seq_id,
        prompt_tokens,
        max_tokens: req.max_tokens,
        temperature: req.temperature,
        top_k: req.top_k,
        top_p: req.top_p,
        seed: None,
        stop_tokens: vec![128001, 128008, 128009],
        event_tx,
    };

    if let Err(e) = state.scheduler.submit(pending) {
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(error_body(&format!("failed to enqueue request: {e}"))),
        )
            .into_response();
    }

    if req.stream {
        return handle_streaming(state, event_rx, prompt_len, true).into_response();
    }

    // Non-streaming: collect all tokens and return.
    handle_non_streaming(state, event_rx, prompt_len, true).await.into_response()
}

async fn chat_completions_handler(
    State(state): State<Arc<BatchedAppState>>,
    Json(req): Json<ChatCompletionRequest>,
) -> impl IntoResponse {
    if let Err(resp) = validate_model_name(req.model.as_deref()) {
        return resp;
    }
    if let Err(resp) = validate_chat_request(&req) {
        return resp;
    }

    let messages: Vec<(String, String)> = req
        .messages
        .iter()
        .map(|m| (m.role.clone(), m.content.clone()))
        .collect();
    let prompt = fracture_generate::apply_chat_template(&messages);
    let prompt_tokens: Vec<u32> = state
        .tokenizer
        .encode(prompt.as_str(), false)
        .map(|enc| enc.get_ids().to_vec())
        .unwrap_or_default();
    let prompt_len = prompt_tokens.len();

    let (event_tx, event_rx) = mpsc::unbounded_channel();
    let seq_id = state.next_seq_id();

    let pending = PendingRequest {
        seq_id,
        prompt_tokens,
        max_tokens: req.max_tokens,
        temperature: req.temperature,
        top_k: req.top_k,
        top_p: req.top_p,
        seed: None,
        stop_tokens: vec![128001, 128008, 128009],
        event_tx,
    };

    if let Err(e) = state.scheduler.submit(pending) {
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(error_body(&format!("failed to enqueue request: {e}"))),
        )
            .into_response();
    }

    if req.stream {
        return handle_streaming(state, event_rx, prompt_len, false).into_response();
    }

    handle_non_streaming(state, event_rx, prompt_len, false).await.into_response()
}

fn handle_streaming(
    state: Arc<BatchedAppState>,
    mut event_rx: mpsc::UnboundedReceiver<GenerationEvent>,
    prompt_len: usize,
    is_completion: bool,
) -> Sse<impl futures_core::Stream<Item = std::result::Result<Event, Infallible>>> {
    state.dashboard.metrics.request_started();

    let stream = async_stream::stream! {
        let id = gen_id();
        let created = unix_timestamp();
        let model = LOADED_MODEL_NAME;
        let object = if is_completion { "text_completion" } else { "chat.completion.chunk" };
        let t_start = Instant::now();
        let mut t_first_token: Option<Instant> = None;
        let mut token_count = 0usize;
        let mut final_reason: &str = "length";

        while let Some(event) = event_rx.recv().await {
            match event {
                GenerationEvent::Token(token_id) => {
                    if t_first_token.is_none() {
                        t_first_token = Some(Instant::now());
                    }
                    token_count += 1;
                    let text = decode_tokens(&state.tokenizer, &[token_id]);
                    let chunk = if is_completion {
                        serde_json::json!({
                            "id": id, "object": object, "created": created, "model": model,
                            "choices": [{"index": 0, "text": text, "finish_reason": null}]
                        })
                    } else {
                        serde_json::json!({
                            "id": id, "object": object, "created": created, "model": model,
                            "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": null}]
                        })
                    };
                    yield Ok::<_, Infallible>(Event::default().data(chunk.to_string()));
                }
                GenerationEvent::Finished { stop_reason, completion_tokens } => {
                    let reason = match stop_reason {
                        fracture_core::StopReason::Stop => "stop",
                        fracture_core::StopReason::Length => "length",
                    };
                    final_reason = reason;
                    token_count = completion_tokens;
                    let final_chunk = if is_completion {
                        serde_json::json!({
                            "id": id, "object": object, "created": created, "model": model,
                            "choices": [{"index": 0, "text": "", "finish_reason": reason}],
                            "usage": {
                                "prompt_tokens": prompt_len,
                                "completion_tokens": completion_tokens,
                                "total_tokens": prompt_len + completion_tokens,
                            }
                        })
                    } else {
                        serde_json::json!({
                            "id": id, "object": object, "created": created, "model": model,
                            "choices": [{"index": 0, "delta": {}, "finish_reason": reason}],
                            "usage": {
                                "prompt_tokens": prompt_len,
                                "completion_tokens": completion_tokens,
                                "total_tokens": prompt_len + completion_tokens,
                            }
                        })
                    };
                    yield Ok(Event::default().data(final_chunk.to_string()));
                    break;
                }
                GenerationEvent::Error(msg) => {
                    final_reason = "error";
                    let err = serde_json::json!({
                        "error": {"message": format!("generation failed: {msg}"), "type": "server_error", "code": null}
                    });
                    yield Ok(Event::default().data(err.to_string()));
                    break;
                }
            }
        }

        // Record completion metrics.
        let total_duration_ms = t_start.elapsed().as_secs_f64() * 1000.0;
        let ttft_ms = t_first_token.map(|t| (t - t_start).as_secs_f64() * 1000.0).unwrap_or(0.0);
        let tps = if total_duration_ms > 0.0 { token_count as f64 / (total_duration_ms / 1000.0) } else { 0.0 };

        let record = RequestRecord {
            id: id.clone(),
            request_type: if is_completion { "completion" } else { "chat" },
            status: if final_reason == "error" { "error" } else { "completed" },
            prompt_tokens: prompt_len,
            completion_tokens: token_count,
            total_tokens: prompt_len + token_count,
            time_to_first_token_ms: ttft_ms,
            total_duration_ms,
            tokens_per_second: tps,
            finish_reason: final_reason,
            temperature: 0.0,
            created_at: String::new(),
        };
        state.dashboard.metrics.record_completion(&record);
        state.dashboard.request_log.push(record);

        yield Ok(Event::default().data("[DONE]"));
    };

    Sse::new(stream)
}

async fn handle_non_streaming(
    state: Arc<BatchedAppState>,
    mut event_rx: mpsc::UnboundedReceiver<GenerationEvent>,
    prompt_len: usize,
    is_completion: bool,
) -> axum::response::Response {
    state.dashboard.metrics.request_started();
    let t_start = Instant::now();
    let mut t_first_token: Option<Instant> = None;
    let mut tokens = Vec::new();
    let mut finish_reason: &'static str = "length";

    while let Some(event) = event_rx.recv().await {
        match event {
            GenerationEvent::Token(t) => {
                if t_first_token.is_none() {
                    t_first_token = Some(Instant::now());
                }
                tokens.push(t);
            }
            GenerationEvent::Finished { stop_reason, .. } => {
                finish_reason = match stop_reason {
                    fracture_core::StopReason::Stop => "stop",
                    fracture_core::StopReason::Length => "length",
                };
                break;
            }
            GenerationEvent::Error(msg) => {
                let total_duration_ms = t_start.elapsed().as_secs_f64() * 1000.0;
                let record = RequestRecord {
                    id: gen_id(),
                    request_type: if is_completion { "completion" } else { "chat" },
                    status: "error",
                    prompt_tokens: prompt_len,
                    completion_tokens: tokens.len(),
                    total_tokens: prompt_len + tokens.len(),
                    time_to_first_token_ms: 0.0,
                    total_duration_ms,
                    tokens_per_second: 0.0,
                    finish_reason: "error",
                    temperature: 0.0,
                    created_at: String::new(),
                };
                state.dashboard.metrics.record_completion(&record);
                state.dashboard.request_log.push(record);
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(error_body(&format!("generation failed: {msg}"))),
                )
                    .into_response();
            }
        }
    }

    // Record successful completion.
    let total_duration_ms = t_start.elapsed().as_secs_f64() * 1000.0;
    let ttft_ms = t_first_token.map(|t| (t - t_start).as_secs_f64() * 1000.0).unwrap_or(0.0);
    let tps = if total_duration_ms > 0.0 { tokens.len() as f64 / (total_duration_ms / 1000.0) } else { 0.0 };
    let req_id = gen_id();

    {
        let record = RequestRecord {
            id: req_id.clone(),
            request_type: if is_completion { "completion" } else { "chat" },
            status: "completed",
            prompt_tokens: prompt_len,
            completion_tokens: tokens.len(),
            total_tokens: prompt_len + tokens.len(),
            time_to_first_token_ms: ttft_ms,
            total_duration_ms,
            tokens_per_second: tps,
            finish_reason,
            temperature: 0.0,
            created_at: String::new(),
        };
        state.dashboard.metrics.record_completion(&record);
        state.dashboard.request_log.push(record);
    }

    let text = decode_tokens(&state.tokenizer, &tokens);

    let response = if is_completion {
        CompletionResponse {
            id: gen_id(),
            object: "text_completion".to_string(),
            created: unix_timestamp(),
            choices: vec![Choice {
                index: 0,
                text: Some(text),
                message: None,
                finish_reason: Some(finish_reason.to_string()),
            }],
            usage: Usage {
                prompt_tokens: prompt_len,
                completion_tokens: tokens.len(),
                total_tokens: prompt_len + tokens.len(),
            },
        }
    } else {
        CompletionResponse {
            id: gen_id(),
            object: "chat.completion".to_string(),
            created: unix_timestamp(),
            choices: vec![Choice {
                index: 0,
                text: None,
                message: Some(ResponseMessage {
                    role: "assistant".to_string(),
                    content: text,
                }),
                finish_reason: Some(finish_reason.to_string()),
            }],
            usage: Usage {
                prompt_tokens: prompt_len,
                completion_tokens: tokens.len(),
                total_tokens: prompt_len + tokens.len(),
            },
        }
    };

    (StatusCode::OK, Json(serde_json::to_value(response).unwrap())).into_response()
}

fn error_body(message: &str) -> serde_json::Value {
    serde_json::json!({
        "error": {
            "message": message,
            "type": "invalid_request_error",
            "code": null
        }
    })
}

fn gen_id() -> String {
    format!("cmpl-{:016x}", rand::random::<u64>())
}

fn unix_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

fn decode_tokens(tokenizer: &Tokenizer, tokens: &[u32]) -> String {
    tokenizer
        .decode(tokens, true)
        .unwrap_or_else(|_| String::new())
}
