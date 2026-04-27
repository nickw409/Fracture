use axum::extract::State;
use axum::http::StatusCode;
use axum::response::sse::{Event, Sse};
use axum::response::{IntoResponse, Json};
use axum::routing::{get, post};
use axum::Router;
use fracture_core::Backend;
use fracture_engine::{Engine, PagedKvCacheManager};
use fracture_generate::{apply_chat_template, GenerationConfig, GenerationLoop, StopReason};
use std::convert::Infallible;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;
use tokio::sync::mpsc;
use tokenizers::Tokenizer;
use tower_http::cors::CorsLayer;

use crate::api::*;
use crate::dashboard::dto::RequestRecord;
use crate::dashboard::routes::{dashboard_routes, DashboardState};
use crate::utils::*;

/// Shared application state passed to all handlers.
pub struct AppState<B: Backend> {
    pub engine: Arc<Engine<B>>,
    pub cache: Mutex<PagedKvCacheManager>,
    pub tokenizer: Tokenizer,
    pub dashboard: Arc<DashboardState>,
}

/// Create the HTTP router with OpenAI-compatible endpoints.
pub fn create_router<B: Backend + 'static>(state: Arc<AppState<B>>) -> Router {
    let dashboard_state = Arc::clone(&state.dashboard);

    let api = Router::new()
        .route("/v1/completions", post(completions_handler::<B>))
        .route("/v1/chat/completions", post(chat_completions_handler::<B>))
        .route("/v1/models", get(models_handler))
        .route("/v1/profile", get(profile_handler))
        .route("/health", get(health_handler))
        .with_state(state);

    api.merge(dashboard_routes(dashboard_state))
        .layer(CorsLayer::permissive())
}

async fn completions_handler<B: Backend + 'static>(
    State(state): State<Arc<AppState<B>>>,
    Json(req): Json<CompletionRequest>,
) -> impl IntoResponse {
    // Validate model name
    if let Err(resp) = validate_model_name(req.model.as_deref()) {
        return resp;
    }

    // Validate request
    if let Err(resp) = validate_completion_request(&req) {
        return resp;
    }

    let encoding = match state.tokenizer.encode(req.prompt.as_str(), false) {
        Ok(enc) => enc,
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(error_body(&format!("tokenization failed: {e}"))),
            )
                .into_response()
        }
    };
    let prompt_tokens: Vec<u32> = encoding.get_ids().to_vec();
    let prompt_len = prompt_tokens.len();

    let config = GenerationConfig {
        max_tokens: req.max_tokens,
        temperature: req.temperature,
        top_k: req.top_k,
        top_p: req.top_p,
        ..Default::default()
    };

    if req.stream {
        return handle_streaming(state, prompt_tokens, config, true).into_response();
    }

    // Non-streaming
    state.dashboard.metrics.request_started();
    let t_start = Instant::now();
    let (tx, mut rx) = mpsc::unbounded_channel();
    let generated = {
        let mut cache = state.cache.lock().unwrap();
        GenerationLoop::generate(state.engine.as_ref(), &prompt_tokens, &config, &mut *cache, &tx)
    };

    match generated {
        Ok(result) => {
            drop(tx);
            let mut tokens = Vec::new();
            while let Some(t) = rx.recv().await {
                tokens.push(t);
            }
            let text = decode_tokens(&state.tokenizer, &tokens);
            let finish_reason: &'static str = match result.stop_reason {
                StopReason::Stop => "stop",
                StopReason::Length => "length",
            };
            let total_duration_ms = t_start.elapsed().as_secs_f64() * 1000.0;
            let tps = if total_duration_ms > 0.0 { tokens.len() as f64 / (total_duration_ms / 1000.0) } else { 0.0 };
            let req_id = gen_id();
            let record = RequestRecord {
                id: req_id.clone(),
                request_type: "completion",
                status: "completed",
                prompt_tokens: prompt_len,
                completion_tokens: tokens.len(),
                total_tokens: prompt_len + tokens.len(),
                time_to_first_token_ms: 0.0,
                total_duration_ms,
                tokens_per_second: tps,
                finish_reason,
                temperature: req.temperature,
                created_at: String::new(),
            };
            state.dashboard.metrics.record_completion(&record);
            state.dashboard.request_log.push(record);
            let response = CompletionResponse {
                id: req_id,
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
            };
            (StatusCode::OK, Json(serde_json::to_value(response).unwrap())).into_response()
        }
        Err(e) => {
            let total_duration_ms = t_start.elapsed().as_secs_f64() * 1000.0;
            let record = RequestRecord {
                id: gen_id(),
                request_type: "completion",
                status: "error",
                prompt_tokens: prompt_len,
                completion_tokens: 0,
                total_tokens: prompt_len,
                time_to_first_token_ms: 0.0,
                total_duration_ms,
                tokens_per_second: 0.0,
                finish_reason: "error",
                temperature: req.temperature,
                created_at: String::new(),
            };
            state.dashboard.metrics.record_completion(&record);
            state.dashboard.request_log.push(record);
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(error_body(&format!("generation failed: {e}"))),
            )
                .into_response()
        }
    }
}

async fn chat_completions_handler<B: Backend + 'static>(
    State(state): State<Arc<AppState<B>>>,
    Json(req): Json<ChatCompletionRequest>,
) -> impl IntoResponse {
    // Validate model name
    if let Err(resp) = validate_model_name(req.model.as_deref()) {
        return resp;
    }

    if let Err(resp) = validate_chat_request(&req) {
        return resp;
    }

    // Apply chat template
    let messages: Vec<(String, String)> = req
        .messages
        .iter()
        .map(|m| (m.role.clone(), m.content.clone()))
        .collect();
    let prompt = apply_chat_template(&messages);

    let encoding = match state.tokenizer.encode(prompt.as_str(), false) {
        Ok(enc) => enc,
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(error_body(&format!("tokenization failed: {e}"))),
            )
                .into_response()
        }
    };
    let prompt_tokens: Vec<u32> = encoding.get_ids().to_vec();
    let prompt_len = prompt_tokens.len();

    let config = GenerationConfig {
        max_tokens: req.max_tokens,
        temperature: req.temperature,
        top_k: req.top_k,
        top_p: req.top_p,
        ..Default::default()
    };

    if req.stream {
        return handle_streaming(state, prompt_tokens, config, false).into_response();
    }

    // Non-streaming
    state.dashboard.metrics.request_started();
    let t_start = Instant::now();
    let (tx, mut rx) = mpsc::unbounded_channel();
    let generated = {
        let mut cache = state.cache.lock().unwrap();
        GenerationLoop::generate(state.engine.as_ref(), &prompt_tokens, &config, &mut *cache, &tx)
    };

    match generated {
        Ok(result) => {
            drop(tx);
            let mut tokens = Vec::new();
            while let Some(t) = rx.recv().await {
                tokens.push(t);
            }
            let text = decode_tokens(&state.tokenizer, &tokens);
            let finish_reason: &'static str = match result.stop_reason {
                StopReason::Stop => "stop",
                StopReason::Length => "length",
            };
            let total_duration_ms = t_start.elapsed().as_secs_f64() * 1000.0;
            let tps = if total_duration_ms > 0.0 { tokens.len() as f64 / (total_duration_ms / 1000.0) } else { 0.0 };
            let req_id = gen_id();
            let record = RequestRecord {
                id: req_id.clone(),
                request_type: "chat",
                status: "completed",
                prompt_tokens: prompt_len,
                completion_tokens: tokens.len(),
                total_tokens: prompt_len + tokens.len(),
                time_to_first_token_ms: 0.0,
                total_duration_ms,
                tokens_per_second: tps,
                finish_reason,
                temperature: req.temperature,
                created_at: String::new(),
            };
            state.dashboard.metrics.record_completion(&record);
            state.dashboard.request_log.push(record);
            let response = CompletionResponse {
                id: req_id,
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
            };
            (StatusCode::OK, Json(serde_json::to_value(response).unwrap())).into_response()
        }
        Err(e) => {
            let total_duration_ms = t_start.elapsed().as_secs_f64() * 1000.0;
            let record = RequestRecord {
                id: gen_id(),
                request_type: "chat",
                status: "error",
                prompt_tokens: prompt_len,
                completion_tokens: 0,
                total_tokens: prompt_len,
                time_to_first_token_ms: 0.0,
                total_duration_ms,
                tokens_per_second: 0.0,
                finish_reason: "error",
                temperature: req.temperature,
                created_at: String::new(),
            };
            state.dashboard.metrics.record_completion(&record);
            state.dashboard.request_log.push(record);
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(error_body(&format!("generation failed: {e}"))),
            )
                .into_response()
        }
    }
}

fn handle_streaming<B: Backend + 'static>(
    state: Arc<AppState<B>>,
    prompt_tokens: Vec<u32>,
    config: GenerationConfig,
    is_completion: bool,
) -> Sse<impl futures_core::Stream<Item = std::result::Result<Event, Infallible>> + use<B>> {
    state.dashboard.metrics.request_started();

    let (tx, mut rx) = mpsc::unbounded_channel();
    let (result_tx, mut result_rx) = mpsc::unbounded_channel();
    let cancel = Arc::new(AtomicBool::new(false));
    let cancel_for_gen = Arc::clone(&cancel);
    let state_for_gen = Arc::clone(&state);
    let prompt_tokens_gen = prompt_tokens.clone();
    let prompt_len = prompt_tokens.len();
    let config_gen = config.clone();
    let tokenizer_for_stream = state.tokenizer.clone();
    let dashboard = Arc::clone(&state.dashboard);

    tokio::task::spawn_blocking(move || {
        let mut cache = state_for_gen.cache.lock().unwrap();
        let result = GenerationLoop::generate_with_cancel(
            state_for_gen.engine.as_ref(),
            &prompt_tokens_gen,
            &config_gen,
            &mut *cache,
            &tx,
            Some(cancel_for_gen),
        );
        let _ = result_tx.send(result);
    });

    let cancel_on_drop = Arc::clone(&cancel);

    let stream = async_stream::stream! {
        struct CancelGuard(Arc<AtomicBool>);
        impl Drop for CancelGuard {
            fn drop(&mut self) {
                self.0.store(true, Ordering::Relaxed);
            }
        }
        let _cancel_guard = CancelGuard(cancel_on_drop);

        // Ensures active_requests is decremented even if the stream is dropped.
        struct MetricsGuard(Arc<DashboardState>, bool);
        impl Drop for MetricsGuard {
            fn drop(&mut self) {
                if !self.1 {
                    self.0.metrics.record_completion(&RequestRecord {
                        id: String::new(),
                        request_type: "chat",
                        status: "cancelled",
                        prompt_tokens: 0,
                        completion_tokens: 0,
                        total_tokens: 0,
                        time_to_first_token_ms: 0.0,
                        total_duration_ms: 0.0,
                        tokens_per_second: 0.0,
                        finish_reason: "cancelled",
                        temperature: 0.0,
                        created_at: String::new(),
                    });
                }
            }
        }
        let mut metrics_guard = MetricsGuard(Arc::clone(&dashboard), false);

        let id = gen_id();
        let created = unix_timestamp();
        let model = LOADED_MODEL_NAME;
        let object = if is_completion { "text_completion" } else { "chat.completion.chunk" };
        let t_start = Instant::now();
        let mut t_first_token: Option<Instant> = None;
        let mut token_count = 0usize;

        while let Some(token_id) = rx.recv().await {
            if t_first_token.is_none() {
                t_first_token = Some(Instant::now());
            }
            token_count += 1;
            let text = decode_tokens(&tokenizer_for_stream, &[token_id]);
            let delta = if is_completion {
                serde_json::json!({
                    "id": id,
                    "object": object,
                    "created": created,
                    "model": model,
                    "choices": [{"index": 0, "text": text, "finish_reason": null}]
                })
            } else {
                serde_json::json!({
                    "id": id,
                    "object": object,
                    "created": created,
                    "model": model,
                    "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": null}]
                })
            };
            yield Ok::<_, Infallible>(Event::default().data(delta.to_string()));
        }

        // Emit final chunk with finish_reason and usage stats
        let gen_result = result_rx.recv().await;
        let (finish_reason, error_msg) = match gen_result {
            Some(Ok(result)) => {
                let reason = match result.stop_reason {
                    StopReason::Stop => "stop",
                    StopReason::Length => "length",
                };
                (reason, None)
            }
            Some(Err(e)) => ("error", Some(e.to_string())),
            None => ("error", Some("generation task dropped".to_string())),
        };

        if let Some(err_msg) = error_msg {
            let error_event = serde_json::json!({
                "error": {
                    "message": format!("generation failed: {err_msg}"),
                    "type": "server_error",
                    "code": null,
                }
            });
            yield Ok(Event::default().data(error_event.to_string()));
        } else {
            let final_chunk = if is_completion {
                serde_json::json!({
                    "id": id,
                    "object": object,
                    "created": created,
                    "model": model,
                    "choices": [{"index": 0, "text": "", "finish_reason": finish_reason}],
                    "usage": {
                        "prompt_tokens": prompt_len,
                        "completion_tokens": token_count,
                        "total_tokens": prompt_len + token_count,
                    }
                })
            } else {
                serde_json::json!({
                    "id": id,
                    "object": object,
                    "created": created,
                    "model": model,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}],
                    "usage": {
                        "prompt_tokens": prompt_len,
                        "completion_tokens": token_count,
                        "total_tokens": prompt_len + token_count,
                    }
                })
            };
            yield Ok(Event::default().data(final_chunk.to_string()));
        }

        // Record completion metrics.
        let total_duration_ms = t_start.elapsed().as_secs_f64() * 1000.0;
        let ttft_ms = t_first_token.map(|t| (t - t_start).as_secs_f64() * 1000.0).unwrap_or(0.0);
        let tps = if total_duration_ms > 0.0 { token_count as f64 / (total_duration_ms / 1000.0) } else { 0.0 };
        let record = RequestRecord {
            id: id.clone(),
            request_type: if is_completion { "completion" } else { "chat" },
            status: if finish_reason == "error" { "error" } else { "completed" },
            prompt_tokens: prompt_len,
            completion_tokens: token_count,
            total_tokens: prompt_len + token_count,
            time_to_first_token_ms: ttft_ms,
            total_duration_ms,
            tokens_per_second: tps,
            finish_reason,
            temperature: 0.0,
            created_at: String::new(),
        };
        dashboard.metrics.record_completion(&record);
        dashboard.request_log.push(record);
        metrics_guard.1 = true;

        yield Ok(Event::default().data("[DONE]"));
    };

    Sse::new(stream)
}

async fn profile_handler() -> impl IntoResponse {
    // TODO: return most recent RequestMetrics + aggregated stats
    // Requires shared state (Arc<Mutex<...>>) wired in when server is fully connected
    (
        StatusCode::OK,
        Json(serde_json::json!({"status": "profiling endpoint ready, metrics collection active on stderr"})),
    )
}

#[cfg(test)]
mod routes_tests;
