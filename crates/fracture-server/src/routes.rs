use axum::extract::State;
use axum::http::StatusCode;
use axum::response::sse::{Event, Sse};
use axum::response::{IntoResponse, Json};
use axum::routing::{get, post};
use axum::Router;
use fracture_core::Backend;
use fracture_engine::{Engine, KvCacheManager};
use fracture_generate::{apply_chat_template, GenerationConfig, GenerationLoop};
use std::convert::Infallible;
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::sync::mpsc;
use tokenizers::Tokenizer;

use crate::api::*;

/// Hardcoded model name for Phase 1 (single-model serving).
/// When AppState carries the loaded model name, this should be replaced.
const LOADED_MODEL_NAME: &str = "llama-3-8b";

/// Shared application state passed to all handlers.
pub struct AppState<B: Backend> {
    pub engine: Arc<Engine<B>>,
    pub cache: Mutex<KvCacheManager>,
    pub tokenizer: Tokenizer,
}

/// Create the HTTP router with OpenAI-compatible endpoints.
pub fn create_router<B: Backend + 'static>(state: Arc<AppState<B>>) -> Router {
    Router::new()
        .route("/v1/completions", post(completions_handler::<B>))
        .route("/v1/chat/completions", post(chat_completions_handler::<B>))
        .route("/v1/models", get(models_handler))
        .route("/v1/profile", get(profile_handler))
        .route("/health", get(health_handler))
        .with_state(state)
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
    let (tx, mut rx) = mpsc::unbounded_channel();
    let generated = {
        let mut cache = state.cache.lock().unwrap();
        GenerationLoop::generate(state.engine.as_ref(), &prompt_tokens, &config, &mut cache, &tx)
    };

    match generated {
        Ok(_) => {
            drop(tx);
            let mut tokens = Vec::new();
            while let Some(t) = rx.recv().await {
                tokens.push(t);
            }
            let text = decode_tokens(&state.tokenizer, &tokens);
            let response = CompletionResponse {
                id: gen_id(),
                object: "text_completion".to_string(),
                created: unix_timestamp(),
                choices: vec![Choice {
                    index: 0,
                    text: Some(text),
                    message: None,
                    finish_reason: Some("stop".to_string()),
                }],
                usage: Usage {
                    prompt_tokens: prompt_len,
                    completion_tokens: tokens.len(),
                    total_tokens: prompt_len + tokens.len(),
                },
            };
            (StatusCode::OK, Json(serde_json::to_value(response).unwrap())).into_response()
        }
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(error_body(&format!("generation failed: {e}"))),
        )
            .into_response(),
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
    let (tx, mut rx) = mpsc::unbounded_channel();
    let generated = {
        let mut cache = state.cache.lock().unwrap();
        GenerationLoop::generate(state.engine.as_ref(), &prompt_tokens, &config, &mut cache, &tx)
    };

    match generated {
        Ok(_) => {
            drop(tx);
            let mut tokens = Vec::new();
            while let Some(t) = rx.recv().await {
                tokens.push(t);
            }
            let text = decode_tokens(&state.tokenizer, &tokens);
            let response = CompletionResponse {
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
                    finish_reason: Some("stop".to_string()),
                }],
                usage: Usage {
                    prompt_tokens: prompt_len,
                    completion_tokens: tokens.len(),
                    total_tokens: prompt_len + tokens.len(),
                },
            };
            (StatusCode::OK, Json(serde_json::to_value(response).unwrap())).into_response()
        }
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(error_body(&format!("generation failed: {e}"))),
        )
            .into_response(),
    }
}

fn handle_streaming<B: Backend + 'static>(
    state: Arc<AppState<B>>,
    prompt_tokens: Vec<u32>,
    config: GenerationConfig,
    is_completion: bool,
) -> Sse<impl futures_core::Stream<Item = std::result::Result<Event, Infallible>> + use<B>> {
    let (tx, mut rx) = mpsc::unbounded_channel();
    let state_for_gen = Arc::clone(&state);
    let prompt_tokens_gen = prompt_tokens.clone();
    let config_gen = config.clone();
    let tokenizer_for_stream = state.tokenizer.clone();

    // Run generation in a blocking task since the engine is synchronous.
    //
    // Known limitation: streaming-error-propagation
    // Generation errors from GenerationLoop::generate are currently silently dropped
    // (`let _ = ...`). When generation fails mid-stream (e.g., GPU OOM during decode),
    // the SSE stream simply ends without notifying the client of the error. This should
    // be changed to emit an SSE error event with the failure details so clients can
    // distinguish a generation failure from normal stream completion.
    //
    // Known limitation: stream-cancellation
    // Client disconnect detection is not yet implemented. When a client disconnects
    // during SSE streaming, the generation loop continues running to completion,
    // wasting GPU compute and leaking KV cache memory. This requires a
    // CancellationToken or similar mechanism to be threaded into GenerationLoop
    // so that the decode loop can exit early on client disconnect.
    tokio::task::spawn_blocking(move || {
        let mut cache = state_for_gen.cache.lock().unwrap();
        let _ = GenerationLoop::generate(
            state_for_gen.engine.as_ref(),
            &prompt_tokens_gen,
            &config_gen,
            &mut cache,
            &tx,
        );
    });

    let stream = async_stream::stream! {
        let id = gen_id();
        let object = if is_completion { "text_completion" } else { "chat.completion.chunk" };

        while let Some(token_id) = rx.recv().await {
            let text = decode_tokens(&tokenizer_for_stream, &[token_id]);
            let delta = if is_completion {
                serde_json::json!({
                    "id": id,
                    "object": object,
                    "choices": [{"index": 0, "text": text}]
                })
            } else {
                serde_json::json!({
                    "id": id,
                    "object": object,
                    "choices": [{"index": 0, "delta": {"content": text}}]
                })
            };
            yield Ok::<_, Infallible>(Event::default().data(delta.to_string()));
        }

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

/// Build an OpenAI-compatible error response body.
fn error_body(message: &str) -> serde_json::Value {
    serde_json::json!({
        "error": {
            "message": message,
            "type": "invalid_request_error",
            "code": null
        }
    })
}

pub(crate) fn validate_model_name(
    model: Option<&str>,
) -> std::result::Result<(), axum::response::Response> {
    if let Some(name) = model {
        if name != LOADED_MODEL_NAME {
            return Err((
                StatusCode::NOT_FOUND,
                Json(error_body(&format!(
                    "The model `{name}` does not exist"
                ))),
            )
                .into_response());
        }
    }
    Ok(())
}

pub(crate) fn validate_completion_request(
    req: &CompletionRequest,
) -> std::result::Result<(), axum::response::Response> {
    if req.prompt.is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(error_body("prompt must not be empty")),
        )
            .into_response());
    }
    validate_sampling_params(req.temperature, req.top_p, req.top_k, req.max_tokens)
}

pub(crate) fn validate_chat_request(
    req: &ChatCompletionRequest,
) -> std::result::Result<(), axum::response::Response> {
    if req.messages.is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(error_body("messages must not be empty")),
        )
            .into_response());
    }
    validate_sampling_params(req.temperature, req.top_p, req.top_k, req.max_tokens)
}

pub(crate) fn validate_sampling_params(
    temperature: f32,
    top_p: f32,
    _top_k: usize,
    max_tokens: usize,
) -> std::result::Result<(), axum::response::Response> {
    if temperature < 0.0 {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(error_body("temperature must be >= 0")),
        )
            .into_response());
    }
    if top_p < 0.0 || top_p > 1.0 {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(error_body("top_p must be in [0, 1]")),
        )
            .into_response());
    }
    if max_tokens == 0 {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(error_body("max_tokens must be > 0")),
        )
            .into_response());
    }
    Ok(())
}

fn decode_tokens(tokenizer: &Tokenizer, tokens: &[u32]) -> String {
    tokenizer
        .decode(tokens, true)
        .unwrap_or_else(|_| String::new())
}

fn gen_id() -> String {
    format!("cmpl-{:016x}", rand::random::<u64>())
}

fn unix_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::{ChatMessage, ChatCompletionRequest, CompletionRequest};
    use http_body_util::BodyExt;

    fn valid_completion_request() -> CompletionRequest {
        CompletionRequest {
            model: None,
            prompt: "Hello world".to_string(),
            max_tokens: 256,
            temperature: 1.0,
            top_p: 1.0,
            top_k: 0,
            stream: false,
        }
    }

    fn valid_chat_request() -> ChatCompletionRequest {
        ChatCompletionRequest {
            model: None,
            messages: vec![ChatMessage {
                role: "user".to_string(),
                content: "Hello".to_string(),
            }],
            max_tokens: 256,
            temperature: 1.0,
            top_p: 1.0,
            top_k: 0,
            stream: false,
        }
    }

    /// Helper: extract the error response JSON from a validation Err.
    /// Parses the response body and returns the parsed JSON value.
    async fn extract_error_json(resp: axum::response::Response) -> serde_json::Value {
        let body_bytes = resp.into_body().collect().await.unwrap().to_bytes();
        serde_json::from_slice(&body_bytes).unwrap()
    }

    /// Helper: assert that an error response uses the OpenAI nested format:
    /// {"error": {"message": "...", "type": "invalid_request_error", "code": null}}
    fn assert_openai_error_format(json: &serde_json::Value) {
        let error_obj = json.get("error").expect("response must have 'error' key");
        assert!(error_obj.is_object(), "error value must be an object, not a string");
        assert!(error_obj.get("message").is_some(), "error object must have 'message'");
        assert_eq!(
            error_obj.get("type").and_then(|v| v.as_str()),
            Some("invalid_request_error")
        );
        assert!(error_obj.get("code").is_some(), "error object must have 'code'");
    }

    #[test]
    fn test_empty_prompt_returns_error() {
        let mut req = valid_completion_request();
        req.prompt = "".to_string();
        assert!(validate_completion_request(&req).is_err());
    }

    #[test]
    fn test_empty_messages_returns_error() {
        let mut req = valid_chat_request();
        req.messages = vec![];
        assert!(validate_chat_request(&req).is_err());
    }

    #[test]
    fn test_negative_temperature_returns_error() {
        assert!(validate_sampling_params(-1.0, 1.0, 0, 256).is_err());
    }

    #[test]
    fn test_top_p_greater_than_one_returns_error() {
        assert!(validate_sampling_params(1.0, 1.5, 0, 256).is_err());
    }

    #[test]
    fn test_top_p_less_than_zero_returns_error() {
        assert!(validate_sampling_params(1.0, -0.1, 0, 256).is_err());
    }

    #[test]
    fn test_max_tokens_zero_returns_error() {
        assert!(validate_sampling_params(1.0, 1.0, 0, 0).is_err());
    }

    #[test]
    fn test_valid_params_pass() {
        assert!(validate_sampling_params(1.0, 1.0, 0, 256).is_ok());
        assert!(validate_sampling_params(0.0, 0.5, 10, 100).is_ok());
    }

    #[test]
    fn test_valid_completion_request_passes() {
        assert!(validate_completion_request(&valid_completion_request()).is_ok());
    }

    #[test]
    fn test_valid_chat_request_passes() {
        assert!(validate_chat_request(&valid_chat_request()).is_ok());
    }

    #[test]
    fn test_valid_params_with_streaming() {
        let mut req = valid_completion_request();
        req.stream = true;
        assert!(validate_completion_request(&req).is_ok());

        let mut chat_req = valid_chat_request();
        chat_req.stream = true;
        assert!(validate_chat_request(&chat_req).is_ok());
    }

    #[test]
    fn test_temperature_zero_is_valid() {
        assert!(validate_sampling_params(0.0, 1.0, 0, 256).is_ok());
    }

    #[test]
    fn test_top_p_boundary_values() {
        assert!(validate_sampling_params(1.0, 0.0, 0, 256).is_ok());
        assert!(validate_sampling_params(1.0, 1.0, 0, 256).is_ok());
    }

    #[test]
    fn test_max_tokens_one_is_valid() {
        assert!(validate_sampling_params(1.0, 1.0, 0, 1).is_ok());
    }

    // --- Gap #6: /v1/models endpoint ---

    #[tokio::test]
    async fn test_models_endpoint_exists() {
        // Call the models_handler directly and verify response JSON.
        let resp = models_handler().await.into_response();
        let body_bytes = resp.into_body().collect().await.unwrap().to_bytes();
        let json: serde_json::Value = serde_json::from_slice(&body_bytes).unwrap();

        assert_eq!(json["object"], "list");
        let data = json["data"].as_array().expect("data must be an array");
        assert_eq!(data.len(), 1);
        assert_eq!(data[0]["id"], "llama-3-8b");
        assert_eq!(data[0]["object"], "model");
        assert!(data[0]["created"].as_u64().is_some());
        assert_eq!(data[0]["owned_by"], "fracture");
    }

    // --- Gap #7: /health endpoint ---

    #[tokio::test]
    async fn test_health_endpoint() {
        let resp = health_handler().await.into_response();
        let body_bytes = resp.into_body().collect().await.unwrap().to_bytes();
        let json: serde_json::Value = serde_json::from_slice(&body_bytes).unwrap();
        assert_eq!(json, serde_json::json!({"status": "ready"}));
    }

    // --- Gap #8: model name mismatch ---

    #[test]
    fn test_model_name_mismatch() {
        let result = validate_model_name(Some("wrong-model"));
        assert!(result.is_err(), "mismatched model name should be rejected");
    }

    // --- Gap #9: model name None passes ---

    #[test]
    fn test_model_name_none_passes() {
        assert!(validate_model_name(None).is_ok());
    }

    #[test]
    fn test_model_name_correct_passes() {
        assert!(validate_model_name(Some("llama-3-8b")).is_ok());
    }

    // --- Gap #10: error response format ---

    #[tokio::test]
    async fn test_error_response_format() {
        // Verify that validation errors use the OpenAI nested error format.

        // Empty prompt error
        let mut req = valid_completion_request();
        req.prompt = "".to_string();
        let err_resp = validate_completion_request(&req).unwrap_err();
        let json = extract_error_json(err_resp).await;
        assert_openai_error_format(&json);

        // Empty messages error
        let mut chat_req = valid_chat_request();
        chat_req.messages = vec![];
        let err_resp = validate_chat_request(&chat_req).unwrap_err();
        let json = extract_error_json(err_resp).await;
        assert_openai_error_format(&json);

        // Negative temperature error
        let err_resp = validate_sampling_params(-1.0, 1.0, 0, 256).unwrap_err();
        let json = extract_error_json(err_resp).await;
        assert_openai_error_format(&json);

        // top_p out of range error
        let err_resp = validate_sampling_params(1.0, 1.5, 0, 256).unwrap_err();
        let json = extract_error_json(err_resp).await;
        assert_openai_error_format(&json);

        // max_tokens=0 error
        let err_resp = validate_sampling_params(1.0, 1.0, 0, 0).unwrap_err();
        let json = extract_error_json(err_resp).await;
        assert_openai_error_format(&json);

        // Model name mismatch error
        let err_resp = validate_model_name(Some("nonexistent")).unwrap_err();
        let json = extract_error_json(err_resp).await;
        assert_openai_error_format(&json);
    }

    // --- Gap #11: streaming not exercised (known limitation) ---

    /// Streaming requires a running engine with a GPU and loaded model weights.
    /// This cannot be tested in unit tests because the Engine requires a concrete
    /// Backend implementation with actual device memory. Streaming correctness
    /// (SSE event format, token-by-token delivery, [DONE] sentinel) must be
    /// validated in integration tests with a fully initialized server.
    #[test]
    #[ignore]
    fn test_streaming_not_exercised_note() {
        // Intentionally empty — this test documents a known gap.
        // Streaming tests require integration test infrastructure with:
        // - A concrete Backend implementation (e.g., CUDA)
        // - Loaded model weights
        // - A running tokio runtime with the full server stack
    }

    // --- Gap #12: concurrent requests not exercised (known limitation) ---

    /// Concurrent request handling requires a running engine and multiple
    /// simultaneous HTTP connections. The current Phase 1 architecture uses a
    /// shared Mutex<KvCacheManager> which serializes all generation requests.
    /// True concurrent request testing requires:
    /// - Multiple tokio tasks issuing requests in parallel
    /// - A concrete Backend with actual GPU resources
    /// - Verification that one request completing/failing doesn't corrupt another
    /// This is deferred to integration testing (and eventually Phase 4 continuous batching).
    #[test]
    #[ignore]
    fn test_concurrent_requests_not_exercised_note() {
        // Intentionally empty — this test documents a known gap.
        // Concurrent request testing requires integration test infrastructure.
    }

    // --- Gap #15: chat template applied correctly ---

    #[test]
    fn test_chat_template_called_correctly() {
        // Verify apply_chat_template produces expected output for server-relevant messages.
        let messages = vec![
            ("system".to_string(), "You are a helpful assistant.".to_string()),
            ("user".to_string(), "What is Rust?".to_string()),
        ];
        let result = apply_chat_template(&messages);

        // Verify structure: begin_of_text, then each message wrapped in header tags,
        // then trailing assistant header.
        assert!(result.starts_with("<|begin_of_text|>"));
        assert!(result.contains("<|start_header_id|>system<|end_header_id|>\n\nYou are a helpful assistant.<|eot_id|>"));
        assert!(result.contains("<|start_header_id|>user<|end_header_id|>\n\nWhat is Rust?<|eot_id|>"));
        assert!(result.ends_with("<|start_header_id|>assistant<|end_header_id|>\n\n"));

        // Single user message (common case for server)
        let messages = vec![("user".to_string(), "Hello".to_string())];
        let result = apply_chat_template(&messages);
        assert_eq!(
            result,
            "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nHello<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        );
    }
}
