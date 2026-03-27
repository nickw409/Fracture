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
        .route("/v1/profile", get(profile_handler))
        .with_state(state)
}

async fn completions_handler<B: Backend + 'static>(
    State(state): State<Arc<AppState<B>>>,
    Json(req): Json<CompletionRequest>,
) -> impl IntoResponse {
    // Validate request
    if let Err(resp) = validate_completion_request(&req) {
        return resp;
    }

    let encoding = match state.tokenizer.encode(req.prompt.as_str(), false) {
        Ok(enc) => enc,
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({"error": format!("tokenization failed: {e}")})),
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
            Json(serde_json::json!({"error": format!("generation failed: {e}")})),
        )
            .into_response(),
    }
}

async fn chat_completions_handler<B: Backend + 'static>(
    State(state): State<Arc<AppState<B>>>,
    Json(req): Json<ChatCompletionRequest>,
) -> impl IntoResponse {
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
                Json(serde_json::json!({"error": format!("tokenization failed: {e}")})),
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
            Json(serde_json::json!({"error": format!("generation failed: {e}")})),
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

    // Run generation in a blocking task since the engine is synchronous
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

pub(crate) fn validate_completion_request(
    req: &CompletionRequest,
) -> std::result::Result<(), axum::response::Response> {
    if req.prompt.is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({"error": "prompt must not be empty"})),
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
            Json(serde_json::json!({"error": "messages must not be empty"})),
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
            Json(serde_json::json!({"error": "temperature must be >= 0"})),
        )
            .into_response());
    }
    if top_p < 0.0 || top_p > 1.0 {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({"error": "top_p must be in [0, 1]"})),
        )
            .into_response());
    }
    if max_tokens == 0 {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({"error": "max_tokens must be > 0"})),
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
}
