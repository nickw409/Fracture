//! Shared utilities for both Phase 3 (routes.rs) and Phase 4 (batched_routes.rs) HTTP handlers.

use axum::http::StatusCode;
use axum::response::{IntoResponse, Json};
use std::time::{SystemTime, UNIX_EPOCH};
use tokenizers::Tokenizer;

use crate::api::*;

/// Hardcoded model name for Phase 1 (single-model serving).
/// When AppState carries the loaded model name, this should be replaced.
pub const LOADED_MODEL_NAME: &str = "llama-3-8b";

/// Build an OpenAI-compatible error response body.
pub fn error_body(message: &str) -> serde_json::Value {
    serde_json::json!({
        "error": {
            "message": message,
            "type": "invalid_request_error",
            "code": null
        }
    })
}

pub fn gen_id() -> String {
    format!("cmpl-{:016x}", rand::random::<u64>())
}

pub fn unix_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

pub fn decode_tokens(tokenizer: &Tokenizer, tokens: &[u32]) -> String {
    tokenizer
        .decode(tokens, true)
        .unwrap_or_else(|_| String::new())
}

pub async fn models_handler() -> impl IntoResponse {
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

pub async fn health_handler() -> impl IntoResponse {
    (
        StatusCode::OK,
        Json(serde_json::json!({"status": "ready"})),
    )
}

pub fn validate_model_name(
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

pub fn validate_completion_request(
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

pub fn validate_chat_request(
    req: &ChatCompletionRequest,
) -> std::result::Result<(), axum::response::Response> {
    if req.messages.is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(error_body("messages must not be empty")),
        )
            .into_response());
    }
    for (i, msg) in req.messages.iter().enumerate() {
        if msg.role.is_empty() {
            return Err((
                StatusCode::BAD_REQUEST,
                Json(error_body(&format!("messages[{i}].role must not be empty"))),
            )
                .into_response());
        }
        if !matches!(msg.role.as_str(), "system" | "user" | "assistant") {
            return Err((
                StatusCode::BAD_REQUEST,
                Json(error_body(&format!(
                    "messages[{i}].role must be 'system', 'user', or 'assistant', got '{}'",
                    msg.role
                ))),
            )
                .into_response());
        }
    }
    validate_sampling_params(req.temperature, req.top_p, req.top_k, req.max_tokens)
}

pub fn validate_sampling_params(
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
