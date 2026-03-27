use axum::{Router, routing::post};

/// Create the HTTP router with OpenAI-compatible endpoints.
pub fn create_router() -> Router {
    Router::new()
        .route("/v1/completions", post(completions_handler))
        .route("/v1/chat/completions", post(chat_completions_handler))
}

async fn completions_handler() -> &'static str {
    // TODO: wire to generation loop
    "not yet implemented"
}

async fn chat_completions_handler() -> &'static str {
    // TODO: wire to generation loop
    "not yet implemented"
}
