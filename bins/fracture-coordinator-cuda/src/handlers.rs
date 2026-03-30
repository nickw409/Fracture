use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Json};
use fracture_coordinator::{
    pipeline::DistributedPipeline,
    registry::PeerRegistry,
    state::SequenceStateManager,
};
use fracture_generate::{Sampler, SamplingParams};
use fracture_server::api::*;
use fracture_server::utils::{decode_tokens, error_body};
use std::sync::Arc;
use tokio::sync::Mutex;
use tokenizers::Tokenizer;

/// Shared state for the coordinator's HTTP handlers.
pub struct CoordState {
    pub pipeline_rx: tokio::sync::watch::Receiver<Arc<DistributedPipeline>>,
    pub registry: Arc<Mutex<PeerRegistry>>,
    pub seq_mgr: Arc<Mutex<SequenceStateManager>>,
    pub tokenizer: Tokenizer,
    pub max_seq_len: usize,
}

/// Llama 3 EOS tokens.
const STOP_TOKENS: &[u32] = &[128001, 128008, 128009];

pub async fn completions_handler(
    State(state): State<Arc<CoordState>>,
    Json(req): Json<CompletionRequest>,
) -> impl IntoResponse {
    // Validate
    if req.prompt.is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Json(error_body("empty prompt")),
        )
            .into_response();
    }
    if req.temperature < 0.0 {
        return (
            StatusCode::BAD_REQUEST,
            Json(error_body("negative temperature")),
        )
            .into_response();
    }

    // Tokenize
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

    if prompt_len > state.max_seq_len {
        return (
            StatusCode::BAD_REQUEST,
            Json(error_body(&format!(
                "prompt length {} exceeds max_seq_len {}",
                prompt_len, state.max_seq_len
            ))),
        )
            .into_response();
    }

    // Run generation through the distributed pipeline
    let pipeline = state.pipeline_rx.borrow().clone();
    let result = distributed_generate(
        &pipeline,
        &state.registry,
        &state.seq_mgr,
        &prompt_tokens,
        req.max_tokens,
        req.temperature,
        req.top_k,
        req.top_p,
    )
    .await;

    match result {
        Ok(generated_tokens) => {
            let text = decode_tokens(&state.tokenizer, &generated_tokens);
            let completion_len = generated_tokens.len();
            let ts = fracture_server::utils::unix_timestamp();
            Json(serde_json::json!({
                "id": format!("cmpl-{}", ts),
                "object": "text_completion",
                "created": ts,
                "choices": [{
                    "index": 0,
                    "text": text,
                    "finish_reason": "stop"
                }],
                "usage": {
                    "prompt_tokens": prompt_len,
                    "completion_tokens": completion_len,
                    "total_tokens": prompt_len + completion_len,
                }
            }))
            .into_response()
        }
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(error_body(&format!("generation failed: {e}"))),
        )
            .into_response(),
    }
}

/// Run generation through the distributed pipeline.
///
/// This is the async equivalent of `GenerationLoop::generate`, using
/// network forward passes instead of local engine calls.
#[allow(clippy::too_many_arguments)]
async fn distributed_generate(
    pipeline: &DistributedPipeline,
    registry: &Mutex<PeerRegistry>,
    seq_mgr: &Mutex<SequenceStateManager>,
    prompt_tokens: &[u32],
    max_tokens: usize,
    temperature: f32,
    top_k: usize,
    top_p: f32,
) -> fracture_core::Result<Vec<u32>> {
    let sampling_params = SamplingParams {
        temperature,
        top_k,
        top_p,
        seed: None,
    };

    // Create sequence and allocate cache on all workers
    let seq_id = {
        let mut mgr = seq_mgr.lock().await;
        mgr.create(
            prompt_tokens.len(),
            max_tokens,
            pipeline.pipeline_order().to_vec(),
        )
    };

    let result = distributed_generate_inner(
        pipeline,
        registry,
        seq_id,
        prompt_tokens,
        max_tokens,
        &sampling_params,
    )
    .await;

    // Always free cache on all workers
    {
        let mut reg = registry.lock().await;
        let _ = pipeline.free_cache(&mut reg, seq_id).await;
    }

    // Update sequence state
    {
        let mut mgr = seq_mgr.lock().await;
        match &result {
            Ok(_) => {
                let _ = mgr.complete(seq_id);
            }
            Err(_) => {
                let _ = mgr.mark_error(seq_id);
            }
        }
        mgr.remove(seq_id);
    }

    result
}

async fn distributed_generate_inner(
    pipeline: &DistributedPipeline,
    registry: &Mutex<PeerRegistry>,
    seq_id: u64,
    prompt_tokens: &[u32],
    max_tokens: usize,
    sampling_params: &SamplingParams,
) -> fracture_core::Result<Vec<u32>> {
    // Allocate cache on all workers
    {
        let mut reg = registry.lock().await;
        pipeline.alloc_cache(&mut reg, seq_id).await?;
    }

    // Prefill
    let positions: Vec<u32> = (0..prompt_tokens.len() as u32).collect();
    let logits = {
        let mut reg = registry.lock().await;
        pipeline
            .forward(&mut reg, seq_id, prompt_tokens, &positions, true)
            .await?
    };

    let mut next_token = Sampler::sample(&logits, sampling_params)?;
    if STOP_TOKENS.contains(&next_token) {
        return Ok(Vec::new());
    }

    let mut generated = vec![next_token];
    let mut pos = prompt_tokens.len() as u32;

    // Decode loop
    for _ in 1..max_tokens {
        let logits = {
            let mut reg = registry.lock().await;
            pipeline
                .forward(&mut reg, seq_id, &[next_token], &[pos], false)
                .await?
        };

        next_token = Sampler::sample(&logits, sampling_params)?;
        if STOP_TOKENS.contains(&next_token) {
            break;
        }

        generated.push(next_token);
        pos += 1;
    }

    Ok(generated)
}
