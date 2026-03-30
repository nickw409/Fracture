//! Fracture coordinator binary (CUDA backend).
//!
//! Listens for worker connections, runs the scheduler to assign layers,
//! orchestrates the distributed pipeline, and serves an OpenAI-compatible
//! HTTP API with async generation through the distributed pipeline.

use anyhow::Result;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Json};
use axum::routing::{get, post};
use axum::Router;
use fracture_coordinator::{
    pipeline::DistributedPipeline,
    registry::PeerRegistry,
    scheduler::{self, SchedulerInput, SchedulingMode, WorkerCapabilities},
    state::SequenceStateManager,
};
use fracture_generate::{Sampler, SamplingParams};
use fracture_protocol::{
    connection::FramedConnection,
    frame::MessageType,
    messages::*,
};
use fracture_server::api::*;
use fracture_server::dashboard::dto::ModelInfo as DashboardModelInfo;
use fracture_server::dashboard::metrics::MetricsCollector;
use fracture_server::dashboard::request_log::RequestLog;
use fracture_server::dashboard::routes::{dashboard_routes, ClusterProvider, DashboardState};
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tokio::net::TcpListener;
use tokio::sync::Mutex;
use tokenizers::Tokenizer;
use tracing_subscriber::EnvFilter;

/// Shared state for the coordinator's HTTP handlers.
struct CoordState {
    pipeline: Arc<DistributedPipeline>,
    registry: Arc<Mutex<PeerRegistry>>,
    seq_mgr: Arc<Mutex<SequenceStateManager>>,
    tokenizer: Tokenizer,
    max_seq_len: usize,
}

/// Coordinator config parsed from CLI args.
struct CoordinatorConfig {
    listen_address: String,
    model_path: String,
    expected_workers: usize,
    http_port: u16,
    max_seq_len: usize,
    scheduling_mode: String,
    tokenizer_path: Option<String>,
    /// Timeout in seconds for waiting for all workers to register.
    /// 0 = no timeout (wait indefinitely).
    acceptance_timeout_secs: u64,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .init();

    let args: Vec<String> = std::env::args().collect();

    let config = CoordinatorConfig {
        listen_address: args
            .iter()
            .position(|a| a == "--listen")
            .and_then(|i| args.get(i + 1).cloned())
            .unwrap_or_else(|| "0.0.0.0:9400".into()),
        model_path: args
            .iter()
            .position(|a| a == "--model")
            .and_then(|i| args.get(i + 1).cloned())
            .expect("--model <path-to-gguf> is required"),
        expected_workers: args
            .iter()
            .position(|a| a == "--workers")
            .and_then(|i| args.get(i + 1))
            .and_then(|p| p.parse().ok())
            .expect("--workers <count> is required"),
        http_port: args
            .iter()
            .position(|a| a == "--http-port")
            .and_then(|i| args.get(i + 1))
            .and_then(|p| p.parse().ok())
            .unwrap_or(8080),
        max_seq_len: args
            .iter()
            .position(|a| a == "--max-seq-len")
            .and_then(|i| args.get(i + 1))
            .and_then(|p| p.parse().ok())
            .unwrap_or(4096),
        scheduling_mode: args
            .iter()
            .position(|a| a == "--scheduling")
            .and_then(|i| args.get(i + 1).cloned())
            .unwrap_or_else(|| "auto".into()),
        tokenizer_path: args
            .iter()
            .position(|a| a == "--tokenizer")
            .and_then(|i| args.get(i + 1).cloned()),
        acceptance_timeout_secs: args
            .iter()
            .position(|a| a == "--acceptance-timeout")
            .and_then(|i| args.get(i + 1))
            .and_then(|p| p.parse().ok())
            .unwrap_or(120), // default 2 minutes
    };

    tracing::info!("Fracture coordinator");
    tracing::info!("listening for workers on {}", config.listen_address);
    tracing::info!("expecting {} workers", config.expected_workers);
    tracing::info!("model: {}", config.model_path);

    // Parse GGUF metadata for model config
    let gguf = fracture_gguf::GgufParser::parse(std::path::Path::new(&config.model_path))?;
    let model_config = gguf.config.clone();
    tracing::info!(
        "model: {}L, d={}, vocab={}",
        model_config.num_layers,
        model_config.hidden_size,
        model_config.vocab_size
    );

    // Listen for worker connections
    let listener = TcpListener::bind(&config.listen_address).await?;
    let mut registry = PeerRegistry::new();

    let timeout_duration = if config.acceptance_timeout_secs > 0 {
        Some(Duration::from_secs(config.acceptance_timeout_secs))
    } else {
        None
    };
    tracing::info!(
        "waiting for {} workers to register (timeout: {})...",
        config.expected_workers,
        timeout_duration.map_or("none".to_string(), |d| format!("{}s", d.as_secs()))
    );

    let accept_start = std::time::Instant::now();

    while registry.active_count() < config.expected_workers {
        if let Some(timeout) = timeout_duration {
            let elapsed = accept_start.elapsed();
            if elapsed >= timeout {
                anyhow::bail!(
                    "timed out waiting for workers: got {}/{} after {}s",
                    registry.active_count(),
                    config.expected_workers,
                    elapsed.as_secs()
                );
            }
        }

        let accept_future = listener.accept();
        let (stream, addr) = if let Some(timeout) = timeout_duration {
            let remaining = timeout.saturating_sub(accept_start.elapsed());
            match tokio::time::timeout(remaining, accept_future).await {
                Ok(result) => result?,
                Err(_) => {
                    anyhow::bail!(
                        "timed out waiting for workers: got {}/{} after {}s",
                        registry.active_count(),
                        config.expected_workers,
                        config.acceptance_timeout_secs
                    );
                }
            }
        } else {
            accept_future.await?
        };
        tracing::info!("worker connected from {addr}");
        let mut conn = FramedConnection::new(stream);

        let (header, payload) = conn.recv().await?;
        if header.msg_type != MessageType::Register {
            tracing::warn!(
                "expected Register from {addr}, got {:?} — dropping",
                header.msg_type
            );
            continue;
        }

        let reg: RegisterPayload = FramedConnection::deserialize_payload(&payload)?;
        tracing::info!(
            "worker '{}' registered: {} ({:.1} GB available, decode={:.2} ms/layer)",
            reg.node_id,
            reg.gpu_model,
            reg.gpu_memory_available as f64 / 1e9,
            reg.decode_ms_per_layer
        );

        let caps = WorkerCapabilities {
            node_id: reg.node_id.clone(),
            gpu_model: reg.gpu_model,
            gpu_memory_available: reg.gpu_memory_available as usize,
            compute_capability: reg.compute_capability,
            decode_ms_per_layer: reg.decode_ms_per_layer,
            prefill_ms_per_layer_128: reg.prefill_ms_per_layer_128,
        };

        registry.register(caps, conn)?;
    }

    tracing::info!(
        "all {} workers registered — running scheduler",
        config.expected_workers
    );

    // Run scheduler
    let scheduling_mode = match config.scheduling_mode.as_str() {
        "auto" => SchedulingMode::Auto,
        "equal" => SchedulingMode::EqualSplit,
        _ => {
            tracing::warn!(
                "unknown scheduling mode '{}', defaulting to auto",
                config.scheduling_mode
            );
            SchedulingMode::Auto
        }
    };

    let scheduler_input = SchedulerInput {
        model_config: model_config.clone(),
        workers: registry.all_capabilities(),
        coordinator_compute: None,
        mode: scheduling_mode,
        max_seq_len: config.max_seq_len,
        hop_latency_ms: 2.0,
    };

    let schedule_result = scheduler::schedule(&scheduler_input)?;

    tracing::info!("scheduler result:");
    for a in &schedule_result.assignments {
        tracing::info!(
            "  {} → layers {:?} ({:?}), {:.1} ms/decode, {:.1} GB weights, {:.1} GB cache",
            a.node_id,
            a.layer_range,
            a.role,
            a.expected_decode_ms,
            a.weight_memory_gb,
            a.cache_memory_gb
        );
    }
    tracing::info!(
        "pipeline: {:.1} ms/token, imbalance={:.2}, bottleneck={}",
        schedule_result.pipeline_decode_ms,
        schedule_result.imbalance_ratio,
        schedule_result.bottleneck_node
    );
    for ex in &schedule_result.excluded_nodes {
        tracing::info!("  excluded: {} ({:?})", ex.node_id, ex.reason);
    }

    // Send RegisterAck to each worker
    for assignment in &schedule_result.assignments {
        let ack = RegisterAckPayload {
            layer_start: assignment.layer_range.start as u32,
            layer_end: assignment.layer_range.end as u32,
            total_layers: model_config.num_layers as u32,
            max_seq_len: config.max_seq_len as u32,
            model_config: model_config.clone(),
        };
        registry.assign(&assignment.node_id, assignment.clone())?;
        let entry = registry.get_mut(&assignment.node_id).ok_or_else(|| {
            anyhow::anyhow!("worker '{}' disappeared", assignment.node_id)
        })?;
        entry
            .connection
            .send(MessageType::RegisterAck, 0, &ack)
            .await?;
        tracing::info!("sent RegisterAck to '{}'", assignment.node_id);
    }

    // Build distributed pipeline
    let pipeline = DistributedPipeline::new(&schedule_result.assignments, model_config.hidden_size)?;
    tracing::info!(
        "distributed pipeline ready with {} stages",
        pipeline.pipeline_order().len()
    );

    // Load tokenizer
    let tokenizer = load_tokenizer(&config)?;

    // Build shared state
    let state = Arc::new(CoordState {
        pipeline: Arc::new(pipeline),
        registry: Arc::new(Mutex::new(registry)),
        seq_mgr: Arc::new(Mutex::new(SequenceStateManager::new())),
        tokenizer,
        max_seq_len: config.max_seq_len,
    });

    // Build dashboard state
    let dashboard_state = Arc::new(DashboardState {
        metrics: Arc::new(MetricsCollector::new()),
        request_log: Arc::new(RequestLog::new()),
        cluster: ClusterProvider::Standalone {
            gpu_name: "distributed".to_string(),
            vram_total_mb: 0,
            vram_used_mb: 0,
            model: DashboardModelInfo {
                name: "llama-3-8b".to_string(),
                parameters: "8B".to_string(),
                layers: model_config.num_layers,
                context_length: config.max_seq_len,
                dtype: "FP16".to_string(),
            },
            total_layers: model_config.num_layers,
        },
        scheduler: None,
    });

    // Build HTTP router
    use tower_http::cors::CorsLayer;
    let router = Router::new()
        .route("/v1/completions", post(completions_handler))
        .route("/v1/models", get(models_handler))
        .route("/health", get(health_handler))
        .with_state(state.clone())
        .merge(dashboard_routes(dashboard_state))
        .layer(CorsLayer::permissive());

    // Start HTTP server
    let http_addr = format!("0.0.0.0:{}", config.http_port);
    let http_listener = tokio::net::TcpListener::bind(&http_addr).await?;
    tracing::info!("HTTP server listening on {}", http_listener.local_addr()?);

    // Serve HTTP with graceful shutdown on ctrl-c
    axum::serve(http_listener, router)
        .with_graceful_shutdown(async {
            tokio::signal::ctrl_c().await.ok();
            tracing::info!("shutting down...");
        })
        .await?;

    // Send Shutdown to all workers
    {
        let pipeline = &state.pipeline;
        let mut reg = state.registry.lock().await;
        for node_id in pipeline.pipeline_order() {
            if let Some(entry) = reg.get_mut(node_id) {
                let _ = entry
                    .connection
                    .send_empty(MessageType::Shutdown, 0)
                    .await;
            }
        }
    }

    tracing::info!("coordinator shut down");
    Ok(())
}

// ── HTTP Handlers ───────────────────────────────────────────────────────

async fn health_handler() -> impl IntoResponse {
    Json(serde_json::json!({"status": "ready"}))
}

async fn models_handler() -> impl IntoResponse {
    Json(serde_json::json!({
        "object": "list",
        "data": [{
            "id": "llama-3-8b",
            "object": "model",
            "created": unix_timestamp(),
            "owned_by": "fracture"
        }]
    }))
}

async fn completions_handler(
    State(state): State<Arc<CoordState>>,
    Json(req): Json<CompletionRequest>,
) -> impl IntoResponse {
    // Validate
    if req.prompt.is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Json(error_json("empty prompt")),
        )
            .into_response();
    }
    if req.temperature < 0.0 {
        return (
            StatusCode::BAD_REQUEST,
            Json(error_json("negative temperature")),
        )
            .into_response();
    }

    // Tokenize
    let encoding = match state.tokenizer.encode(req.prompt.as_str(), false) {
        Ok(enc) => enc,
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(error_json(&format!("tokenization failed: {e}"))),
            )
                .into_response()
        }
    };
    let prompt_tokens: Vec<u32> = encoding.get_ids().to_vec();
    let prompt_len = prompt_tokens.len();

    if prompt_len > state.max_seq_len {
        return (
            StatusCode::BAD_REQUEST,
            Json(error_json(&format!(
                "prompt length {} exceeds max_seq_len {}",
                prompt_len, state.max_seq_len
            ))),
        )
            .into_response();
    }

    // Run generation through the distributed pipeline
    let result = distributed_generate(
        &state.pipeline,
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
            Json(serde_json::json!({
                "id": format!("cmpl-{}", unix_timestamp()),
                "object": "text_completion",
                "created": unix_timestamp(),
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
            Json(error_json(&format!("generation failed: {e}"))),
        )
            .into_response(),
    }
}

// ── Distributed Generation ──────────────────────────────────────────────

/// Llama 3 EOS tokens.
const STOP_TOKENS: &[u32] = &[128001, 128008, 128009];

/// Run generation through the distributed pipeline.
///
/// This is the async equivalent of `GenerationLoop::generate`, using
/// network forward passes instead of local engine calls.
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
        pipeline.alloc_cache(&mut reg, seq_id, 0).await?;
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

// ── Helpers ─────────────────────────────────────────────────────────────

fn load_tokenizer(config: &CoordinatorConfig) -> Result<Tokenizer> {
    if let Some(ref path) = config.tokenizer_path {
        return Tokenizer::from_file(path)
            .map_err(|e| anyhow::anyhow!("failed to load tokenizer: {e}"));
    }
    let model_dir = std::path::Path::new(&config.model_path)
        .parent()
        .unwrap_or(std::path::Path::new("."));
    let tokenizer_file = model_dir.join("tokenizer.json");
    if tokenizer_file.exists() {
        Tokenizer::from_file(&tokenizer_file)
            .map_err(|e| anyhow::anyhow!("failed to load tokenizer: {e}"))
    } else {
        anyhow::bail!(
            "no tokenizer found. Provide --tokenizer <path> or place tokenizer.json next to the model file"
        );
    }
}

fn decode_tokens(tokenizer: &Tokenizer, tokens: &[u32]) -> String {
    tokenizer
        .decode(tokens, true)
        .unwrap_or_else(|_| String::new())
}

fn unix_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs()
}

fn error_json(message: &str) -> serde_json::Value {
    serde_json::json!({
        "error": {
            "message": message,
            "type": "invalid_request_error",
            "code": null
        }
    })
}
