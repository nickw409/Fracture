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
use fracture_server::{create_batched_router, BatchedAppState, SchedulerHandle};
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tokio::net::TcpListener;
use tokio::sync::Mutex;
use tokenizers::Tokenizer;
use tracing_subscriber::EnvFilter;

mod distributed_loop;

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
    /// Use the batched scheduler loop (Phase 4) instead of the
    /// one-at-a-time distributed_generate path.
    batched: bool,
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
        batched: args.iter().any(|a| a == "--batched"),
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

    // Shared state — created before HTTP starts so dashboard is available immediately.
    let registry = Arc::new(Mutex::new(PeerRegistry::new()));
    let tokenizer = load_tokenizer(&config)?;

    // Dashboard cluster snapshot via watch channel (starts empty, filled by worker acceptance).
    use fracture_server::dashboard::dto::{ClusterResponse as DashboardClusterResponse, WorkerInfo as DashboardWorkerInfo};
    let empty_cluster = build_cluster_snapshot(&registry, &model_config, config.max_seq_len).await;
    let (cluster_tx, cluster_rx) = tokio::sync::watch::channel(empty_cluster);

    let dashboard_state = Arc::new(DashboardState {
        metrics: Arc::new(MetricsCollector::new()),
        request_log: Arc::new(RequestLog::new()),
        cluster: ClusterProvider::Live(cluster_rx),
        scheduler: None,
    });

    // Pipeline watch channel — starts with an empty placeholder, replaced when workers connect.
    let empty_pipeline = Arc::new(DistributedPipeline::empty(model_config.hidden_size));
    let (pipeline_tx, pipeline_rx) = tokio::sync::watch::channel(Arc::clone(&empty_pipeline));

    // Build HTTP router immediately.
    use tower_http::cors::CorsLayer;
    let router: Router = if config.batched {
        tracing::info!("using batched scheduler loop (Phase 4)");
        let handle = distributed_loop::start_distributed_loop(
            Arc::clone(&empty_pipeline),
            Arc::clone(&registry),
            distributed_loop::DistributedLoopConfig::default(),
            pipeline_rx,
        );
        let batched_state = Arc::new(BatchedAppState::new(handle, tokenizer, Arc::clone(&dashboard_state)));
        create_batched_router(batched_state)
    } else {
        tracing::info!("using sequential distributed_generate");
        let state = Arc::new(CoordState {
            pipeline: Arc::clone(&empty_pipeline),
            registry: Arc::clone(&registry),
            seq_mgr: Arc::new(Mutex::new(SequenceStateManager::new())),
            tokenizer,
            max_seq_len: config.max_seq_len,
        });
        Router::new()
            .route("/v1/completions", post(completions_handler))
            .route("/v1/models", get(models_handler))
            .route("/health", get(health_handler))
            .with_state(state)
            .merge(dashboard_routes(Arc::clone(&dashboard_state)))
            .layer(CorsLayer::permissive())
    };

    // Start HTTP server immediately.
    let http_addr = format!("0.0.0.0:{}", config.http_port);
    let http_listener = tokio::net::TcpListener::bind(&http_addr).await?;
    tracing::info!("HTTP server listening on {}", http_listener.local_addr()?);

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

    // Spawn worker acceptance + pipeline setup in the background.
    let listener = TcpListener::bind(&config.listen_address).await?;
    tracing::info!("listening for workers on {}", config.listen_address);
    {
        let registry = Arc::clone(&registry);
        let model_config = model_config.clone();
        let pipeline_tx = pipeline_tx.clone();
        let cluster_tx_bg = cluster_tx.clone();
        let expected_workers = config.expected_workers;
        let max_seq_len = config.max_seq_len;
        let acceptance_timeout_secs = config.acceptance_timeout_secs;

        tokio::spawn(async move {
            if let Err(e) = accept_and_setup_pipeline(
                &listener,
                &registry,
                &model_config,
                scheduling_mode.clone(),
                expected_workers,
                max_seq_len,
                acceptance_timeout_secs,
                &pipeline_tx,
            ).await {
                tracing::error!("worker acceptance failed: {e}");
                return;
            }

            // Refresh cluster snapshot now that workers are ready.
            let snap = build_cluster_snapshot(&registry, &model_config, max_seq_len).await;
            let _ = cluster_tx_bg.send(snap);

            // Periodic cluster snapshot refresh.
            let mut interval = tokio::time::interval(Duration::from_secs(2));
            loop {
                interval.tick().await;
                let snap = build_cluster_snapshot(&registry, &model_config, max_seq_len).await;
                if cluster_tx_bg.send(snap).is_err() {
                    break;
                }
            }
        });
    }

    // Serve HTTP with graceful shutdown on ctrl-c.
    axum::serve(http_listener, router)
        .with_graceful_shutdown(async {
            tokio::signal::ctrl_c().await.ok();
            tracing::info!("shutting down...");
        })
        .await?;

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

/// Accept workers, run scheduler, set up pipeline, and broadcast via watch channel.
async fn accept_and_setup_pipeline(
    listener: &TcpListener,
    registry: &Mutex<PeerRegistry>,
    model_config: &fracture_core::ModelConfig,
    scheduling_mode: SchedulingMode,
    expected_workers: usize,
    max_seq_len: usize,
    acceptance_timeout_secs: u64,
    pipeline_tx: &tokio::sync::watch::Sender<Arc<DistributedPipeline>>,
) -> Result<()> {
    let timeout_duration = if acceptance_timeout_secs > 0 {
        Some(Duration::from_secs(acceptance_timeout_secs))
    } else {
        None
    };
    tracing::info!(
        "waiting for {} workers to register (timeout: {})...",
        expected_workers,
        timeout_duration.map_or("none".to_string(), |d| format!("{}s", d.as_secs()))
    );

    let accept_start = std::time::Instant::now();

    loop {
        {
            let reg = registry.lock().await;
            if reg.active_count() >= expected_workers {
                break;
            }
        }

        if let Some(timeout) = timeout_duration {
            if accept_start.elapsed() >= timeout {
                let reg = registry.lock().await;
                anyhow::bail!(
                    "timed out waiting for workers: got {}/{} after {}s",
                    reg.active_count(),
                    expected_workers,
                    timeout.as_secs()
                );
            }
        }

        let accept_future = listener.accept();
        let (stream, addr) = if let Some(timeout) = timeout_duration {
            let remaining = timeout.saturating_sub(accept_start.elapsed());
            match tokio::time::timeout(remaining, accept_future).await {
                Ok(result) => result?,
                Err(_) => {
                    let reg = registry.lock().await;
                    anyhow::bail!(
                        "timed out waiting for workers: got {}/{} after {}s",
                        reg.active_count(),
                        expected_workers,
                        acceptance_timeout_secs
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

        let reg_msg: RegisterPayload = FramedConnection::deserialize_payload(&payload)?;
        tracing::info!(
            "worker '{}' registered: {} ({:.1} GB available, decode={:.2} ms/layer)",
            reg_msg.node_id,
            reg_msg.gpu_model,
            reg_msg.gpu_memory_available as f64 / 1e9,
            reg_msg.decode_ms_per_layer
        );

        let caps = WorkerCapabilities {
            node_id: reg_msg.node_id.clone(),
            gpu_model: reg_msg.gpu_model,
            gpu_memory_available: reg_msg.gpu_memory_available as usize,
            compute_capability: reg_msg.compute_capability,
            decode_ms_per_layer: reg_msg.decode_ms_per_layer,
            prefill_ms_per_layer_128: reg_msg.prefill_ms_per_layer_128,
        };

        let mut reg = registry.lock().await;
        reg.register(caps, conn)?;
    }

    tracing::info!("all {} workers registered — running scheduler", expected_workers);

    // Run scheduler.
    let mut reg = registry.lock().await;
    let scheduler_input = SchedulerInput {
        model_config: model_config.clone(),
        workers: reg.all_capabilities(),
        coordinator_compute: None,
        mode: scheduling_mode,
        max_seq_len,
        hop_latency_ms: 2.0,
    };

    let schedule_result = scheduler::schedule(&scheduler_input)?;

    tracing::info!("scheduler result:");
    for a in &schedule_result.assignments {
        tracing::info!(
            "  {} → layers {:?} ({:?}), {:.1} ms/decode, {:.1} GB weights, {:.1} GB cache",
            a.node_id, a.layer_range, a.role, a.expected_decode_ms, a.weight_memory_gb, a.cache_memory_gb
        );
    }

    // Send RegisterAck to each worker.
    for assignment in &schedule_result.assignments {
        let ack = RegisterAckPayload {
            layer_start: assignment.layer_range.start as u32,
            layer_end: assignment.layer_range.end as u32,
            total_layers: model_config.num_layers as u32,
            max_seq_len: max_seq_len as u32,
            model_config: model_config.clone(),
        };
        reg.assign(&assignment.node_id, assignment.clone())?;
        let entry = reg.get_mut(&assignment.node_id).ok_or_else(|| {
            anyhow::anyhow!("worker '{}' disappeared", assignment.node_id)
        })?;
        entry.connection.send(MessageType::RegisterAck, 0, &ack).await?;
        tracing::info!("sent RegisterAck to '{}'", assignment.node_id);
    }

    // Wait for WorkerReady from each.
    tracing::info!("waiting for workers to finish weight loading...");
    for assignment in &schedule_result.assignments {
        let entry = reg.get_mut(&assignment.node_id).ok_or_else(|| {
            anyhow::anyhow!("worker '{}' disappeared", assignment.node_id)
        })?;
        let (header, _) = entry.connection.recv().await?;
        if header.msg_type != MessageType::WorkerReady {
            anyhow::bail!(
                "expected WorkerReady from '{}', got {:?}",
                assignment.node_id, header.msg_type
            );
        }
        tracing::info!("worker '{}' ready", assignment.node_id);
    }

    // Refresh last_heartbeat so workers aren't immediately marked dead
    // by the distributed loop's heartbeat checker (which started earlier).
    let now = std::time::Instant::now();
    for assignment in &schedule_result.assignments {
        if let Some(entry) = reg.get_mut(&assignment.node_id) {
            entry.last_heartbeat = now;
        }
    }
    drop(reg);

    // Build pipeline and broadcast.
    let pipeline = Arc::new(DistributedPipeline::new(
        &schedule_result.assignments,
        model_config.hidden_size,
    )?);
    tracing::info!("distributed pipeline ready with {} stages", pipeline.pipeline_order().len());
    let _ = pipeline_tx.send(pipeline);

    Ok(())
}

async fn build_cluster_snapshot(
    registry: &Mutex<PeerRegistry>,
    model_config: &fracture_core::ModelConfig,
    max_seq_len: usize,
) -> fracture_server::dashboard::dto::ClusterResponse {
    use fracture_server::dashboard::dto::{
        ClusterResponse as CR, ModelInfo as MI, WorkerInfo as WI,
    };
    let reg = registry.lock().await;
    let order = reg.pipeline_order();
    let workers: Vec<WI> = order
        .iter()
        .enumerate()
        .filter_map(|(i, node_id)| {
            let entry = reg.get(node_id)?;
            let (layer_start, layer_end) = entry
                .assignment
                .as_ref()
                .map(|a| (a.layer_range.start, a.layer_range.end.saturating_sub(1)))
                .unwrap_or((0, 0));
            let role = if i == 0 {
                "head"
            } else if i == order.len() - 1 {
                "tail"
            } else {
                "middle"
            };
            let status = match entry.status {
                fracture_coordinator::registry::WorkerStatus::Ready => "active",
                fracture_coordinator::registry::WorkerStatus::Dead => "dead",
                _ => "calibrating",
            };
            let vram_total_mb =
                (entry.capabilities.gpu_memory_available / (1024 * 1024)) as u64;
            let vram_used_mb = entry.gpu_memory_used / (1024 * 1024);
            let heartbeat_age_ms = entry.last_heartbeat.elapsed().as_millis() as u64;
            Some(WI {
                id: i,
                role,
                address: entry.capabilities.node_id.clone(),
                gpu: entry.capabilities.gpu_model.clone(),
                vram_total_mb,
                vram_used_mb,
                layers: [layer_start, layer_end],
                status,
                last_heartbeat_ms: heartbeat_age_ms,
                calibration_ms_per_layer: entry.capabilities.decode_ms_per_layer as f64,
            })
        })
        .collect();
    CR {
        mode: "distributed",
        num_workers: workers.len(),
        workers,
        scheduling_mode: "auto",
        model: MI {
            name: "llama-3-8b".to_string(),
            parameters: "8B".to_string(),
            layers: model_config.num_layers,
            context_length: max_seq_len,
            dtype: "FP16".to_string(),
        },
    }
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

// ── Fault Tolerance: Reconfiguration + Reconnection ────────────────────

/// Reconfigure the pipeline with whatever workers are currently alive.
///
/// Re-runs the scheduler, sends Reconfigure to surviving workers, waits
/// for WorkerReady, and broadcasts the new pipeline via the watch channel.
async fn reconfigure_pipeline(
    registry: &Mutex<PeerRegistry>,
    model_config: &fracture_core::ModelConfig,
    scheduling_mode: SchedulingMode,
    max_seq_len: usize,
    pipeline_tx: &tokio::sync::watch::Sender<Arc<DistributedPipeline>>,
) -> Result<()> {
    let mut reg = registry.lock().await;

    let caps = reg.all_capabilities();
    if caps.is_empty() {
        anyhow::bail!("no surviving workers — cannot reconfigure");
    }

    tracing::info!(
        "reconfiguring pipeline with {} surviving worker(s)",
        caps.len()
    );

    let input = SchedulerInput {
        model_config: model_config.clone(),
        workers: caps,
        coordinator_compute: None,
        mode: scheduling_mode,
        max_seq_len,
        hop_latency_ms: 2.0,
    };

    let result = scheduler::schedule(&input)?;

    tracing::info!("new schedule:");
    for a in &result.assignments {
        tracing::info!(
            "  {} → layers {:?} ({:?})",
            a.node_id, a.layer_range, a.role
        );
    }

    // Send Reconfigure (same payload as RegisterAck) to each surviving worker.
    for assignment in &result.assignments {
        let payload = RegisterAckPayload {
            layer_start: assignment.layer_range.start as u32,
            layer_end: assignment.layer_range.end as u32,
            total_layers: model_config.num_layers as u32,
            max_seq_len: max_seq_len as u32,
            model_config: model_config.clone(),
        };
        reg.assign(&assignment.node_id, assignment.clone())?;
        let entry = reg.get_mut(&assignment.node_id).ok_or_else(|| {
            anyhow::anyhow!("worker '{}' disappeared during reconfigure", assignment.node_id)
        })?;
        entry
            .connection
            .send(MessageType::Reconfigure, 0, &payload)
            .await?;
        tracing::info!("sent Reconfigure to '{}'", assignment.node_id);
    }

    // Wait for WorkerReady from each reconfigured worker.
    for assignment in &result.assignments {
        let entry = reg.get_mut(&assignment.node_id).ok_or_else(|| {
            anyhow::anyhow!("worker '{}' disappeared", assignment.node_id)
        })?;
        let (header, _) = entry.connection.recv().await?;
        if header.msg_type != MessageType::WorkerReady {
            anyhow::bail!(
                "expected WorkerReady from '{}' after reconfigure, got {:?}",
                assignment.node_id, header.msg_type
            );
        }
        tracing::info!("worker '{}' ready after reconfigure", assignment.node_id);
    }

    // Build new pipeline and broadcast it.
    let new_pipeline = DistributedPipeline::new(
        &result.assignments,
        model_config.hidden_size,
    )?;
    let new_pipeline = Arc::new(new_pipeline);
    let _ = pipeline_tx.send(new_pipeline);
    tracing::info!("pipeline reconfigured successfully");

    Ok(())
}

/// Background task that accepts new worker connections for reconnection.
///
/// Workers that previously died can reconnect by opening a new TCP connection
/// and sending Register. This task processes the registration and triggers
/// pipeline reconfiguration.
async fn reconnection_listener(
    listener: TcpListener,
    registry: Arc<Mutex<PeerRegistry>>,
    model_config: fracture_core::ModelConfig,
    scheduling_mode: SchedulingMode,
    max_seq_len: usize,
    pipeline_tx: tokio::sync::watch::Sender<Arc<DistributedPipeline>>,
) {
    loop {
        let (stream, addr) = match listener.accept().await {
            Ok(pair) => pair,
            Err(e) => {
                tracing::error!("reconnection listener accept error: {e}");
                continue;
            }
        };

        tracing::info!("worker reconnecting from {addr}");
        let mut conn = FramedConnection::new(stream);

        let (header, payload) = match conn.recv().await {
            Ok(frame) => frame,
            Err(e) => {
                tracing::warn!("failed to read Register from {addr}: {e}");
                continue;
            }
        };

        if header.msg_type != MessageType::Register {
            tracing::warn!(
                "expected Register from reconnecting {addr}, got {:?}",
                header.msg_type
            );
            continue;
        }

        let reg_payload: RegisterPayload = match FramedConnection::deserialize_payload(&payload) {
            Ok(p) => p,
            Err(e) => {
                tracing::warn!("failed to deserialize Register from {addr}: {e}");
                continue;
            }
        };

        tracing::info!(
            "reconnecting worker '{}': {} ({:.1} GB)",
            reg_payload.node_id,
            reg_payload.gpu_model,
            reg_payload.gpu_memory_available as f64 / 1e9,
        );

        let caps = WorkerCapabilities {
            node_id: reg_payload.node_id.clone(),
            gpu_model: reg_payload.gpu_model,
            gpu_memory_available: reg_payload.gpu_memory_available as usize,
            compute_capability: reg_payload.compute_capability,
            decode_ms_per_layer: reg_payload.decode_ms_per_layer,
            prefill_ms_per_layer_128: reg_payload.prefill_ms_per_layer_128,
        };

        {
            let mut reg = registry.lock().await;
            if let Err(e) = reg.register(caps, conn) {
                tracing::error!("failed to re-register '{}': {e}", reg_payload.node_id);
                continue;
            }
        }

        // Trigger reconfiguration with the new worker included.
        if let Err(e) = reconfigure_pipeline(
            &registry,
            &model_config,
            scheduling_mode.clone(),
            max_seq_len,
            &pipeline_tx,
        )
        .await
        {
            tracing::error!("reconfiguration after reconnect failed: {e}");
        }
    }
}
