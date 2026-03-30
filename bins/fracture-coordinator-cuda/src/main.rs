//! Fracture coordinator binary (CUDA backend).
//!
//! Listens for worker connections, runs the scheduler to assign layers,
//! orchestrates the distributed pipeline, and serves an OpenAI-compatible
//! HTTP API with async generation through the distributed pipeline.

use anyhow::Result;
use axum::routing::{get, post};
use axum::Router;
use fracture_coordinator::{
    pipeline::DistributedPipeline,
    registry::PeerRegistry,
    scheduler::SchedulingMode,
    state::SequenceStateManager,
};
use fracture_server::dashboard::metrics::MetricsCollector;
use fracture_server::dashboard::request_log::RequestLog;
use fracture_server::dashboard::routes::{dashboard_routes, ClusterProvider, DashboardState};
use fracture_server::{create_batched_router, BatchedAppState};
use fracture_server::utils::{health_handler, models_handler};
use std::sync::Arc;
use std::time::Duration;
use tokio::net::TcpListener;
use tokio::sync::Mutex;
use tracing_subscriber::EnvFilter;

mod admin;
mod distributed_loop;
mod handlers;
mod pipeline_setup;

use handlers::{completions_handler, CoordState};

/// Coordinator config parsed from CLI args.
struct CoordinatorConfig {
    listen_address: String,
    model_path: String,
    /// Minimum workers before building initial pipeline. Default: 1.
    /// Additional workers join dynamically via FT-7.
    min_workers: usize,
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
    /// Starting term for this coordinator instance. Workers with a higher
    /// term will reject connections from this coordinator (FT-13).
    term: u64,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .init();

    // Load config file (fracture.env) → CLI flags as fallback.
    let args: Vec<String> = std::env::args().collect();
    let (cfg, cfg_path) = fracture_core::env_config::load_config(&args);
    if let Some(path) = cfg_path {
        tracing::info!("loaded config from {path}");
    }

    let config = CoordinatorConfig {
        listen_address: cfg
            .get_or_flag("FRACTURE_LISTEN", &args, "--listen")
            .unwrap_or("0.0.0.0:9400")
            .to_string(),
        model_path: cfg
            .get_or_flag("FRACTURE_MODEL", &args, "--model")
            .expect("--model or FRACTURE_MODEL is required")
            .to_string(),
        min_workers: cfg
            .get_or_flag("FRACTURE_MIN_WORKERS", &args, "--min-workers")
            .and_then(|p| p.parse().ok())
            .unwrap_or(1),
        http_port: cfg
            .get_or_flag("FRACTURE_HTTP_PORT", &args, "--http-port")
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
        term: args
            .iter()
            .position(|a| a == "--term")
            .and_then(|i| args.get(i + 1))
            .and_then(|p| p.parse().ok())
            .unwrap_or(0),
    };

    tracing::info!("Fracture coordinator (term={})", config.term);
    tracing::info!("listening for workers on {}", config.listen_address);
    tracing::info!("min workers before pipeline: {}", config.min_workers);
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
    let tokenizer = pipeline_setup::load_tokenizer(
        &config.model_path,
        config.tokenizer_path.as_deref(),
    )?;

    // Dashboard cluster snapshot via watch channel (starts empty, filled by worker acceptance).
    let empty_cluster = pipeline_setup::build_cluster_snapshot(&registry, &model_config, config.max_seq_len).await;
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
        let loop_config = distributed_loop::DistributedLoopConfig {
            model_config: Some(model_config.clone()),
            scheduling_mode: match config.scheduling_mode.as_str() {
                "equal" => fracture_coordinator::scheduler::SchedulingMode::EqualSplit,
                _ => fracture_coordinator::scheduler::SchedulingMode::Auto,
            },
            max_seq_len: config.max_seq_len,
            ..distributed_loop::DistributedLoopConfig::default()
        };
        let handle = distributed_loop::start_distributed_loop(
            Arc::clone(&empty_pipeline),
            Arc::clone(&registry),
            loop_config,
            pipeline_rx,
        );
        let batched_state = Arc::new(BatchedAppState::new(handle, tokenizer, Arc::clone(&dashboard_state)));
        // Admin API for cluster management (rebalance, drain, cluster info).
        let (admin_rebalance_tx, _admin_rebalance_rx) = tokio::sync::mpsc::unbounded_channel();
        let admin_state = Arc::new(admin::AdminState {
            registry: Arc::clone(&registry),
            rebalance_tx: admin_rebalance_tx,
        });
        create_batched_router(batched_state)
            .merge(admin::admin_routes(admin_state))
    } else {
        tracing::info!("using sequential distributed_generate");
        let state = Arc::new(CoordState {
            pipeline_rx,
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
        let expected_workers = config.min_workers;
        let max_seq_len = config.max_seq_len;
        let acceptance_timeout_secs = config.acceptance_timeout_secs;

        let scheduling_mode_for_recon = scheduling_mode.clone();
        tokio::spawn(async move {
            if let Err(e) = pipeline_setup::accept_and_setup_pipeline(
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
            let snap = pipeline_setup::build_cluster_snapshot(&registry, &model_config, max_seq_len).await;
            let _ = cluster_tx_bg.send(snap);

            // Spawn reconnection listener — reuses the worker TCP listener so
            // that workers that die and restart can reconnect.
            tokio::spawn(pipeline_setup::reconnection_listener(
                listener,
                Arc::clone(&registry),
                model_config.clone(),
                scheduling_mode_for_recon,
                max_seq_len,
                pipeline_tx,
                config.listen_address.clone(),
            ));

            // Periodic cluster snapshot refresh.
            let mut interval = tokio::time::interval(Duration::from_secs(2));
            loop {
                interval.tick().await;
                let snap = pipeline_setup::build_cluster_snapshot(&registry, &model_config, max_seq_len).await;
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
