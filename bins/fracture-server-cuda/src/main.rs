use anyhow::Result;
use fracture_core::Backend;
use fracture_cuda::CudaBackend;
use fracture_engine::{Engine, PagedKvCacheManager};
use fracture_gguf::WeightStore;
use fracture_server::dashboard::dto::ModelInfo;
use fracture_server::dashboard::metrics::MetricsCollector;
use fracture_server::dashboard::request_log::RequestLog;
use fracture_server::{create_router, AppState, ClusterProvider, DashboardState};
use std::sync::{Arc, Mutex};
use tokenizers::Tokenizer;
use tracing_subscriber::EnvFilter;

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .init();

    // Parse CLI args
    let args: Vec<String> = std::env::args().collect();
    let model_path = args
        .iter()
        .position(|a| a == "--model")
        .and_then(|i| args.get(i + 1))
        .expect("usage: fracture-server-cuda --model <path-to-gguf> [--port <port>] [--tokenizer <path>]");

    let port: u16 = args
        .iter()
        .position(|a| a == "--port")
        .and_then(|i| args.get(i + 1))
        .and_then(|p| p.parse().ok())
        .unwrap_or(8080);

    let tokenizer_path = args
        .iter()
        .position(|a| a == "--tokenizer")
        .and_then(|i| args.get(i + 1).cloned());

    let max_seq_len_override: Option<usize> = args
        .iter()
        .position(|a| a == "--max-seq-len")
        .and_then(|i| args.get(i + 1))
        .and_then(|p| p.parse().ok());

    tracing::info!("Fracture inference server (CUDA backend)");
    tracing::info!("model: {model_path}");

    // Initialize CUDA backend
    let mut backend = CudaBackend::new(0)?;
    tracing::info!(
        "GPU: {} ({:.1} GB total, {:.1} GB available)",
        backend.device_name(),
        backend.total_memory() as f64 / (1024.0 * 1024.0 * 1024.0),
        backend.available_memory() as f64 / (1024.0 * 1024.0 * 1024.0),
    );

    // Load model weights
    tracing::info!("loading model weights...");
    let weights = WeightStore::load(std::path::Path::new(model_path), &backend, None)?;
    let mut config = weights.config.clone();

    // Clamp max_seq_len to avoid OOM on contiguous KV cache allocation.
    // The GGUF may report 128K+ but the contiguous cache pre-allocates the full length.
    if let Some(override_len) = max_seq_len_override {
        config.max_seq_len = override_len;
    } else if config.max_seq_len > 4096 {
        tracing::warn!(
            "clamping max_seq_len from {} to 4096 (use --max-seq-len to override)",
            config.max_seq_len
        );
        config.max_seq_len = 4096;
    }
    tracing::info!(
        "model loaded: {}L, d={}, vocab={}",
        config.num_layers,
        config.hidden_size,
        config.vocab_size
    );

    // Pre-compute RoPE frequencies
    backend.precompute_rope_freqs(config.head_dim, config.rope_theta)?;

    // Create engine with full layer range
    let layer_range = 0..config.num_layers;
    let engine = Arc::new(Engine::new(backend, weights, layer_range));

    // Create paged KV cache manager (single-sequence legacy server; one block-pool sized
    // for max_seq_len + safety margin). 16 is the hardcoded BLOCK_SIZE.
    let num_blocks = config.max_seq_len.div_ceil(16) + 2;
    let cache = PagedKvCacheManager::new(
        num_blocks,
        config.num_layers,
        config.num_kv_heads,
        config.head_dim,
        engine.backend(),
    )?;

    // Load tokenizer
    let tokenizer = if let Some(path) = tokenizer_path {
        Tokenizer::from_file(&path).map_err(|e| anyhow::anyhow!("failed to load tokenizer: {e}"))?
    } else {
        // Try loading from same directory as model
        let model_dir = std::path::Path::new(model_path).parent().unwrap_or(std::path::Path::new("."));
        let tokenizer_file = model_dir.join("tokenizer.json");
        if tokenizer_file.exists() {
            Tokenizer::from_file(&tokenizer_file)
                .map_err(|e| anyhow::anyhow!("failed to load tokenizer: {e}"))?
        } else {
            anyhow::bail!(
                "no tokenizer found. Provide --tokenizer <path> or place tokenizer.json next to the model file"
            );
        }
    };

    // Build dashboard state
    let dashboard_state = Arc::new(DashboardState {
        metrics: Arc::new(MetricsCollector::new()),
        request_log: Arc::new(RequestLog::new()),
        cluster: ClusterProvider::Standalone {
            gpu_name: engine.backend().device_name().to_string(),
            vram_total_mb: (engine.backend().total_memory() / (1024 * 1024)) as u64,
            vram_used_mb: ((engine.backend().total_memory() - engine.backend().available_memory())
                / (1024 * 1024)) as u64,
            model: ModelInfo {
                name: "llama-3-8b".to_string(),
                parameters: "8B".to_string(),
                layers: config.num_layers,
                context_length: config.max_seq_len,
                dtype: "FP16".to_string(),
            },
            total_layers: config.num_layers,
        },
        scheduler: None,
    });

    // Build app state and router
    let state = Arc::new(AppState {
        engine,
        cache: Mutex::new(cache),
        tokenizer,
        dashboard: dashboard_state,
    });

    let router = create_router(state);

    // Start server
    let addr = format!("0.0.0.0:{port}");
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    tracing::info!("listening on {}", listener.local_addr()?);
    axum::serve(listener, router).await?;

    Ok(())
}
