//! In-flight cancellation: dropping the SSE receiver mid-decode must abort the
//! sequence and return its KV blocks within 2 seconds.
//!
//! Two tests, one per route:
//!  - `dropping_sse_receiver_frees_paged_blocks_non_batched` covers
//!    `routes::create_router` (Mutex-serialized path with GenerationLoop +
//!    cooperative cancel via AtomicBool).
//!  - `dropping_sse_receiver_frees_paged_blocks_batched` covers
//!    `batched_routes::create_batched_router` (scheduler-driven path that
//!    reaps disconnected sequences via `event_tx.is_closed()`).
//!
//! Both routes back onto a `PagedKvCacheManager` so the same probe —
//! `num_free_blocks()` — works for both.

use fracture_core::{Backend, DType, ModelConfig};
use fracture_cuda::CudaBackend;
use fracture_engine::{Engine, PagedKvCacheManager};
use fracture_gguf::{LayerWeights, WeightStore};
use fracture_server::dashboard::dto::ModelInfo;
use fracture_server::dashboard::metrics::MetricsCollector;
use fracture_server::dashboard::request_log::RequestLog;
use fracture_server::{
    create_batched_router, create_router, start_scheduler_loop, AppState, BatchedAppState,
    ClusterProvider, DashboardState, SchedulerLoopConfig,
};
use half::f16;
use rand::Rng;
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokenizers::Tokenizer;
use tokio::sync::oneshot;
use tokio::time::sleep;

// ---------------------------------------------------------------------------
// Tiny model bootstrap (lifted from gpu_integration.rs).
// ---------------------------------------------------------------------------

fn test_config() -> ModelConfig {
    ModelConfig {
        hidden_size: 64,
        num_layers: 2,
        num_q_heads: 4,
        num_kv_heads: 2,
        head_dim: 16,
        intermediate_size: 128,
        vocab_size: 256,
        rope_theta: 10000.0,
        rms_norm_eps: 1e-5,
        max_seq_len: 512,
    }
}

fn random_fp16_bytes(numel: usize) -> Vec<u8> {
    let mut rng = rand::rng();
    let mut bytes = Vec::with_capacity(numel * 2);
    for _ in 0..numel {
        let val: f32 = rng.random_range(-0.1..0.1);
        let fp16 = f16::from_f32(val);
        bytes.extend_from_slice(&fp16.to_le_bytes());
    }
    bytes
}

fn alloc_random_tensor(
    backend: &CudaBackend,
    shape: &[usize],
) -> fracture_core::Result<fracture_core::DeviceTensor> {
    let tensor = backend.alloc(shape, DType::FP16)?;
    let numel: usize = shape.iter().product();
    let data = random_fp16_bytes(numel);
    backend.copy_to_device(&tensor, &data)?;
    Ok(tensor)
}

fn build_test_weights(backend: &CudaBackend, cfg: &ModelConfig) -> fracture_core::Result<WeightStore> {
    let token_embedding = alloc_random_tensor(backend, &[cfg.vocab_size, cfg.hidden_size])?;
    let mut layers = Vec::new();
    for _ in 0..cfg.num_layers {
        layers.push(LayerWeights {
            q_proj: alloc_random_tensor(backend, &[cfg.hidden_size, cfg.hidden_size])?,
            k_proj: alloc_random_tensor(backend, &[cfg.num_kv_heads * cfg.head_dim, cfg.hidden_size])?,
            v_proj: alloc_random_tensor(backend, &[cfg.num_kv_heads * cfg.head_dim, cfg.hidden_size])?,
            o_proj: alloc_random_tensor(backend, &[cfg.hidden_size, cfg.hidden_size])?,
            gate_proj: alloc_random_tensor(backend, &[cfg.intermediate_size, cfg.hidden_size])?,
            up_proj: alloc_random_tensor(backend, &[cfg.intermediate_size, cfg.hidden_size])?,
            down_proj: alloc_random_tensor(backend, &[cfg.hidden_size, cfg.intermediate_size])?,
            attn_norm: alloc_random_tensor(backend, &[cfg.hidden_size])?,
            ffn_norm: alloc_random_tensor(backend, &[cfg.hidden_size])?,
        });
    }
    let output_norm = alloc_random_tensor(backend, &[cfg.hidden_size])?;
    let lm_head = alloc_random_tensor(backend, &[cfg.vocab_size, cfg.hidden_size])?;
    Ok(WeightStore { config: cfg.clone(), token_embedding, layers, output_norm, lm_head })
}

/// Build a minimal byte-level tokenizer that produces non-empty token ID streams
/// for arbitrary ASCII prompts.
fn make_tiny_tokenizer() -> Tokenizer {
    use tokenizers::models::bpe::BPE;
    let model = BPE::default();
    let mut tok = Tokenizer::new(model);
    let tokens: Vec<tokenizers::AddedToken> = (0..256u32)
        .map(|i| tokenizers::AddedToken::from(format!("{}", i as u8 as char), false))
        .collect();
    tok.add_tokens(&tokens);
    tok
}

fn make_dashboard_state() -> Arc<DashboardState> {
    Arc::new(DashboardState {
        metrics: Arc::new(MetricsCollector::new()),
        request_log: Arc::new(RequestLog::new()),
        cluster: ClusterProvider::Standalone {
            gpu_name: "test".to_string(),
            vram_total_mb: 1024,
            vram_used_mb: 0,
            model: ModelInfo {
                name: "test".to_string(),
                parameters: "tiny".to_string(),
                layers: 2,
                context_length: 512,
                dtype: "FP16".to_string(),
            },
            total_layers: 2,
        },
        scheduler: None,
    })
}

// ---------------------------------------------------------------------------
// Probe: a callable closure that returns the cache's free block count.
// Both modes share `PagedKvCacheManager`, so the probe works in either case;
// only the storage of the Arc differs.
// ---------------------------------------------------------------------------

/// Two-mode probe: either a direct sync closure (non-batched, owns the Arc)
/// or a SchedulerHandle (batched, must round-trip through the scheduler).
enum FreeBlocksProbe {
    Sync(Arc<dyn Fn() -> usize + Send + Sync>),
    Scheduler(fracture_server::SchedulerHandle),
}

impl FreeBlocksProbe {
    async fn read(&self) -> usize {
        match self {
            FreeBlocksProbe::Sync(f) => {
                let f = Arc::clone(f);
                tokio::task::spawn_blocking(move || f()).await.unwrap()
            }
            FreeBlocksProbe::Scheduler(h) => {
                h.snapshot().await.expect("snapshot").free_blocks
            }
        }
    }
}

struct TestServer {
    client: reqwest::Client,
    base_url: String,
    free_blocks: FreeBlocksProbe,
    shutdown: Option<oneshot::Sender<()>>,
    join: Option<tokio::task::JoinHandle<()>>,
}

impl Drop for TestServer {
    fn drop(&mut self) {
        if let Some(tx) = self.shutdown.take() {
            let _ = tx.send(());
        }
        if let Some(j) = self.join.take() {
            j.abort();
        }
    }
}

async fn bind_random_port() -> (tokio::net::TcpListener, String) {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.expect("bind");
    let port = listener.local_addr().expect("local_addr").port();
    let base_url = format!("http://127.0.0.1:{port}");
    (listener, base_url)
}

fn make_engine_and_cache() -> (Arc<Engine<CudaBackend>>, PagedKvCacheManager, ModelConfig) {
    let mut backend = CudaBackend::new(0).expect("cuda backend");
    let cfg = test_config();
    backend.precompute_rope_freqs(cfg.head_dim, cfg.rope_theta).expect("rope");
    let weights = build_test_weights(&backend, &cfg).expect("weights");
    let num_blocks = cfg.max_seq_len.div_ceil(16) + 8;
    let cache = PagedKvCacheManager::new(
        num_blocks, cfg.num_layers, cfg.num_kv_heads, cfg.head_dim, &backend,
    ).expect("paged cache");
    let engine = Arc::new(Engine::new(backend, weights, 0..cfg.num_layers));
    (engine, cache, cfg)
}

async fn spawn_tiny_server_non_batched() -> TestServer {
    let (engine, cache, _cfg) = make_engine_and_cache();
    let cache = Mutex::new(cache);
    let tokenizer = make_tiny_tokenizer();
    let dashboard = make_dashboard_state();

    let state = Arc::new(AppState { engine, cache, tokenizer, dashboard });

    // Probe: lock the AppState's cache mutex and read num_free_blocks.
    let state_for_probe = Arc::clone(&state);
    let free_blocks = FreeBlocksProbe::Sync(Arc::new(move || {
        let guard = state_for_probe.cache.lock().unwrap();
        guard.num_free_blocks()
    }));

    let router = create_router(state);

    let (listener, base_url) = bind_random_port().await;
    let (shutdown_tx, shutdown_rx) = oneshot::channel();
    let join = tokio::spawn(async move {
        let _ = axum::serve(listener, router)
            .with_graceful_shutdown(async move {
                let _ = shutdown_rx.await;
            })
            .await;
    });

    let client = reqwest::Client::builder()
        .pool_max_idle_per_host(0)
        .build()
        .expect("reqwest client");

    TestServer {
        client,
        base_url,
        free_blocks,
        shutdown: Some(shutdown_tx),
        join: Some(join),
    }
}

async fn spawn_tiny_server_batched() -> TestServer {
    let (engine, cache, _cfg) = make_engine_and_cache();
    let tokenizer = make_tiny_tokenizer();
    let dashboard = make_dashboard_state();

    // The scheduler_loop owns the cache via Arc<StdMutex<PagedKvCacheManager>>;
    // it doesn't expose the Arc, so we keep our own clone for the probe by
    // wrapping the PagedKvCacheManager *before* handing it over. To make this
    // work, re-create the wrapper externally and have start_scheduler_loop
    // accept the un-wrapped manager (current API). Trade-off: probe will
    // request a snapshot via SchedulerHandle which carries free-block info.
    let scheduler_config = SchedulerLoopConfig {
        max_batch_size: 4,
        max_batch_tokens: 512,
        max_prefill_tokens: 256,
        block_pool_reserve: 0.0,
    };
    let scheduler = start_scheduler_loop(Arc::clone(&engine), cache, scheduler_config);
    let free_blocks = FreeBlocksProbe::Scheduler(scheduler.clone());

    let state = Arc::new(BatchedAppState::new(scheduler, tokenizer, dashboard));
    let router = create_batched_router(state);

    let (listener, base_url) = bind_random_port().await;
    let (shutdown_tx, shutdown_rx) = oneshot::channel();
    let join = tokio::spawn(async move {
        let _ = axum::serve(listener, router)
            .with_graceful_shutdown(async move {
                let _ = shutdown_rx.await;
            })
            .await;
    });

    let client = reqwest::Client::builder()
        .pool_max_idle_per_host(0)
        .build()
        .expect("reqwest client");

    TestServer {
        client,
        base_url,
        free_blocks,
        shutdown: Some(shutdown_tx),
        join: Some(join),
    }
}

async fn probe_free_blocks(server: &TestServer) -> usize {
    server.free_blocks.read().await
}

// ---------------------------------------------------------------------------
// Helper: open an SSE stream, wait for the first chunk, then drop it.
// Asserts the request was actually accepted (status 200) so a misconfigured
// test doesn't silently look like a successful cancellation.
// ---------------------------------------------------------------------------

async fn open_and_drop_sse(server: &TestServer) {
    let body = serde_json::json!({
        "model": "llama-3-8b",
        "prompt": "abcdefgh",
        "max_tokens": 200,
        "temperature": 0.0,
        "top_p": 1.0,
        "stream": true,
    });
    let response = server
        .client
        .post(format!("{}/v1/completions", server.base_url))
        .json(&body)
        .send()
        .await
        .expect("send");
    assert!(
        response.status().is_success(),
        "SSE request did not return 2xx (got {})",
        response.status()
    );
    let mut response = response;
    // Wait for at least one chunk to land — proves the server actually started
    // decoding (and therefore allocated cache blocks). Using a timeout so the
    // test fails loudly if streaming never yields.
    let first = tokio::time::timeout(Duration::from_secs(10), response.chunk()).await;
    match first {
        Ok(Ok(Some(_chunk))) => {}
        Ok(Ok(None)) => panic!("stream closed before any chunk arrived"),
        Ok(Err(e)) => panic!("error reading first chunk: {e}"),
        Err(_) => panic!("timed out waiting for first SSE chunk"),
    }
    drop(response);
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn dropping_sse_receiver_frees_paged_blocks_non_batched() {
    let server = spawn_tiny_server_non_batched().await;
    let initial_free = probe_free_blocks(&server).await;
    assert!(initial_free > 0, "test setup: expected free blocks at startup");

    open_and_drop_sse(&server).await;

    // Allow up to 2s for the server to observe disconnect, exit decode, and
    // free the cache handle.
    let deadline = std::time::Instant::now() + Duration::from_secs(2);
    let mut current = probe_free_blocks(&server).await;
    while current < initial_free && std::time::Instant::now() < deadline {
        sleep(Duration::from_millis(50)).await;
        current = probe_free_blocks(&server).await;
    }

    let final_free = probe_free_blocks(&server).await;
    assert_eq!(
        final_free, initial_free,
        "non-batched: cancelled sequence did not return its blocks; \
         leaked {} blocks (initial={initial_free}, final={final_free})",
        initial_free.saturating_sub(final_free)
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn dropping_sse_receiver_frees_paged_blocks_batched() {
    let server = spawn_tiny_server_batched().await;
    let initial_free = probe_free_blocks(&server).await;
    assert!(initial_free > 0, "test setup: expected free blocks at startup");

    open_and_drop_sse(&server).await;

    let deadline = std::time::Instant::now() + Duration::from_secs(2);
    let mut current = probe_free_blocks(&server).await;
    while current < initial_free && std::time::Instant::now() < deadline {
        sleep(Duration::from_millis(50)).await;
        current = probe_free_blocks(&server).await;
    }

    let final_free = probe_free_blocks(&server).await;
    assert_eq!(
        final_free, initial_free,
        "batched: cancelled sequence did not return its blocks; \
         leaked {} blocks (initial={initial_free}, final={final_free})",
        initial_free.saturating_sub(final_free)
    );
}
