//! End-to-end tests for the distributed inference pipeline.
//!
//! These tests spawn the actual coordinator and worker binaries as child
//! processes, exercise them over HTTP, and verify correctness.
//!
//! **Excluded from normal test runs** — all tests are `#[ignore]`.
//! Run with:
//!   cargo nextest run -p fracture-coordinator-cuda --run-ignored all
//!
//! Prerequisites:
//!   - GPU available (CUDA)
//!   - Release binaries built: cargo build --release -p fracture-coordinator-cuda -p fracture-worker-cuda
//!   - GGUF model at models/llama-3.1-8b-instruct-f16.gguf
//!   - tokenizer.json in the models/ directory

use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

/// Resolve a path relative to the workspace root (two levels up from this crate).
fn workspace_root() -> PathBuf {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    manifest.parent().unwrap().parent().unwrap().to_path_buf()
}

fn coord_bin() -> PathBuf {
    workspace_root().join("target/release/fracture-coordinator-cuda")
}
fn worker_bin() -> PathBuf {
    workspace_root().join("target/release/fracture-worker-cuda")
}
fn model_path() -> PathBuf {
    workspace_root().join("models/llama-3.1-8b-instruct-f16.gguf")
}

/// Guard that kills child processes on drop.
struct ProcessGuard {
    coordinator: Child,
    worker: Child,
}

impl Drop for ProcessGuard {
    fn drop(&mut self) {
        let _ = self.coordinator.kill();
        let _ = self.worker.kill();
        let _ = self.coordinator.wait();
        let _ = self.worker.wait();
    }
}

fn spawn_pipeline(coord_port: u16, http_port: u16) -> ProcessGuard {
    let (guard, _) = spawn_pipeline_timed(coord_port, http_port);
    guard
}

fn spawn_pipeline_timed(coord_port: u16, http_port: u16) -> (ProcessGuard, Duration) {
    let setup_start = Instant::now();
    let bin_c = coord_bin();
    let bin_w = worker_bin();
    let model = model_path();

    let coordinator = Command::new(&bin_c)
        .args([
            "--model", model.to_str().unwrap(),
            "--listen", &format!("127.0.0.1:{coord_port}"),
            "--workers", "1",
            "--http-port", &http_port.to_string(),
            "--scheduling", "equal",
        ])
        .env("RUST_LOG", "info")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap_or_else(|e| panic!("failed to spawn {}: {e}", bin_c.display()));

    std::thread::sleep(Duration::from_secs(1));

    let worker = Command::new(&bin_w)
        .args([
            "--model", model.to_str().unwrap(),
            "--coordinator", &format!("127.0.0.1:{coord_port}"),
            "--node-id", "e2e-worker",
        ])
        .env("RUST_LOG", "info")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap_or_else(|e| panic!("failed to spawn {}: {e}", bin_w.display()));

    let guard = ProcessGuard { coordinator, worker };

    // Wait for HTTP readiness
    let client = reqwest::blocking::Client::new();
    loop {
        if setup_start.elapsed() > Duration::from_secs(60) {
            panic!("HTTP server not ready within 60 seconds");
        }
        if let Ok(resp) = client
            .get(format!("http://127.0.0.1:{http_port}/health"))
            .send()
        {
            if resp.status().is_success() {
                break;
            }
        }
        std::thread::sleep(Duration::from_millis(500));
    }

    let setup_duration = setup_start.elapsed();
    (guard, setup_duration)
}

fn send_completion(http_port: u16, prompt: &str, max_tokens: usize, temperature: f32) -> serde_json::Value {
    let client = reqwest::blocking::Client::new();
    let resp = client
        .post(format!("http://127.0.0.1:{http_port}/v1/completions"))
        .json(&serde_json::json!({
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }))
        .send()
        .expect("HTTP request failed");
    assert!(resp.status().is_success(), "HTTP {}", resp.status());
    resp.json().expect("invalid JSON response")
}

/// Tests the full inference pipeline: health, models, greedy completion,
/// determinism, long generation, and error handling.
#[test]
#[ignore = "e2e: requires GPU, release binaries, and GGUF model"]
fn test_e2e_inference_pipeline() {
    let _guard = spawn_pipeline(9410, 8091);

    // Health
    let client = reqwest::blocking::Client::new();
    let health: serde_json::Value = client
        .get("http://127.0.0.1:8091/health")
        .send().unwrap().json().unwrap();
    assert_eq!(health["status"], "ready");

    // Models
    let models: serde_json::Value = client
        .get("http://127.0.0.1:8091/v1/models")
        .send().unwrap().json().unwrap();
    assert_eq!(models["object"], "list");
    assert!(!models["data"].as_array().unwrap().is_empty());

    // Greedy completion
    let resp = send_completion(8091, "The capital of France is", 20, 0.0);
    let text = resp["choices"][0]["text"].as_str().unwrap();
    assert!(text.contains("Paris"), "should mention Paris: {text}");
    let prompt_tokens = resp["usage"]["prompt_tokens"].as_u64().unwrap();
    let completion_tokens = resp["usage"]["completion_tokens"].as_u64().unwrap();
    assert!(prompt_tokens > 0);
    assert_eq!(completion_tokens, 20);
    assert_eq!(
        resp["usage"]["total_tokens"].as_u64().unwrap(),
        prompt_tokens + completion_tokens
    );

    // Greedy determinism: two identical requests produce identical output
    let resp1 = send_completion(8091, "Once upon a time", 30, 0.0);
    let resp2 = send_completion(8091, "Once upon a time", 30, 0.0);
    let text1 = resp1["choices"][0]["text"].as_str().unwrap();
    let text2 = resp2["choices"][0]["text"].as_str().unwrap();
    assert_eq!(text1, text2, "greedy should be deterministic:\n  1: {text1}\n  2: {text2}");

    // Long generation
    let resp = send_completion(8091, "Write a detailed essay about", 200, 0.0);
    let ct = resp["usage"]["completion_tokens"].as_u64().unwrap();
    assert!(ct >= 100, "long generation should produce many tokens: {ct}");

    // Empty prompt rejected
    let err_resp = client
        .post("http://127.0.0.1:8091/v1/completions")
        .json(&serde_json::json!({"prompt": "", "max_tokens": 10}))
        .send().unwrap();
    assert_eq!(err_resp.status(), 400);
}

/// Tests the full worker lifecycle: calibration, registration, scheduling,
/// weight loading, and serving — by verifying the pipeline becomes
/// operational (HTTP health returns ready) and can serve a request.
/// If any of calibration, registration, or scheduling failed, the
/// pipeline would never reach the ready state.
#[test]
#[ignore = "e2e: requires GPU, release binaries, and GGUF model"]
fn test_e2e_worker_lifecycle() {
    let _guard = spawn_pipeline(9411, 8092);

    // If we get here, the full lifecycle succeeded:
    // worker calibration → Register → coordinator scheduling → RegisterAck →
    // worker weight loading → serve loop ready → HTTP server up

    // Verify inference works (proves the whole chain is functional)
    let resp = send_completion(8092, "Hello", 5, 0.0);
    let tokens = resp["usage"]["completion_tokens"].as_u64().unwrap();
    assert!(tokens > 0, "should generate tokens after full lifecycle setup");
}

/// Benchmark: measures pipeline setup latency (calibration + registration +
/// scheduling + weight loading). Asserts < 30 seconds per the Phase 3 arch doc.
///
/// This is a benchmark, not a correctness test. Run separately:
///   cargo nextest run -p fracture-coordinator-cuda --run-ignored all -E 'test(bench)'
#[test]
#[ignore = "benchmark: requires GPU, release binaries, and GGUF model"]
fn bench_pipeline_setup_latency() {
    let (_guard, setup_duration) = spawn_pipeline_timed(9412, 8093);

    eprintln!(
        "pipeline setup latency: {:.1}s (threshold: 30s)",
        setup_duration.as_secs_f64()
    );

    assert!(
        setup_duration < Duration::from_secs(30),
        "pipeline setup took {:.1}s, exceeds 30s threshold",
        setup_duration.as_secs_f64()
    );
}
