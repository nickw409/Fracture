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
    spawn_pipeline_inner(coord_port, http_port, false)
}

fn spawn_batched_pipeline(coord_port: u16, http_port: u16) -> ProcessGuard {
    let (guard, _) = spawn_pipeline_inner(coord_port, http_port, true);
    guard
}

fn spawn_pipeline_inner(coord_port: u16, http_port: u16, batched: bool) -> (ProcessGuard, Duration) {
    let setup_start = Instant::now();
    let bin_c = coord_bin();
    let bin_w = worker_bin();
    let model = model_path();

    let mut args = vec![
        "--model".to_string(), model.to_str().unwrap().to_string(),
        "--listen".to_string(), format!("127.0.0.1:{coord_port}"),
        "--workers".to_string(), "1".to_string(),
        "--http-port".to_string(), http_port.to_string(),
        "--scheduling".to_string(), "equal".to_string(),
    ];
    if batched {
        args.push("--batched".to_string());
    }

    let coordinator = Command::new(&bin_c)
        .args(&args)
        .env("RUST_LOG", "info")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
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
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .unwrap_or_else(|e| panic!("failed to spawn {}: {e}", bin_w.display()));

    let guard = ProcessGuard { coordinator, worker };

    // Wait for pipeline readiness by polling a real completion request.
    // Health returns "ready" immediately (HTTP starts before workers connect),
    // but the pipeline is only usable after workers finish weight loading.
    loop {
        if setup_start.elapsed() > Duration::from_secs(60) {
            panic!("pipeline not ready within 60 seconds");
        }
        let client = reqwest::blocking::Client::builder()
            .timeout(Duration::from_secs(10))
            .build()
            .unwrap();
        let result = client
            .post(format!("http://127.0.0.1:{http_port}/v1/completions"))
            .json(&serde_json::json!({
                "prompt": "ready check",
                "max_tokens": 1,
                "temperature": 0.0,
            }))
            .send();
        if let Ok(resp) = result {
            if resp.status().is_success() {
                break;
            }
        }
        std::thread::sleep(Duration::from_secs(1));
    }

    let setup_duration = setup_start.elapsed();
    (guard, setup_duration)
}

fn send_completion(http_port: u16, prompt: &str, max_tokens: usize, temperature: f32) -> serde_json::Value {
    let client = reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(120))
        .build()
        .unwrap();
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

/// Tests that the coordinator handles acceptance timeout gracefully when
/// not enough workers connect within the timeout window.
///
/// Spawns only the coordinator (no workers) with `--workers 2
/// --acceptance-timeout 3`. The HTTP server starts immediately (serving
/// the dashboard) but requests fail because no pipeline is available.
#[test]
#[ignore = "e2e: requires GPU, release binaries, and GGUF model"]
fn test_e2e_acceptance_timeout() {
    let bin_c = coord_bin();
    let model = model_path();

    let mut coordinator = Command::new(&bin_c)
        .args([
            "--model", model.to_str().unwrap(),
            "--listen", "127.0.0.1:9413",
            "--workers", "2",
            "--http-port", "8094",
            "--acceptance-timeout", "3",
        ])
        .env("RUST_LOG", "info")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .unwrap_or_else(|e| panic!("failed to spawn {}: {e}", bin_c.display()));

    // Wait for the acceptance timeout to fire (3s + margin).
    std::thread::sleep(Duration::from_secs(5));

    // The HTTP server should be up (serving dashboard) but completions should fail
    // because no workers connected and no pipeline was built.
    let client = reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(5))
        .build()
        .unwrap();
    let resp = client
        .post("http://127.0.0.1:8094/v1/completions")
        .json(&serde_json::json!({
            "prompt": "test",
            "max_tokens": 1,
            "temperature": 0.0,
        }))
        .send();

    // The coordinator either: (a) isn't serving HTTP yet, (b) returns a non-success
    // status because the pipeline is empty, or (c) has exited. All are acceptable.
    match resp {
        Err(_) => {} // Connection refused or timeout — acceptable.
        Ok(r) => {
            assert!(
                !r.status().is_success() || {
                    let body: serde_json::Value = r.json().unwrap_or_default();
                    body.get("error").is_some()
                },
                "with 0 workers, completions should not succeed"
            );
        }
    }

    let _ = coordinator.kill();
    let _ = coordinator.wait();
}

/// Tests that the pipeline can serve multiple concurrent completion requests
/// without errors or data corruption. Three requests are sent in parallel
/// threads; all must complete successfully with non-empty generated text.
#[test]
#[ignore = "e2e: requires GPU, release binaries, and GGUF model"]
fn test_e2e_concurrent_sequences() {
    let _guard = spawn_pipeline(9414, 8095);

    let prompts = [
        ("The speed of light is", 20usize),
        ("Water is composed of", 20usize),
        ("The Great Wall of China was built", 20usize),
    ];

    let results: Vec<serde_json::Value> = std::thread::scope(|s| {
        let handles: Vec<_> = prompts
            .iter()
            .map(|(prompt, max_tokens)| {
                s.spawn(move || send_completion(8095, prompt, *max_tokens, 0.0))
            })
            .collect();
        handles.into_iter().map(|h| h.join().expect("thread panicked")).collect()
    });

    for (i, resp) in results.iter().enumerate() {
        let text = resp["choices"][0]["text"].as_str().unwrap_or_else(|| {
            panic!("request {i} missing choices[0].text: {resp}")
        });
        assert!(!text.is_empty(), "request {i} returned empty text");

        let completion_tokens = resp["usage"]["completion_tokens"].as_u64().unwrap_or_else(|| {
            panic!("request {i} missing usage.completion_tokens: {resp}")
        });
        assert!(completion_tokens > 0, "request {i} produced no tokens");

        let prompt_tokens = resp["usage"]["prompt_tokens"].as_u64().unwrap();
        let total_tokens = resp["usage"]["total_tokens"].as_u64().unwrap();
        assert_eq!(
            total_tokens,
            prompt_tokens + completion_tokens,
            "request {i} usage totals inconsistent"
        );
    }
}

/// Tests that the pipeline remains stable over a longer generation run.
/// Requests 500 tokens and verifies that at least 400 tokens were produced,
/// all usage fields are present, and the response is well-formed.
#[test]
#[ignore = "e2e: requires GPU, release binaries, and GGUF model"]
fn test_e2e_long_generation_stability() {
    let _guard = spawn_pipeline(9415, 8096);

    let resp = send_completion(8096, "Write a detailed essay about the history of mathematics", 500, 0.0);

    // Verify all usage fields are present and sensible.
    let prompt_tokens = resp["usage"]["prompt_tokens"].as_u64().unwrap_or_else(|| {
        panic!("missing usage.prompt_tokens: {resp}")
    });
    let completion_tokens = resp["usage"]["completion_tokens"].as_u64().unwrap_or_else(|| {
        panic!("missing usage.completion_tokens: {resp}")
    });
    let total_tokens = resp["usage"]["total_tokens"].as_u64().unwrap_or_else(|| {
        panic!("missing usage.total_tokens: {resp}")
    });

    assert!(prompt_tokens > 0, "prompt_tokens should be > 0");
    assert!(
        completion_tokens >= 400,
        "expected >= 400 completion tokens for long generation, got {completion_tokens}"
    );
    assert_eq!(
        total_tokens,
        prompt_tokens + completion_tokens,
        "total_tokens != prompt_tokens + completion_tokens"
    );

    // Verify choices structure.
    let text = resp["choices"][0]["text"].as_str().unwrap_or_else(|| {
        panic!("missing choices[0].text: {resp}")
    });
    assert!(!text.is_empty(), "generated text should not be empty");

    // Verify top-level response fields.
    assert!(resp["id"].as_str().is_some(), "missing id field");
    assert!(resp["object"].as_str().is_some(), "missing object field");
    assert!(resp["created"].as_u64().is_some(), "missing created field");
}

/// Tests that a running pipeline correctly reports an error or a changed health
/// state when the worker process is killed mid-request.
///
/// Spawns coordinator + worker on unique ports (9416/8097). A background thread
/// starts a long generation (500 tokens). While that request is in flight we kill
/// the worker and confirm that either (a) the HTTP request returns a non-2xx
/// status, (b) the JSON body contains an error field, or (c) a subsequent health
/// check no longer reports status=ready.
#[test]
#[ignore = "e2e: requires GPU, release binaries, and GGUF model"]
fn test_e2e_network_failure_detection() {
    // We need direct access to the worker process so we can kill it.
    let bin_c = coord_bin();
    let bin_w = worker_bin();
    let model = model_path();
    const COORD_PORT: u16 = 9416;
    const HTTP_PORT: u16 = 8097;

    let coordinator = Command::new(&bin_c)
        .args([
            "--model", model.to_str().unwrap(),
            "--listen", &format!("127.0.0.1:{COORD_PORT}"),
            "--workers", "1",
            "--http-port", &HTTP_PORT.to_string(),
            "--scheduling", "equal",
        ])
        .env("RUST_LOG", "info")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap_or_else(|e| panic!("failed to spawn coordinator: {e}"));

    std::thread::sleep(Duration::from_secs(1));

    let mut worker = Command::new(&bin_w)
        .args([
            "--model", model.to_str().unwrap(),
            "--coordinator", &format!("127.0.0.1:{COORD_PORT}"),
            "--node-id", "e2e-failure-worker",
        ])
        .env("RUST_LOG", "info")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap_or_else(|e| panic!("failed to spawn worker: {e}"));

    // Wrap coordinator in guard so it's always killed on drop.
    let guard = ProcessGuard { coordinator, worker: {
        // Temporarily move worker out — we'll put a placeholder in.
        // We need to own the worker separately to kill it mid-test.
        // Rebuild the guard manually below after the test.
        Command::new("true").spawn().unwrap()
    }};

    // Wait for HTTP readiness.
    let client = reqwest::blocking::Client::new();
    let setup_start = Instant::now();
    loop {
        if setup_start.elapsed() > Duration::from_secs(60) {
            let _ = worker.kill();
            let _ = worker.wait();
            panic!("HTTP server not ready within 60 seconds");
        }
        if let Ok(resp) = client.get(format!("http://127.0.0.1:{HTTP_PORT}/health")).send() {
            if resp.status().is_success() {
                break;
            }
        }
        std::thread::sleep(Duration::from_millis(500));
    }

    // Start a long generation in a background thread.
    let gen_thread = std::thread::spawn(move || {
        let c = reqwest::blocking::Client::builder()
            .timeout(Duration::from_secs(60))
            .build()
            .unwrap();
        c.post(format!("http://127.0.0.1:{HTTP_PORT}/v1/completions"))
            .json(&serde_json::json!({
                "prompt": "Write a very long essay about the history of computing",
                "max_tokens": 500,
                "temperature": 0.0,
            }))
            .send()
    });

    // Give the generation a moment to start, then kill the worker.
    std::thread::sleep(Duration::from_millis(500));
    let _ = worker.kill();
    let _ = worker.wait();

    // The background request should either fail at the transport layer or
    // return a non-success HTTP status.
    match gen_thread.join().expect("generation thread panicked") {
        Err(_transport_err) => {
            // Transport-level failure is expected after worker death.
        }
        Ok(resp) => {
            // If we got a response, it should indicate failure.
            let status = resp.status();
            if status.is_success() {
                // The coordinator might have returned a successful response
                // before realising the worker died. Check for an error field
                // in the JSON, or check that health now reports not-ready.
                let body: serde_json::Value = resp.json().unwrap_or(serde_json::json!({}));
                let has_error = body.get("error").is_some()
                    || body["choices"][0]["finish_reason"]
                        .as_str()
                        .map_or(false, |r| r == "error");
                if !has_error {
                    // Verify health degraded after the kill.
                    std::thread::sleep(Duration::from_secs(2));
                    let health_resp = client
                        .get(format!("http://127.0.0.1:{HTTP_PORT}/health"))
                        .send();
                    match health_resp {
                        Err(_) => {} // coordinator also down — acceptable
                        Ok(h) => {
                            let health: serde_json::Value =
                                h.json().unwrap_or(serde_json::json!({}));
                            assert_ne!(
                                health["status"].as_str().unwrap_or(""),
                                "ready",
                                "health should no longer be 'ready' after worker killed"
                            );
                        }
                    }
                }
            }
            // Non-success HTTP (4xx / 5xx) is the expected path — no assertion needed.
        }
    }

    // Ensure coordinator is cleaned up (guard.coordinator is already in the ProcessGuard).
    drop(guard);
}

/// Cross-machine inference test using environment variables.
///
/// Set FRACTURE_COORD_HOST (e.g. "192.168.1.10:8091") to point at a running
/// coordinator HTTP endpoint. If not set the test is skipped.
///
/// This test is always `#[ignore]` — it must be run explicitly:
///   FRACTURE_COORD_HOST=host:port cargo nextest run ... --run-ignored all
///       -E 'test(cross_machine)'
#[test]
#[ignore = "cross-machine: requires FRACTURE_COORD_HOST env var pointing at running coordinator"]
fn test_e2e_cross_machine_inference() {
    let coord_host = match std::env::var("FRACTURE_COORD_HOST") {
        Ok(h) if !h.is_empty() => h,
        _ => {
            eprintln!("FRACTURE_COORD_HOST not set — skipping cross-machine test");
            return;
        }
    };

    let base_url = if coord_host.starts_with("http://") || coord_host.starts_with("https://") {
        coord_host.clone()
    } else {
        format!("http://{coord_host}")
    };

    let client = reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(120))
        .build()
        .unwrap();

    // Health check first.
    let health: serde_json::Value = client
        .get(format!("{base_url}/health"))
        .send()
        .expect("health check failed")
        .json()
        .expect("health response is not JSON");
    assert_eq!(
        health["status"].as_str().unwrap_or(""),
        "ready",
        "coordinator at {base_url} is not ready: {health}"
    );

    // Send a greedy completion and validate the response.
    let resp = client
        .post(format!("{base_url}/v1/completions"))
        .json(&serde_json::json!({
            "prompt": "The capital of France is",
            "max_tokens": 10,
            "temperature": 0.0,
        }))
        .send()
        .expect("completion request failed");

    assert!(
        resp.status().is_success(),
        "completion returned HTTP {}: check coordinator logs",
        resp.status()
    );

    let body: serde_json::Value = resp.json().expect("response is not JSON");

    // Validate OpenAI-compatible response structure.
    assert!(body["id"].as_str().is_some(), "missing id field: {body}");
    assert_eq!(body["object"].as_str().unwrap_or(""), "text_completion", "wrong object type: {body}");
    assert!(body["created"].as_u64().is_some(), "missing created field: {body}");

    let text = body["choices"][0]["text"]
        .as_str()
        .unwrap_or_else(|| panic!("missing choices[0].text: {body}"));
    assert!(!text.is_empty(), "generated text is empty: {body}");

    let prompt_tokens = body["usage"]["prompt_tokens"].as_u64().unwrap_or_else(|| {
        panic!("missing usage.prompt_tokens: {body}")
    });
    let completion_tokens = body["usage"]["completion_tokens"].as_u64().unwrap_or_else(|| {
        panic!("missing usage.completion_tokens: {body}")
    });
    let total_tokens = body["usage"]["total_tokens"].as_u64().unwrap_or_else(|| {
        panic!("missing usage.total_tokens: {body}")
    });

    assert!(prompt_tokens > 0, "prompt_tokens should be > 0");
    assert!(completion_tokens > 0, "completion_tokens should be > 0");
    assert_eq!(
        total_tokens,
        prompt_tokens + completion_tokens,
        "usage totals inconsistent: {body}"
    );

    // Check it says something sensible (Paris is the capital of France).
    assert!(
        text.to_lowercase().contains("paris"),
        "expected 'Paris' in greedy completion for 'The capital of France is'; got: {text:?}"
    );
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

// ── Phase 4 Step 4: Distributed Batching Validation ──────────────────────

/// Validates that the distributed batched pipeline (--batched) produces
/// identical greedy output to the sequential distributed_generate path.
///
/// Runs the two pipelines sequentially (not simultaneously) to avoid OOM
/// from loading the model twice on one GPU.
///
/// This is the key validation for Phase 4 Step 4 item 6.
#[test]
#[ignore = "e2e: requires GPU, release binaries, and GGUF model"]
fn test_e2e_distributed_batched_matches_sequential() {
    let prompts = [
        "The capital of France is",
        "In the year 2025, artificial intelligence",
    ];

    // Collect sequential results first, then tear down.
    let seq_results: Vec<serde_json::Value> = {
        let _guard = spawn_pipeline(9420, 8100);
        prompts.iter().map(|p| send_completion(8100, p, 20, 0.0)).collect()
    };
    // _guard dropped → coordinator + worker killed. Brief pause for GPU memory release
    // and OS port cleanup before reusing the same ports.
    std::thread::sleep(Duration::from_secs(2));

    // Now run batched on the same ports (safe since previous processes are dead).
    let bat_results: Vec<serde_json::Value> = {
        let _guard = spawn_batched_pipeline(9420, 8100);
        prompts.iter().map(|p| send_completion(8100, p, 20, 0.0)).collect()
    };

    for (i, prompt) in prompts.iter().enumerate() {
        let seq_text = seq_results[i]["choices"][0]["text"].as_str().unwrap();
        let bat_text = bat_results[i]["choices"][0]["text"].as_str().unwrap();

        assert_eq!(
            seq_text, bat_text,
            "batched and sequential must produce identical greedy output for prompt: {prompt:?}\n  sequential: {seq_text:?}\n  batched:    {bat_text:?}"
        );

        let seq_ct = seq_results[i]["usage"]["completion_tokens"].as_u64().unwrap();
        let bat_ct = bat_results[i]["usage"]["completion_tokens"].as_u64().unwrap();
        assert_eq!(seq_ct, bat_ct, "completion_tokens must match");
    }
}

/// Validates concurrent requests through the batched distributed pipeline.
/// Multiple greedy requests sent simultaneously must all produce correct
/// output with no cross-contamination.
#[test]
#[ignore = "e2e: requires GPU, release binaries, and GGUF model"]
fn test_e2e_distributed_batched_concurrent() {
    let _guard = spawn_batched_pipeline(9422, 8102);

    let prompts = [
        "The capital of France is",
        "Water boils at a temperature of",
        "The largest planet in the solar system is",
    ];

    // Send all concurrently
    let handles: Vec<_> = prompts
        .iter()
        .map(|&prompt| {
            std::thread::spawn(move || send_completion(8102, prompt, 20, 0.0))
        })
        .collect();

    let results: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();

    // All must succeed with valid output
    for (i, resp) in results.iter().enumerate() {
        let text = resp["choices"][0]["text"].as_str().unwrap();
        assert!(!text.is_empty(), "prompt {i} returned empty text");
        let ct = resp["usage"]["completion_tokens"].as_u64().unwrap();
        assert!(ct > 0, "prompt {i} generated 0 tokens");
        let pt = resp["usage"]["prompt_tokens"].as_u64().unwrap();
        let tt = resp["usage"]["total_tokens"].as_u64().unwrap();
        assert_eq!(tt, pt + ct, "prompt {i}: total_tokens mismatch");
    }

    // Send same prompts sequentially and verify identical greedy output
    for (i, &prompt) in prompts.iter().enumerate() {
        let seq_resp = send_completion(8102, prompt, 20, 0.0);
        let seq_text = seq_resp["choices"][0]["text"].as_str().unwrap();
        let conc_text = results[i]["choices"][0]["text"].as_str().unwrap();
        assert_eq!(
            seq_text, conc_text,
            "concurrent vs sequential mismatch for prompt {i}: {prompt:?}"
        );
    }
}

/// Benchmark: measures throughput (tokens/second) of the batched distributed
/// pipeline under concurrent load.
///
/// Run separately:
///   cargo nextest run -p fracture-coordinator-cuda --run-ignored all -E 'test(bench_batched)'
#[test]
#[ignore = "benchmark: requires GPU, release binaries, and GGUF model"]
fn bench_distributed_batched_throughput() {
    let _guard = spawn_batched_pipeline(9423, 8103);

    // Warmup
    send_completion(8103, "Warmup prompt for the model", 10, 0.0);

    let num_requests = 3;
    let max_tokens = 20;
    let prompt = "Explain the concept of distributed computing in simple terms.";

    let start = Instant::now();

    let handles: Vec<_> = (0..num_requests)
        .map(|_| {
            std::thread::spawn(move || send_completion(8103, prompt, max_tokens, 0.0))
        })
        .collect();

    let results: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();

    let elapsed = start.elapsed();

    let total_generated: u64 = results
        .iter()
        .map(|r| r["usage"]["completion_tokens"].as_u64().unwrap())
        .sum();

    let tokens_per_sec = total_generated as f64 / elapsed.as_secs_f64();

    eprintln!(
        "distributed batched throughput: {num_requests} concurrent requests × {max_tokens} max_tokens"
    );
    eprintln!(
        "  total tokens: {total_generated}, wall time: {:.2}s, throughput: {:.1} tok/s",
        elapsed.as_secs_f64(),
        tokens_per_sec,
    );

    // Sanity: all requests completed
    assert_eq!(results.len(), num_requests);
    for (i, r) in results.iter().enumerate() {
        let ct = r["usage"]["completion_tokens"].as_u64().unwrap();
        assert!(ct > 0, "request {i} generated 0 tokens");
    }
}

// ── Phase 4 Step 5: Fault Tolerance E2E Tests ────────────────────────────

/// Verifies that the batched distributed pipeline detects worker death via
/// heartbeat and returns errors to in-flight requests.
///
/// Uses a single worker — after killing it, all requests should fail.
/// The coordinator's heartbeat (integrated in the distributed loop) marks
/// the worker as dead and the pipeline as degraded.
#[test]
#[ignore = "e2e: requires GPU, release binaries, and GGUF model"]
fn test_e2e_batched_worker_death_aborts_requests() {
    let bin_c = coord_bin();
    let bin_w = worker_bin();
    let model = model_path();
    const COORD_PORT: u16 = 9424;
    const HTTP_PORT: u16 = 8104;

    // Spawn coordinator in batched mode.
    let coordinator = Command::new(&bin_c)
        .args([
            "--model", model.to_str().unwrap(),
            "--listen", &format!("127.0.0.1:{COORD_PORT}"),
            "--workers", "1",
            "--http-port", &HTTP_PORT.to_string(),
            "--scheduling", "equal",
            "--batched",
        ])
        .env("RUST_LOG", "info")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .unwrap_or_else(|e| panic!("failed to spawn coordinator: {e}"));

    std::thread::sleep(Duration::from_secs(1));

    let mut worker = Command::new(&bin_w)
        .args([
            "--model", model.to_str().unwrap(),
            "--coordinator", &format!("127.0.0.1:{COORD_PORT}"),
            "--node-id", "fault-worker",
        ])
        .env("RUST_LOG", "info")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .unwrap_or_else(|e| panic!("failed to spawn worker: {e}"));

    // Guard for coordinator cleanup; we manage the worker manually.
    let _guard = ProcessGuard {
        coordinator,
        worker: Command::new("true").spawn().unwrap(),
    };

    // Wait for pipeline to be fully ready by polling completions (health
    // returns "ready" before workers finish connecting).
    let start = Instant::now();
    loop {
        if start.elapsed() > Duration::from_secs(60) {
            let _ = worker.kill();
            let _ = worker.wait();
            panic!("pipeline not ready within 60 seconds");
        }
        let client = reqwest::blocking::Client::builder()
            .timeout(Duration::from_secs(10))
            .build()
            .unwrap();
        let result = client
            .post(format!("http://127.0.0.1:{HTTP_PORT}/v1/completions"))
            .json(&serde_json::json!({
                "prompt": "Hello",
                "max_tokens": 3,
                "temperature": 0.0,
            }))
            .send();
        if let Ok(resp) = result {
            if resp.status().is_success() { break; }
        }
        std::thread::sleep(Duration::from_secs(1));
    }

    // Kill the worker.
    let _ = worker.kill();
    let _ = worker.wait();

    // Wait for the heartbeat to detect the dead worker (up to 20s = 5s interval × 3 missed + margin).
    std::thread::sleep(Duration::from_secs(20));

    // Requests should now fail (500 or error in body).
    let error_client = reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(30))
        .build()
        .unwrap();
    let resp = error_client
        .post(format!("http://127.0.0.1:{HTTP_PORT}/v1/completions"))
        .json(&serde_json::json!({
            "prompt": "test after kill",
            "max_tokens": 5,
            "temperature": 0.0,
        }))
        .send();

    match resp {
        Err(_) => {} // Transport failure — acceptable.
        Ok(r) => {
            // Either HTTP 500 or a response with an error body.
            if r.status().is_success() {
                let body: serde_json::Value = r.json().unwrap_or_default();
                assert!(
                    body.get("error").is_some() || body["choices"][0]["text"].as_str().unwrap_or("").is_empty(),
                    "expected error after worker death, got: {body}"
                );
            }
            // Non-success status (500) is the expected path.
        }
    }
}

/// Verifies worker reconnection: kill the worker, restart it, verify the
/// coordinator re-registers it, reconfigures the pipeline, and inference
/// resumes.
///
/// This tests the full fault-tolerance cycle with a single GPU:
/// 1. Pipeline healthy → inference works
/// 2. Kill worker → pipeline degraded
/// 3. Restart worker → reconnects via reconnection_listener
/// 4. Coordinator sends Reconfigure → worker reloads weights → sends WorkerReady
/// 5. Pipeline restored → inference works again
#[test]
#[ignore = "e2e: requires GPU, release binaries, and GGUF model"]
fn test_e2e_worker_reconnection_recovery() {
    let bin_c = coord_bin();
    let bin_w = worker_bin();
    let model = model_path();
    const COORD_PORT: u16 = 9425;
    const HTTP_PORT: u16 = 8105;

    // Spawn coordinator in batched mode.
    let coordinator = Command::new(&bin_c)
        .args([
            "--model", model.to_str().unwrap(),
            "--listen", &format!("127.0.0.1:{COORD_PORT}"),
            "--workers", "1",
            "--http-port", &HTTP_PORT.to_string(),
            "--scheduling", "equal",
            "--batched",
        ])
        .env("RUST_LOG", "info")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .unwrap_or_else(|e| panic!("failed to spawn coordinator: {e}"));

    std::thread::sleep(Duration::from_secs(1));

    let mut worker = Command::new(&bin_w)
        .args([
            "--model", model.to_str().unwrap(),
            "--coordinator", &format!("127.0.0.1:{COORD_PORT}"),
            "--node-id", "reconnect-worker",
        ])
        .env("RUST_LOG", "info")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .unwrap_or_else(|e| panic!("failed to spawn worker: {e}"));

    let _guard = ProcessGuard {
        coordinator,
        worker: Command::new("true").spawn().unwrap(),
    };

    // Wait for pipeline to be fully ready by polling completions.
    let start = Instant::now();
    loop {
        if start.elapsed() > Duration::from_secs(60) {
            let _ = worker.kill();
            let _ = worker.wait();
            panic!("pipeline not ready within 60 seconds");
        }
        let client = reqwest::blocking::Client::builder()
            .timeout(Duration::from_secs(10))
            .build()
            .unwrap();
        let result = client
            .post(format!("http://127.0.0.1:{HTTP_PORT}/v1/completions"))
            .json(&serde_json::json!({
                "prompt": "test",
                "max_tokens": 1,
                "temperature": 0.0,
            }))
            .send();
        if let Ok(resp) = result {
            if resp.status().is_success() { break; }
        }
        std::thread::sleep(Duration::from_secs(1));
    }

    // Step 1: Verify inference works.
    let resp = send_completion(HTTP_PORT, "The capital of France is", 10, 0.0);
    let text_before = resp["choices"][0]["text"].as_str().unwrap().to_string();
    assert!(!text_before.is_empty(), "should generate text before kill");

    // Step 2: Kill the worker.
    let _ = worker.kill();
    let _ = worker.wait();
    // Brief pause for GPU memory release.
    std::thread::sleep(Duration::from_secs(2));

    // Step 3: Restart the worker (same node-id, connects to same coordinator).
    let mut worker2 = Command::new(&bin_w)
        .args([
            "--model", model.to_str().unwrap(),
            "--coordinator", &format!("127.0.0.1:{COORD_PORT}"),
            "--node-id", "reconnect-worker",
        ])
        .env("RUST_LOG", "info")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .unwrap_or_else(|e| panic!("failed to restart worker: {e}"));

    // Step 4: Wait for the pipeline to recover. The reconnection_listener
    // accepts the new connection, re-registers the worker, reconfigures the
    // pipeline, and the distributed loop swaps to the new pipeline.
    // Give it up to 60s (weight loading + reconfiguration).
    let recovery_start = Instant::now();
    let mut recovered = false;
    while recovery_start.elapsed() < Duration::from_secs(60) {
        let client = reqwest::blocking::Client::builder()
            .timeout(Duration::from_secs(15))
            .build()
            .unwrap();
        let result = client
            .post(format!("http://127.0.0.1:{HTTP_PORT}/v1/completions"))
            .json(&serde_json::json!({
                "prompt": "The capital of France is",
                "max_tokens": 10,
                "temperature": 0.0,
            }))
            .send();

        if let Ok(r) = result {
            if r.status().is_success() {
                let body: serde_json::Value = r.json().unwrap_or_default();
                if body["choices"][0]["text"].as_str().map_or(false, |t| !t.is_empty()) {
                    recovered = true;
                    // Step 5: Verify output is identical (greedy determinism).
                    let text_after = body["choices"][0]["text"].as_str().unwrap();
                    assert_eq!(
                        text_before, text_after,
                        "greedy output should be identical after recovery"
                    );
                    break;
                }
            }
        }
        std::thread::sleep(Duration::from_secs(2));
    }

    let _ = worker2.kill();
    let _ = worker2.wait();

    assert!(recovered, "pipeline did not recover within 60 seconds after worker reconnection");
}
