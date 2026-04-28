//! Fixture-model distributed e2e. Spawns a coordinator and one worker over
//! localhost using the committed `tests/fixtures/tiny-llama.gguf` model and
//! sends a single completion request through the HTTP API.
//!
//! Always runs (no `#[ignore]`) so distributed correctness has a default
//! gate. Does NOT verify token correctness against PyTorch — only that the
//! distributed pipeline executes end-to-end and returns a well-formed
//! OpenAI-compatible response.

use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf()
}

fn fixture_path() -> PathBuf {
    workspace_root().join("tests/fixtures/tiny-llama.gguf")
}

fn release_bin(name: &str) -> PathBuf {
    workspace_root().join(format!("target/release/{name}"))
}

/// Guard that kills the wrapped child process on drop.
struct ChildGuard(Child);
impl Drop for ChildGuard {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

/// Build a minimal byte-level tokenizer covering the fixture's 256-token
/// vocab and write it to `path` as a HuggingFace tokenizer.json.
fn write_fixture_tokenizer(path: &std::path::Path) {
    use tokenizers::models::bpe::BPE;
    use tokenizers::{AddedToken, Tokenizer};

    let model = BPE::default();
    let mut tok = Tokenizer::new(model);
    let tokens: Vec<AddedToken> = (0..256u32)
        .map(|i| AddedToken::from(format!("{}", i as u8 as char), false))
        .collect();
    tok.add_tokens(&tokens);
    tok.save(path, false).expect("save fixture tokenizer.json");
}

#[test]
fn fixture_distributed_single_request() {
    let fixture = fixture_path();
    assert!(
        fixture.exists(),
        "fixture missing at {}; run scripts/build_fixture_model.py",
        fixture.display()
    );

    // High, unused ports to avoid clashing with the model-based e2e tests.
    const COORD_PORT: u16 = 19500;
    const HTTP_PORT: u16 = 19502;

    // Stage a tokenizer.json in a tempdir (the fixture has GGUF tokenizer
    // metadata but no separate tokenizer.json; the binaries require one).
    let tmp = std::env::temp_dir().join(format!(
        "fracture-fixture-e2e-{}",
        std::process::id()
    ));
    std::fs::create_dir_all(&tmp).expect("create temp dir");
    let tokenizer_path = tmp.join("tokenizer.json");
    write_fixture_tokenizer(&tokenizer_path);

    let _coord = ChildGuard(
        Command::new(release_bin("fracture-coordinator-cuda"))
            .args([
                "--model",
                fixture.to_str().unwrap(),
                "--tokenizer",
                tokenizer_path.to_str().unwrap(),
                "--listen",
                &format!("127.0.0.1:{COORD_PORT}"),
                "--http-port",
                &HTTP_PORT.to_string(),
                "--min-workers",
                "1",
                "--scheduling",
                "equal",
                "--max-seq-len",
                "256",
            ])
            .env("RUST_LOG", "info")
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .expect("spawn coordinator"),
    );

    // Give the coordinator a moment to bind its protocol listener before the
    // worker tries to connect. Probing the port doesn't help: the coordinator
    // treats any incoming connection as a registration attempt and bails on
    // EOF, killing the pipeline setup task.
    std::thread::sleep(Duration::from_millis(500));

    let _worker = ChildGuard(
        Command::new(release_bin("fracture-worker-cuda"))
            .args([
                "--model",
                fixture.to_str().unwrap(),
                "--coordinator",
                &format!("127.0.0.1:{COORD_PORT}"),
                "--node-id",
                "fixture-e2e-worker",
            ])
            .env("RUST_LOG", "info")
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .expect("spawn worker"),
    );

    // Poll a real completion request — health returns "ready" before the
    // worker finishes registering, so probing with an actual request is the
    // most reliable readiness signal.
    let client = reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(10))
        .build()
        .unwrap();
    let setup_start = Instant::now();
    let body = serde_json::json!({
        "prompt": "abcd",
        "max_tokens": 4,
        "temperature": 0.0,
        "stream": false,
    });
    let final_resp = loop {
        if setup_start.elapsed() > Duration::from_secs(60) {
            panic!("distributed pipeline not ready within 60s for fixture e2e");
        }
        match client
            .post(format!("http://127.0.0.1:{HTTP_PORT}/v1/completions"))
            .json(&body)
            .send()
        {
            Ok(resp) if resp.status().is_success() => break resp,
            _ => std::thread::sleep(Duration::from_millis(500)),
        }
    };

    let json: serde_json::Value = final_resp.json().expect("response is not JSON");

    // OpenAI-compatible response structure.
    assert!(json["id"].as_str().is_some(), "missing id: {json}");
    assert_eq!(
        json["object"].as_str().unwrap_or(""),
        "text_completion",
        "wrong object type: {json}"
    );
    assert!(
        json["choices"][0]["text"].is_string(),
        "missing choices[0].text: {json}"
    );

    let prompt_tokens = json["usage"]["prompt_tokens"]
        .as_u64()
        .unwrap_or_else(|| panic!("missing usage.prompt_tokens: {json}"));
    let completion_tokens = json["usage"]["completion_tokens"]
        .as_u64()
        .unwrap_or_else(|| panic!("missing usage.completion_tokens: {json}"));
    let total_tokens = json["usage"]["total_tokens"]
        .as_u64()
        .unwrap_or_else(|| panic!("missing usage.total_tokens: {json}"));
    assert!(prompt_tokens > 0, "prompt_tokens should be > 0: {json}");
    assert!(completion_tokens > 0, "completion_tokens should be > 0: {json}");
    assert_eq!(
        total_tokens,
        prompt_tokens + completion_tokens,
        "usage totals inconsistent: {json}"
    );

    // Cleanup tempdir on success; on panic the OS reaps it eventually.
    let _ = std::fs::remove_dir_all(&tmp);
}
