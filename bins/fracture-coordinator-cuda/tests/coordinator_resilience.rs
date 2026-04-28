//! Coordinator resilience: a stray TCP connection to the protocol port must
//! not poison the worker-acceptance loop.
//!
//! Reproduces the bug where probing the protocol port (e.g. a health check, a
//! load balancer, or our own readiness probes) caused `accept_and_setup_pipeline`
//! to bail on EOF, terminating the spawned task and preventing
//! `reconnection_listener` from ever being spawned. After that point, the
//! coordinator can never accept a worker registration until restarted.

use std::net::TcpStream;
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

struct ChildGuard(Child);
impl Drop for ChildGuard {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

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
fn stray_tcp_probe_does_not_poison_accept_loop() {
    let fixture = fixture_path();
    assert!(
        fixture.exists(),
        "fixture missing at {}; run scripts/build_fixture_model.py",
        fixture.display()
    );

    // Distinct ports from e2e_fixture.rs so the two tests don't clash if run
    // concurrently (the e2e-distributed group serializes them, but defensive).
    const COORD_PORT: u16 = 19510;
    const HTTP_PORT: u16 = 19512;

    let tmp = std::env::temp_dir().join(format!(
        "fracture-resilience-{}",
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

    // Wait for the coordinator's HTTP port to bind (proves the binary started)
    // and the protocol port shortly thereafter. We don't probe the protocol
    // port for readiness — that probe is itself the bug trigger.
    std::thread::sleep(Duration::from_millis(500));

    // Send a single stray TCP probe to the protocol port and immediately drop
    // it. This is the bug trigger: a load balancer health check, a port
    // scanner, or our own readiness probe would do exactly this.
    let coord_addr = format!("127.0.0.1:{COORD_PORT}");
    let probe = TcpStream::connect_timeout(
        &coord_addr.parse().unwrap(),
        Duration::from_secs(1),
    )
    .expect("stray probe must connect (coordinator port should be bound)");
    drop(probe);
    // Give the coordinator a moment to react to the dropped connection.
    std::thread::sleep(Duration::from_millis(100));

    // Now spawn a real worker. With the fix in place, the coordinator's accept
    // loop survived the probes and is still listening, so the worker registers
    // normally. Without the fix, the worker's TCP connect either fails (port
    // released after task death) or hangs (data buffered with no reader).
    let _worker = ChildGuard(
        Command::new(release_bin("fracture-worker-cuda"))
            .args([
                "--model",
                fixture.to_str().unwrap(),
                "--coordinator",
                &format!("127.0.0.1:{COORD_PORT}"),
                "--node-id",
                "resilience-test-worker",
            ])
            .env("RUST_LOG", "info")
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .expect("spawn worker"),
    );

    // Verify end-to-end: a successful completion proves the worker registered,
    // the pipeline came up, and HTTP serves requests.
    let client = reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(10))
        .build()
        .unwrap();
    let body = serde_json::json!({
        "prompt": "abcd",
        "max_tokens": 4,
        "temperature": 0.0,
        "stream": false,
    });
    let setup_start = Instant::now();
    let resp = loop {
        if setup_start.elapsed() > Duration::from_secs(45) {
            panic!(
                "distributed pipeline did not become ready within 45s after stray probes — \
                 likely indicates the accept loop was poisoned"
            );
        }
        match client
            .post(format!("http://127.0.0.1:{HTTP_PORT}/v1/completions"))
            .json(&body)
            .send()
        {
            Ok(r) if r.status().is_success() => break r,
            _ => std::thread::sleep(Duration::from_millis(500)),
        }
    };

    let json: serde_json::Value = resp.json().expect("response not JSON");
    assert!(
        json["choices"][0]["text"].is_string(),
        "missing choices[0].text: {json}"
    );

    let _ = std::fs::remove_dir_all(&tmp);
}
