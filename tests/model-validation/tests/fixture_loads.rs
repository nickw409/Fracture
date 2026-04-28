//! Sanity test: fixture GGUF loads via fracture-gguf and matches the JSON config.

use std::path::PathBuf;

#[test]
fn test_fixture_loads_with_expected_config() {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("tests/fixtures/tiny-llama.gguf");
    assert!(
        path.exists(),
        "fixture missing at {}; run scripts/build_fixture_model.py",
        path.display()
    );

    let backend = match fracture_cuda::CudaBackend::new(0) {
        Ok(b) => b,
        Err(_) => {
            eprintln!("skip: CUDA unavailable");
            return;
        }
    };
    let weights = fracture_gguf::WeightStore::load(&path, &backend, None)
        .expect("fixture load failed");
    let cfg = &weights.config;

    assert_eq!(cfg.num_layers, 4, "num_layers");
    assert_eq!(cfg.hidden_size, 128, "hidden_size");
    assert_eq!(cfg.num_q_heads, 4, "num_q_heads");
    assert_eq!(cfg.num_kv_heads, 2, "num_kv_heads");
    assert_eq!(cfg.head_dim, 32, "head_dim");
    assert_eq!(cfg.intermediate_size, 256, "intermediate_size");
    assert_eq!(cfg.vocab_size, 256, "vocab_size");
}
