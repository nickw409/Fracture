//! Regression tests for inputs that previously crashed the GGUF parser fuzzer.
//!
//! Inputs live in tests/fuzz_regressions/inputs/. Each file is fed through the
//! same public entry the fuzz target uses; the test passes as long as no input
//! causes a panic. We accept either Ok or Err results — the contract is "no
//! panic on adversarial input."

use std::path::PathBuf;

#[test]
fn no_panic_on_committed_fuzz_regressions() {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fuzz_regressions/inputs");
    if !dir.exists() {
        return; // No regressions yet — that's fine.
    }
    for entry in std::fs::read_dir(&dir).expect("read fuzz_regressions/inputs") {
        let path = entry.unwrap().path();
        if !path.is_file() {
            continue;
        }
        // Skip the .gitkeep marker (and any other dotfiles).
        if path
            .file_name()
            .and_then(|n| n.to_str())
            .map(|n| n.starts_with('.'))
            .unwrap_or(false)
        {
            continue;
        }
        let bytes = std::fs::read(&path)
            .unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
        // The contract: this call must not panic. Result is otherwise ignored.
        let _ = fracture_gguf::parse_header_from_bytes(&bytes);
    }
}
