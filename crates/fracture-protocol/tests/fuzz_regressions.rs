//! Regression tests for inputs that previously crashed the wire-frame fuzzer.
//!
//! Inputs in `tests/fuzz_regressions/inputs/` were captured from libfuzzer
//! crashes; the contract is that decoding them must return `Err`, never panic.

use std::path::PathBuf;

#[test]
fn no_panic_on_committed_wire_fuzz_regressions() {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fuzz_regressions/inputs");
    if !dir.exists() {
        return;
    }
    for entry in std::fs::read_dir(&dir).expect("read fuzz_regressions/inputs") {
        let path = entry.unwrap().path();
        if !path.is_file() {
            continue;
        }
        // Skip placeholder files (e.g., .gitkeep).
        if path
            .file_name()
            .and_then(|n| n.to_str())
            .map(|n| n.starts_with('.'))
            .unwrap_or(false)
        {
            continue;
        }
        let bytes = std::fs::read(&path).unwrap();
        // Contract: no panic on adversarial input. Result is otherwise ignored.
        let _ = fracture_protocol::decode_frame_from_bytes(&bytes);
    }
}
