//! Scan library crate source files for panicking patterns (.unwrap(), .expect(), panic!())
//! outside of test modules. This enforces the "no panics in library crates" invariant.
//!
//! Known exceptions:
//! - `lock().unwrap()` on Mutex — poisoning implies a prior panic; re-panicking is idiomatic
//! - `to_value(response).unwrap()` — serializing a known-good struct cannot fail
//! - `.unwrap()` with a safety comment (`// safe:`) indicating the invariant was checked

use std::fs;
use std::path::Path;

/// Patterns that indicate potential panics in library code.
const PANIC_PATTERNS: &[&str] = &[".unwrap()", ".expect(", "panic!("];

/// Lines matching these patterns are known-acceptable and should be skipped.
const ALLOWED_PATTERNS: &[&str] = &[
    "lock().unwrap()",              // Mutex poisoning — idiomatic Rust
    "to_value(response).unwrap()",  // serde on a known-valid struct
    "duration_since(UNIX_EPOCH)",   // SystemTime is always after epoch
    "// safe:",                     // Developer explicitly documented safety
];

/// Library crates that must not contain panicking calls outside test modules.
const LIBRARY_CRATES: &[&str] = &[
    "crates/fracture-core/src",
    "crates/fracture-engine/src",
    "crates/fracture-generate/src",
    "crates/fracture-server/src",
    "crates/fracture-gguf/src",
];

#[test]
fn test_no_panics_in_library_crates() {
    let workspace_root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap();
    let mut violations = Vec::new();

    for crate_src in LIBRARY_CRATES {
        let src_dir = workspace_root.join(crate_src);
        if !src_dir.exists() {
            continue;
        }
        scan_dir(&src_dir, &mut violations);
    }

    if !violations.is_empty() {
        let report: Vec<String> = violations
            .iter()
            .map(|(file, line_num, line)| format!("  {}:{}: {}", file, line_num, line.trim()))
            .collect();
        panic!(
            "Found {} panicking pattern(s) in library crate source (outside #[cfg(test)]):\n{}",
            violations.len(),
            report.join("\n")
        );
    }
}

fn scan_dir(dir: &Path, violations: &mut Vec<(String, usize, String)>) {
    for entry in fs::read_dir(dir).unwrap() {
        let entry = entry.unwrap();
        let path = entry.path();
        if path.is_dir() {
            scan_dir(&path, violations);
        } else if path.extension().is_some_and(|e| e == "rs") {
            scan_file(&path, violations);
        }
    }
}

fn scan_file(path: &Path, violations: &mut Vec<(String, usize, String)>) {
    // Skip dedicated test modules (files named *_tests.rs).
    if let Some(stem) = path.file_stem().and_then(|s| s.to_str()) {
        if stem.ends_with("_tests") {
            return;
        }
    }

    let content = fs::read_to_string(path).unwrap();
    let lines: Vec<&str> = content.lines().collect();

    // Find the line where #[cfg(test)] appears — everything after is test code.
    let test_start = lines
        .iter()
        .position(|line| line.trim() == "#[cfg(test)]");

    let end = test_start.unwrap_or(lines.len());

    for (line_num_0, &line) in lines[..end].iter().enumerate() {
        let line_num = line_num_0 + 1;
        let trimmed = line.trim();

        // Skip comments
        if trimmed.starts_with("//") || trimmed.starts_with("/*") || trimmed.starts_with('*') {
            continue;
        }

        // Check for panic patterns
        let has_panic_pattern = PANIC_PATTERNS.iter().any(|p| line.contains(p));
        if !has_panic_pattern {
            continue;
        }

        // Skip known-acceptable patterns
        let is_allowed = ALLOWED_PATTERNS.iter().any(|p| line.contains(p));
        if is_allowed {
            continue;
        }

        violations.push((
            path.display().to_string(),
            line_num,
            line.to_string(),
        ));
    }
}
