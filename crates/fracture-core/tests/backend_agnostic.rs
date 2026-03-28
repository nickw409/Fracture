/// Verify that engine, generate, and server crates never depend on backend crates.
/// This is the critical architectural invariant of Fracture.
#[test]
fn test_no_backend_dependencies() {
    // This test is also referenced as closing the "backend-agnostic" Cargo.toml gap.
    let crates = [
        "crates/fracture-engine/Cargo.toml",
        "crates/fracture-generate/Cargo.toml",
        "crates/fracture-server/Cargo.toml",
    ];

    let backend_names = ["fracture-cuda", "fracture-metal"];

    let workspace_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap();

    for crate_path in &crates {
        let full_path = workspace_root.join(crate_path);
        let contents = std::fs::read_to_string(&full_path)
            .unwrap_or_else(|e| panic!("failed to read {}: {e}", full_path.display()));

        for backend in &backend_names {
            assert!(
                !contents.contains(backend),
                "{crate_path} must not depend on {backend}, but found it in Cargo.toml"
            );
        }
    }
}

/// Verify that engine code never accesses raw device pointers — only
/// uses DeviceTensor metadata fields (id, shape, dtype, numel, size_bytes).
/// Scans engine source for patterns like `as *mut`, `as *const`, `unsafe`.
#[test]
fn test_engine_never_accesses_device_pointers() {
    let workspace_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap();

    let engine_src = workspace_root.join("crates/fracture-engine/src");
    let forbidden = ["as *mut", "as *const", "unsafe "];

    for entry in std::fs::read_dir(&engine_src).unwrap() {
        let path = entry.unwrap().path();
        if path.extension().map_or(true, |e| e != "rs") {
            continue;
        }
        let content = std::fs::read_to_string(&path).unwrap();

        // Skip test code
        let end = content.find("#[cfg(test)]").unwrap_or(content.len());
        let production_code = &content[..end];

        for pattern in &forbidden {
            assert!(
                !production_code.contains(pattern),
                "{} contains '{}' — engine must not access raw pointers",
                path.display(),
                pattern
            );
        }
    }
}
