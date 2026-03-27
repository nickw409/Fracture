/// Verify that engine, generate, and server crates never depend on backend crates.
/// This is the critical architectural invariant of Fracture.
#[test]
fn test_no_backend_dependencies() {
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
