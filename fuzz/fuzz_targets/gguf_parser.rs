#![no_main]
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    // Parser must never panic on arbitrary bytes — only return Err.
    let _ = fracture_gguf::parse_header_from_bytes(data);
});
