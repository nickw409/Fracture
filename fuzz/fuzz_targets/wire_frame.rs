#![no_main]
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    // Wire-frame decoder must never panic on arbitrary bytes — only return Err.
    let _ = fracture_protocol::decode_frame_from_bytes(data);
});
