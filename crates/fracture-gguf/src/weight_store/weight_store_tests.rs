use super::*;
use byteorder::{LittleEndian, WriteBytesExt};
use fracture_core::{DType, DeviceTimer, TensorId};
use std::io::Write;
use std::sync::atomic::{AtomicU64, Ordering};

struct MockBackend {
    next_id: AtomicU64,
}

impl MockBackend {
    fn new() -> Self {
        Self {
            next_id: AtomicU64::new(1),
        }
    }
}

impl Backend for MockBackend {
    fn alloc(&self, shape: &[usize], dtype: DType) -> Result<DeviceTensor> {
        let id = self.next_id.fetch_add(1, Ordering::SeqCst);
        Ok(DeviceTensor::new(TensorId(id), shape.to_vec(), dtype))
    }

    fn free(&self, _tensor: &DeviceTensor) -> Result<()> {
        Ok(())
    }

    fn copy_to_device(&self, _dst: &DeviceTensor, _src: &[u8]) -> Result<()> {
        Ok(())
    }

    fn copy_to_host(&self, _src: &DeviceTensor, _dst: &mut [u8]) -> Result<()> {
        Ok(())
    }

    fn matmul(
        &self,
        _a: &DeviceTensor,
        _b: &DeviceTensor,
        _out: &DeviceTensor,
    ) -> Result<()> {
        unimplemented!()
    }

    fn rmsnorm(
        &self,
        _input: &DeviceTensor,
        _weight: &DeviceTensor,
        _eps: f64,
        _out: &DeviceTensor,
    ) -> Result<()> {
        unimplemented!()
    }

    fn rope(
        &self,
        _q: &DeviceTensor,
        _k: &DeviceTensor,
        _positions: &[u32],
        _theta: f64,
        _head_dim: usize,
    ) -> Result<()> {
        unimplemented!()
    }

    fn attention(
        &self,
        _q: &DeviceTensor,
        _k_cache: &DeviceTensor,
        _v_cache: &DeviceTensor,
        _num_kv_heads: usize,
        _start_pos: usize,
        _out: &DeviceTensor,
    ) -> Result<()> {
        unimplemented!()
    }

    fn silu_mul(
        &self,
        _gate: &DeviceTensor,
        _up: &DeviceTensor,
        _out: &DeviceTensor,
    ) -> Result<()> {
        unimplemented!()
    }

    fn embedding(
        &self,
        _token_ids: &[u32],
        _embedding_table: &DeviceTensor,
        _out: &DeviceTensor,
    ) -> Result<()> {
        unimplemented!()
    }

    fn add(
        &self,
        _a: &DeviceTensor,
        _b: &DeviceTensor,
        _out: &DeviceTensor,
    ) -> Result<()> {
        unimplemented!()
    }

    fn copy_rows(
        &self,
        _src: &DeviceTensor,
        _dst: &DeviceTensor,
        _src_offset: usize,
        _dst_offset: usize,
        _count: usize,
    ) -> Result<()> {
        unimplemented!()
    }

    fn device_name(&self) -> &str {
        "mock"
    }

    fn total_memory(&self) -> usize {
        1 << 30
    }

    fn available_memory(&self) -> usize {
        1 << 30
    }

    fn synchronize(&self) -> Result<()> {
        Ok(())
    }

    fn create_timer(&self) -> Result<DeviceTimer> {
        Ok(DeviceTimer(0))
    }

    fn start_timer(&self, _timer: &DeviceTimer) -> Result<()> {
        Ok(())
    }

    fn stop_timer(&self, _timer: &DeviceTimer) -> Result<f32> {
        Ok(0.0)
    }

    fn destroy_timer(&self, _timer: &DeviceTimer) -> Result<()> {
        Ok(())
    }
}

// ── GGUF builder helpers ──────────────────────────────────────────

fn write_gguf_string(buf: &mut Vec<u8>, s: &str) {
    buf.write_u64::<LittleEndian>(s.len() as u64).unwrap();
    buf.write_all(s.as_bytes()).unwrap();
}

fn write_metadata_kv_u32(buf: &mut Vec<u8>, key: &str, val: u32) {
    write_gguf_string(buf, key);
    buf.write_u32::<LittleEndian>(4).unwrap(); // Uint32
    buf.write_u32::<LittleEndian>(val).unwrap();
}

fn write_metadata_kv_f32(buf: &mut Vec<u8>, key: &str, val: f32) {
    write_gguf_string(buf, key);
    buf.write_u32::<LittleEndian>(6).unwrap(); // Float32
    buf.write_f32::<LittleEndian>(val).unwrap();
}

fn write_metadata_kv_string(buf: &mut Vec<u8>, key: &str, val: &str) {
    write_gguf_string(buf, key);
    buf.write_u32::<LittleEndian>(8).unwrap(); // String
    write_gguf_string(buf, val);
}

fn write_metadata_kv_string_array(buf: &mut Vec<u8>, key: &str, vals: &[&str]) {
    write_gguf_string(buf, key);
    buf.write_u32::<LittleEndian>(9).unwrap(); // Array
    buf.write_u32::<LittleEndian>(8).unwrap(); // String element type
    buf.write_u64::<LittleEndian>(vals.len() as u64).unwrap();
    for s in vals {
        write_gguf_string(buf, s);
    }
}

fn write_tensor_info(buf: &mut Vec<u8>, name: &str, shape: &[u64], dtype: u32, offset: u64) {
    write_gguf_string(buf, name);
    buf.write_u32::<LittleEndian>(shape.len() as u32).unwrap();
    // Write dims in GGUF order (innermost first = reversed)
    for &dim in shape.iter().rev() {
        buf.write_u64::<LittleEndian>(dim).unwrap();
    }
    buf.write_u32::<LittleEndian>(dtype).unwrap();
    buf.write_u64::<LittleEndian>(offset).unwrap();
}

fn align_offset(offset: usize, alignment: usize) -> usize {
    (offset + alignment - 1) & !(alignment - 1)
}

/// Build a complete GGUF with N layers.
/// hidden_size=64, num_q_heads=4, num_kv_heads=2, head_dim=16, ffn=128, vocab=4
/// All tensors FP16 (dtype code 1).
fn build_complete_gguf(
    num_layers: usize,
    extra_tensors: &[(&str, &[u64])],
) -> Vec<u8> {
    // Layer tensor suffixes and their shapes:
    // attn_q.weight: [hidden, hidden] = [64, 64]
    // attn_k.weight: [kv_heads * head_dim, hidden] = [32, 64]
    // attn_v.weight: [kv_heads * head_dim, hidden] = [32, 64]
    // attn_output.weight: [hidden, hidden] = [64, 64]
    // ffn_gate.weight: [ffn, hidden] = [128, 64]
    // ffn_up.weight: [ffn, hidden] = [128, 64]
    // ffn_down.weight: [hidden, ffn] = [64, 128]
    // attn_norm.weight: [hidden] = [64]
    // ffn_norm.weight: [hidden] = [64]

    let hidden: u64 = 64;
    let kv_dim: u64 = 32; // num_kv_heads * head_dim = 2 * 16
    let ffn: u64 = 128;
    let vocab: u64 = 4;

    struct TensorSpec {
        name: String,
        shape: Vec<u64>,
    }

    let mut tensors: Vec<TensorSpec> = Vec::new();

    // Global tensors
    tensors.push(TensorSpec {
        name: "token_embd.weight".into(),
        shape: vec![vocab, hidden],
    });
    tensors.push(TensorSpec {
        name: "output_norm.weight".into(),
        shape: vec![hidden],
    });
    tensors.push(TensorSpec {
        name: "output.weight".into(),
        shape: vec![vocab, hidden],
    });

    // Per-layer tensors
    for layer in 0..num_layers {
        let layer_tensors: Vec<(&str, Vec<u64>)> = vec![
            ("attn_q.weight", vec![hidden, hidden]),
            ("attn_k.weight", vec![kv_dim, hidden]),
            ("attn_v.weight", vec![kv_dim, hidden]),
            ("attn_output.weight", vec![hidden, hidden]),
            ("ffn_gate.weight", vec![ffn, hidden]),
            ("ffn_up.weight", vec![ffn, hidden]),
            ("ffn_down.weight", vec![hidden, ffn]),
            ("attn_norm.weight", vec![hidden]),
            ("ffn_norm.weight", vec![hidden]),
        ];
        for (suffix, shape) in layer_tensors {
            tensors.push(TensorSpec {
                name: format!("blk.{layer}.{suffix}"),
                shape,
            });
        }
    }

    // Extra tensors
    for &(name, shape) in extra_tensors {
        tensors.push(TensorSpec {
            name: name.to_string(),
            shape: shape.to_vec(),
        });
    }

    // Compute offsets (all FP16, 2 bytes per element)
    let mut offsets: Vec<u64> = Vec::new();
    let mut data_size: u64 = 0;
    for t in &tensors {
        offsets.push(data_size);
        let numel: u64 = t.shape.iter().product();
        data_size += numel * 2; // FP16
    }

    let tensor_count = tensors.len();
    let metadata_count = 8;

    let mut buf = Vec::new();

    // Header
    buf.write_u32::<LittleEndian>(0x46554747).unwrap(); // GGUF_MAGIC
    buf.write_u32::<LittleEndian>(3).unwrap(); // version
    buf.write_u64::<LittleEndian>(tensor_count as u64).unwrap();
    buf.write_u64::<LittleEndian>(metadata_count as u64).unwrap();

    // Metadata
    write_metadata_kv_string(&mut buf, "general.architecture", "llama");
    write_metadata_kv_u32(&mut buf, "llama.embedding_length", hidden as u32);
    write_metadata_kv_u32(&mut buf, "llama.block_count", num_layers as u32);
    write_metadata_kv_u32(&mut buf, "llama.attention.head_count", 4);
    write_metadata_kv_u32(&mut buf, "llama.attention.head_count_kv", 2);
    write_metadata_kv_u32(&mut buf, "llama.feed_forward_length", ffn as u32);
    write_metadata_kv_f32(&mut buf, "llama.rope.freq_base", 500000.0);
    write_metadata_kv_string_array(
        &mut buf,
        "tokenizer.ggml.tokens",
        &["a", "b", "c", "d"],
    );

    // Tensor infos
    for (i, t) in tensors.iter().enumerate() {
        write_tensor_info(&mut buf, &t.name, &t.shape, 1, offsets[i]); // 1 = FP16
    }

    // Align
    let current = buf.len();
    let aligned = align_offset(current, 32);
    buf.resize(aligned, 0);

    // Tensor data
    buf.extend(vec![0u8; data_size as usize]);

    buf
}

/// Like build_complete_gguf but allows omitting specific tensor names.
fn build_gguf_without(num_layers: usize, omit: &[&str]) -> Vec<u8> {
    let hidden: u64 = 64;
    let kv_dim: u64 = 32;
    let ffn: u64 = 128;
    let vocab: u64 = 4;

    struct TensorSpec {
        name: String,
        shape: Vec<u64>,
    }

    let mut tensors: Vec<TensorSpec> = Vec::new();

    let globals: Vec<(&str, Vec<u64>)> = vec![
        ("token_embd.weight", vec![vocab, hidden]),
        ("output_norm.weight", vec![hidden]),
        ("output.weight", vec![vocab, hidden]),
    ];

    for (name, shape) in globals {
        if !omit.contains(&name) {
            tensors.push(TensorSpec {
                name: name.into(),
                shape,
            });
        }
    }

    for layer in 0..num_layers {
        let layer_tensors: Vec<(&str, Vec<u64>)> = vec![
            ("attn_q.weight", vec![hidden, hidden]),
            ("attn_k.weight", vec![kv_dim, hidden]),
            ("attn_v.weight", vec![kv_dim, hidden]),
            ("attn_output.weight", vec![hidden, hidden]),
            ("ffn_gate.weight", vec![ffn, hidden]),
            ("ffn_up.weight", vec![ffn, hidden]),
            ("ffn_down.weight", vec![hidden, ffn]),
            ("attn_norm.weight", vec![hidden]),
            ("ffn_norm.weight", vec![hidden]),
        ];
        for (suffix, shape) in layer_tensors {
            let name = format!("blk.{layer}.{suffix}");
            if !omit.contains(&name.as_str()) {
                tensors.push(TensorSpec { name, shape });
            }
        }
    }

    let mut offsets: Vec<u64> = Vec::new();
    let mut data_size: u64 = 0;
    for t in &tensors {
        offsets.push(data_size);
        let numel: u64 = t.shape.iter().product();
        data_size += numel * 2;
    }

    let tensor_count = tensors.len();
    let metadata_count = 8;

    let mut buf = Vec::new();

    buf.write_u32::<LittleEndian>(0x46554747).unwrap();
    buf.write_u32::<LittleEndian>(3).unwrap();
    buf.write_u64::<LittleEndian>(tensor_count as u64).unwrap();
    buf.write_u64::<LittleEndian>(metadata_count as u64).unwrap();

    write_metadata_kv_string(&mut buf, "general.architecture", "llama");
    write_metadata_kv_u32(&mut buf, "llama.embedding_length", hidden as u32);
    write_metadata_kv_u32(&mut buf, "llama.block_count", num_layers as u32);
    write_metadata_kv_u32(&mut buf, "llama.attention.head_count", 4);
    write_metadata_kv_u32(&mut buf, "llama.attention.head_count_kv", 2);
    write_metadata_kv_u32(&mut buf, "llama.feed_forward_length", ffn as u32);
    write_metadata_kv_f32(&mut buf, "llama.rope.freq_base", 500000.0);
    write_metadata_kv_string_array(
        &mut buf,
        "tokenizer.ggml.tokens",
        &["a", "b", "c", "d"],
    );

    for (i, t) in tensors.iter().enumerate() {
        write_tensor_info(&mut buf, &t.name, &t.shape, 1, offsets[i]);
    }

    let current = buf.len();
    let aligned = align_offset(current, 32);
    buf.resize(aligned, 0);

    buf.extend(vec![0u8; data_size as usize]);

    buf
}

fn write_gguf_to_file(data: &[u8]) -> (tempfile::TempDir, std::path::PathBuf) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("test.gguf");
    std::fs::write(&path, data).unwrap();
    (dir, path)
}

#[test]
fn test_weight_store_load_full() {
    let data = build_complete_gguf(2, &[]);
    let (_dir, path) = write_gguf_to_file(&data);
    let backend = MockBackend::new();

    let store = WeightStore::load(&path, &backend, None).unwrap();
    assert_eq!(store.config.num_layers, 2);
    assert_eq!(store.config.hidden_size, 64);
    assert_eq!(store.layers.len(), 2);
}

#[test]
fn test_weight_store_layer_range() {
    let data = build_complete_gguf(2, &[]);
    let (_dir, path) = write_gguf_to_file(&data);
    let backend = MockBackend::new();

    let store = WeightStore::load(&path, &backend, Some(0..1)).unwrap();
    assert_eq!(store.layers.len(), 1);
}

#[test]
fn test_weight_store_missing_tensor() {
    let data = build_gguf_without(2, &["blk.0.attn_q.weight"]);
    let (_dir, path) = write_gguf_to_file(&data);
    let backend = MockBackend::new();

    let result = WeightStore::load(&path, &backend, None);
    assert!(result.is_err());
    let err = result.err().unwrap();
    assert!(
        matches!(err, FractureError::WeightLoad(_)),
        "expected WeightLoad, got: {err}"
    );
    assert!(
        err.to_string().contains("attn_q.weight"),
        "expected mention of attn_q.weight in: {err}"
    );
}

#[test]
fn test_weight_store_unknown_tensor_warning() {
    // Extra tensor "custom.weight" should be ignored (load succeeds).
    let data = build_complete_gguf(2, &[("custom.weight", &[16, 16])]);
    let (_dir, path) = write_gguf_to_file(&data);
    let backend = MockBackend::new();

    let store = WeightStore::load(&path, &backend, None).unwrap();
    assert_eq!(store.layers.len(), 2);
}

#[test]
fn test_weight_field_shapes_correct() {
    // Build a 1-layer GGUF with known shapes and verify each LayerWeights field.
    // Config: hidden=64, num_q_heads=4, num_kv_heads=2, head_dim=16, ffn=128, vocab=4
    let data = build_complete_gguf(1, &[]);
    let (_dir, path) = write_gguf_to_file(&data);
    let backend = MockBackend::new();

    let store = WeightStore::load(&path, &backend, None).unwrap();
    assert_eq!(store.layers.len(), 1);

    let layer = &store.layers[0];

    // Q projection: [hidden, hidden] = [64, 64]
    assert_eq!(layer.q_proj.shape, vec![64, 64], "q_proj shape mismatch");
    // K projection: [kv_dim, hidden] = [32, 64]
    assert_eq!(layer.k_proj.shape, vec![32, 64], "k_proj shape mismatch");
    // V projection: [kv_dim, hidden] = [32, 64]
    assert_eq!(layer.v_proj.shape, vec![32, 64], "v_proj shape mismatch");
    // Output projection: [hidden, hidden] = [64, 64]
    assert_eq!(layer.o_proj.shape, vec![64, 64], "o_proj shape mismatch");
    // Gate projection: [ffn, hidden] = [128, 64]
    assert_eq!(layer.gate_proj.shape, vec![128, 64], "gate_proj shape mismatch");
    // Up projection: [ffn, hidden] = [128, 64]
    assert_eq!(layer.up_proj.shape, vec![128, 64], "up_proj shape mismatch");
    // Down projection: [hidden, ffn] = [64, 128]
    assert_eq!(layer.down_proj.shape, vec![64, 128], "down_proj shape mismatch");
    // Attention norm: [hidden] = [64]
    assert_eq!(layer.attn_norm.shape, vec![64], "attn_norm shape mismatch");
    // FFN norm: [hidden] = [64]
    assert_eq!(layer.ffn_norm.shape, vec![64], "ffn_norm shape mismatch");

    // Global tensors
    assert_eq!(store.token_embedding.shape, vec![4, 64], "token_embedding shape mismatch");
    assert_eq!(store.output_norm.shape, vec![64], "output_norm shape mismatch");
    assert_eq!(store.lm_head.shape, vec![4, 64], "lm_head shape mismatch");
}

/// FailingMockBackend: fails on the Nth alloc call with a WeightLoad-like error.
struct FailingMockBackend {
    next_id: AtomicU64,
    alloc_count: std::sync::atomic::AtomicU64,
    fail_at: u64,
}

impl FailingMockBackend {
    fn new(fail_at: u64) -> Self {
        Self {
            next_id: AtomicU64::new(1),
            alloc_count: AtomicU64::new(0),
            fail_at,
        }
    }
}

impl Backend for FailingMockBackend {
    fn alloc(&self, shape: &[usize], dtype: DType) -> Result<DeviceTensor> {
        let count = self.alloc_count.fetch_add(1, Ordering::SeqCst);
        if count >= self.fail_at {
            return Err(FractureError::Backend("alloc failed: out of memory".into()));
        }
        let id = self.next_id.fetch_add(1, Ordering::SeqCst);
        Ok(DeviceTensor::new(TensorId(id), shape.to_vec(), dtype))
    }

    fn free(&self, _tensor: &DeviceTensor) -> Result<()> {
        Ok(())
    }

    fn copy_to_device(&self, _dst: &DeviceTensor, _src: &[u8]) -> Result<()> {
        Ok(())
    }

    fn copy_to_host(&self, _src: &DeviceTensor, _dst: &mut [u8]) -> Result<()> {
        Ok(())
    }

    fn matmul(&self, _a: &DeviceTensor, _b: &DeviceTensor, _out: &DeviceTensor) -> Result<()> {
        unimplemented!()
    }

    fn rmsnorm(&self, _input: &DeviceTensor, _weight: &DeviceTensor, _eps: f64, _out: &DeviceTensor) -> Result<()> {
        unimplemented!()
    }

    fn rope(&self, _q: &DeviceTensor, _k: &DeviceTensor, _positions: &[u32], _theta: f64, _head_dim: usize) -> Result<()> {
        unimplemented!()
    }

    fn attention(&self, _q: &DeviceTensor, _k_cache: &DeviceTensor, _v_cache: &DeviceTensor, _num_kv_heads: usize, _start_pos: usize, _out: &DeviceTensor) -> Result<()> {
        unimplemented!()
    }

    fn silu_mul(&self, _gate: &DeviceTensor, _up: &DeviceTensor, _out: &DeviceTensor) -> Result<()> {
        unimplemented!()
    }

    fn embedding(&self, _token_ids: &[u32], _embedding_table: &DeviceTensor, _out: &DeviceTensor) -> Result<()> {
        unimplemented!()
    }

    fn add(&self, _a: &DeviceTensor, _b: &DeviceTensor, _out: &DeviceTensor) -> Result<()> {
        unimplemented!()
    }

    fn copy_rows(&self, _src: &DeviceTensor, _dst: &DeviceTensor, _src_offset: usize, _dst_offset: usize, _count: usize) -> Result<()> {
        unimplemented!()
    }

    fn device_name(&self) -> &str {
        "failing-mock"
    }

    fn total_memory(&self) -> usize {
        1 << 30
    }

    fn available_memory(&self) -> usize {
        1 << 30
    }

    fn synchronize(&self) -> Result<()> {
        Ok(())
    }

    fn create_timer(&self) -> Result<DeviceTimer> {
        Ok(DeviceTimer(0))
    }

    fn start_timer(&self, _timer: &DeviceTimer) -> Result<()> {
        Ok(())
    }

    fn stop_timer(&self, _timer: &DeviceTimer) -> Result<f32> {
        Ok(0.0)
    }

    fn destroy_timer(&self, _timer: &DeviceTimer) -> Result<()> {
        Ok(())
    }
}

#[test]
fn test_weight_loading_alignment() {
    // GGUF file alignment (default 32 bytes, spec mentions 256-byte alignment for some files)
    // is handled by the parser at the file level: tensor_data_offset is aligned so that
    // tensor data starts at a properly aligned position within the mmap. Individual tensor
    // offsets within tensor_data are relative to this aligned start.
    //
    // This alignment is a GGUF file format concern, NOT a GPU allocation concern.
    // cudaMalloc returns 256-byte aligned pointers by default — the backend handles
    // device memory alignment independently of GGUF file alignment.
    //
    // Verify that the parser sets tensor_data_offset to an aligned position.
    let data = build_complete_gguf(1, &[]);
    let (_dir, path) = write_gguf_to_file(&data);

    let gguf = crate::parser::GgufParser::parse(&path).unwrap();
    assert_eq!(
        gguf.tensor_data_offset % 32,
        0,
        "tensor_data_offset should be 32-byte aligned, got {}",
        gguf.tensor_data_offset
    );
}

#[test]
fn test_weight_tensor_size_validation() {
    // Verify that upload_tensor checks byte range against mmap bounds and that the
    // error message includes the tensor name. This is already triggered by
    // test_truncated_gguf, but here we add an explicit assertion about the tensor name.
    let mut data = build_complete_gguf(2, &[]);
    // Truncate significantly to ensure at least one tensor exceeds the mmap bounds
    let truncate_to = data.len().saturating_sub(1000);
    data.truncate(truncate_to);

    let (_dir, path) = write_gguf_to_file(&data);
    let backend = MockBackend::new();

    let result = WeightStore::load(&path, &backend, None);
    assert!(result.is_err());
    let err = result.err().unwrap();
    assert!(
        matches!(err, FractureError::WeightLoad(_)),
        "expected WeightLoad, got: {err}"
    );
    let msg = err.to_string();
    assert!(
        msg.contains("exceeds mmap"),
        "expected 'exceeds mmap' in: {msg}"
    );
    // The error should contain the name of the tensor that exceeded bounds
    // (some tensor name like "blk.1.ffn_down.weight" or similar)
    assert!(
        msg.contains(".weight") || msg.contains("token_embd") || msg.contains("output"),
        "expected tensor name in error message: {msg}"
    );
}

#[test]
fn test_weight_alloc_failure() {
    // Use FailingMockBackend that fails on the 3rd alloc call.
    // The first alloc is token_embedding, second is output_norm, third is lm_head.
    // Failing at alloc #3 (0-indexed: fail_at=2) should cause a Backend error
    // that propagates as a load failure.
    let data = build_complete_gguf(1, &[]);
    let (_dir, path) = write_gguf_to_file(&data);
    let backend = FailingMockBackend::new(2);

    let result = WeightStore::load(&path, &backend, None);
    assert!(result.is_err(), "expected alloc failure to propagate");
    let err = result.err().unwrap();
    assert!(
        matches!(err, FractureError::Backend(_)),
        "expected Backend error from failed alloc, got: {err}"
    );
}

#[test]
fn test_weight_name_to_field_mapping() {
    // Load a 1-layer GGUF with MockBackend, verify each LayerWeights field
    // has the correct shape from the GGUF tensor info.
    let data = build_complete_gguf(1, &[]);
    let (_dir, path) = write_gguf_to_file(&data);
    let backend = MockBackend::new();

    let store = WeightStore::load(&path, &backend, None).unwrap();
    assert_eq!(store.layers.len(), 1);
    let layer = &store.layers[0];

    // hidden=64, kv_dim=32 (num_kv_heads=2, head_dim=16), ffn=128
    assert_eq!(layer.q_proj.shape, vec![64, 64], "q_proj: [hidden, hidden]");
    assert_eq!(layer.k_proj.shape, vec![32, 64], "k_proj: [kv_dim, hidden]");
    assert_eq!(layer.v_proj.shape, vec![32, 64], "v_proj: [kv_dim, hidden]");
    assert_eq!(layer.o_proj.shape, vec![64, 64], "o_proj: [hidden, hidden]");
    assert_eq!(layer.gate_proj.shape, vec![128, 64], "gate_proj: [ffn, hidden]");
    assert_eq!(layer.up_proj.shape, vec![128, 64], "up_proj: [ffn, hidden]");
    assert_eq!(layer.down_proj.shape, vec![64, 128], "down_proj: [hidden, ffn]");
    assert_eq!(layer.attn_norm.shape, vec![64], "attn_norm: [hidden]");
    assert_eq!(layer.ffn_norm.shape, vec![64], "ffn_norm: [hidden]");
}

#[test]
fn test_weight_store_layer_range_validation() {
    // Load with layer_range=Some(0..10) on a 2-layer model.
    // Should fail because range end (10) exceeds layer count (2).
    let data = build_complete_gguf(2, &[]);
    let (_dir, path) = write_gguf_to_file(&data);
    let backend = MockBackend::new();

    let result = WeightStore::load(&path, &backend, Some(0..10));
    assert!(result.is_err());
    let err = result.err().unwrap();
    assert!(
        matches!(err, FractureError::WeightLoad(_)),
        "expected WeightLoad, got: {err}"
    );
    let msg = err.to_string();
    assert!(
        msg.contains("layer_range end 10") && msg.contains("2"),
        "expected error about range exceeding layer count: {msg}"
    );
}

/// A MockBackend that captures the bytes passed to copy_to_device for each tensor.
struct CapturingMockBackend {
    next_id: AtomicU64,
    /// Maps TensorId -> (shape, dtype, bytes) for each copy_to_device call.
    captured: std::sync::Mutex<HashMap<u64, (Vec<usize>, DType, Vec<u8>)>>,
}

impl CapturingMockBackend {
    fn new() -> Self {
        Self {
            next_id: AtomicU64::new(1),
            captured: std::sync::Mutex::new(HashMap::new()),
        }
    }

    fn get_captured(&self, id: u64) -> Option<(Vec<usize>, DType, Vec<u8>)> {
        self.captured.lock().unwrap().get(&id).cloned()
    }
}

impl Backend for CapturingMockBackend {
    fn alloc(&self, shape: &[usize], dtype: DType) -> Result<DeviceTensor> {
        let id = self.next_id.fetch_add(1, Ordering::SeqCst);
        // Pre-register with empty bytes so we know shape/dtype
        self.captured
            .lock()
            .unwrap()
            .insert(id, (shape.to_vec(), dtype, Vec::new()));
        Ok(DeviceTensor::new(TensorId(id), shape.to_vec(), dtype))
    }

    fn free(&self, _tensor: &DeviceTensor) -> Result<()> {
        Ok(())
    }

    fn copy_to_device(&self, dst: &DeviceTensor, src: &[u8]) -> Result<()> {
        let mut map = self.captured.lock().unwrap();
        if let Some(entry) = map.get_mut(&dst.id.0) {
            entry.2 = src.to_vec();
        }
        Ok(())
    }

    fn copy_to_host(&self, _src: &DeviceTensor, _dst: &mut [u8]) -> Result<()> {
        Ok(())
    }

    fn matmul(
        &self,
        _a: &DeviceTensor,
        _b: &DeviceTensor,
        _out: &DeviceTensor,
    ) -> Result<()> {
        unimplemented!()
    }

    fn rmsnorm(
        &self,
        _input: &DeviceTensor,
        _weight: &DeviceTensor,
        _eps: f64,
        _out: &DeviceTensor,
    ) -> Result<()> {
        unimplemented!()
    }

    fn rope(
        &self,
        _q: &DeviceTensor,
        _k: &DeviceTensor,
        _positions: &[u32],
        _theta: f64,
        _head_dim: usize,
    ) -> Result<()> {
        unimplemented!()
    }

    fn attention(
        &self,
        _q: &DeviceTensor,
        _k_cache: &DeviceTensor,
        _v_cache: &DeviceTensor,
        _num_kv_heads: usize,
        _start_pos: usize,
        _out: &DeviceTensor,
    ) -> Result<()> {
        unimplemented!()
    }

    fn silu_mul(
        &self,
        _gate: &DeviceTensor,
        _up: &DeviceTensor,
        _out: &DeviceTensor,
    ) -> Result<()> {
        unimplemented!()
    }

    fn embedding(
        &self,
        _token_ids: &[u32],
        _embedding_table: &DeviceTensor,
        _out: &DeviceTensor,
    ) -> Result<()> {
        unimplemented!()
    }

    fn add(
        &self,
        _a: &DeviceTensor,
        _b: &DeviceTensor,
        _out: &DeviceTensor,
    ) -> Result<()> {
        unimplemented!()
    }

    fn copy_rows(
        &self,
        _src: &DeviceTensor,
        _dst: &DeviceTensor,
        _src_offset: usize,
        _dst_offset: usize,
        _count: usize,
    ) -> Result<()> {
        unimplemented!()
    }

    fn device_name(&self) -> &str {
        "capturing-mock"
    }

    fn total_memory(&self) -> usize {
        1 << 30
    }

    fn available_memory(&self) -> usize {
        1 << 30
    }

    fn synchronize(&self) -> Result<()> {
        Ok(())
    }

    fn create_timer(&self) -> Result<DeviceTimer> {
        Ok(DeviceTimer(0))
    }

    fn start_timer(&self, _timer: &DeviceTimer) -> Result<()> {
        Ok(())
    }

    fn stop_timer(&self, _timer: &DeviceTimer) -> Result<f32> {
        Ok(0.0)
    }

    fn destroy_timer(&self, _timer: &DeviceTimer) -> Result<()> {
        Ok(())
    }
}

/// Build a GGUF where norm weights (attn_norm, ffn_norm, output_norm) are stored as F32
/// while all other tensors remain FP16.
fn build_gguf_with_f32_norms(num_layers: usize) -> Vec<u8> {
    let hidden: u64 = 64;
    let kv_dim: u64 = 32;
    let ffn: u64 = 128;
    let vocab: u64 = 4;

    struct TensorSpec {
        name: String,
        shape: Vec<u64>,
        dtype_code: u32, // 0=FP32, 1=FP16
        byte_size: u64,
    }

    let mut tensors: Vec<TensorSpec> = Vec::new();

    // Global tensors
    tensors.push(TensorSpec {
        name: "token_embd.weight".into(),
        shape: vec![vocab, hidden],
        dtype_code: 1,
        byte_size: vocab * hidden * 2,
    });
    // output_norm is F32
    tensors.push(TensorSpec {
        name: "output_norm.weight".into(),
        shape: vec![hidden],
        dtype_code: 0, // F32
        byte_size: hidden * 4,
    });
    tensors.push(TensorSpec {
        name: "output.weight".into(),
        shape: vec![vocab, hidden],
        dtype_code: 1,
        byte_size: vocab * hidden * 2,
    });

    for layer in 0..num_layers {
        // FP16 tensors
        for (suffix, shape) in &[
            ("attn_q.weight", vec![hidden, hidden]),
            ("attn_k.weight", vec![kv_dim, hidden]),
            ("attn_v.weight", vec![kv_dim, hidden]),
            ("attn_output.weight", vec![hidden, hidden]),
            ("ffn_gate.weight", vec![ffn, hidden]),
            ("ffn_up.weight", vec![ffn, hidden]),
            ("ffn_down.weight", vec![hidden, ffn]),
        ] {
            let numel: u64 = shape.iter().product();
            tensors.push(TensorSpec {
                name: format!("blk.{layer}.{suffix}"),
                shape: shape.clone(),
                dtype_code: 1,
                byte_size: numel * 2,
            });
        }
        // F32 norm tensors
        for suffix in &["attn_norm.weight", "ffn_norm.weight"] {
            tensors.push(TensorSpec {
                name: format!("blk.{layer}.{suffix}"),
                shape: vec![hidden],
                dtype_code: 0, // F32
                byte_size: hidden * 4,
            });
        }
    }

    // Compute offsets
    let mut offsets: Vec<u64> = Vec::new();
    let mut data_size: u64 = 0;
    for t in &tensors {
        offsets.push(data_size);
        data_size += t.byte_size;
    }

    let tensor_count = tensors.len();
    let metadata_count = 8;

    let mut buf = Vec::new();

    buf.write_u32::<LittleEndian>(0x46554747).unwrap();
    buf.write_u32::<LittleEndian>(3).unwrap();
    buf.write_u64::<LittleEndian>(tensor_count as u64).unwrap();
    buf.write_u64::<LittleEndian>(metadata_count as u64).unwrap();

    write_metadata_kv_string(&mut buf, "general.architecture", "llama");
    write_metadata_kv_u32(&mut buf, "llama.embedding_length", hidden as u32);
    write_metadata_kv_u32(&mut buf, "llama.block_count", num_layers as u32);
    write_metadata_kv_u32(&mut buf, "llama.attention.head_count", 4);
    write_metadata_kv_u32(&mut buf, "llama.attention.head_count_kv", 2);
    write_metadata_kv_u32(&mut buf, "llama.feed_forward_length", ffn as u32);
    write_metadata_kv_f32(&mut buf, "llama.rope.freq_base", 500000.0);
    write_metadata_kv_string_array(
        &mut buf,
        "tokenizer.ggml.tokens",
        &["a", "b", "c", "d"],
    );

    for (i, t) in tensors.iter().enumerate() {
        write_tensor_info(&mut buf, &t.name, &t.shape, t.dtype_code, offsets[i]);
    }

    let current = buf.len();
    let aligned = align_offset(current, 32);
    buf.resize(aligned, 0);

    // Write tensor data. For F32 norm tensors, write known f32 values.
    let data_start = buf.len();
    buf.extend(vec![0u8; data_size as usize]);

    // Write known F32 values into the output_norm and per-layer norm tensors
    // We'll write [1.0, 2.0, 3.0, ...] as f32 into each F32 tensor
    for (i, t) in tensors.iter().enumerate() {
        if t.dtype_code == 0 {
            // F32 tensor
            let start = data_start + offsets[i] as usize;
            let numel = (t.byte_size / 4) as usize;
            for j in 0..numel {
                let val = (j + 1) as f32;
                let bytes = val.to_le_bytes();
                let off = start + j * 4;
                buf[off..off + 4].copy_from_slice(&bytes);
            }
        }
    }

    buf
}

#[test]
fn test_f32_to_fp16_conversion() {
    let data = build_gguf_with_f32_norms(1);
    let (_dir, path) = write_gguf_to_file(&data);
    let backend = CapturingMockBackend::new();

    let store = WeightStore::load(&path, &backend, None).unwrap();

    // output_norm should have been converted from F32 to FP16
    let norm_id = store.output_norm.id.0;
    let (shape, dtype, bytes) = backend.get_captured(norm_id).unwrap();
    assert_eq!(shape, vec![64]);
    assert_eq!(dtype, DType::FP16, "output_norm should be allocated as FP16");
    // 64 elements * 2 bytes = 128 bytes (FP16), NOT 64*4=256 (FP32)
    assert_eq!(bytes.len(), 64 * 2, "expected FP16 byte count (128), got {}", bytes.len());

    // Verify the FP16 values are correct conversions from [1.0, 2.0, ..., 64.0]
    for i in 0..64usize {
        let expected_f32 = (i + 1) as f32;
        let expected_f16 = half::f16::from_f32(expected_f32);
        let actual_bytes = [bytes[i * 2], bytes[i * 2 + 1]];
        let actual_f16 = half::f16::from_le_bytes(actual_bytes);
        assert_eq!(
            actual_f16, expected_f16,
            "element {i}: expected f16({expected_f32}) = {expected_f16}, got {actual_f16}"
        );
    }

    // Also check per-layer norms
    let layer = &store.layers[0];
    let attn_norm_id = layer.attn_norm.id.0;
    let (_, dtype, bytes) = backend.get_captured(attn_norm_id).unwrap();
    assert_eq!(dtype, DType::FP16, "attn_norm should be FP16");
    assert_eq!(bytes.len(), 64 * 2);

    let ffn_norm_id = layer.ffn_norm.id.0;
    let (_, dtype, bytes) = backend.get_captured(ffn_norm_id).unwrap();
    assert_eq!(dtype, DType::FP16, "ffn_norm should be FP16");
    assert_eq!(bytes.len(), 64 * 2);
}

/// Build a minimal GGUF where norm tensors use BF16 (dtype_code=30) instead of F32.
fn build_gguf_with_bf16_norms(num_layers: usize) -> Vec<u8> {
    let hidden: u64 = 64;
    let kv_dim: u64 = 32;
    let ffn: u64 = 128;
    let vocab: u64 = 4;

    struct TensorSpec {
        name: String,
        shape: Vec<u64>,
        dtype_code: u32,
        byte_size: u64,
    }

    let mut tensors: Vec<TensorSpec> = Vec::new();

    // Global tensors
    tensors.push(TensorSpec {
        name: "token_embd.weight".into(),
        shape: vec![vocab, hidden],
        dtype_code: 1, // FP16
        byte_size: vocab * hidden * 2,
    });
    // output_norm is BF16
    tensors.push(TensorSpec {
        name: "output_norm.weight".into(),
        shape: vec![hidden],
        dtype_code: 30, // BF16
        byte_size: hidden * 2,
    });
    tensors.push(TensorSpec {
        name: "output.weight".into(),
        shape: vec![vocab, hidden],
        dtype_code: 1,
        byte_size: vocab * hidden * 2,
    });

    for layer in 0..num_layers {
        // FP16 tensors
        for (suffix, shape) in &[
            ("attn_q.weight", vec![hidden, hidden]),
            ("attn_k.weight", vec![kv_dim, hidden]),
            ("attn_v.weight", vec![kv_dim, hidden]),
            ("attn_output.weight", vec![hidden, hidden]),
            ("ffn_gate.weight", vec![ffn, hidden]),
            ("ffn_up.weight", vec![ffn, hidden]),
            ("ffn_down.weight", vec![hidden, ffn]),
        ] {
            let numel: u64 = shape.iter().product();
            tensors.push(TensorSpec {
                name: format!("blk.{layer}.{suffix}"),
                shape: shape.clone(),
                dtype_code: 1,
                byte_size: numel * 2,
            });
        }
        // BF16 norm tensors
        for suffix in &["attn_norm.weight", "ffn_norm.weight"] {
            tensors.push(TensorSpec {
                name: format!("blk.{layer}.{suffix}"),
                shape: vec![hidden],
                dtype_code: 30, // BF16
                byte_size: hidden * 2,
            });
        }
    }

    // Compute offsets
    let mut offsets: Vec<u64> = Vec::new();
    let mut data_size: u64 = 0;
    for t in &tensors {
        offsets.push(data_size);
        data_size += t.byte_size;
    }

    let tensor_count = tensors.len();
    let metadata_count = 8;

    let mut buf = Vec::new();

    buf.write_u32::<LittleEndian>(0x46554747).unwrap();
    buf.write_u32::<LittleEndian>(3).unwrap();
    buf.write_u64::<LittleEndian>(tensor_count as u64).unwrap();
    buf.write_u64::<LittleEndian>(metadata_count as u64).unwrap();

    write_metadata_kv_string(&mut buf, "general.architecture", "llama");
    write_metadata_kv_u32(&mut buf, "llama.embedding_length", hidden as u32);
    write_metadata_kv_u32(&mut buf, "llama.block_count", num_layers as u32);
    write_metadata_kv_u32(&mut buf, "llama.attention.head_count", 4);
    write_metadata_kv_u32(&mut buf, "llama.attention.head_count_kv", 2);
    write_metadata_kv_u32(&mut buf, "llama.feed_forward_length", ffn as u32);
    write_metadata_kv_f32(&mut buf, "llama.rope.freq_base", 500000.0);
    write_metadata_kv_string_array(
        &mut buf,
        "tokenizer.ggml.tokens",
        &["a", "b", "c", "d"],
    );

    for (i, t) in tensors.iter().enumerate() {
        write_tensor_info(&mut buf, &t.name, &t.shape, t.dtype_code, offsets[i]);
    }

    let current = buf.len();
    let aligned = align_offset(current, 32);
    buf.resize(aligned, 0);

    // Write tensor data. For BF16 norm tensors, write known bf16 values.
    let data_start = buf.len();
    buf.extend(vec![0u8; data_size as usize]);

    // Write known BF16 values [1.0, 2.0, 3.0, ...] into each BF16 tensor
    for (i, t) in tensors.iter().enumerate() {
        if t.dtype_code == 30 {
            let start = data_start + offsets[i] as usize;
            let numel = (t.byte_size / 2) as usize;
            for j in 0..numel {
                let val = (j + 1) as f32;
                let bf = half::bf16::from_f32(val);
                let bytes = bf.to_le_bytes();
                let off = start + j * 2;
                buf[off..off + 2].copy_from_slice(&bytes);
            }
        }
    }

    buf
}

#[test]
fn test_bf16_to_fp16_conversion() {
    let data = build_gguf_with_bf16_norms(1);
    let (_dir, path) = write_gguf_to_file(&data);
    let backend = CapturingMockBackend::new();

    let store = WeightStore::load(&path, &backend, None).unwrap();

    // output_norm should have been converted from BF16 to FP16
    let norm_id = store.output_norm.id.0;
    let (shape, dtype, bytes) = backend.get_captured(norm_id).unwrap();
    assert_eq!(shape, vec![64]);
    assert_eq!(dtype, DType::FP16, "output_norm should be allocated as FP16");
    // 64 elements * 2 bytes = 128 bytes (FP16)
    assert_eq!(bytes.len(), 64 * 2, "expected FP16 byte count (128), got {}", bytes.len());

    // Verify the FP16 values are correct conversions from BF16 [1.0, 2.0, ..., 64.0]
    for i in 0..64usize {
        let expected_f32 = (i + 1) as f32;
        // BF16→F32→FP16 round-trip
        let bf = half::bf16::from_f32(expected_f32);
        let expected_f16 = half::f16::from_f32(bf.to_f32());
        let actual_bytes = [bytes[i * 2], bytes[i * 2 + 1]];
        let actual_f16 = half::f16::from_le_bytes(actual_bytes);
        assert_eq!(
            actual_f16, expected_f16,
            "element {i}: expected f16 from bf16({expected_f32}) = {expected_f16}, got {actual_f16}"
        );
    }

    // Also check per-layer norms
    let layer = &store.layers[0];
    let attn_norm_id = layer.attn_norm.id.0;
    let (_, dtype, bytes) = backend.get_captured(attn_norm_id).unwrap();
    assert_eq!(dtype, DType::FP16, "attn_norm should be FP16");
    assert_eq!(bytes.len(), 64 * 2);

    let ffn_norm_id = layer.ffn_norm.id.0;
    let (_, dtype, bytes) = backend.get_captured(ffn_norm_id).unwrap();
    assert_eq!(dtype, DType::FP16, "ffn_norm should be FP16");
    assert_eq!(bytes.len(), 64 * 2);
}

#[test]
fn test_reverse_qk_permutation_correctness() {
    // Test with 2 heads, head_dim=4, hidden=8
    // Each head has head_dim=4 rows of hidden*2=16 bytes each
    // head_bytes = head_dim * row_bytes = 4 * 16 = 64
    // total = 2 heads * 64 = 128 bytes
    let num_heads = 2;
    let head_dim = 4;
    let hidden = 8;
    let _half_dim = head_dim / 2; // 2
    let row_bytes = hidden * 2; // 16 bytes per row (FP16)
    let total_bytes = num_heads * head_dim * row_bytes; // 128

    // Create interleaved input where:
    //   even rows (0, 2) contain first-half data
    //   odd rows (1, 3) contain second-half data
    // For head 0:
    //   row 0 (even, i=0, first-half row 0): fill with 0x10
    //   row 1 (odd,  i=0, second-half row 0): fill with 0x20
    //   row 2 (even, i=1, first-half row 1): fill with 0x30
    //   row 3 (odd,  i=1, second-half row 1): fill with 0x40
    // For head 1:
    //   row 0: 0x50, row 1: 0x60, row 2: 0x70, row 3: 0x80
    let mut input = vec![0u8; total_bytes];
    let fill_values: [u8; 8] = [0x10, 0x20, 0x30, 0x40, 0x50, 0x60, 0x70, 0x80];
    for (row_idx, &val) in fill_values.iter().enumerate() {
        let start = row_idx * row_bytes;
        for b in &mut input[start..start + row_bytes] {
            *b = val;
        }
    }

    let output = reverse_qk_permutation(&input, num_heads, head_dim, hidden);
    assert_eq!(output.len(), total_bytes);

    // Expected output after de-interleaving:
    // Head 0 (rows 0-3 of output):
    //   row 0 = first-half row 0 (from even row 0) = 0x10
    //   row 1 = first-half row 1 (from even row 2) = 0x30
    //   row 2 = second-half row 0 (from odd row 1) = 0x20
    //   row 3 = second-half row 1 (from odd row 3) = 0x40
    // Head 1 (rows 4-7 of output):
    //   row 4 = 0x50, row 5 = 0x70, row 6 = 0x60, row 7 = 0x80
    let expected_vals: [u8; 8] = [0x10, 0x30, 0x20, 0x40, 0x50, 0x70, 0x60, 0x80];
    for (row_idx, &expected_val) in expected_vals.iter().enumerate() {
        let start = row_idx * row_bytes;
        for (byte_idx, &b) in output[start..start + row_bytes].iter().enumerate() {
            assert_eq!(
                b, expected_val,
                "row {row_idx}, byte {byte_idx}: expected 0x{expected_val:02x}, got 0x{b:02x}"
            );
        }
    }
}

#[test]
fn test_weight_store_missing_global_tensor() {
    // Missing token_embd.weight
    let data = build_gguf_without(1, &["token_embd.weight"]);
    let (_dir, path) = write_gguf_to_file(&data);
    let backend = MockBackend::new();

    let result = WeightStore::load(&path, &backend, None);
    assert!(result.is_err());
    let err = result.err().unwrap();
    assert!(
        matches!(err, FractureError::WeightLoad(_)),
        "expected WeightLoad, got: {err}"
    );
    assert!(
        err.to_string().contains("token_embd.weight"),
        "expected mention of token_embd.weight in: {err}"
    );

    // Missing output_norm.weight
    let data = build_gguf_without(1, &["output_norm.weight"]);
    let (_dir, path) = write_gguf_to_file(&data);
    let backend = MockBackend::new();

    let result = WeightStore::load(&path, &backend, None);
    assert!(result.is_err());
    let err = result.err().unwrap();
    assert!(
        matches!(err, FractureError::WeightLoad(_)),
        "expected WeightLoad, got: {err}"
    );
    assert!(
        err.to_string().contains("output_norm.weight"),
        "expected mention of output_norm.weight in: {err}"
    );

    // Missing output.weight
    let data = build_gguf_without(1, &["output.weight"]);
    let (_dir, path) = write_gguf_to_file(&data);
    let backend = MockBackend::new();

    let result = WeightStore::load(&path, &backend, None);
    assert!(result.is_err());
    let err = result.err().unwrap();
    assert!(
        matches!(err, FractureError::WeightLoad(_)),
        "expected WeightLoad, got: {err}"
    );
    assert!(
        err.to_string().contains("output.weight"),
        "expected mention of output.weight in: {err}"
    );
}

#[test]
fn test_truncated_gguf() {
    let mut data = build_complete_gguf(2, &[]);
    // Truncate: remove last 1000 bytes of tensor data
    let truncate_to = data.len().saturating_sub(1000);
    data.truncate(truncate_to);

    let (_dir, path) = write_gguf_to_file(&data);
    let backend = MockBackend::new();

    let result = WeightStore::load(&path, &backend, None);
    assert!(result.is_err());
    let err = result.err().unwrap();
    assert!(
        matches!(err, FractureError::WeightLoad(_)),
        "expected WeightLoad, got: {err}"
    );
    assert!(
        err.to_string().contains("exceeds mmap"),
        "expected 'exceeds mmap' in: {err}"
    );
}

/// Build a GGUF identical to build_complete_gguf(1, &[]) but with known non-zero
/// FP16 data in the q_proj and k_proj tensors of layer 0. Returns the buffer plus
/// the byte ranges [q_start, q_end) and [k_start, k_end) relative to the start of
/// the tensor data section (i.e. relative to `tensor_data_offset`).
///
/// Config: hidden=64, num_q_heads=4, head_dim=16, num_kv_heads=2, ffn=128, vocab=4.
/// q shape: [64, 64], k shape: [32, 64] (both FP16).
fn build_gguf_with_nonzero_qk() -> (Vec<u8>, usize, usize, usize, usize) {
    let hidden: u64 = 64;
    let kv_dim: u64 = 32; // num_kv_heads * head_dim = 2 * 16
    let ffn: u64 = 128;
    let vocab: u64 = 4;

    struct TensorSpec {
        name: String,
        shape: Vec<u64>,
    }

    let mut tensors: Vec<TensorSpec> = Vec::new();
    tensors.push(TensorSpec { name: "token_embd.weight".into(), shape: vec![vocab, hidden] });
    tensors.push(TensorSpec { name: "output_norm.weight".into(), shape: vec![hidden] });
    tensors.push(TensorSpec { name: "output.weight".into(), shape: vec![vocab, hidden] });

    // Layer 0 tensors in order they appear in the file
    let layer_defs: Vec<(&str, Vec<u64>)> = vec![
        ("attn_q.weight",    vec![hidden, hidden]),
        ("attn_k.weight",    vec![kv_dim, hidden]),
        ("attn_v.weight",    vec![kv_dim, hidden]),
        ("attn_output.weight", vec![hidden, hidden]),
        ("ffn_gate.weight",  vec![ffn, hidden]),
        ("ffn_up.weight",    vec![ffn, hidden]),
        ("ffn_down.weight",  vec![hidden, ffn]),
        ("attn_norm.weight", vec![hidden]),
        ("ffn_norm.weight",  vec![hidden]),
    ];
    for (suffix, shape) in &layer_defs {
        tensors.push(TensorSpec { name: format!("blk.0.{suffix}"), shape: shape.clone() });
    }

    // Compute offsets (all FP16, 2 bytes each)
    let mut offsets: Vec<u64> = Vec::new();
    let mut data_size: u64 = 0;
    for t in &tensors {
        offsets.push(data_size);
        let numel: u64 = t.shape.iter().product();
        data_size += numel * 2;
    }

    let tensor_count = tensors.len();
    let metadata_count: u64 = 8;

    let mut buf = Vec::new();
    buf.write_u32::<LittleEndian>(0x46554747).unwrap();
    buf.write_u32::<LittleEndian>(3).unwrap();
    buf.write_u64::<LittleEndian>(tensor_count as u64).unwrap();
    buf.write_u64::<LittleEndian>(metadata_count).unwrap();

    write_metadata_kv_string(&mut buf, "general.architecture", "llama");
    write_metadata_kv_u32(&mut buf, "llama.embedding_length", hidden as u32);
    write_metadata_kv_u32(&mut buf, "llama.block_count", 1);
    write_metadata_kv_u32(&mut buf, "llama.attention.head_count", 4);
    write_metadata_kv_u32(&mut buf, "llama.attention.head_count_kv", 2);
    write_metadata_kv_u32(&mut buf, "llama.feed_forward_length", ffn as u32);
    write_metadata_kv_f32(&mut buf, "llama.rope.freq_base", 500000.0);
    write_metadata_kv_string_array(&mut buf, "tokenizer.ggml.tokens", &["a", "b", "c", "d"]);

    for (i, t) in tensors.iter().enumerate() {
        write_tensor_info(&mut buf, &t.name, &t.shape, 1, offsets[i]);
    }

    let current = buf.len();
    let tensor_data_offset = align_offset(current, 32);
    buf.resize(tensor_data_offset, 0);

    // Append all-zeros tensor data initially
    buf.extend(vec![0u8; data_size as usize]);

    // Write distinct non-zero FP16 patterns into q and k tensors.
    // q is tensor index 3 (after token_embd, output_norm, output), k is index 4.
    let q_idx = 3usize;
    let k_idx = 4usize;

    let q_start = tensor_data_offset + offsets[q_idx] as usize;
    let q_numel = tensors[q_idx].shape.iter().product::<u64>() as usize;
    let _q_end = q_start + q_numel * 2;

    let k_start = tensor_data_offset + offsets[k_idx] as usize;
    let k_numel = tensors[k_idx].shape.iter().product::<u64>() as usize;
    let _k_end = k_start + k_numel * 2;

    // Fill q bytes: each row gets a unique repeating byte pattern based on row index.
    // row_bytes = hidden * 2 = 128, num_rows = hidden = 64
    let q_row_bytes = (hidden as usize) * 2;
    let q_rows = hidden as usize;
    for row in 0..q_rows {
        let fill = ((row + 1) as u8).wrapping_mul(3);
        let start = q_start + row * q_row_bytes;
        for b in &mut buf[start..start + q_row_bytes] {
            *b = fill;
        }
    }

    // Fill k bytes: each row gets a unique repeating byte pattern based on row index.
    // row_bytes = hidden * 2 = 128, num_rows = kv_dim = 32
    let k_row_bytes = (hidden as usize) * 2;
    let k_rows = kv_dim as usize;
    for row in 0..k_rows {
        let fill = ((row + 1) as u8).wrapping_mul(5);
        let start = k_start + row * k_row_bytes;
        for b in &mut buf[start..start + k_row_bytes] {
            *b = fill;
        }
    }

    // Return: (bytes, q_raw_start_relative_to_tdo, q_raw_end_relative_to_tdo, same for k)
    let q_rel_start = offsets[q_idx] as usize;
    let q_rel_end = q_rel_start + q_numel * 2;
    let k_rel_start = offsets[k_idx] as usize;
    let k_rel_end = k_rel_start + k_numel * 2;

    (buf, q_rel_start, q_rel_end, k_rel_start, k_rel_end)
}

#[test]
fn test_qk_permutation_applied_during_load() {
    // Config: hidden=64, num_q_heads=4, head_dim=16, num_kv_heads=2, ffn=128, vocab=4
    let (data, q_rel_start, q_rel_end, k_rel_start, k_rel_end) =
        build_gguf_with_nonzero_qk();
    let (_dir, path) = write_gguf_to_file(&data);
    let backend = CapturingMockBackend::new();

    let store = WeightStore::load(&path, &backend, None).unwrap();
    assert_eq!(store.layers.len(), 1);

    let layer = &store.layers[0];

    // Determine tensor_data_offset from the file (we know it's 32-byte aligned past header).
    // Re-derive it from the data we wrote — use GgufParser directly.
    let gguf = crate::parser::GgufParser::parse(&path).unwrap();
    let tdo = gguf.tensor_data_offset;
    let mmap_bytes = &gguf.mmap[..];

    // Fetch the raw GGUF bytes for q and k (relative to tensor_data_offset)
    let q_raw = &mmap_bytes[tdo + q_rel_start..tdo + q_rel_end];
    let k_raw = &mmap_bytes[tdo + k_rel_start..tdo + k_rel_end];

    // Compute the expected permuted output using the same function WeightStore calls
    let num_q_heads = 4usize;
    let num_kv_heads = 2usize;
    let head_dim = 16usize;
    let hidden = 64usize;

    let expected_q = reverse_qk_permutation(q_raw, num_q_heads, head_dim, hidden);
    let expected_k = reverse_qk_permutation(k_raw, num_kv_heads, head_dim, hidden);

    // The raw input is non-zero, so the permuted output should differ from raw for non-trivial
    // permutations. Verify by checking at least one row that differs.
    let row_bytes = hidden * 2; // 128 bytes per row in the weight matrix
    let half = head_dim / 2; // 8
    // In GGUF interleaved layout, head 0 row 0 (even) maps to original row 0.
    // Head 0 row 1 (odd) maps to original row half (=8).
    // For our fill pattern: GGUF row 0 = fill 3, GGUF row 1 = fill 6.
    // After de-interleaving: original row 0 = fill 3 (from GGUF row 0), original row 8 = fill 6.
    // GGUF row 2 (even, i=1) -> original row 1: fill 9.
    // So expected_q row 0 should all be 0x03, row 8 should all be 0x06.
    assert!(
        expected_q[0..row_bytes].iter().all(|&b| b == 0x03),
        "expected row 0 of de-interleaved q to be all 0x03"
    );
    assert!(
        expected_q[half * row_bytes..(half + 1) * row_bytes].iter().all(|&b| b == 0x06),
        "expected row {} of de-interleaved q to be all 0x06 (from GGUF odd row 1)",
        half
    );

    // Get captured bytes for q_proj and k_proj
    let q_id = layer.q_proj.id.0;
    let k_id = layer.k_proj.id.0;

    let (_, _, captured_q_bytes) = backend.get_captured(q_id).unwrap();
    let (_, _, captured_k_bytes) = backend.get_captured(k_id).unwrap();

    // The captured bytes should match the expected permuted output exactly.
    assert_eq!(
        captured_q_bytes, expected_q,
        "q_proj bytes should match de-interleaved permutation output"
    );
    assert_eq!(
        captured_k_bytes, expected_k,
        "k_proj bytes should match de-interleaved permutation output"
    );

    // Confirm the raw bytes differ from the permuted bytes (proves the permutation did something)
    assert_ne!(
        captured_q_bytes, q_raw,
        "q_proj captured bytes should differ from raw GGUF bytes (permutation was applied)"
    );
}

#[test]
fn test_weight_store_layer_range_loads_globals() {
    // Load a 2-layer GGUF with layer_range=Some(0..1) and verify that global tensors
    // (token_embedding, output_norm, lm_head) are always present with correct shapes.
    // Config: hidden=64, vocab=4.
    let data = build_complete_gguf(2, &[]);
    let (_dir, path) = write_gguf_to_file(&data);
    let backend = MockBackend::new();

    let store = WeightStore::load(&path, &backend, Some(0..1)).unwrap();

    // Only 1 layer loaded
    assert_eq!(store.layers.len(), 1, "expected 1 layer with layer_range 0..1");

    // Global tensors must be present regardless of layer_range
    assert_eq!(
        store.token_embedding.shape,
        vec![4, 64],
        "token_embedding shape should be [vocab=4, hidden=64]"
    );
    assert_eq!(
        store.output_norm.shape,
        vec![64],
        "output_norm shape should be [hidden=64]"
    );
    assert_eq!(
        store.lm_head.shape,
        vec![4, 64],
        "lm_head shape should be [vocab=4, hidden=64]"
    );

    // Layer 0's weights are also present
    let layer = &store.layers[0];
    assert_eq!(layer.q_proj.shape, vec![64, 64], "q_proj shape");
    assert_eq!(layer.k_proj.shape, vec![32, 64], "k_proj shape");
}

/// Build a GGUF with `general.alignment=64` in metadata.
/// Identical to build_complete_gguf(1, &[]) except for the extra alignment key
/// and the tensor data section aligned to 64 bytes.
fn build_gguf_with_custom_alignment(alignment: u32) -> Vec<u8> {
    let hidden: u64 = 64;
    let kv_dim: u64 = 32;
    let ffn: u64 = 128;
    let vocab: u64 = 4;

    struct TensorSpec {
        name: String,
        shape: Vec<u64>,
    }

    let mut tensors: Vec<TensorSpec> = Vec::new();
    tensors.push(TensorSpec { name: "token_embd.weight".into(), shape: vec![vocab, hidden] });
    tensors.push(TensorSpec { name: "output_norm.weight".into(), shape: vec![hidden] });
    tensors.push(TensorSpec { name: "output.weight".into(), shape: vec![vocab, hidden] });

    let layer_defs: Vec<(&str, Vec<u64>)> = vec![
        ("attn_q.weight",    vec![hidden, hidden]),
        ("attn_k.weight",    vec![kv_dim, hidden]),
        ("attn_v.weight",    vec![kv_dim, hidden]),
        ("attn_output.weight", vec![hidden, hidden]),
        ("ffn_gate.weight",  vec![ffn, hidden]),
        ("ffn_up.weight",    vec![ffn, hidden]),
        ("ffn_down.weight",  vec![hidden, ffn]),
        ("attn_norm.weight", vec![hidden]),
        ("ffn_norm.weight",  vec![hidden]),
    ];
    for (suffix, shape) in &layer_defs {
        tensors.push(TensorSpec { name: format!("blk.0.{suffix}"), shape: shape.clone() });
    }

    let mut offsets: Vec<u64> = Vec::new();
    let mut data_size: u64 = 0;
    for t in &tensors {
        offsets.push(data_size);
        let numel: u64 = t.shape.iter().product();
        data_size += numel * 2; // FP16
    }

    let tensor_count = tensors.len();
    // 9 metadata entries: the standard 8 plus general.alignment
    let metadata_count: u64 = 9;

    let mut buf = Vec::new();
    buf.write_u32::<LittleEndian>(0x46554747).unwrap();
    buf.write_u32::<LittleEndian>(3).unwrap();
    buf.write_u64::<LittleEndian>(tensor_count as u64).unwrap();
    buf.write_u64::<LittleEndian>(metadata_count).unwrap();

    write_metadata_kv_string(&mut buf, "general.architecture", "llama");
    write_metadata_kv_u32(&mut buf, "llama.embedding_length", hidden as u32);
    write_metadata_kv_u32(&mut buf, "llama.block_count", 1);
    write_metadata_kv_u32(&mut buf, "llama.attention.head_count", 4);
    write_metadata_kv_u32(&mut buf, "llama.attention.head_count_kv", 2);
    write_metadata_kv_u32(&mut buf, "llama.feed_forward_length", ffn as u32);
    write_metadata_kv_f32(&mut buf, "llama.rope.freq_base", 500000.0);
    write_metadata_kv_string_array(&mut buf, "tokenizer.ggml.tokens", &["a", "b", "c", "d"]);
    // The alignment metadata key
    write_metadata_kv_u32(&mut buf, "general.alignment", alignment);

    for (i, t) in tensors.iter().enumerate() {
        write_tensor_info(&mut buf, &t.name, &t.shape, 1, offsets[i]);
    }

    // Align to the custom alignment value
    let current = buf.len();
    let aligned = align_offset(current, alignment as usize);
    buf.resize(aligned, 0);

    buf.extend(vec![0u8; data_size as usize]);

    buf
}

#[test]
fn test_custom_alignment() {
    // Build a GGUF with general.alignment=64 and verify the parser respects it:
    // tensor_data_offset must be 64-byte aligned, and WeightStore::load must succeed
    // (i.e. all tensor byte ranges are valid under the 64-byte-aligned data section).
    let alignment = 64u32;
    let data = build_gguf_with_custom_alignment(alignment);
    let (_dir, path) = write_gguf_to_file(&data);

    // Verify parser uses the custom alignment
    let gguf = crate::parser::GgufParser::parse(&path).unwrap();
    assert_eq!(
        gguf.tensor_data_offset % alignment as usize,
        0,
        "tensor_data_offset {} should be {}-byte aligned",
        gguf.tensor_data_offset,
        alignment
    );

    // Verify WeightStore::load succeeds and shapes are correct
    let backend = MockBackend::new();
    let store = WeightStore::load(&path, &backend, None).unwrap();

    assert_eq!(store.config.num_layers, 1);
    assert_eq!(store.config.hidden_size, 64);
    assert_eq!(store.layers.len(), 1);

    // Global tensors accessible
    assert_eq!(store.token_embedding.shape, vec![4, 64]);
    assert_eq!(store.output_norm.shape, vec![64]);
    assert_eq!(store.lm_head.shape, vec![4, 64]);

    // Layer tensors accessible
    let layer = &store.layers[0];
    assert_eq!(layer.q_proj.shape, vec![64, 64]);
    assert_eq!(layer.k_proj.shape, vec![32, 64]);
    assert_eq!(layer.attn_norm.shape, vec![64]);
}
