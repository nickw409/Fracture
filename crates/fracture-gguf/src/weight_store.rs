use std::collections::HashMap;

use fracture_core::{Backend, DeviceTensor, FractureError, ModelConfig, Result};

use crate::parser::{GgufParser, TensorInfo};

/// Per-layer weight tensors on device.
pub struct LayerWeights {
    pub q_proj: DeviceTensor,
    pub k_proj: DeviceTensor,
    pub v_proj: DeviceTensor,
    pub o_proj: DeviceTensor,
    pub gate_proj: DeviceTensor,
    pub up_proj: DeviceTensor,
    pub down_proj: DeviceTensor,
    pub attn_norm: DeviceTensor,
    pub ffn_norm: DeviceTensor,
}

/// Holds all model weights on device, organized for fast access during inference.
pub struct WeightStore {
    pub config: ModelConfig,
    pub token_embedding: DeviceTensor,
    pub layers: Vec<LayerWeights>,
    pub output_norm: DeviceTensor,
    pub lm_head: DeviceTensor,
}

/// Intermediate builder for a single layer's weights during loading.
/// Fields are Option so we can populate them incrementally as we encounter tensors.
struct LayerWeightsBuilder {
    q_proj: Option<DeviceTensor>,
    k_proj: Option<DeviceTensor>,
    v_proj: Option<DeviceTensor>,
    o_proj: Option<DeviceTensor>,
    gate_proj: Option<DeviceTensor>,
    up_proj: Option<DeviceTensor>,
    down_proj: Option<DeviceTensor>,
    attn_norm: Option<DeviceTensor>,
    ffn_norm: Option<DeviceTensor>,
}

impl LayerWeightsBuilder {
    fn new() -> Self {
        Self {
            q_proj: None,
            k_proj: None,
            v_proj: None,
            o_proj: None,
            gate_proj: None,
            up_proj: None,
            down_proj: None,
            attn_norm: None,
            ffn_norm: None,
        }
    }

    fn build(self, layer_idx: usize) -> Result<LayerWeights> {
        let require = |opt: Option<DeviceTensor>, name: &str| -> Result<DeviceTensor> {
            opt.ok_or_else(|| {
                FractureError::WeightLoad(format!(
                    "missing required tensor: blk.{layer_idx}.{name}"
                ))
            })
        };

        Ok(LayerWeights {
            q_proj: require(self.q_proj, "attn_q.weight")?,
            k_proj: require(self.k_proj, "attn_k.weight")?,
            v_proj: require(self.v_proj, "attn_v.weight")?,
            o_proj: require(self.o_proj, "attn_output.weight")?,
            gate_proj: require(self.gate_proj, "ffn_gate.weight")?,
            up_proj: require(self.up_proj, "ffn_up.weight")?,
            down_proj: require(self.down_proj, "ffn_down.weight")?,
            attn_norm: require(self.attn_norm, "attn_norm.weight")?,
            ffn_norm: require(self.ffn_norm, "ffn_norm.weight")?,
        })
    }
}

/// Upload a tensor from the mmap region to the device backend, returning the DeviceTensor handle.
fn upload_tensor<B: Backend>(
    backend: &B,
    tensor: &TensorInfo,
    mmap: &[u8],
    tensor_data_offset: usize,
) -> Result<DeviceTensor> {
    let numel: usize = tensor.shape.iter().product();
    let byte_size = numel * tensor.dtype.size_bytes();
    let start = tensor_data_offset + tensor.offset as usize;
    let end = start + byte_size;

    if end > mmap.len() {
        return Err(FractureError::WeightLoad(format!(
            "tensor '{}' data [{start}..{end}) exceeds mmap length {}",
            tensor.name,
            mmap.len()
        )));
    }

    let slice = &mmap[start..end];
    let device_tensor = backend.alloc(&tensor.shape, tensor.dtype)?;
    backend.copy_to_device(&device_tensor, slice)?;
    Ok(device_tensor)
}

impl WeightStore {
    /// Load weights from a GGUF file onto the given backend.
    ///
    /// If `layer_range` is `Some`, only layers within that range are loaded. The resulting
    /// `WeightStore::layers` vec will contain exactly the layers in the range, indexed from 0.
    /// Global tensors (token_embedding, output_norm, lm_head) are always loaded.
    pub fn load<B: Backend>(
        path: &std::path::Path,
        backend: &B,
        layer_range: Option<std::ops::Range<usize>>,
    ) -> Result<Self> {
        let gguf = GgufParser::parse(path)?;

        let num_layers = gguf.config.num_layers;
        let range = layer_range.unwrap_or(0..num_layers);

        if range.end > num_layers {
            return Err(FractureError::WeightLoad(format!(
                "layer_range end {} exceeds model layer count {num_layers}",
                range.end
            )));
        }

        // Build fast name -> TensorInfo lookup
        let tensor_map: HashMap<&str, &TensorInfo> = gguf
            .tensors
            .iter()
            .map(|t| (t.name.as_str(), t))
            .collect();

        let mmap = &gguf.mmap[..];
        let data_offset = gguf.tensor_data_offset;

        // Helper to load a required global tensor
        let load_required = |name: &str| -> Result<DeviceTensor> {
            let info = tensor_map.get(name).ok_or_else(|| {
                FractureError::WeightLoad(format!("missing required tensor: {name}"))
            })?;
            upload_tensor(backend, info, mmap, data_offset)
        };

        // Global tensors
        let token_embedding = load_required("token_embd.weight")?;
        let output_norm = load_required("output_norm.weight")?;
        let lm_head = load_required("output.weight")?;

        // Per-layer tensors
        let layer_count = range.end - range.start;
        let mut layer_builders: Vec<LayerWeightsBuilder> =
            (0..layer_count).map(|_| LayerWeightsBuilder::new()).collect();

        // Layer tensor name suffixes and their corresponding builder fields
        let layer_fields: &[(&str, fn(&mut LayerWeightsBuilder, DeviceTensor))] = &[
            ("attn_q.weight", |b, t| b.q_proj = Some(t)),
            ("attn_k.weight", |b, t| b.k_proj = Some(t)),
            ("attn_v.weight", |b, t| b.v_proj = Some(t)),
            ("attn_output.weight", |b, t| b.o_proj = Some(t)),
            ("ffn_gate.weight", |b, t| b.gate_proj = Some(t)),
            ("ffn_up.weight", |b, t| b.up_proj = Some(t)),
            ("ffn_down.weight", |b, t| b.down_proj = Some(t)),
            ("attn_norm.weight", |b, t| b.attn_norm = Some(t)),
            ("ffn_norm.weight", |b, t| b.ffn_norm = Some(t)),
        ];

        // Track which tensor names we've consumed so we can warn about unknowns
        let mut consumed: std::collections::HashSet<&str> = std::collections::HashSet::new();
        consumed.insert("token_embd.weight");
        consumed.insert("output_norm.weight");
        consumed.insert("output.weight");

        for layer_idx in range.clone() {
            let builder_idx = layer_idx - range.start;
            for &(suffix, setter) in layer_fields {
                let name = format!("blk.{layer_idx}.{suffix}");
                if let Some(info) = tensor_map.get(name.as_str()) {
                    let device_tensor = upload_tensor(backend, info, mmap, data_offset)?;
                    setter(&mut layer_builders[builder_idx], device_tensor);
                    consumed.insert(info.name.as_str());
                }
                // Missing layer tensors will be caught by build() below
            }
        }

        // Also mark out-of-range layer tensors as consumed (not unknown, just skipped)
        for tensor in &gguf.tensors {
            if tensor.name.starts_with("blk.") {
                consumed.insert(tensor.name.as_str());
            }
        }

        // Warn about unknown tensor names
        for tensor in &gguf.tensors {
            if !consumed.contains(tensor.name.as_str()) {
                tracing::warn!("unknown tensor name in GGUF file: '{}'", tensor.name);
            }
        }

        // Build all layers, failing on any missing required tensor
        let layers: Vec<LayerWeights> = layer_builders
            .into_iter()
            .enumerate()
            .map(|(i, b)| b.build(range.start + i))
            .collect::<Result<Vec<_>>>()?;

        tracing::info!(
            "loaded {} layers ({}-{}) with {} total tensors onto {}",
            layers.len(),
            range.start,
            range.end,
            consumed.len(),
            backend.device_name(),
        );

        Ok(Self {
            config: gguf.config,
            token_embedding,
            layers,
            output_norm,
            lm_head,
        })
    }
}

#[cfg(test)]
mod tests {
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
        for &dim in shape {
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
        buf.write_u32::<LittleEndian>(0x46475547).unwrap(); // GGUF_MAGIC
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

        buf.write_u32::<LittleEndian>(0x46475547).unwrap();
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
}
