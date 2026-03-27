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
