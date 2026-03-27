use serde::Serialize;

/// Opaque handle to a GPU timer. Like TensorId — just a u64.
/// The backend maps this to actual timer resources (e.g., CUDA events).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DeviceTimer(pub u64);

/// Timing profile for a single transformer layer.
#[derive(Debug, Clone, Serialize)]
pub struct LayerProfile {
    pub layer_idx: usize,
    pub total_ms: f32,
    pub rmsnorm_attn_ms: f32,
    pub qkv_proj_ms: f32,
    pub rope_ms: f32,
    pub attention_ms: f32,
    pub output_proj_ms: f32,
    pub rmsnorm_ffn_ms: f32,
    pub gate_up_proj_ms: f32,
    pub silu_mul_ms: f32,
    pub down_proj_ms: f32,
}

/// Timing profile for a complete forward pass.
#[derive(Debug, Clone, Serialize)]
pub struct ForwardProfile {
    pub total_ms: f32,
    pub prefill: bool,
    pub seq_len: usize,
    pub layer_profiles: Vec<LayerProfile>,
}

/// Per-request generation metrics.
#[derive(Debug, Clone, Serialize)]
pub struct RequestMetrics {
    pub request_id: String,
    pub prompt_tokens: usize,
    pub generated_tokens: usize,
    pub ttft_ms: f64,
    pub total_ms: f64,
    pub tokens_per_sec: f64,
    pub avg_decode_ms: f64,
    pub peak_vram_mb: f64,
    pub kv_cache_tokens: usize,
}
