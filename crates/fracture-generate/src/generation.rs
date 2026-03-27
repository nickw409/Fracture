use fracture_core::{Backend, Result};
use fracture_engine::Engine;
use tokio::sync::mpsc;

/// Configuration for a generation request.
#[derive(Debug, Clone)]
pub struct GenerationConfig {
    pub max_tokens: usize,
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub stop_tokens: Vec<u32>,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            max_tokens: 256,
            temperature: 1.0,
            top_k: 0,
            top_p: 1.0,
            stop_tokens: vec![128001, 128008, 128009], // Llama 3 EOS tokens
        }
    }
}

/// Orchestrates tokenization, prefill, decode loop, and streaming.
pub struct GenerationLoop;

impl GenerationLoop {
    /// Generate tokens from a prompt, streaming results through the channel.
    pub async fn generate<B: Backend>(
        _engine: &Engine<B>,
        _prompt: &str,
        _config: GenerationConfig,
        _tx: mpsc::Sender<String>,
    ) -> Result<Vec<u32>> {
        // TODO: tokenize → prefill → decode loop → detokenize → stream
        Err(fracture_core::FractureError::Generation(
            "not yet implemented".into(),
        ))
    }
}
