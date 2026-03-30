mod backend;
mod dtype;
mod error;
mod model_config;
mod profiling;
mod tensor;
pub mod turboquant;

pub use backend::Backend;
pub use dtype::DType;
pub use error::{FractureError, Result};
pub use model_config::ModelConfig;
pub use profiling::{DeviceTimer, ForwardProfile, LayerProfile, RequestMetrics};
pub use tensor::{DeviceTensor, TensorId};
pub use turboquant::TurboQuantConfig;

/// Why generation stopped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StopReason {
    /// Hit an EOS/stop token.
    Stop,
    /// Reached max_tokens limit.
    Length,
}
