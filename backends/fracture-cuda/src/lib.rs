pub mod ffi;
mod nvtx;
mod timers;

use ffi::*;
use fracture_core::{FractureError, Result, TensorId};
use std::collections::HashMap;
use std::ffi::{c_int, c_void, CStr};
use std::sync::Mutex;

/// Check a CUDA runtime call, mapping errors to FractureError.
macro_rules! cuda_check {
    ($call:expr) => {{
        let err = unsafe { $call };
        if err != CUDA_SUCCESS {
            let msg = unsafe { CStr::from_ptr(cudaGetErrorString(err)) };
            return Err(FractureError::Backend(format!(
                "CUDA error ({}): {}",
                err,
                msg.to_string_lossy()
            )));
        }
    }};
}

/// Check a cuBLAS call, mapping errors to FractureError.
macro_rules! cublas_check {
    ($call:expr) => {{
        let status = unsafe { $call };
        if status != CUBLAS_STATUS_SUCCESS {
            return Err(FractureError::Backend(format!(
                "cuBLAS error: status {}",
                status
            )));
        }
    }};
}

mod backend;

#[cfg(test)]
mod tests;

/// Internal state protected by a mutex for tensor registry mutations.
struct CudaState {
    tensors: HashMap<u64, *mut c_void>,
    next_id: u64,
}

// Raw pointers are safe to send between threads when access is synchronized.
unsafe impl Send for CudaState {}

/// CUDA backend implementing the Backend trait.
///
/// Manages CUDA device, stream, cuBLAS handle, and a registry mapping
/// TensorId values to device pointers.
pub struct CudaBackend {
    #[allow(dead_code)] // retained for multi-GPU support
    device_id: c_int,
    stream: cudaStream_t,
    cublas_handle: cublasHandle_t,
    state: Mutex<CudaState>,
    timer_manager: timers::TimerManager,
    device_name: String,
    total_memory: usize,
    rope_freq_table: Option<*mut c_void>,
}

// The raw pointers (stream, cublas_handle, rope_freq_table) are CUDA handles
// that are safe to use from any thread when properly synchronized.
unsafe impl Send for CudaBackend {}
unsafe impl Sync for CudaBackend {}

impl CudaBackend {
    /// Create a new CUDA backend on the specified device.
    pub fn new(device_id: i32) -> Result<Self> {
        cuda_check!(cudaSetDevice(device_id));

        let mut stream: cudaStream_t = std::ptr::null_mut();
        cuda_check!(cudaStreamCreate(&mut stream));

        let mut cublas_handle: cublasHandle_t = std::ptr::null_mut();
        cublas_check!(cublasCreate_v2(&mut cublas_handle));
        cublas_check!(cublasSetStream_v2(cublas_handle, stream));

        let mut props = cudaDeviceProp::default();
        cuda_check!(cudaGetDeviceProperties(&mut props, device_id));
        let device_name = unsafe {
            CStr::from_ptr(props.name_ptr())
                .to_string_lossy()
                .into_owned()
        };
        let total_memory = props.total_global_mem();

        tracing::info!(
            "CUDA backend initialized: {} ({:.1} GB)",
            device_name,
            total_memory as f64 / (1024.0 * 1024.0 * 1024.0)
        );

        Ok(Self {
            device_id,
            stream,
            cublas_handle,
            state: Mutex::new(CudaState {
                tensors: HashMap::new(),
                next_id: 1,
            }),
            timer_manager: timers::TimerManager::new(),
            device_name,
            total_memory,
            rope_freq_table: None,
        })
    }

    /// Look up the device pointer for a TensorId.
    pub fn get_ptr(&self, id: TensorId) -> Result<*mut c_void> {
        let state = self.state.lock().unwrap();
        state
            .tensors
            .get(&id.0)
            .copied()
            .ok_or_else(|| FractureError::TensorNotFound(format!("tensor id {}", id.0)))
    }

    /// Get the CUDA stream handle (for direct FFI calls in tests).
    pub fn stream(&self) -> cudaStream_t {
        self.stream
    }

    /// Pre-compute RoPE frequency table and store on GPU.
    /// freq[i] = 1.0 / (theta ^ (2i / head_dim)) for i in 0..head_dim/2
    pub fn precompute_rope_freqs(&mut self, head_dim: usize, theta: f64) -> Result<()> {
        let half_dim = head_dim / 2;
        let mut freqs = vec![0.0f32; half_dim];
        for (i, freq) in freqs.iter_mut().enumerate().take(half_dim) {
            *freq = 1.0 / theta.powf(2.0 * i as f64 / head_dim as f64) as f32;
        }

        let size = half_dim * std::mem::size_of::<f32>();
        let mut dev_ptr: *mut c_void = std::ptr::null_mut();
        cuda_check!(cudaMalloc(&mut dev_ptr, size));
        cuda_check!(cudaMemcpy(
            dev_ptr,
            freqs.as_ptr() as *const c_void,
            size,
            CUDA_MEMCPY_HOST_TO_DEVICE
        ));

        if let Some(old) = self.rope_freq_table {
            unsafe { cudaFree(old) };
        }
        self.rope_freq_table = Some(dev_ptr);
        Ok(())
    }

    fn rope_freq_ptr(&self) -> Result<*const c_void> {
        self.rope_freq_table.map(|p| p as *const c_void).ok_or_else(|| {
            FractureError::Backend("RoPE frequency table not initialized. Call precompute_rope_freqs() first.".into())
        })
    }
}

impl Drop for CudaBackend {
    fn drop(&mut self) {
        // Free all remaining tensors.
        let state = self.state.lock().unwrap();
        for (_, ptr) in state.tensors.iter() {
            unsafe { cudaFree(*ptr) };
        }
        drop(state);

        if let Some(freq_ptr) = self.rope_freq_table {
            unsafe { cudaFree(freq_ptr) };
        }

        unsafe {
            cublasDestroy_v2(self.cublas_handle);
            cudaStreamDestroy(self.stream);
        }
    }
}
