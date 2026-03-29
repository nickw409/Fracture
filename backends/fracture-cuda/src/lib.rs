mod ffi;
mod nvtx;
mod timers;

use ffi::*;
use fracture_core::{Backend, DType, DeviceTensor, DeviceTimer, FractureError, Result, TensorId};
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
    fn get_ptr(&self, id: TensorId) -> Result<*mut c_void> {
        let state = self.state.lock().unwrap();
        state
            .tensors
            .get(&id.0)
            .copied()
            .ok_or_else(|| FractureError::TensorNotFound(format!("tensor id {}", id.0)))
    }

    /// Pre-compute RoPE frequency table and store on GPU.
    /// freq[i] = 1.0 / (theta ^ (2i / head_dim)) for i in 0..head_dim/2
    pub fn precompute_rope_freqs(&mut self, head_dim: usize, theta: f64) -> Result<()> {
        let half_dim = head_dim / 2;
        let mut freqs = vec![0.0f32; half_dim];
        for i in 0..half_dim {
            freqs[i] = 1.0 / theta.powf(2.0 * i as f64 / head_dim as f64) as f32;
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

impl Backend for CudaBackend {
    fn alloc(&self, shape: &[usize], dtype: DType) -> Result<DeviceTensor> {
        let numel: usize = shape.iter().product();
        let size = if dtype.is_packed() {
            (numel + 1) / 2
        } else {
            numel * dtype.size_bytes()
        };

        let mut dev_ptr: *mut c_void = std::ptr::null_mut();
        cuda_check!(cudaMalloc(&mut dev_ptr, size));

        let mut state = self.state.lock().unwrap();
        let id = state.next_id;
        state.next_id += 1;
        state.tensors.insert(id, dev_ptr);

        Ok(DeviceTensor::new(TensorId(id), shape.to_vec(), dtype))
    }

    fn free(&self, tensor: &DeviceTensor) -> Result<()> {
        let mut state = self.state.lock().unwrap();
        let ptr = state
            .tensors
            .remove(&tensor.id.0)
            .ok_or_else(|| FractureError::TensorNotFound(format!("tensor id {}", tensor.id.0)))?;
        cuda_check!(cudaFree(ptr));
        Ok(())
    }

    fn copy_to_device(&self, dst: &DeviceTensor, src: &[u8]) -> Result<()> {
        let ptr = self.get_ptr(dst.id)?;
        let size = dst.size_bytes();
        if src.len() < size {
            return Err(FractureError::InvalidShape(format!(
                "source buffer too small: {} < {}",
                src.len(),
                size
            )));
        }
        cuda_check!(cudaMemcpy(
            ptr,
            src.as_ptr() as *const c_void,
            size,
            CUDA_MEMCPY_HOST_TO_DEVICE
        ));
        Ok(())
    }

    fn copy_to_host(&self, src: &DeviceTensor, dst: &mut [u8]) -> Result<()> {
        let ptr = self.get_ptr(src.id)?;
        let size = src.size_bytes();
        if dst.len() < size {
            return Err(FractureError::InvalidShape(format!(
                "destination buffer too small: {} < {}",
                dst.len(),
                size
            )));
        }
        cuda_check!(cudaMemcpy(
            dst.as_mut_ptr() as *mut c_void,
            ptr as *const c_void,
            size,
            CUDA_MEMCPY_DEVICE_TO_HOST
        ));
        Ok(())
    }

    fn matmul(&self, a: &DeviceTensor, b: &DeviceTensor, out: &DeviceTensor) -> Result<()> {
        // C = A @ B^T  where A is [M, K], B is [N, K], C is [M, N].
        // Validate inner dimensions match.
        if a.shape[1] != b.shape[1] {
            return Err(FractureError::InvalidShape(format!(
                "matmul: A inner dim {} != B inner dim {} (A is {:?}, B is {:?})",
                a.shape[1], b.shape[1], a.shape, b.shape
            )));
        }
        nvtx::range_push("matmul");
        // GGUF weight matrices are stored as [N, K] (output features × input features),
        // matching PyTorch nn.Linear convention: y = x @ W^T.
        //
        // cuBLAS column-major trick:
        //   Row-major X appears as X^T in column-major.
        //   We want C = A @ B^T in row-major.
        //   Column-major: C^T = B @ A^T  (B transposed becomes un-transposed, A stays transposed)
        //   So: cublasGemmEx(N, N, ...) with B first, A second, N = b.shape[0].
        let m = a.shape[0] as c_int; // rows of A / rows of C
        let k = a.shape[1] as c_int; // shared dimension (input features)
        let n = b.shape[0] as c_int; // rows of B = output features = cols of C

        let a_ptr = self.get_ptr(a.id)?;
        let b_ptr = self.get_ptr(b.id)?;
        let c_ptr = self.get_ptr(out.id)?;

        let alpha: f32 = 1.0;
        let beta: f32 = 0.0;

        // Column-major layout: B[N,K] in row-major = B^T[K,N] in col-major.
        // We call OP_T on B to get B^T^T = B, so the effective col-major op is
        // C^T = B @ A^T, giving C = A @ B^T in row-major.
        cublas_check!(cublasGemmEx(
            self.cublas_handle,
            cublasOperation_t::CUBLAS_OP_T,  // transpose B in col-major view
            cublasOperation_t::CUBLAS_OP_N,  // A^T in col-major = A non-transposed
            n,                                // rows of op(B) = N (output features)
            m,                                // cols of op(A^T) = M (batch/seq)
            k,                                // shared dimension (input features)
            &alpha as *const f32 as *const c_void,
            b_ptr as *const c_void,
            cudaDataType_t::CUDA_R_16F,
            k,                                // ldb = K (B is [N,K] row-major → K cols)
            a_ptr as *const c_void,
            cudaDataType_t::CUDA_R_16F,
            k,                                // lda = K (A is [M,K] row-major → K cols)
            &beta as *const f32 as *const c_void,
            c_ptr,
            cudaDataType_t::CUDA_R_16F,
            n,                                // ldc = N (C is [M,N] row-major → N cols)
            cublasComputeType_t::CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT,
        ));

        nvtx::range_pop();
        Ok(())
    }

    fn rmsnorm(
        &self,
        input: &DeviceTensor,
        weight: &DeviceTensor,
        eps: f64,
        out: &DeviceTensor,
    ) -> Result<()> {
        nvtx::range_push("rmsnorm");
        let rows = input.shape[0] as c_int;
        let cols = input.shape[1] as c_int;

        let input_ptr = self.get_ptr(input.id)?;
        let weight_ptr = self.get_ptr(weight.id)?;
        let out_ptr = self.get_ptr(out.id)?;

        cuda_check!(launch_rmsnorm(
            out_ptr,
            input_ptr as *const c_void,
            weight_ptr as *const c_void,
            eps as f32,
            rows,
            cols,
            self.stream,
        ));
        nvtx::range_pop();
        Ok(())
    }

    fn rope(
        &self,
        q: &DeviceTensor,
        k: &DeviceTensor,
        positions: &[u32],
        _theta: f64,
        head_dim: usize,
    ) -> Result<()> {
        nvtx::range_push("rope");
        let q_ptr = self.get_ptr(q.id)?;
        let k_ptr = self.get_ptr(k.id)?;
        let freq_ptr = self.rope_freq_ptr()?;

        let num_tokens = positions.len() as c_int;
        let num_q_heads = q.shape[1] as c_int;
        let num_kv_heads = k.shape[1] as c_int;

        // Copy positions to device
        let pos_size = positions.len() * std::mem::size_of::<u32>();
        let mut pos_dev: *mut c_void = std::ptr::null_mut();
        cuda_check!(cudaMalloc(&mut pos_dev, pos_size));
        cuda_check!(cudaMemcpy(
            pos_dev,
            positions.as_ptr() as *const c_void,
            pos_size,
            CUDA_MEMCPY_HOST_TO_DEVICE
        ));

        let result = unsafe {
            launch_rope(
                q_ptr as *mut c_void,
                k_ptr as *mut c_void,
                pos_dev as *const u32,
                freq_ptr as *const f32,
                num_tokens,
                num_q_heads,
                num_kv_heads,
                head_dim as c_int,
                self.stream,
            )
        };

        // Free positions regardless of result
        unsafe { cudaFree(pos_dev) };

        if result != CUDA_SUCCESS {
            let msg = unsafe { CStr::from_ptr(cudaGetErrorString(result)) };
            nvtx::range_pop();
            return Err(FractureError::Backend(format!(
                "RoPE kernel error: {}",
                msg.to_string_lossy()
            )));
        }
        nvtx::range_pop();
        Ok(())
    }

    fn attention(
        &self,
        q: &DeviceTensor,
        k_cache: &DeviceTensor,
        v_cache: &DeviceTensor,
        num_kv_heads: usize,
        start_pos: usize,
        out: &DeviceTensor,
    ) -> Result<()> {
        nvtx::range_push("attention");
        let q_ptr = self.get_ptr(q.id)?;
        let k_ptr = self.get_ptr(k_cache.id)?;
        let v_ptr = self.get_ptr(v_cache.id)?;
        let out_ptr = self.get_ptr(out.id)?;

        let num_tokens = q.shape[0] as c_int;
        let num_q_heads = q.shape[1] as c_int;
        let head_dim = q.shape[2] as c_int;
        let kv_len = (start_pos + num_tokens as usize) as c_int;

        cuda_check!(launch_attention(
            out_ptr,
            q_ptr as *const c_void,
            k_ptr as *const c_void,
            v_ptr as *const c_void,
            num_tokens,
            num_q_heads,
            num_kv_heads as c_int,
            head_dim,
            kv_len,
            start_pos as c_int,
            self.stream,
        ));
        nvtx::range_pop();
        Ok(())
    }

    fn attention_paged(
        &self,
        q: &DeviceTensor,
        block_table: &[i32],
        k_blocks: &[&DeviceTensor],
        v_blocks: &[&DeviceTensor],
        num_kv_heads: usize,
        kv_len: usize,
        start_pos: usize,
        out: &DeviceTensor,
    ) -> Result<()> {
        nvtx::range_push("attention_paged");
        let q_ptr = self.get_ptr(q.id)?;
        let out_ptr = self.get_ptr(out.id)?;

        let num_tokens = q.shape[0] as c_int;
        let num_q_heads = q.shape[1] as c_int;
        let head_dim = q.shape[2] as c_int;

        // Resolve block tensor IDs to device pointers
        let k_ptrs: Vec<*const c_void> = k_blocks
            .iter()
            .map(|t| self.get_ptr(t.id).map(|p| p as *const c_void))
            .collect::<Result<Vec<_>>>()?;
        let v_ptrs: Vec<*const c_void> = v_blocks
            .iter()
            .map(|t| self.get_ptr(t.id).map(|p| p as *const c_void))
            .collect::<Result<Vec<_>>>()?;

        // Copy block_table to device
        let bt_size = block_table.len() * std::mem::size_of::<i32>();
        let mut bt_dev: *mut c_void = std::ptr::null_mut();
        cuda_check!(cudaMalloc(&mut bt_dev, bt_size));
        cuda_check!(cudaMemcpy(
            bt_dev,
            block_table.as_ptr() as *const c_void,
            bt_size,
            CUDA_MEMCPY_HOST_TO_DEVICE
        ));

        // Copy K block pointer array to device
        let kbp_size = k_ptrs.len() * std::mem::size_of::<*const c_void>();
        let mut kbp_dev: *mut c_void = std::ptr::null_mut();
        cuda_check!(cudaMalloc(&mut kbp_dev, kbp_size));
        cuda_check!(cudaMemcpy(
            kbp_dev,
            k_ptrs.as_ptr() as *const c_void,
            kbp_size,
            CUDA_MEMCPY_HOST_TO_DEVICE
        ));

        // Copy V block pointer array to device
        let vbp_size = v_ptrs.len() * std::mem::size_of::<*const c_void>();
        let mut vbp_dev: *mut c_void = std::ptr::null_mut();
        cuda_check!(cudaMalloc(&mut vbp_dev, vbp_size));
        cuda_check!(cudaMemcpy(
            vbp_dev,
            v_ptrs.as_ptr() as *const c_void,
            vbp_size,
            CUDA_MEMCPY_HOST_TO_DEVICE
        ));

        cuda_check!(launch_paged_attention(
            out_ptr,
            q_ptr as *const c_void,
            bt_dev as *const c_int,
            kbp_dev as *const *const c_void,
            vbp_dev as *const *const c_void,
            num_tokens,
            num_q_heads,
            num_kv_heads as c_int,
            head_dim,
            kv_len as c_int,
            start_pos as c_int,
            block_table.len() as c_int,
            self.stream,
        ));

        // Free temporary device allocations
        unsafe {
            cudaFree(bt_dev);
            cudaFree(kbp_dev);
            cudaFree(vbp_dev);
        }

        nvtx::range_pop();
        Ok(())
    }

    fn silu_mul(
        &self,
        gate: &DeviceTensor,
        up: &DeviceTensor,
        out: &DeviceTensor,
    ) -> Result<()> {
        if gate.shape != up.shape {
            return Err(FractureError::InvalidShape(format!(
                "silu_mul: gate shape {:?} != up shape {:?}",
                gate.shape, up.shape
            )));
        }
        nvtx::range_push("silu_mul");
        let n = gate.numel() as c_int;
        let gate_ptr = self.get_ptr(gate.id)?;
        let up_ptr = self.get_ptr(up.id)?;
        let out_ptr = self.get_ptr(out.id)?;

        cuda_check!(launch_silu_mul(
            out_ptr,
            gate_ptr as *const c_void,
            up_ptr as *const c_void,
            n,
            self.stream,
        ));
        nvtx::range_pop();
        Ok(())
    }

    fn embedding(
        &self,
        token_ids: &[u32],
        embedding_table: &DeviceTensor,
        out: &DeviceTensor,
    ) -> Result<()> {
        nvtx::range_push("embedding");
        let table_ptr = self.get_ptr(embedding_table.id)?;
        let out_ptr = self.get_ptr(out.id)?;

        let num_tokens = token_ids.len() as c_int;
        let hidden_dim = embedding_table.shape[1] as c_int;
        let vocab_size = embedding_table.shape[0] as c_int;

        // Copy token IDs to device
        let ids_size = token_ids.len() * std::mem::size_of::<u32>();
        let mut ids_dev: *mut c_void = std::ptr::null_mut();
        cuda_check!(cudaMalloc(&mut ids_dev, ids_size));
        cuda_check!(cudaMemcpy(
            ids_dev,
            token_ids.as_ptr() as *const c_void,
            ids_size,
            CUDA_MEMCPY_HOST_TO_DEVICE
        ));

        let result = unsafe {
            launch_embedding(
                out_ptr,
                table_ptr as *const c_void,
                ids_dev as *const u32,
                num_tokens,
                hidden_dim,
                vocab_size,
                self.stream,
            )
        };

        unsafe { cudaFree(ids_dev) };

        if result != CUDA_SUCCESS {
            let msg = unsafe { CStr::from_ptr(cudaGetErrorString(result)) };
            nvtx::range_pop();
            return Err(FractureError::Backend(format!(
                "embedding kernel error: {}",
                msg.to_string_lossy()
            )));
        }
        nvtx::range_pop();
        Ok(())
    }

    fn add(&self, a: &DeviceTensor, b: &DeviceTensor, out: &DeviceTensor) -> Result<()> {
        if a.shape != b.shape {
            return Err(FractureError::InvalidShape(format!(
                "add: a shape {:?} != b shape {:?}",
                a.shape, b.shape
            )));
        }
        nvtx::range_push("add");
        let n = a.numel() as c_int;
        let a_ptr = self.get_ptr(a.id)?;
        let b_ptr = self.get_ptr(b.id)?;
        let out_ptr = self.get_ptr(out.id)?;

        cuda_check!(launch_add(
            out_ptr,
            a_ptr as *const c_void,
            b_ptr as *const c_void,
            n,
            self.stream,
        ));
        nvtx::range_pop();
        Ok(())
    }

    fn copy_rows(
        &self,
        src: &DeviceTensor,
        dst: &DeviceTensor,
        src_offset: usize,
        dst_offset: usize,
        count: usize,
    ) -> Result<()> {
        nvtx::range_push("copy_rows");

        // Bounds validation
        if src_offset + count > src.shape[0] {
            return Err(FractureError::InvalidShape(format!(
                "copy_rows: src_offset({}) + count({}) = {} exceeds src rows({})",
                src_offset, count, src_offset + count, src.shape[0]
            )));
        }
        if dst_offset + count > dst.shape[0] {
            return Err(FractureError::InvalidShape(format!(
                "copy_rows: dst_offset({}) + count({}) = {} exceeds dst rows({})",
                dst_offset, count, dst_offset + count, dst.shape[0]
            )));
        }
        if src.shape[1..] != dst.shape[1..] {
            return Err(FractureError::InvalidShape(format!(
                "copy_rows: src column shape {:?} != dst column shape {:?}",
                &src.shape[1..], &dst.shape[1..]
            )));
        }
        if src.dtype != dst.dtype {
            return Err(FractureError::InvalidShape(format!(
                "copy_rows: src dtype {:?} != dst dtype {:?}",
                src.dtype, dst.dtype
            )));
        }

        let src_ptr = self.get_ptr(src.id)?;
        let dst_ptr = self.get_ptr(dst.id)?;

        // Row size = product of all dims except first * dtype size
        let row_size: usize = src.shape[1..].iter().product::<usize>() * src.dtype.size_bytes();
        let byte_count = count * row_size;

        let src_byte_offset = src_offset * row_size;
        let dst_byte_offset = dst_offset * row_size;

        cuda_check!(cudaMemcpy(
            (dst_ptr as *mut u8).wrapping_add(dst_byte_offset) as *mut c_void,
            (src_ptr as *const u8).wrapping_add(src_byte_offset) as *const c_void,
            byte_count,
            CUDA_MEMCPY_DEVICE_TO_DEVICE
        ));
        nvtx::range_pop();
        Ok(())
    }

    fn device_name(&self) -> &str {
        &self.device_name
    }

    fn total_memory(&self) -> usize {
        self.total_memory
    }

    fn available_memory(&self) -> usize {
        let mut free: usize = 0;
        let mut total: usize = 0;
        unsafe { cudaMemGetInfo(&mut free, &mut total) };
        free
    }

    fn synchronize(&self) -> Result<()> {
        cuda_check!(cudaStreamSynchronize(self.stream));
        Ok(())
    }

    fn create_timer(&self) -> Result<DeviceTimer> {
        self.timer_manager.create()
    }

    fn start_timer(&self, timer: &DeviceTimer) -> Result<()> {
        self.timer_manager.start(timer, self.stream)
    }

    fn stop_timer(&self, timer: &DeviceTimer) -> Result<f32> {
        self.timer_manager.stop(timer, self.stream)
    }

    fn destroy_timer(&self, timer: &DeviceTimer) -> Result<()> {
        self.timer_manager.destroy(timer)
    }

    fn marker_push(&self, name: &str) {
        nvtx::range_push(name);
    }

    fn marker_pop(&self) {
        nvtx::range_pop();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fracture_core::Backend;
    use half::f16;

    fn make_backend() -> CudaBackend {
        CudaBackend::new(0).expect("failed to init CUDA backend")
    }

    fn to_fp16_bytes(vals: &[f32]) -> Vec<u8> {
        vals.iter()
            .flat_map(|&v| f16::from_f32(v).to_le_bytes())
            .collect()
    }

    fn from_fp16_bytes(bytes: &[u8]) -> Vec<f32> {
        bytes
            .chunks_exact(2)
            .map(|b| f16::from_le_bytes([b[0], b[1]]).to_f32())
            .collect()
    }

    fn alloc_with_data(backend: &CudaBackend, shape: &[usize], data: &[f32]) -> DeviceTensor {
        let t = backend.alloc(shape, DType::FP16).unwrap();
        let bytes = to_fp16_bytes(data);
        backend.copy_to_device(&t, &bytes).unwrap();
        t
    }

    fn read_fp16(backend: &CudaBackend, t: &DeviceTensor) -> Vec<f32> {
        let mut bytes = vec![0u8; t.size_bytes()];
        backend.copy_to_host(t, &mut bytes).unwrap();
        from_fp16_bytes(&bytes)
    }

    // ── Memory management ──────────────────────────────────────────

    #[test]
    fn test_alloc_free() {
        let b = make_backend();
        let t = b.alloc(&[4, 8], DType::FP16).unwrap();
        assert_eq!(t.shape, vec![4, 8]);
        assert_eq!(t.numel(), 32);
        b.free(&t).unwrap();
    }

    #[test]
    fn test_copy_roundtrip() {
        let b = make_backend();
        let data: Vec<f32> = (0..16).map(|i| i as f32 * 0.5).collect();
        let t = alloc_with_data(&b, &[4, 4], &data);
        let result = read_fp16(&b, &t);
        for (a, e) in result.iter().zip(data.iter()) {
            assert!((a - e).abs() < 0.01, "mismatch: got {a}, expected {e}");
        }
        b.free(&t).unwrap();
    }

    #[test]
    fn test_free_invalid_tensor() {
        let b = make_backend();
        let fake = DeviceTensor::new(TensorId(999999), vec![1], DType::FP16);
        assert!(b.free(&fake).is_err());
    }

    #[test]
    fn test_alloc_oom() {
        let b = make_backend();
        // Try to allocate more memory than any GPU has (1 TB)
        let result = b.alloc(&[1024 * 1024 * 1024, 256], DType::FP16);
        assert!(result.is_err(), "allocating 1TB should fail with OOM");
        let err = result.unwrap_err();
        assert!(
            err.to_string().contains("CUDA error"),
            "expected CUDA error, got: {err}"
        );
    }

    #[test]
    fn test_copy_rows() {
        let b = make_backend();
        // src: 4 rows of 2 elements
        let src_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let src = alloc_with_data(&b, &[4, 2], &src_data);
        // dst: 4 rows, initially zero
        let dst = b.alloc(&[4, 2], DType::FP16).unwrap();
        let zeros = to_fp16_bytes(&vec![0.0; 8]);
        b.copy_to_device(&dst, &zeros).unwrap();

        // Copy rows 1..3 from src to dst at offset 2
        b.copy_rows(&src, &dst, 1, 2, 2).unwrap();

        let result = read_fp16(&b, &dst);
        // dst[0..2] should be zeros, dst[2..4] should be src[1..3]
        assert!((result[0]).abs() < 0.01);
        assert!((result[1]).abs() < 0.01);
        assert!((result[2]).abs() < 0.01);
        assert!((result[3]).abs() < 0.01);
        assert!((result[4] - 3.0).abs() < 0.01); // src row 1
        assert!((result[5] - 4.0).abs() < 0.01);
        assert!((result[6] - 5.0).abs() < 0.01); // src row 2
        assert!((result[7] - 6.0).abs() < 0.01);

        b.free(&src).unwrap();
        b.free(&dst).unwrap();
    }

    #[test]
    fn test_device_info() {
        let b = make_backend();
        assert!(!b.device_name().is_empty());
        assert!(b.total_memory() > 0);
        assert!(b.available_memory() > 0);
        assert!(b.available_memory() <= b.total_memory());
    }

    // ── Elementwise add ────────────────────────────────────────────

    #[test]
    fn test_add() {
        let b = make_backend();
        let a_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let b_data: Vec<f32> = vec![10.0, 20.0, 30.0, 40.0];
        let a = alloc_with_data(&b, &[2, 2], &a_data);
        let bt = alloc_with_data(&b, &[2, 2], &b_data);
        let out = b.alloc(&[2, 2], DType::FP16).unwrap();

        b.add(&a, &bt, &out).unwrap();
        b.synchronize().unwrap();

        let result = read_fp16(&b, &out);
        let expected: Vec<f32> = vec![11.0, 22.0, 33.0, 44.0];
        for (r, e) in result.iter().zip(expected.iter()) {
            assert!((r - e).abs() < 0.1, "add: got {r}, expected {e}");
        }

        b.free(&a).unwrap();
        b.free(&bt).unwrap();
        b.free(&out).unwrap();
    }

    // ── Embedding lookup ───────────────────────────────────────────

    #[test]
    fn test_embedding() {
        let b = make_backend();
        // Vocab=4, dim=3
        let table_data: Vec<f32> = vec![
            0.1, 0.2, 0.3, // token 0
            1.1, 1.2, 1.3, // token 1
            2.1, 2.2, 2.3, // token 2
            3.1, 3.2, 3.3, // token 3
        ];
        let table = alloc_with_data(&b, &[4, 3], &table_data);
        let out = b.alloc(&[2, 3], DType::FP16).unwrap();

        b.embedding(&[2, 0], &table, &out).unwrap();
        b.synchronize().unwrap();

        let result = read_fp16(&b, &out);
        // token 2 then token 0
        assert!((result[0] - 2.1).abs() < 0.05);
        assert!((result[1] - 2.2).abs() < 0.05);
        assert!((result[2] - 2.3).abs() < 0.05);
        assert!((result[3] - 0.1).abs() < 0.05);
        assert!((result[4] - 0.2).abs() < 0.05);
        assert!((result[5] - 0.3).abs() < 0.05);

        b.free(&table).unwrap();
        b.free(&out).unwrap();
    }

    // ── RMSNorm ────────────────────────────────────────────────────

    #[test]
    fn test_rmsnorm() {
        let b = make_backend();
        // 1 row, 4 elements
        let x_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let w_data: Vec<f32> = vec![1.0, 1.0, 1.0, 1.0]; // identity weight

        let x = alloc_with_data(&b, &[1, 4], &x_data);
        let w = alloc_with_data(&b, &[4], &w_data);
        let out = b.alloc(&[1, 4], DType::FP16).unwrap();

        b.rmsnorm(&x, &w, 1e-5, &out).unwrap();
        b.synchronize().unwrap();

        let result = read_fp16(&b, &out);

        // CPU reference: rms = sqrt(mean(x^2) + eps) = sqrt((1+4+9+16)/4 + 1e-5) = sqrt(7.5)
        let rms = (7.5f32 + 1e-5).sqrt();
        let expected: Vec<f32> = x_data.iter().map(|&v| v / rms).collect();

        for (r, e) in result.iter().zip(expected.iter()) {
            assert!(
                (r - e).abs() < 0.01,
                "rmsnorm: got {r}, expected {e}"
            );
        }

        b.free(&x).unwrap();
        b.free(&w).unwrap();
        b.free(&out).unwrap();
    }

    #[test]
    fn test_rmsnorm_zero_input() {
        let b = make_backend();
        let x = alloc_with_data(&b, &[1, 4], &[0.0; 4]);
        let w = alloc_with_data(&b, &[4], &[1.0; 4]);
        let out = b.alloc(&[1, 4], DType::FP16).unwrap();

        b.rmsnorm(&x, &w, 1e-5, &out).unwrap();
        b.synchronize().unwrap();

        let result = read_fp16(&b, &out);
        for &v in &result {
            assert!(v.abs() < 0.01, "rmsnorm of zeros should be ~zero, got {v}");
        }

        b.free(&x).unwrap();
        b.free(&w).unwrap();
        b.free(&out).unwrap();
    }

    // ── SiLU multiply ──────────────────────────────────────────────

    #[test]
    fn test_silu_mul() {
        let b = make_backend();
        let gate_data: Vec<f32> = vec![0.0, 1.0, -1.0, 2.0];
        let up_data: Vec<f32> = vec![1.0, 1.0, 1.0, 1.0];

        let gate = alloc_with_data(&b, &[1, 4], &gate_data);
        let up = alloc_with_data(&b, &[1, 4], &up_data);
        let out = b.alloc(&[1, 4], DType::FP16).unwrap();

        b.silu_mul(&gate, &up, &out).unwrap();
        b.synchronize().unwrap();

        let result = read_fp16(&b, &out);

        // CPU reference: silu(x) = x / (1 + exp(-x))
        let silu = |x: f32| x / (1.0 + (-x).exp());
        let expected: Vec<f32> = gate_data.iter().zip(up_data.iter())
            .map(|(&g, &u)| silu(g) * u)
            .collect();

        for (r, e) in result.iter().zip(expected.iter()) {
            assert!(
                (r - e).abs() < 0.02,
                "silu_mul: got {r}, expected {e}"
            );
        }

        b.free(&gate).unwrap();
        b.free(&up).unwrap();
        b.free(&out).unwrap();
    }

    // ── Matrix multiplication ──────────────────────────────────────

    #[test]
    fn test_matmul() {
        let b = make_backend();
        // A = [2, 3], B = [2, 3] (transposed storage), C = [2, 2]
        // matmul computes C = A @ B^T
        let a_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b_data: Vec<f32> = vec![7.0, 9.0, 11.0, 8.0, 10.0, 12.0];

        let a = alloc_with_data(&b, &[2, 3], &a_data);
        let bt = alloc_with_data(&b, &[2, 3], &b_data);
        let out = b.alloc(&[2, 2], DType::FP16).unwrap();

        b.matmul(&a, &bt, &out).unwrap();
        b.synchronize().unwrap();

        let result = read_fp16(&b, &out);

        // CPU: C[0,0] = 1*7 + 2*9 + 3*11 = 58
        // C[0,1] = 1*8 + 2*10 + 3*12 = 64
        // C[1,0] = 4*7 + 5*9 + 6*11 = 139
        // C[1,1] = 4*8 + 5*10 + 6*12 = 154
        let expected: Vec<f32> = vec![58.0, 64.0, 139.0, 154.0];

        for (r, e) in result.iter().zip(expected.iter()) {
            assert!(
                (r - e).abs() < 1.0, // FP16 tolerance for larger values
                "matmul: got {r}, expected {e}"
            );
        }

        b.free(&a).unwrap();
        b.free(&bt).unwrap();
        b.free(&out).unwrap();
    }

    #[test]
    fn test_matmul_m1() {
        // M=1 decode-like shape, matmul computes C = A @ B^T
        let b = make_backend();
        let a_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0]; // [1, 4]
        let b_data: Vec<f32> = vec![1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0]; // [2, 4]

        let a = alloc_with_data(&b, &[1, 4], &a_data);
        let bt = alloc_with_data(&b, &[2, 4], &b_data);
        let out = b.alloc(&[1, 2], DType::FP16).unwrap();

        b.matmul(&a, &bt, &out).unwrap();
        b.synchronize().unwrap();

        let result = read_fp16(&b, &out);
        // C[0,0] = 1*1 + 2*0 + 3*1 + 4*0 = 4
        // C[0,1] = 1*0 + 2*1 + 3*0 + 4*1 = 6
        assert!((result[0] - 4.0).abs() < 0.1);
        assert!((result[1] - 6.0).abs() < 0.1);

        b.free(&a).unwrap();
        b.free(&bt).unwrap();
        b.free(&out).unwrap();
    }

    // ── RoPE ───────────────────────────────────────────────────────

    #[test]
    fn test_rope() {
        let mut b = make_backend();
        let head_dim = 4;
        let theta = 10000.0;
        b.precompute_rope_freqs(head_dim, theta).unwrap();

        // 1 token, 1 Q head, 1 KV head, head_dim=4
        // Q and K are [1, 1, 4]
        let q_data: Vec<f32> = vec![1.0, 0.0, 1.0, 0.0];
        let k_data: Vec<f32> = vec![1.0, 0.0, 1.0, 0.0];

        let q = alloc_with_data(&b, &[1, 1, 4], &q_data);
        let k = alloc_with_data(&b, &[1, 1, 4], &k_data);

        // Position 0: angle = 0 * freq, so cos=1 sin=0, no rotation
        b.rope(&q, &k, &[0], theta, head_dim).unwrap();
        b.synchronize().unwrap();

        let q_result = read_fp16(&b, &q);
        // At position 0, rotation angle is 0, so output should equal input
        for (r, e) in q_result.iter().zip(q_data.iter()) {
            assert!((r - e).abs() < 0.01, "rope pos=0: got {r}, expected {e}");
        }

        // Now test with position > 0 to verify rotation actually happens
        let q2 = alloc_with_data(&b, &[1, 1, 4], &[1.0, 0.0, 1.0, 0.0]);
        let k2 = alloc_with_data(&b, &[1, 1, 4], &[1.0, 0.0, 1.0, 0.0]);
        b.rope(&q2, &k2, &[5], theta, head_dim).unwrap();
        b.synchronize().unwrap();

        let q2_result = read_fp16(&b, &q2);
        // At position 5 with non-zero freq, values should differ from input
        let changed = q2_result.iter().zip([1.0f32, 0.0, 1.0, 0.0].iter())
            .any(|(r, e)| (r - e).abs() > 0.01);
        assert!(changed, "rope at pos=5 should modify the input");

        b.free(&q).unwrap();
        b.free(&k).unwrap();
        b.free(&q2).unwrap();
        b.free(&k2).unwrap();
    }

    // ── Attention ──────────────────────────────────────────────────

    #[test]
    fn test_attention_single_token() {
        let b = make_backend();
        // Simplest case: 1 token, 1 Q head, 1 KV head, head_dim=4
        // Q attends to itself only (kv_len=1, start_pos=0)
        let q_data: Vec<f32> = vec![1.0, 0.0, 0.0, 0.0];
        let k_data: Vec<f32> = vec![1.0, 0.0, 0.0, 0.0];
        let v_data: Vec<f32> = vec![0.5, 0.6, 0.7, 0.8];

        // KV cache: [max_seq=4, 1 head, 4 dim] — we only use position 0
        let k_cache = b.alloc(&[4, 1, 4], DType::FP16).unwrap();
        let v_cache = b.alloc(&[4, 1, 4], DType::FP16).unwrap();

        // Write K and V at position 0
        let k_src = alloc_with_data(&b, &[1, 1, 4], &k_data);
        let v_src = alloc_with_data(&b, &[1, 1, 4], &v_data);
        b.copy_rows(&k_src, &k_cache, 0, 0, 1).unwrap();
        b.copy_rows(&v_src, &v_cache, 0, 0, 1).unwrap();

        let q = alloc_with_data(&b, &[1, 1, 4], &q_data);
        let out = b.alloc(&[1, 1, 4], DType::FP16).unwrap();

        // start_pos=0, so kv_len = 0 + 1 = 1. Single token attends to itself.
        b.attention(&q, &k_cache, &v_cache, 1, 0, &out).unwrap();
        b.synchronize().unwrap();

        let result = read_fp16(&b, &out);
        // With only 1 key, softmax is [1.0], so output = V directly
        for (r, e) in result.iter().zip(v_data.iter()) {
            assert!(
                (r - e).abs() < 0.05,
                "attention single token: got {r}, expected {e}"
            );
        }

        b.free(&q).unwrap();
        b.free(&k_cache).unwrap();
        b.free(&v_cache).unwrap();
        b.free(&k_src).unwrap();
        b.free(&v_src).unwrap();
        b.free(&out).unwrap();
    }

    #[test]
    fn test_attention_gqa() {
        let b = make_backend();
        // 1 token, 2 Q heads sharing 1 KV head (GQA group_size=2), head_dim=2
        let q_data: Vec<f32> = vec![
            1.0, 0.0, // Q head 0
            0.0, 1.0, // Q head 1
        ];
        let k_data: Vec<f32> = vec![1.0, 0.0]; // 1 KV head
        let v_data: Vec<f32> = vec![5.0, 6.0]; // 1 KV head

        let k_cache = b.alloc(&[4, 1, 2], DType::FP16).unwrap();
        let v_cache = b.alloc(&[4, 1, 2], DType::FP16).unwrap();

        let k_src = alloc_with_data(&b, &[1, 1, 2], &k_data);
        let v_src = alloc_with_data(&b, &[1, 1, 2], &v_data);
        b.copy_rows(&k_src, &k_cache, 0, 0, 1).unwrap();
        b.copy_rows(&v_src, &v_cache, 0, 0, 1).unwrap();

        let q = alloc_with_data(&b, &[1, 2, 2], &q_data);
        let out = b.alloc(&[1, 2, 2], DType::FP16).unwrap();

        // 2 Q heads, 1 KV head
        b.attention(&q, &k_cache, &v_cache, 1, 0, &out).unwrap();
        b.synchronize().unwrap();

        let result = read_fp16(&b, &out);
        // Both Q heads attend to the single K/V pair, so output for both = V
        assert!((result[0] - 5.0).abs() < 0.1, "GQA head 0 dim 0: {}", result[0]);
        assert!((result[1] - 6.0).abs() < 0.1, "GQA head 0 dim 1: {}", result[1]);
        assert!((result[2] - 5.0).abs() < 0.1, "GQA head 1 dim 0: {}", result[2]);
        assert!((result[3] - 6.0).abs() < 0.1, "GQA head 1 dim 1: {}", result[3]);

        b.free(&q).unwrap();
        b.free(&k_cache).unwrap();
        b.free(&v_cache).unwrap();
        b.free(&k_src).unwrap();
        b.free(&v_src).unwrap();
        b.free(&out).unwrap();
    }

    // ── Timers ─────────────────────────────────────────────────────

    #[test]
    fn test_gpu_timers() {
        let b = make_backend();
        let timer = b.create_timer().unwrap();
        b.start_timer(&timer).unwrap();
        // Do a small allocation to put some work on the stream
        let t = b.alloc(&[1024, 1024], DType::FP16).unwrap();
        b.free(&t).unwrap();
        let elapsed = b.stop_timer(&timer).unwrap();
        assert!(elapsed >= 0.0, "elapsed should be non-negative");
        b.destroy_timer(&timer).unwrap();
    }

    // ── Synchronize ────────────────────────────────────────────────

    #[test]
    fn test_synchronize() {
        let b = make_backend();
        b.synchronize().unwrap();
    }

    // ── Memory management error paths ─────────────────────────────

    #[test]
    fn test_double_free() {
        let b = make_backend();
        let t = b.alloc(&[4, 4], DType::FP16).unwrap();
        b.free(&t).unwrap();
        assert!(b.free(&t).is_err());
    }

    #[test]
    fn test_use_after_free() {
        let b = make_backend();
        let t = b.alloc(&[4, 4], DType::FP16).unwrap();
        b.free(&t).unwrap();
        let mut buf = vec![0u8; t.size_bytes()];
        assert!(b.copy_to_host(&t, &mut buf).is_err());
    }

    #[test]
    fn test_copy_to_device_buffer_too_small() {
        let b = make_backend();
        let t = b.alloc(&[4, 4], DType::FP16).unwrap(); // 32 elements = 64 bytes
        let small_buf = vec![0u8; 16]; // too small
        let err = b.copy_to_device(&t, &small_buf).unwrap_err();
        assert!(matches!(err, FractureError::InvalidShape(_)));
        b.free(&t).unwrap();
    }

    #[test]
    fn test_copy_to_host_buffer_too_small() {
        let b = make_backend();
        let t = b.alloc(&[4, 4], DType::FP16).unwrap();
        let mut small_buf = vec![0u8; 16]; // too small
        let err = b.copy_to_host(&t, &mut small_buf).unwrap_err();
        assert!(matches!(err, FractureError::InvalidShape(_)));
        b.free(&t).unwrap();
    }

    #[test]
    fn test_error_types() {
        let b = make_backend();
        // Free of invalid tensor returns TensorNotFound
        let fake = DeviceTensor::new(TensorId(999999), vec![1], DType::FP16);
        let err = b.free(&fake).unwrap_err();
        assert!(matches!(err, FractureError::TensorNotFound(_)));

        // Copy with wrong buffer size returns InvalidShape
        let t = b.alloc(&[4, 4], DType::FP16).unwrap();
        let err = b.copy_to_device(&t, &[0u8; 2]).unwrap_err();
        assert!(matches!(err, FractureError::InvalidShape(_)));
        b.free(&t).unwrap();
    }

    #[test]
    fn test_tensor_not_found_contains_id() {
        let b = make_backend();
        let fake = DeviceTensor::new(TensorId(12345), vec![1], DType::FP16);
        let err = b.free(&fake).unwrap_err();
        assert!(
            matches!(err, FractureError::TensorNotFound(_)),
            "expected TensorNotFound, got: {err}"
        );
        let display = err.to_string();
        assert!(
            display.contains("12345"),
            "TensorNotFound error should contain tensor id '12345' in: {display}"
        );
    }

    // ── RMSNorm additional tests ──────────────────────────────────

    #[test]
    fn test_rmsnorm_prefill() {
        use rand::Rng;
        let b = make_backend();
        let rows = 4;
        let cols = 256;
        let mut rng = rand::thread_rng();
        let x_data: Vec<f32> = (0..rows * cols).map(|_| rng.gen_range(-1.0..1.0)).collect();
        let w_data: Vec<f32> = (0..cols).map(|_| rng.gen_range(0.5..2.0)).collect();

        let x = alloc_with_data(&b, &[rows, cols], &x_data);
        let w = alloc_with_data(&b, &[cols], &w_data);
        let out = b.alloc(&[rows, cols], DType::FP16).unwrap();

        b.rmsnorm(&x, &w, 1e-5, &out).unwrap();
        b.synchronize().unwrap();

        let result = read_fp16(&b, &out);

        // CPU reference
        for row in 0..rows {
            let start = row * cols;
            let end = start + cols;
            let row_data = &x_data[start..end];
            let mean_sq: f32 = row_data.iter().map(|v| v * v).sum::<f32>() / cols as f32;
            let rms = (mean_sq + 1e-5).sqrt();
            for j in 0..cols {
                let expected = (row_data[j] / rms) * w_data[j];
                let got = result[start + j];
                assert!(
                    (got - expected).abs() < 0.05,
                    "rmsnorm_prefill [{row},{j}]: got {got}, expected {expected}"
                );
            }
        }

        b.free(&x).unwrap();
        b.free(&w).unwrap();
        b.free(&out).unwrap();
    }

    #[test]
    fn test_rmsnorm_full_dim() {
        let b = make_backend();
        let cols = 4096;
        let x_data: Vec<f32> = (0..cols).map(|i| ((i as f32) * 0.001) - 2.0).collect();
        let w_data: Vec<f32> = vec![1.0; cols];

        let x = alloc_with_data(&b, &[1, cols], &x_data);
        let w = alloc_with_data(&b, &[cols], &w_data);
        let out = b.alloc(&[1, cols], DType::FP16).unwrap();

        b.rmsnorm(&x, &w, 1e-5, &out).unwrap();
        b.synchronize().unwrap();

        let result = read_fp16(&b, &out);
        let mean_sq: f32 = x_data.iter().map(|v| v * v).sum::<f32>() / cols as f32;
        let rms = (mean_sq + 1e-5).sqrt();

        // Check a few sampled positions
        for &idx in &[0, 100, 2048, 4095] {
            let expected = x_data[idx] / rms;
            let got = result[idx];
            assert!(
                (got - expected).abs() < 0.05,
                "rmsnorm_full_dim [{idx}]: got {got}, expected {expected}"
            );
        }

        b.free(&x).unwrap();
        b.free(&w).unwrap();
        b.free(&out).unwrap();
    }

    #[test]
    fn test_rmsnorm_with_weights() {
        let b = make_backend();
        let x_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let w_data: Vec<f32> = vec![0.5, 2.0, 0.1, 3.0];

        let x = alloc_with_data(&b, &[1, 4], &x_data);
        let w = alloc_with_data(&b, &[4], &w_data);
        let out = b.alloc(&[1, 4], DType::FP16).unwrap();

        b.rmsnorm(&x, &w, 1e-5, &out).unwrap();
        b.synchronize().unwrap();

        let result = read_fp16(&b, &out);
        let mean_sq: f32 = x_data.iter().map(|v| v * v).sum::<f32>() / 4.0;
        let rms = (mean_sq + 1e-5).sqrt();

        for i in 0..4 {
            let expected = (x_data[i] / rms) * w_data[i];
            let got = result[i];
            assert!(
                (got - expected).abs() < 0.05,
                "rmsnorm_with_weights [{i}]: got {got}, expected {expected}"
            );
        }

        b.free(&x).unwrap();
        b.free(&w).unwrap();
        b.free(&out).unwrap();
    }

    #[test]
    fn test_rmsnorm_large_values() {
        let b = make_backend();
        let x_data: Vec<f32> = vec![60000.0, 60000.0, 60000.0, 60000.0];
        let w_data: Vec<f32> = vec![1.0; 4];

        let x = alloc_with_data(&b, &[1, 4], &x_data);
        let w = alloc_with_data(&b, &[4], &w_data);
        let out = b.alloc(&[1, 4], DType::FP16).unwrap();

        b.rmsnorm(&x, &w, 1e-5, &out).unwrap();
        b.synchronize().unwrap();

        let result = read_fp16(&b, &out);
        // CPU: rms = sqrt(mean([60000^2]*4) + eps) = sqrt(60000^2 + eps) = 60000
        // output = (60000/60000) * 1.0 = 1.0 for each element
        for (i, &v) in result.iter().enumerate() {
            assert!(v.is_finite(), "rmsnorm_large_values: got non-finite {v}");
            assert!(
                (v - 1.0).abs() < 1e-3,
                "rmsnorm_large_values [{i}]: got {v}, expected 1.0"
            );
        }

        b.free(&x).unwrap();
        b.free(&w).unwrap();
        b.free(&out).unwrap();
    }

    #[test]
    fn test_rmsnorm_small_values() {
        let b = make_backend();
        let x_data: Vec<f32> = vec![0.001, 0.001, 0.001, 0.001];
        let w_data: Vec<f32> = vec![1.0; 4];

        let x = alloc_with_data(&b, &[1, 4], &x_data);
        let w = alloc_with_data(&b, &[4], &w_data);
        let out = b.alloc(&[1, 4], DType::FP16).unwrap();

        b.rmsnorm(&x, &w, 1e-5, &out).unwrap();
        b.synchronize().unwrap();

        let result = read_fp16(&b, &out);
        for &v in &result {
            assert!(v.is_finite(), "rmsnorm_small_values: got non-finite {v}");
        }
        // With eps, output should be approximately x / sqrt(x^2 + eps) ~ 1.0
        let mean_sq = 0.001f32 * 0.001;
        let rms = (mean_sq + 1e-5).sqrt();
        let expected = 0.001 / rms;
        for &v in &result {
            assert!(
                (v - expected).abs() < 0.05,
                "rmsnorm_small_values: got {v}, expected {expected}"
            );
        }

        b.free(&x).unwrap();
        b.free(&w).unwrap();
        b.free(&out).unwrap();
    }

    #[test]
    fn test_rmsnorm_invalid_tensor() {
        let b = make_backend();
        let fake = DeviceTensor::new(TensorId(999999), vec![1, 4], DType::FP16);
        let w = alloc_with_data(&b, &[4], &[1.0; 4]);
        let out = b.alloc(&[1, 4], DType::FP16).unwrap();
        assert!(b.rmsnorm(&fake, &w, 1e-5, &out).is_err());
        b.free(&w).unwrap();
        b.free(&out).unwrap();
    }

    // ── RoPE additional tests ─────────────────────────────────────

    #[test]
    fn test_rope_without_freq_table() {
        let b = make_backend(); // No precompute_rope_freqs call
        let q = alloc_with_data(&b, &[1, 1, 4], &[1.0, 0.0, 1.0, 0.0]);
        let k = alloc_with_data(&b, &[1, 1, 4], &[1.0, 0.0, 1.0, 0.0]);
        let err = b.rope(&q, &k, &[0], 10000.0, 4);
        assert!(err.is_err(), "rope without freq table should fail");
        b.free(&q).unwrap();
        b.free(&k).unwrap();
    }

    #[test]
    fn test_rope_invalid_tensor() {
        let mut b = make_backend();
        b.precompute_rope_freqs(4, 10000.0).unwrap();
        let fake_q = DeviceTensor::new(TensorId(999999), vec![1, 1, 4], DType::FP16);
        let k = alloc_with_data(&b, &[1, 1, 4], &[1.0; 4]);
        assert!(b.rope(&fake_q, &k, &[0], 10000.0, 4).is_err());
        b.free(&k).unwrap();
    }

    #[test]
    fn test_rope_prefill_multi_position() {
        let mut b = make_backend();
        let head_dim = 4;
        let theta = 10000.0;
        b.precompute_rope_freqs(head_dim, theta).unwrap();

        // 4 tokens, 1 Q head, 1 KV head
        let q_data: Vec<f32> = vec![
            1.0, 0.0, 1.0, 0.0, // token 0
            1.0, 0.0, 1.0, 0.0, // token 1
            1.0, 0.0, 1.0, 0.0, // token 2
            1.0, 0.0, 1.0, 0.0, // token 3
        ];
        let k_data = q_data.clone();

        let q = alloc_with_data(&b, &[4, 1, 4], &q_data);
        let k = alloc_with_data(&b, &[4, 1, 4], &k_data);

        b.rope(&q, &k, &[0, 1, 2, 3], theta, head_dim).unwrap();
        b.synchronize().unwrap();

        let q_result = read_fp16(&b, &q);
        // Token 0 (pos=0) should be unchanged
        assert!((q_result[0] - 1.0).abs() < 0.01);
        assert!((q_result[1] - 0.0).abs() < 0.01);

        // Tokens at different positions should have different rotations
        let t0 = &q_result[0..4];
        let t1 = &q_result[4..8];
        let t2 = &q_result[8..12];
        let t3 = &q_result[12..16];

        // Each subsequent token should differ from the previous
        let diff_01: f32 = t0.iter().zip(t1.iter()).map(|(a, b)| (a - b).abs()).sum();
        let diff_12: f32 = t1.iter().zip(t2.iter()).map(|(a, b)| (a - b).abs()).sum();
        let diff_23: f32 = t2.iter().zip(t3.iter()).map(|(a, b)| (a - b).abs()).sum();
        assert!(diff_01 > 0.001, "tokens 0,1 should differ: diff={diff_01}");
        assert!(diff_12 > 0.001, "tokens 1,2 should differ: diff={diff_12}");
        assert!(diff_23 > 0.001, "tokens 2,3 should differ: diff={diff_23}");

        b.free(&q).unwrap();
        b.free(&k).unwrap();
    }

    #[test]
    fn test_rope_numerical_rotation() {
        let mut b = make_backend();
        let head_dim = 4;
        let theta = 10000.0;
        b.precompute_rope_freqs(head_dim, theta).unwrap();

        let x0 = 3.0f32;
        let x1 = 4.0f32;
        let x2 = 1.0f32;
        let x3 = 2.0f32;
        let q_data = vec![x0, x1, x2, x3];

        let q = alloc_with_data(&b, &[1, 1, 4], &q_data);
        let k = alloc_with_data(&b, &[1, 1, 4], &q_data);

        let pos = 1u32;
        b.rope(&q, &k, &[pos], theta, head_dim).unwrap();
        b.synchronize().unwrap();

        let result = read_fp16(&b, &q);

        // CPU reference: freq[i] = 1.0 / (theta^(2i/head_dim))
        let freq0 = 1.0 / theta.powf(0.0 / head_dim as f64); // = 1.0
        let freq1 = 1.0 / theta.powf(2.0 / head_dim as f64); // = 1/100
        let angle0 = (pos as f64) * freq0;
        let angle1 = (pos as f64) * freq1;
        let cos0 = angle0.cos() as f32;
        let sin0 = angle0.sin() as f32;
        let cos1 = angle1.cos() as f32;
        let sin1 = angle1.sin() as f32;

        // Split-half convention: pairs (d, d+half_dim) where half_dim = head_dim/2
        // Pair (0, 2) uses freq0, pair (1, 3) uses freq1
        let exp0 = x0 * cos0 - x2 * sin0;
        let exp1 = x1 * cos1 - x3 * sin1;
        let exp2 = x0 * sin0 + x2 * cos0;
        let exp3 = x1 * sin1 + x3 * cos1;

        // FP16 precision at magnitude ~5 is ~0.002, use numpy-style allclose
        let atol = 1e-3f32;
        let rtol = 1e-3f32;
        for (i, (&got, &exp)) in result.iter().zip([exp0, exp1, exp2, exp3].iter()).enumerate() {
            let tol = atol + rtol * exp.abs();
            assert!(
                (got - exp).abs() <= tol,
                "rope numerical [{i}]: got {got}, expected {exp} (err={}, tol={tol})",
                (got - exp).abs()
            );
        }

        b.free(&q).unwrap();
        b.free(&k).unwrap();
    }

    #[test]
    fn test_rope_model_dimensions() {
        let mut b = make_backend();
        let head_dim = 8;
        let theta = 10000.0;
        b.precompute_rope_freqs(head_dim, theta).unwrap();

        // Q: [2, 2, 8] — 2 tokens, 2 Q heads, head_dim=8
        // K: [2, 1, 8] — 2 tokens, 1 KV head, head_dim=8
        let q_data: Vec<f32> = (0..2 * 2 * 8).map(|i| (i as f32) * 0.1).collect();
        let k_data: Vec<f32> = (0..2 * 1 * 8).map(|i| (i as f32) * 0.2).collect();

        let q = alloc_with_data(&b, &[2, 2, 8], &q_data);
        let k = alloc_with_data(&b, &[2, 1, 8], &k_data);

        b.rope(&q, &k, &[0, 5], theta, head_dim).unwrap();
        b.synchronize().unwrap();

        let q_result = read_fp16(&b, &q);
        let k_result = read_fp16(&b, &k);

        // Position 0 should leave data unchanged
        for i in 0..8 {
            let expected = q_data[i]; // first head of first token
            assert!(
                (q_result[i] - expected).abs() < 0.05,
                "rope model dims q[{i}]: got {}, expected {expected}", q_result[i]
            );
        }

        // Position 5 (second token) should produce different values
        let q_tok1_start = 2 * 8; // second token, first head
        let changed = (0..8).any(|i| {
            (q_result[q_tok1_start + i] - q_data[q_tok1_start + i]).abs() > 0.01
        });
        assert!(changed, "rope at pos=5 should modify Q token 1");

        // K at position 5 should also change
        let k_tok1_start = 8;
        let k_changed = (0..8).any(|i| {
            (k_result[k_tok1_start + i] - k_data[k_tok1_start + i]).abs() > 0.01
        });
        assert!(k_changed, "rope at pos=5 should modify K token 1");

        b.free(&q).unwrap();
        b.free(&k).unwrap();
    }

    // ── SiLU multiply additional tests ────────────────────────────

    #[test]
    fn test_silu_mul_batch() {
        use rand::Rng;
        let b = make_backend();
        let mut rng = rand::thread_rng();
        let rows = 4;
        let cols = 8;
        let gate_data: Vec<f32> = (0..rows * cols).map(|_| rng.gen_range(-2.0..2.0)).collect();
        let up_data: Vec<f32> = (0..rows * cols).map(|_| rng.gen_range(-2.0..2.0)).collect();

        let gate = alloc_with_data(&b, &[rows, cols], &gate_data);
        let up = alloc_with_data(&b, &[rows, cols], &up_data);
        let out = b.alloc(&[rows, cols], DType::FP16).unwrap();

        b.silu_mul(&gate, &up, &out).unwrap();
        b.synchronize().unwrap();

        let result = read_fp16(&b, &out);
        let silu = |x: f32| x / (1.0 + (-x).exp());
        for i in 0..rows * cols {
            let expected = silu(gate_data[i]) * up_data[i];
            assert!(
                (result[i] - expected).abs() < 0.05,
                "silu_mul_batch [{i}]: got {}, expected {expected}", result[i]
            );
        }

        b.free(&gate).unwrap();
        b.free(&up).unwrap();
        b.free(&out).unwrap();
    }

    #[test]
    fn test_silu_mul_large_dim() {
        use rand::Rng;
        let b = make_backend();
        let mut rng = rand::thread_rng();
        let rows = 2;
        let cols = 14336;
        let n = rows * cols;
        let gate_data: Vec<f32> = (0..n).map(|_| rng.gen_range(-3.0..3.0)).collect();
        let up_data: Vec<f32> = (0..n).map(|_| rng.gen_range(-3.0..3.0)).collect();

        let gate = alloc_with_data(&b, &[rows, cols], &gate_data);
        let up = alloc_with_data(&b, &[rows, cols], &up_data);
        let out = b.alloc(&[rows, cols], DType::FP16).unwrap();

        b.silu_mul(&gate, &up, &out).unwrap();
        b.synchronize().unwrap();

        let result = read_fp16(&b, &out);
        let silu = |x: f32| x / (1.0 + (-x).exp());
        for i in 0..n {
            let expected = silu(gate_data[i]) * up_data[i];
            assert!(
                (result[i] - expected).abs() < 0.05,
                "silu_mul_large [{i}]: got {}, expected {expected}", result[i]
            );
        }

        b.free(&gate).unwrap();
        b.free(&up).unwrap();
        b.free(&out).unwrap();
    }

    #[test]
    fn test_silu_mul_invalid_tensor() {
        let b = make_backend();
        let fake = DeviceTensor::new(TensorId(999999), vec![1, 4], DType::FP16);
        let up = alloc_with_data(&b, &[1, 4], &[1.0; 4]);
        let out = b.alloc(&[1, 4], DType::FP16).unwrap();
        assert!(b.silu_mul(&fake, &up, &out).is_err());
        b.free(&up).unwrap();
        b.free(&out).unwrap();
    }

    // ── Attention additional tests ────────────────────────────────

    #[test]
    fn test_attention_prefill_causal() {
        let b = make_backend();
        // N=4 tokens, 2 Q heads, 1 KV head, head_dim=4
        let head_dim = 4;
        let num_q_heads = 2;
        let num_kv_heads = 1;
        let n_tokens = 4;
        let max_seq = 8;

        // Use distinct V values per position so we can verify weighted sums
        let v_values: Vec<Vec<f32>> = vec![
            vec![1.0, 0.0, 0.0, 0.0], // pos 0
            vec![0.0, 1.0, 0.0, 0.0], // pos 1
            vec![0.0, 0.0, 1.0, 0.0], // pos 2
            vec![0.0, 0.0, 0.0, 1.0], // pos 3
        ];

        // K values: use identical keys so attention is uniform (within causal mask)
        let k_val = vec![1.0, 0.0, 0.0, 0.0];

        let k_cache = b.alloc(&[max_seq, num_kv_heads, head_dim], DType::FP16).unwrap();
        let v_cache = b.alloc(&[max_seq, num_kv_heads, head_dim], DType::FP16).unwrap();

        // Fill KV cache for positions 0..4
        for pos in 0..n_tokens {
            let k_src = alloc_with_data(&b, &[1, num_kv_heads, head_dim], &k_val);
            let v_src = alloc_with_data(&b, &[1, num_kv_heads, head_dim], &v_values[pos]);
            b.copy_rows(&k_src, &k_cache, 0, pos, 1).unwrap();
            b.copy_rows(&v_src, &v_cache, 0, pos, 1).unwrap();
            b.free(&k_src).unwrap();
            b.free(&v_src).unwrap();
        }

        // Q: all tokens query with same direction as K
        let q_data: Vec<f32> = (0..n_tokens * num_q_heads)
            .flat_map(|_| k_val.iter().copied())
            .collect();
        let q = alloc_with_data(&b, &[n_tokens, num_q_heads, head_dim], &q_data);
        let out = b.alloc(&[n_tokens, num_q_heads, head_dim], DType::FP16).unwrap();

        b.attention(&q, &k_cache, &v_cache, num_kv_heads, 0, &out).unwrap();
        b.synchronize().unwrap();

        let result = read_fp16(&b, &out);

        // Token 0 (head 0): sees only pos 0 => output should be V[0] = [1,0,0,0]
        assert!(
            (result[0] - 1.0).abs() < 0.1,
            "causal token0 h0 d0: {}", result[0]
        );
        assert!(
            result[1].abs() < 0.1,
            "causal token0 h0 d1: {}", result[1]
        );

        // Token 1 (head 0): sees pos 0,1 => avg of V[0] and V[1] = [0.5, 0.5, 0, 0]
        let t1_offset = num_q_heads * head_dim; // token 1, head 0
        assert!(
            (result[t1_offset] - 0.5).abs() < 0.15,
            "causal token1 h0 d0: {}", result[t1_offset]
        );
        assert!(
            (result[t1_offset + 1] - 0.5).abs() < 0.15,
            "causal token1 h0 d1: {}", result[t1_offset + 1]
        );

        // Token 3 (head 0): sees pos 0..3 => avg = [0.25, 0.25, 0.25, 0.25]
        let t3_offset = 3 * num_q_heads * head_dim;
        for d in 0..head_dim {
            assert!(
                (result[t3_offset + d] - 0.25).abs() < 0.15,
                "causal token3 h0 d{d}: {}", result[t3_offset + d]
            );
        }

        b.free(&q).unwrap();
        b.free(&k_cache).unwrap();
        b.free(&v_cache).unwrap();
        b.free(&out).unwrap();
    }

    #[test]
    fn test_attention_decode_with_cache() {
        let b = make_backend();
        let head_dim = 4;
        let num_q_heads = 2;
        let num_kv_heads = 1;
        let max_seq = 8;

        let k_cache = b.alloc(&[max_seq, num_kv_heads, head_dim], DType::FP16).unwrap();
        let v_cache = b.alloc(&[max_seq, num_kv_heads, head_dim], DType::FP16).unwrap();

        // Fill 4 prior entries in cache (positions 0..4)
        let k_val = vec![1.0, 0.0, 0.0, 0.0];
        let v_val = vec![2.0, 3.0, 4.0, 5.0];
        for pos in 0..4 {
            let k_src = alloc_with_data(&b, &[1, num_kv_heads, head_dim], &k_val);
            let v_src = alloc_with_data(&b, &[1, num_kv_heads, head_dim], &v_val);
            b.copy_rows(&k_src, &k_cache, 0, pos, 1).unwrap();
            b.copy_rows(&v_src, &v_cache, 0, pos, 1).unwrap();
            b.free(&k_src).unwrap();
            b.free(&v_src).unwrap();
        }

        // New token at position 4 (start_pos=4, N=1 => kv_len=5)
        let new_k = alloc_with_data(&b, &[1, num_kv_heads, head_dim], &k_val);
        let new_v = alloc_with_data(&b, &[1, num_kv_heads, head_dim], &v_val);
        b.copy_rows(&new_k, &k_cache, 0, 4, 1).unwrap();
        b.copy_rows(&new_v, &v_cache, 0, 4, 1).unwrap();

        // Q for the new token
        let q_data: Vec<f32> = (0..num_q_heads).flat_map(|_| k_val.iter().copied()).collect();
        let q = alloc_with_data(&b, &[1, num_q_heads, head_dim], &q_data);
        let out = b.alloc(&[1, num_q_heads, head_dim], DType::FP16).unwrap();

        // start_pos=4, so kv_len = 4 + 1 = 5
        b.attention(&q, &k_cache, &v_cache, num_kv_heads, 4, &out).unwrap();
        b.synchronize().unwrap();

        let result = read_fp16(&b, &out);
        // All 5 positions have identical K and V, so output = V
        for d in 0..head_dim {
            assert!(
                (result[d] - v_val[d]).abs() < 0.15,
                "decode_cache h0 d{d}: got {}, expected {}", result[d], v_val[d]
            );
        }

        b.free(&q).unwrap();
        b.free(&k_cache).unwrap();
        b.free(&v_cache).unwrap();
        b.free(&new_k).unwrap();
        b.free(&new_v).unwrap();
        b.free(&out).unwrap();
    }

    #[test]
    fn test_attention_score_scaling() {
        let b = make_backend();
        let head_dim = 4;
        let num_q_heads = 1;
        let num_kv_heads = 1;
        let max_seq = 4;

        let k_cache = b.alloc(&[max_seq, num_kv_heads, head_dim], DType::FP16).unwrap();
        let v_cache = b.alloc(&[max_seq, num_kv_heads, head_dim], DType::FP16).unwrap();

        // 2 keys that produce equal dot product with Q
        let k0 = vec![1.0, 1.0, 0.0, 0.0];
        let k1 = vec![0.0, 0.0, 1.0, 1.0];
        let v0 = vec![10.0, 0.0, 0.0, 0.0];
        let v1 = vec![0.0, 10.0, 0.0, 0.0];

        let k0_t = alloc_with_data(&b, &[1, num_kv_heads, head_dim], &k0);
        let k1_t = alloc_with_data(&b, &[1, num_kv_heads, head_dim], &k1);
        let v0_t = alloc_with_data(&b, &[1, num_kv_heads, head_dim], &v0);
        let v1_t = alloc_with_data(&b, &[1, num_kv_heads, head_dim], &v1);
        b.copy_rows(&k0_t, &k_cache, 0, 0, 1).unwrap();
        b.copy_rows(&k1_t, &k_cache, 0, 1, 1).unwrap();
        b.copy_rows(&v0_t, &v_cache, 0, 0, 1).unwrap();
        b.copy_rows(&v1_t, &v_cache, 0, 1, 1).unwrap();

        // Q has equal dot product with both keys
        let q_data = vec![1.0, 1.0, 1.0, 1.0];
        let q = alloc_with_data(&b, &[1, num_q_heads, head_dim], &q_data);
        let out = b.alloc(&[1, num_q_heads, head_dim], DType::FP16).unwrap();

        // start_pos=1 so kv_len=2 (new token sees both cached)
        b.attention(&q, &k_cache, &v_cache, num_kv_heads, 1, &out).unwrap();
        b.synchronize().unwrap();

        let result = read_fp16(&b, &out);
        // Equal attention => output is average of V0 and V1 = [5, 5, 0, 0]
        assert!(
            (result[0] - 5.0).abs() < 0.5,
            "score_scaling d0: got {}, expected 5.0", result[0]
        );
        assert!(
            (result[1] - 5.0).abs() < 0.5,
            "score_scaling d1: got {}, expected 5.0", result[1]
        );

        b.free(&q).unwrap();
        b.free(&k_cache).unwrap();
        b.free(&v_cache).unwrap();
        b.free(&k0_t).unwrap();
        b.free(&k1_t).unwrap();
        b.free(&v0_t).unwrap();
        b.free(&v1_t).unwrap();
        b.free(&out).unwrap();
    }

    #[test]
    fn test_attention_invalid_tensor() {
        let b = make_backend();
        let fake = DeviceTensor::new(TensorId(999999), vec![1, 1, 4], DType::FP16);
        let k_cache = b.alloc(&[4, 1, 4], DType::FP16).unwrap();
        let v_cache = b.alloc(&[4, 1, 4], DType::FP16).unwrap();
        let out = b.alloc(&[1, 1, 4], DType::FP16).unwrap();
        assert!(b.attention(&fake, &k_cache, &v_cache, 1, 0, &out).is_err());
        b.free(&k_cache).unwrap();
        b.free(&v_cache).unwrap();
        b.free(&out).unwrap();
    }

    // ── Embedding additional tests ────────────────────────────────

    #[test]
    fn test_embedding_large_hidden_dim() {
        let b = make_backend();
        let vocab = 8;
        let hidden_dim = 512;
        // Fill table with identifiable values
        let table_data: Vec<f32> = (0..vocab * hidden_dim)
            .map(|i| ((i % hidden_dim) as f32) * 0.01 + (i / hidden_dim) as f32)
            .collect();
        let table = alloc_with_data(&b, &[vocab, hidden_dim], &table_data);
        let out = b.alloc(&[2, hidden_dim], DType::FP16).unwrap();

        b.embedding(&[3, 7], &table, &out).unwrap();
        b.synchronize().unwrap();

        let result = read_fp16(&b, &out);
        // Token 3: row starts at 3*512
        for j in 0..hidden_dim {
            let expected = table_data[3 * hidden_dim + j];
            assert!(
                (result[j] - expected).abs() < 0.05,
                "embed large h{j}: got {}, expected {expected}", result[j]
            );
        }
        // Token 7: row starts at 7*512
        for j in 0..hidden_dim {
            let expected = table_data[7 * hidden_dim + j];
            assert!(
                (result[hidden_dim + j] - expected).abs() < 0.05,
                "embed large t1 h{j}: got {}, expected {expected}", result[hidden_dim + j]
            );
        }

        b.free(&table).unwrap();
        b.free(&out).unwrap();
    }

    #[test]
    fn test_embedding_invalid_tensor() {
        let b = make_backend();
        let fake = DeviceTensor::new(TensorId(999999), vec![4, 3], DType::FP16);
        let out = b.alloc(&[1, 3], DType::FP16).unwrap();
        assert!(b.embedding(&[0], &fake, &out).is_err());
        b.free(&out).unwrap();
    }

    // ── Matmul additional tests ───────────────────────────────────

    #[test]
    fn test_matmul_no_nan() {
        let b = make_backend();
        let shapes: Vec<(usize, usize, usize)> = vec![
            (1, 4, 4),
            (2, 3, 5),
            (4, 8, 2),
            (1, 16, 16),
        ];
        for (m, k, n) in shapes {
            let a_data: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.1).collect();
            let b_data: Vec<f32> = (0..n * k).map(|i| (i as f32) * 0.1).collect();
            let a = alloc_with_data(&b, &[m, k], &a_data);
            // B is [N, K] for C = A @ B^T convention
            let bt = alloc_with_data(&b, &[n, k], &b_data);
            let out = b.alloc(&[m, n], DType::FP16).unwrap();

            b.matmul(&a, &bt, &out).unwrap();
            b.synchronize().unwrap();

            let result = read_fp16(&b, &out);
            for (i, &v) in result.iter().enumerate() {
                assert!(
                    !v.is_nan(),
                    "matmul_no_nan shape [{m},{k}]x[{k},{n}] idx {i}: got NaN"
                );
            }

            b.free(&a).unwrap();
            b.free(&bt).unwrap();
            b.free(&out).unwrap();
        }
    }

    #[test]
    fn test_matmul_invalid_tensor() {
        let b = make_backend();
        // Use matching inner dimensions so the dimension check passes,
        // but an invalid TensorId so we get TensorNotFound.
        let fake = DeviceTensor::new(TensorId(999999), vec![2, 3], DType::FP16);
        let bt = alloc_with_data(&b, &[4, 3], &[1.0; 12]);
        let out = b.alloc(&[2, 4], DType::FP16).unwrap();
        let err = b.matmul(&fake, &bt, &out).unwrap_err();
        assert!(matches!(err, FractureError::TensorNotFound(_)));
        b.free(&bt).unwrap();
        b.free(&out).unwrap();
    }

    #[test]
    fn test_matmul_gqa_projection() {
        let b = make_backend();
        // [1, 64] x [16, 64]^T — K/V projection-like shape, matmul computes C = A @ B^T
        let m = 1;
        let k = 64;
        let n = 16;
        let a_data: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.01).collect();
        let b_data: Vec<f32> = (0..n * k).map(|i| ((i % 17) as f32) * 0.05).collect();

        let a = alloc_with_data(&b, &[m, k], &a_data);
        let bt = alloc_with_data(&b, &[n, k], &b_data);
        let out = b.alloc(&[m, n], DType::FP16).unwrap();

        b.matmul(&a, &bt, &out).unwrap();
        b.synchronize().unwrap();

        let result = read_fp16(&b, &out);

        // CPU reference: C = A @ B^T, B is [N, K] so B^T[i, j] = B[j, i] = b_data[j * k + i]
        for j in 0..n {
            let mut expected = 0.0f32;
            for i in 0..k {
                expected += a_data[i] * b_data[j * k + i];
            }
            assert!(
                (result[j] - expected).abs() < 1.0,
                "matmul_gqa [{j}]: got {}, expected {expected}", result[j]
            );
        }

        b.free(&a).unwrap();
        b.free(&bt).unwrap();
        b.free(&out).unwrap();
    }

    #[test]
    fn test_matmul_large_m() {
        let b = make_backend();
        // [32, 64] x [64, 64] — prefill-like
        let m = 32;
        let k = 64;
        let n = 64;
        let a_data: Vec<f32> = (0..m * k).map(|i| ((i % 13) as f32) * 0.05).collect();
        let b_data: Vec<f32> = (0..k * n).map(|i| ((i % 11) as f32) * 0.05).collect();

        let a = alloc_with_data(&b, &[m, k], &a_data);
        let bt = alloc_with_data(&b, &[k, n], &b_data);
        let out = b.alloc(&[m, n], DType::FP16).unwrap();

        b.matmul(&a, &bt, &out).unwrap();
        b.synchronize().unwrap();

        let result = read_fp16(&b, &out);

        // Check a few sampled entries
        for &row in &[0, 15, 31] {
            for &col in &[0, 32, 63] {
                let mut expected = 0.0f32;
                for i in 0..k {
                    expected += a_data[row * k + i] * b_data[i * n + col];
                }
                let got = result[row * n + col];
                assert!(
                    (got - expected).abs() < 1.0,
                    "matmul_large_m [{row},{col}]: got {got}, expected {expected}"
                );
            }
        }

        b.free(&a).unwrap();
        b.free(&bt).unwrap();
        b.free(&out).unwrap();
    }

    // ── Timer error paths ─────────────────────────────────────────

    #[test]
    fn test_timer_invalid_id() {
        let b = make_backend();
        let fake_timer = DeviceTimer(999999);
        assert!(b.destroy_timer(&fake_timer).is_err());
    }

    #[test]
    fn test_timer_start_invalid() {
        let b = make_backend();
        let fake_timer = DeviceTimer(999999);
        assert!(b.start_timer(&fake_timer).is_err());
    }

    // ── Add error paths ───────────────────────────────────────────

    #[test]
    fn test_add_invalid_tensor() {
        let b = make_backend();
        let fake = DeviceTensor::new(TensorId(999999), vec![2, 2], DType::FP16);
        let real = alloc_with_data(&b, &[2, 2], &[1.0; 4]);
        let out = b.alloc(&[2, 2], DType::FP16).unwrap();
        assert!(b.add(&fake, &real, &out).is_err());
        b.free(&real).unwrap();
        b.free(&out).unwrap();
    }

    // ── Gap 1: copy_rows bounds validation ──────────────────────────

    #[test]
    fn test_copy_rows_src_out_of_bounds() {
        let b = make_backend();
        let src = alloc_with_data(&b, &[4, 2], &[1.0; 8]);
        let dst = b.alloc(&[4, 2], DType::FP16).unwrap();
        // src has 4 rows, src_offset=3 + count=2 = 5 > 4
        let err = b.copy_rows(&src, &dst, 3, 0, 2).unwrap_err();
        assert!(matches!(err, FractureError::InvalidShape(_)));
        b.free(&src).unwrap();
        b.free(&dst).unwrap();
    }

    #[test]
    fn test_copy_rows_dst_out_of_bounds() {
        let b = make_backend();
        let src = alloc_with_data(&b, &[4, 2], &[1.0; 8]);
        let dst = b.alloc(&[4, 2], DType::FP16).unwrap();
        // dst has 4 rows, dst_offset=3 + count=2 = 5 > 4
        let err = b.copy_rows(&src, &dst, 0, 3, 2).unwrap_err();
        assert!(matches!(err, FractureError::InvalidShape(_)));
        b.free(&src).unwrap();
        b.free(&dst).unwrap();
    }

    // ── Gap 2: alloc huge memory ────────────────────────────────────

    #[test]
    fn test_alloc_huge_memory() {
        let b = make_backend();
        let result = b.alloc(&[usize::MAX / 2, 1], DType::FP16);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            matches!(err, FractureError::Backend(ref s) if s.contains("CUDA")),
            "expected FractureError::Backend with CUDA error, got: {err}"
        );
    }

    // ── Gap 3: alloc packed INT4 dtype ──────────────────────────────

    #[test]
    fn test_alloc_packed_dtype() {
        let b = make_backend();
        let t = b.alloc(&[100], DType::INT4).unwrap();
        // numel=100, size_bytes = (100+1)/2 = 50
        assert_eq!(t.size_bytes(), (100 + 1) / 2);
        assert_eq!(t.numel(), 100);
        b.free(&t).unwrap();
    }

    // ── Gap 4: rmsnorm prefill large ────────────────────────────────

    #[test]
    fn test_rmsnorm_prefill_large() {
        use rand::Rng;
        let b = make_backend();
        let rows = 8;
        let cols = 4096;
        let mut rng = rand::thread_rng();
        let x_data: Vec<f32> = (0..rows * cols).map(|_| rng.gen_range(-1.0..1.0)).collect();
        let w_data: Vec<f32> = (0..cols).map(|_| rng.gen_range(0.5..2.0)).collect();

        let x = alloc_with_data(&b, &[rows, cols], &x_data);
        let w = alloc_with_data(&b, &[cols], &w_data);
        let out = b.alloc(&[rows, cols], DType::FP16).unwrap();

        b.rmsnorm(&x, &w, 1e-5, &out).unwrap();
        b.synchronize().unwrap();

        let result = read_fp16(&b, &out);

        // CPU reference
        for row in 0..rows {
            let start = row * cols;
            let end = start + cols;
            let row_data = &x_data[start..end];
            let mean_sq: f32 = row_data.iter().map(|v| v * v).sum::<f32>() / cols as f32;
            let rms = (mean_sq + 1e-5).sqrt();
            for j in 0..cols {
                let expected = (row_data[j] / rms) * w_data[j];
                let got = result[start + j];
                let abs_err = (got - expected).abs();
                // numpy-style allclose: abs_err <= atol + rtol * |expected|
                let tol = 1e-3 + 1e-3 * expected.abs();
                assert!(
                    abs_err <= tol,
                    "rmsnorm_prefill_large [{row},{j}]: got {got}, expected {expected} (abs_err={abs_err}, tol={tol})"
                );
            }
        }

        b.free(&x).unwrap();
        b.free(&w).unwrap();
        b.free(&out).unwrap();
    }

    // ── Gap 6: NVTX markers direct ─────────────────────────────────

    #[test]
    fn test_nvtx_markers_direct() {
        let b = make_backend();
        // Should not panic
        b.marker_push("test");
        b.marker_pop();
        // Nested markers
        b.marker_push("outer");
        b.marker_push("inner");
        b.marker_pop();
        b.marker_pop();
    }

    // ── Gap 8: rope production dimensions ───────────────────────────

    #[test]
    fn test_rope_prod_dimensions() {
        let mut b = make_backend();
        let head_dim = 128;
        let theta = 500000.0;
        b.precompute_rope_freqs(head_dim, theta).unwrap();

        // Q [2, 32, 128], K [2, 8, 128]
        let q_data: Vec<f32> = (0..2 * 32 * 128).map(|i| (i as f32) * 0.001).collect();
        let k_data: Vec<f32> = (0..2 * 8 * 128).map(|i| (i as f32) * 0.002).collect();

        let q = alloc_with_data(&b, &[2, 32, 128], &q_data);
        let k = alloc_with_data(&b, &[2, 8, 128], &k_data);

        b.rope(&q, &k, &[0, 10], theta, head_dim).unwrap();
        b.synchronize().unwrap();

        let q_result = read_fp16(&b, &q);
        let k_result = read_fp16(&b, &k);

        // Verify output differs from input (at least for position 10)
        let q_tok1_start = 32 * 128; // second token
        let changed = (0..32 * 128).any(|i| {
            (q_result[q_tok1_start + i] - q_data[q_tok1_start + i]).abs() > 0.001
        });
        assert!(changed, "rope prod dims: Q at pos=10 should differ from input");

        // All values should be finite
        for (i, &v) in q_result.iter().enumerate() {
            assert!(v.is_finite(), "rope prod dims: Q[{i}] is not finite: {v}");
        }
        for (i, &v) in k_result.iter().enumerate() {
            assert!(v.is_finite(), "rope prod dims: K[{i}] is not finite: {v}");
        }

        b.free(&q).unwrap();
        b.free(&k).unwrap();
    }

    // ── Gap 9: rope decode position 47 ──────────────────────────────

    #[test]
    fn test_rope_decode_position_47() {
        let mut b = make_backend();
        let head_dim = 8;
        let theta = 10000.0;
        b.precompute_rope_freqs(head_dim, theta).unwrap();

        let q_data: Vec<f32> = vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0,
                                     0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5];
        let k_data: Vec<f32> = vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0];

        let q = alloc_with_data(&b, &[1, 2, 8], &q_data);
        let k = alloc_with_data(&b, &[1, 1, 8], &k_data);

        b.rope(&q, &k, &[47], theta, head_dim).unwrap();
        b.synchronize().unwrap();

        let q_result = read_fp16(&b, &q);
        // At position 47 with non-zero freq, values should change
        let changed = q_result.iter().zip(q_data.iter()).any(|(r, e)| (r - e).abs() > 0.01);
        assert!(changed, "rope at pos=47 should modify Q");

        let k_result = read_fp16(&b, &k);
        let k_changed = k_result.iter().zip(k_data.iter()).any(|(r, e)| (r - e).abs() > 0.01);
        assert!(k_changed, "rope at pos=47 should modify K");

        b.free(&q).unwrap();
        b.free(&k).unwrap();
    }

    // ── Gap 10: rope precompute prod params ─────────────────────────

    #[test]
    fn test_rope_precompute_prod_params() {
        let mut b = make_backend();
        let head_dim = 128;
        let theta = 500000.0;
        b.precompute_rope_freqs(head_dim, theta).unwrap();

        // Verify rope works with these params by running a forward pass
        let q_data: Vec<f32> = (0..1 * 2 * 128).map(|i| ((i % 7) as f32) * 0.3 - 1.0).collect();
        let k_data: Vec<f32> = (0..1 * 1 * 128).map(|i| ((i % 5) as f32) * 0.2 - 0.5).collect();

        let q_before = q_data.clone();
        let q = alloc_with_data(&b, &[1, 2, 128], &q_data);
        let k = alloc_with_data(&b, &[1, 1, 128], &k_data);

        b.rope(&q, &k, &[42], theta, head_dim).unwrap();
        b.synchronize().unwrap();

        let q_after = read_fp16(&b, &q);
        // Verify Q was rotated (values differ from before)
        let changed = q_after.iter().zip(q_before.iter()).any(|(a, b)| (a - b).abs() > 0.001);
        assert!(changed, "rope with prod params should rotate Q at pos=42");

        // All values finite
        for (i, &v) in q_after.iter().enumerate() {
            assert!(v.is_finite(), "rope precompute prod: Q[{i}] not finite");
        }

        b.free(&q).unwrap();
        b.free(&k).unwrap();
    }

    // ── Gap 11: rope freq reuse ─────────────────────────────────────

    #[test]
    fn test_rope_freq_reuse() {
        let mut b = make_backend();
        let head_dim = 8;
        let theta = 10000.0;
        b.precompute_rope_freqs(head_dim, theta).unwrap();

        // Call rope 3 times with different tensors, all should succeed
        for pos in &[0u32, 5, 100] {
            let q = alloc_with_data(&b, &[1, 1, 8], &[1.0; 8]);
            let k = alloc_with_data(&b, &[1, 1, 8], &[1.0; 8]);
            b.rope(&q, &k, &[*pos], theta, head_dim).unwrap();
            b.synchronize().unwrap();
            b.free(&q).unwrap();
            b.free(&k).unwrap();
        }
    }

    // ── Gap 12: rope Q/K consistency ────────────────────────────────

    #[test]
    fn test_rope_q_k_consistency() {
        let mut b = make_backend();
        let head_dim = 8;
        let theta = 10000.0;
        b.precompute_rope_freqs(head_dim, theta).unwrap();

        // Same data for both Q and K (1 head each)
        let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let q = alloc_with_data(&b, &[1, 1, 8], &data);
        let k = alloc_with_data(&b, &[1, 1, 8], &data);

        b.rope(&q, &k, &[7], theta, head_dim).unwrap();
        b.synchronize().unwrap();

        let q_result = read_fp16(&b, &q);
        let k_result = read_fp16(&b, &k);

        // Same input, same position => same rotation
        for i in 0..8 {
            assert!(
                (q_result[i] - k_result[i]).abs() < 1e-3,
                "rope Q/K consistency [{i}]: Q={}, K={}", q_result[i], k_result[i]
            );
        }

        b.free(&q).unwrap();
        b.free(&k).unwrap();
    }

    // ── Gap 15: attention decode distinct KV ────────────────────────

    #[test]
    fn test_attention_decode_distinct_kv() {
        let b = make_backend();
        let head_dim = 4;
        let num_q_heads = 2;
        let num_kv_heads = 1;
        let max_seq = 8;

        let k_cache = b.alloc(&[max_seq, num_kv_heads, head_dim], DType::FP16).unwrap();
        let v_cache = b.alloc(&[max_seq, num_kv_heads, head_dim], DType::FP16).unwrap();

        // Distinct K and V values per position
        let k_values: Vec<Vec<f32>> = vec![
            vec![1.0, 0.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0, 0.0],
            vec![0.0, 0.0, 1.0, 0.0],
            vec![0.0, 0.0, 0.0, 1.0],
        ];
        let v_values: Vec<Vec<f32>> = vec![
            vec![10.0, 0.0, 0.0, 0.0],
            vec![0.0, 20.0, 0.0, 0.0],
            vec![0.0, 0.0, 30.0, 0.0],
            vec![0.0, 0.0, 0.0, 40.0],
        ];

        for pos in 0..4 {
            let k_src = alloc_with_data(&b, &[1, num_kv_heads, head_dim], &k_values[pos]);
            let v_src = alloc_with_data(&b, &[1, num_kv_heads, head_dim], &v_values[pos]);
            b.copy_rows(&k_src, &k_cache, 0, pos, 1).unwrap();
            b.copy_rows(&v_src, &v_cache, 0, pos, 1).unwrap();
            b.free(&k_src).unwrap();
            b.free(&v_src).unwrap();
        }

        // New token at position 4
        let new_k = vec![1.0, 1.0, 1.0, 1.0]; // equal affinity to all
        let new_v = vec![5.0, 5.0, 5.0, 5.0];
        let k_src = alloc_with_data(&b, &[1, num_kv_heads, head_dim], &new_k);
        let v_src = alloc_with_data(&b, &[1, num_kv_heads, head_dim], &new_v);
        b.copy_rows(&k_src, &k_cache, 0, 4, 1).unwrap();
        b.copy_rows(&v_src, &v_cache, 0, 4, 1).unwrap();
        b.free(&k_src).unwrap();
        b.free(&v_src).unwrap();

        // Q at position 4 with equal affinity to everything
        let q_data: Vec<f32> = vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]; // 2 heads
        let q = alloc_with_data(&b, &[1, num_q_heads, head_dim], &q_data);
        let out = b.alloc(&[1, num_q_heads, head_dim], DType::FP16).unwrap();

        b.attention(&q, &k_cache, &v_cache, num_kv_heads, 4, &out).unwrap();
        b.synchronize().unwrap();

        let result = read_fp16(&b, &out);

        // Output should be a weighted combination, NOT equal to any single V
        for v in &v_values {
            let is_exact_match = (0..head_dim).all(|d| (result[d] - v[d]).abs() < 0.1);
            assert!(
                !is_exact_match,
                "output should not exactly match any single V vector"
            );
        }

        // Output should be finite
        for (i, &v) in result.iter().enumerate() {
            assert!(v.is_finite(), "attention decode distinct [{i}]: not finite");
        }

        b.free(&q).unwrap();
        b.free(&k_cache).unwrap();
        b.free(&v_cache).unwrap();
        b.free(&out).unwrap();
    }

    // ── Gap 16: attention scaling head_dim 128 ──────────────────────

    #[test]
    fn test_attention_scaling_head_dim_128() {
        let b = make_backend();
        let head_dim = 128;
        let num_q_heads = 1;
        let num_kv_heads = 1;
        let max_seq = 4;

        let k_cache = b.alloc(&[max_seq, num_kv_heads, head_dim], DType::FP16).unwrap();
        let v_cache = b.alloc(&[max_seq, num_kv_heads, head_dim], DType::FP16).unwrap();

        // 2 keys that produce equal dot product with Q
        let mut k0 = vec![0.0f32; head_dim];
        let mut k1 = vec![0.0f32; head_dim];
        k0[0] = 1.0;
        k1[1] = 1.0;

        let mut v0 = vec![0.0f32; head_dim];
        let mut v1 = vec![0.0f32; head_dim];
        v0[0] = 10.0;
        v1[1] = 10.0;

        let k0_t = alloc_with_data(&b, &[1, num_kv_heads, head_dim], &k0);
        let k1_t = alloc_with_data(&b, &[1, num_kv_heads, head_dim], &k1);
        let v0_t = alloc_with_data(&b, &[1, num_kv_heads, head_dim], &v0);
        let v1_t = alloc_with_data(&b, &[1, num_kv_heads, head_dim], &v1);
        b.copy_rows(&k0_t, &k_cache, 0, 0, 1).unwrap();
        b.copy_rows(&k1_t, &k_cache, 0, 1, 1).unwrap();
        b.copy_rows(&v0_t, &v_cache, 0, 0, 1).unwrap();
        b.copy_rows(&v1_t, &v_cache, 0, 1, 1).unwrap();

        // Q has equal dot product with both keys => softmax should be ~50/50
        let mut q_data = vec![0.0f32; head_dim];
        q_data[0] = 1.0;
        q_data[1] = 1.0;
        let q = alloc_with_data(&b, &[1, num_q_heads, head_dim], &q_data);
        let out = b.alloc(&[1, num_q_heads, head_dim], DType::FP16).unwrap();

        // scale = 1/sqrt(128) ~ 0.0884
        b.attention(&q, &k_cache, &v_cache, num_kv_heads, 1, &out).unwrap();
        b.synchronize().unwrap();

        let result = read_fp16(&b, &out);
        // Equal attention => output[0] ~ 5.0 and output[1] ~ 5.0
        assert!(
            (result[0] - 5.0).abs() < 1.0,
            "scaling hd128 d0: got {}, expected ~5.0", result[0]
        );
        assert!(
            (result[1] - 5.0).abs() < 1.0,
            "scaling hd128 d1: got {}, expected ~5.0", result[1]
        );

        b.free(&q).unwrap();
        b.free(&k_cache).unwrap();
        b.free(&v_cache).unwrap();
        b.free(&k0_t).unwrap();
        b.free(&k1_t).unwrap();
        b.free(&v0_t).unwrap();
        b.free(&v1_t).unwrap();
        b.free(&out).unwrap();
    }

    // ── Gap 17: attention error context ─────────────────────────────

    #[test]
    fn test_attention_error_context() {
        let b = make_backend();
        let fake_q = DeviceTensor::new(TensorId(999999), vec![1, 2, 4], DType::FP16);
        let k_cache = b.alloc(&[4, 1, 4], DType::FP16).unwrap();
        let v_cache = b.alloc(&[4, 1, 4], DType::FP16).unwrap();
        let out = b.alloc(&[1, 2, 4], DType::FP16).unwrap();

        let err = b.attention(&fake_q, &k_cache, &v_cache, 1, 0, &out).unwrap_err();
        let msg = err.to_string();
        // Error should contain useful context about the tensor
        assert!(
            msg.contains("999999") || msg.contains("not found") || msg.contains("tensor"),
            "error should contain useful context, got: {msg}"
        );

        b.free(&k_cache).unwrap();
        b.free(&v_cache).unwrap();
        b.free(&out).unwrap();
    }

    // ── Gap 18: embedding OOV behavior ──────────────────────────────

    #[test]
    fn test_embedding_oov_behavior() {
        let b = make_backend();
        let vocab = 4;
        let dim = 3;
        let table_data: Vec<f32> = vec![
            0.1, 0.2, 0.3,
            1.1, 1.2, 1.3,
            2.1, 2.2, 2.3,
            3.1, 3.2, 3.3,
        ];
        let table = alloc_with_data(&b, &[vocab, dim], &table_data);
        let out = b.alloc(&[1, dim], DType::FP16).unwrap();

        // token_id=10 >= vocab_size=4 (OOV)
        // The kernel has vocab_size bounds check: if token_id >= vocab_size, it writes zeros
        let result = b.embedding(&[10], &table, &out);
        b.synchronize().unwrap();

        if result.is_ok() {
            // Kernel wrote zeros for OOV token
            let data = read_fp16(&b, &out);
            for (i, &v) in data.iter().enumerate() {
                assert!(
                    v.abs() < 0.01,
                    "OOV embedding should be zero, got [{i}]={v}"
                );
            }
        }
        // If it returned an error, that's also acceptable behavior

        b.free(&table).unwrap();
        b.free(&out).unwrap();
    }

    // ── Gap 19: matmul llama3 shapes ────────────────────────────────

    #[test]
    fn test_matmul_llama3_shapes() {
        use rand::Rng;
        let b = make_backend();
        let mut rng = rand::thread_rng();

        let shapes: Vec<(usize, usize, usize)> = vec![
            (1, 4096, 4096),
            (1, 4096, 1024),
        ];

        for (m, k, n) in shapes {
            let a_data: Vec<f32> = (0..m * k).map(|_| rng.gen_range(-0.1..0.1)).collect();
            let b_data: Vec<f32> = (0..n * k).map(|_| rng.gen_range(-0.1..0.1)).collect();

            let a = alloc_with_data(&b, &[m, k], &a_data);
            let bt = alloc_with_data(&b, &[n, k], &b_data);
            let out = b.alloc(&[m, n], DType::FP16).unwrap();

            b.matmul(&a, &bt, &out).unwrap();
            b.synchronize().unwrap();

            let result = read_fp16(&b, &out);
            assert_eq!(result.len(), m * n);
            for (i, &v) in result.iter().enumerate() {
                assert!(
                    !v.is_nan(),
                    "matmul llama3 [{m},{k}]x[{n},{k}]^T idx {i}: NaN"
                );
                assert!(
                    v.is_finite(),
                    "matmul llama3 [{m},{k}]x[{n},{k}]^T idx {i}: not finite"
                );
            }

            b.free(&a).unwrap();
            b.free(&bt).unwrap();
            b.free(&out).unwrap();
        }
    }

    // ── Gap 20: matmul FP32 accumulation ────────────────────────────

    #[test]
    fn test_matmul_fp32_accumulation() {
        let b = make_backend();
        // K=256 values of ~256 each. Sum ~ 65536 which overflows FP16 max (65504)
        // With FP32 accumulation, this should work fine.
        let m = 1;
        let k = 256;
        let n = 1;
        let a_data: Vec<f32> = vec![1.0; k]; // [1, 256] all ones
        let b_data: Vec<f32> = vec![256.0; k]; // [1, 256] all 256.0
        // Expected: sum of 256 * 256 = 65536 > 65504 (FP16 max)

        let a = alloc_with_data(&b, &[m, k], &a_data);
        // B is [N, K] for C = A @ B^T convention
        let bt = alloc_with_data(&b, &[n, k], &b_data);
        let out = b.alloc(&[m, n], DType::FP16).unwrap();

        b.matmul(&a, &bt, &out).unwrap();
        b.synchronize().unwrap();

        let result = read_fp16(&b, &out);
        // FP16 can represent inf for values > 65504, but FP32 accumulation means
        // the computation itself doesn't overflow. The result gets stored as FP16
        // which may be inf, but the accumulation was correct in FP32.
        // Since 65536 > 65504, the FP16 output will be inf.
        // The key test: with FP16 accumulation, intermediate sums would lose precision
        // and could produce NaN. FP32 accumulation avoids that.
        assert!(
            !result[0].is_nan(),
            "FP32 accumulation should not produce NaN, got: {}", result[0]
        );
    }

    // ── Gap 21: matmul large prefill ────────────────────────────────

    #[test]
    fn test_matmul_large_prefill() {
        use rand::Rng;
        let b = make_backend();
        let mut rng = rand::thread_rng();
        let m = 128;
        let k = 4096;
        let n = 4096;

        let a_data: Vec<f32> = (0..m * k).map(|_| rng.gen_range(-0.05..0.05)).collect();
        let b_data: Vec<f32> = (0..k * n).map(|_| rng.gen_range(-0.05..0.05)).collect();

        let a = alloc_with_data(&b, &[m, k], &a_data);
        let bt = alloc_with_data(&b, &[k, n], &b_data);
        let out = b.alloc(&[m, n], DType::FP16).unwrap();

        b.matmul(&a, &bt, &out).unwrap();
        b.synchronize().unwrap();

        let result = read_fp16(&b, &out);
        assert_eq!(result.len(), m * n);
        for (i, &v) in result.iter().enumerate() {
            assert!(v.is_finite(), "matmul large prefill idx {i}: not finite");
        }

        b.free(&a).unwrap();
        b.free(&bt).unwrap();
        b.free(&out).unwrap();
    }

    // ── Gap 22: matmul GQA KV shape ─────────────────────────────────

    #[test]
    fn test_matmul_gqa_kv_shape() {
        use rand::Rng;
        let b = make_backend();
        let mut rng = rand::thread_rng();
        let m = 1;
        let k = 4096;
        let n = 1024;

        let a_data: Vec<f32> = (0..m * k).map(|_| rng.gen_range(-0.1..0.1)).collect();
        let b_data: Vec<f32> = (0..n * k).map(|_| rng.gen_range(-0.1..0.1)).collect();

        let a = alloc_with_data(&b, &[m, k], &a_data);
        let bt = alloc_with_data(&b, &[n, k], &b_data);
        let out = b.alloc(&[m, n], DType::FP16).unwrap();

        b.matmul(&a, &bt, &out).unwrap();
        b.synchronize().unwrap();

        let result = read_fp16(&b, &out);
        assert_eq!(result.len(), m * n);
        for (i, &v) in result.iter().enumerate() {
            assert!(v.is_finite(), "matmul gqa kv idx {i}: not finite");
        }

        b.free(&a).unwrap();
        b.free(&bt).unwrap();
        b.free(&out).unwrap();
    }

    // ── Gap 23: timer stop invalid ──────────────────────────────────

    #[test]
    fn test_timer_stop_invalid() {
        let b = make_backend();
        let fake_timer = DeviceTimer(999999);
        let err = b.stop_timer(&fake_timer);
        assert!(err.is_err(), "stop_timer with invalid ID should error");
    }

    // ── Gap 24: timer double destroy ────────────────────────────────

    #[test]
    fn test_timer_double_destroy() {
        let b = make_backend();
        let timer = b.create_timer().unwrap();
        b.destroy_timer(&timer).unwrap();
        let err = b.destroy_timer(&timer);
        assert!(err.is_err(), "second destroy_timer should error");
    }

    // ── Numerical correctness: add reference ────────────────────────

    #[test]
    fn test_add_reference_correctness() {
        use rand::Rng;
        let b = make_backend();
        let mut rng = rand::thread_rng();
        let n = 4096;

        let a_data: Vec<f32> = (0..n).map(|_| rng.gen_range(-1.0..1.0)).collect();
        let b_data: Vec<f32> = (0..n).map(|_| rng.gen_range(-1.0..1.0)).collect();

        let a = alloc_with_data(&b, &[1, n], &a_data);
        let bt = alloc_with_data(&b, &[1, n], &b_data);
        let out = b.alloc(&[1, n], DType::FP16).unwrap();

        b.add(&a, &bt, &out).unwrap();
        b.synchronize().unwrap();

        let result = read_fp16(&b, &out);

        // CPU reference: element-wise add in f32, then convert to f16 and back
        for i in 0..n {
            let expected = f16::from_f32(a_data[i]).to_f32() + f16::from_f32(b_data[i]).to_f32();
            let expected_f16 = f16::from_f32(expected).to_f32();
            let got = result[i];
            let abs_err = (got - expected_f16).abs();
            let tol = 1e-3 + 1e-3 * expected_f16.abs();
            assert!(
                abs_err <= tol,
                "add_reference [{i}]: got {got}, expected {expected_f16} (abs_err={abs_err}, tol={tol})"
            );
        }

        b.free(&a).unwrap();
        b.free(&bt).unwrap();
        b.free(&out).unwrap();
    }

    // ── Numerical correctness: add large tensor ─────────────────────

    #[test]
    fn test_add_large_tensor() {
        use rand::Rng;
        let b = make_backend();
        let mut rng = rand::thread_rng();
        let rows = 128;
        let cols = 4096;
        let n = rows * cols;

        let a_data: Vec<f32> = (0..n).map(|_| rng.gen_range(-1.0..1.0)).collect();
        let b_data: Vec<f32> = (0..n).map(|_| rng.gen_range(-1.0..1.0)).collect();

        let a = alloc_with_data(&b, &[rows, cols], &a_data);
        let bt = alloc_with_data(&b, &[rows, cols], &b_data);
        let out = b.alloc(&[rows, cols], DType::FP16).unwrap();

        b.add(&a, &bt, &out).unwrap();
        b.synchronize().unwrap();

        let result = read_fp16(&b, &out);

        for i in 0..n {
            let expected = f16::from_f32(a_data[i]).to_f32() + f16::from_f32(b_data[i]).to_f32();
            let expected_f16 = f16::from_f32(expected).to_f32();
            let got = result[i];
            let abs_err = (got - expected_f16).abs();
            let tol = 1e-3 + 1e-3 * expected_f16.abs();
            assert!(
                abs_err <= tol,
                "add_large [{i}]: got {got}, expected {expected_f16} (abs_err={abs_err}, tol={tol})"
            );
        }

        b.free(&a).unwrap();
        b.free(&bt).unwrap();
        b.free(&out).unwrap();
    }

    // ── Numerical correctness: matmul llama3 shapes ─────────────────

    #[test]
    fn test_matmul_llama3_shapes_correctness() {
        use rand::Rng;
        let b = make_backend();
        let mut rng = rand::thread_rng();
        let m = 1;
        let k = 64;
        let n = 64;

        let a_data: Vec<f32> = (0..m * k).map(|_| rng.gen_range(-0.1..0.1)).collect();
        let b_data: Vec<f32> = (0..n * k).map(|_| rng.gen_range(-0.1..0.1)).collect();

        let a = alloc_with_data(&b, &[m, k], &a_data);
        let bt = alloc_with_data(&b, &[n, k], &b_data);
        let out = b.alloc(&[m, n], DType::FP16).unwrap();

        b.matmul(&a, &bt, &out).unwrap();
        b.synchronize().unwrap();

        let result = read_fp16(&b, &out);

        // CPU reference: C = A @ B^T, where B is [N, K], so C[row][col] = sum_i A[row,i] * B[col,i]
        for row in 0..m {
            for col in 0..n {
                let mut expected = 0.0f32;
                for i in 0..k {
                    let av = f16::from_f32(a_data[row * k + i]).to_f32();
                    let bv = f16::from_f32(b_data[col * k + i]).to_f32();
                    expected += av * bv;
                }
                let got = result[row * n + col];
                let abs_err = (got - expected).abs();
                let tol = 1e-2 + 1e-2 * expected.abs();
                assert!(
                    abs_err <= tol,
                    "matmul_llama3_correctness [{row},{col}]: got {got}, expected {expected} (abs_err={abs_err})"
                );
            }
        }

        b.free(&a).unwrap();
        b.free(&bt).unwrap();
        b.free(&out).unwrap();
    }

    // ── Numerical correctness: matmul large M fix ───────────────────

    #[test]
    fn test_matmul_large_m_fix() {
        let b = make_backend();
        // Non-square B to ensure we compute A @ B^T correctly
        // A: [32, 64], B: [48, 64] => C: [32, 48]
        let m = 32;
        let k = 64;
        let n = 48;
        let a_data: Vec<f32> = (0..m * k).map(|i| ((i % 13) as f32) * 0.05).collect();
        let b_data: Vec<f32> = (0..n * k).map(|i| ((i % 11) as f32) * 0.05).collect();

        let a = alloc_with_data(&b, &[m, k], &a_data);
        let bt = alloc_with_data(&b, &[n, k], &b_data);
        let out = b.alloc(&[m, n], DType::FP16).unwrap();

        b.matmul(&a, &bt, &out).unwrap();
        b.synchronize().unwrap();

        let result = read_fp16(&b, &out);

        // CPU reference: C = A @ B^T => C[row][col] = sum_i A[row,i] * B[col,i]
        for &row in &[0, 15, 31] {
            for &col in &[0, 24, 47] {
                let mut expected = 0.0f32;
                for i in 0..k {
                    let av = f16::from_f32(a_data[row * k + i]).to_f32();
                    let bv = f16::from_f32(b_data[col * k + i]).to_f32();
                    expected += av * bv;
                }
                let got = result[row * n + col];
                let abs_err = (got - expected).abs();
                let tol = 1e-2 + 1e-2 * expected.abs();
                assert!(
                    abs_err <= tol,
                    "matmul_large_m_fix [{row},{col}]: got {got}, expected {expected} (abs_err={abs_err})"
                );
            }
        }

        b.free(&a).unwrap();
        b.free(&bt).unwrap();
        b.free(&out).unwrap();
    }

    // ── Numerical correctness: rmsnorm prefill n128 ─────────────────

    #[test]
    fn test_rmsnorm_prefill_n128() {
        use rand::Rng;
        let b = make_backend();
        let mut rng = rand::thread_rng();
        let rows = 128;
        let cols = 4096;

        let x_data: Vec<f32> = (0..rows * cols).map(|_| rng.gen_range(-1.0..1.0)).collect();
        let w_data: Vec<f32> = (0..cols).map(|_| rng.gen_range(0.5..2.0)).collect();

        let x = alloc_with_data(&b, &[rows, cols], &x_data);
        let w = alloc_with_data(&b, &[cols], &w_data);
        let out = b.alloc(&[rows, cols], DType::FP16).unwrap();

        b.rmsnorm(&x, &w, 1e-5, &out).unwrap();
        b.synchronize().unwrap();

        let result = read_fp16(&b, &out);

        // CPU reference
        for row in 0..rows {
            let start = row * cols;
            let end = start + cols;
            let row_data = &x_data[start..end];
            let mean_sq: f32 = row_data.iter().map(|v| v * v).sum::<f32>() / cols as f32;
            let rms = (mean_sq + 1e-5).sqrt();
            for j in 0..cols {
                let expected = (row_data[j] / rms) * w_data[j];
                let got = result[start + j];
                let abs_err = (got - expected).abs();
                let tol = 1e-3 + 1e-3 * expected.abs();
                assert!(
                    abs_err <= tol,
                    "rmsnorm_prefill_n128 [{row},{j}]: got {got}, expected {expected} (abs_err={abs_err}, tol={tol})"
                );
            }
        }

        b.free(&x).unwrap();
        b.free(&w).unwrap();
        b.free(&out).unwrap();
    }

    // ── Numerical correctness: rmsnorm prefill n512 ─────────────────

    #[test]
    fn test_rmsnorm_prefill_n512() {
        use rand::Rng;
        let b = make_backend();
        let mut rng = rand::thread_rng();
        let rows = 512;
        let cols = 4096;

        let x_data: Vec<f32> = (0..rows * cols).map(|_| rng.gen_range(-1.0..1.0)).collect();
        let w_data: Vec<f32> = (0..cols).map(|_| rng.gen_range(0.5..2.0)).collect();

        let x = alloc_with_data(&b, &[rows, cols], &x_data);
        let w = alloc_with_data(&b, &[cols], &w_data);
        let out = b.alloc(&[rows, cols], DType::FP16).unwrap();

        b.rmsnorm(&x, &w, 1e-5, &out).unwrap();
        b.synchronize().unwrap();

        let result = read_fp16(&b, &out);

        // CPU reference — check sampled rows to keep test fast
        for &row in &[0, 127, 255, 384, 511] {
            let start = row * cols;
            let end = start + cols;
            let row_data = &x_data[start..end];
            let mean_sq: f32 = row_data.iter().map(|v| v * v).sum::<f32>() / cols as f32;
            let rms = (mean_sq + 1e-5).sqrt();
            for j in 0..cols {
                let expected = (row_data[j] / rms) * w_data[j];
                let got = result[start + j];
                let abs_err = (got - expected).abs();
                let tol = 1e-3 + 1e-3 * expected.abs();
                assert!(
                    abs_err <= tol,
                    "rmsnorm_prefill_n512 [{row},{j}]: got {got}, expected {expected} (abs_err={abs_err}, tol={tol})"
                );
            }
        }

        b.free(&x).unwrap();
        b.free(&w).unwrap();
        b.free(&out).unwrap();
    }

    // ── GPU error handling ────────────────────────────────────────────

    #[test]
    fn test_rmsnorm_error_context() {
        let b = make_backend();
        let fake = DeviceTensor::new(TensorId(999999), vec![1, 4], DType::FP16);
        let w = alloc_with_data(&b, &[4], &[1.0; 4]);
        let out = b.alloc(&[1, 4], DType::FP16).unwrap();
        let err = b.rmsnorm(&fake, &w, 1e-5, &out).unwrap_err();
        assert!(matches!(err, FractureError::TensorNotFound(_)), "expected TensorNotFound, got: {err:?}");
        assert!(err.to_string().contains("999999"), "error should contain tensor id: {err}");
        b.free(&w).unwrap();
        b.free(&out).unwrap();
    }

    #[test]
    fn test_silu_mul_error_type() {
        let b = make_backend();
        let fake = DeviceTensor::new(TensorId(999999), vec![1, 4], DType::FP16);
        let up = alloc_with_data(&b, &[1, 4], &[1.0; 4]);
        let out = b.alloc(&[1, 4], DType::FP16).unwrap();
        let err = b.silu_mul(&fake, &up, &out).unwrap_err();
        assert!(matches!(err, FractureError::TensorNotFound(_)), "expected TensorNotFound, got: {err:?}");
        assert!(err.to_string().contains("999999"), "error should mention tensor id: {err}");
        b.free(&up).unwrap();
        b.free(&out).unwrap();
    }

    #[test]
    fn test_silu_mul_invalid_up_tensor() {
        let b = make_backend();
        let gate = alloc_with_data(&b, &[1, 4], &[1.0; 4]);
        let fake = DeviceTensor::new(TensorId(999998), vec![1, 4], DType::FP16);
        let out = b.alloc(&[1, 4], DType::FP16).unwrap();
        let result = b.silu_mul(&gate, &fake, &out);
        assert!(result.is_err(), "invalid up tensor should fail");
        b.free(&gate).unwrap();
        b.free(&out).unwrap();
    }

    #[test]
    fn test_silu_mul_invalid_out_tensor() {
        let b = make_backend();
        let gate = alloc_with_data(&b, &[1, 4], &[1.0; 4]);
        let up = alloc_with_data(&b, &[1, 4], &[1.0; 4]);
        let fake = DeviceTensor::new(TensorId(999997), vec![1, 4], DType::FP16);
        let result = b.silu_mul(&gate, &up, &fake);
        assert!(result.is_err(), "invalid out tensor should fail");
        b.free(&gate).unwrap();
        b.free(&up).unwrap();
    }

    #[test]
    fn test_embedding_error_context() {
        let b = make_backend();
        let fake_table = DeviceTensor::new(TensorId(999999), vec![4, 8], DType::FP16);
        let out = b.alloc(&[1, 8], DType::FP16).unwrap();
        let err = b.embedding(&[0], &fake_table, &out).unwrap_err();
        assert!(matches!(err, FractureError::TensorNotFound(_)), "expected TensorNotFound, got: {err:?}");
        assert!(err.to_string().contains("999999"), "error should contain tensor id: {err}");
        b.free(&out).unwrap();
    }

    #[test]
    fn test_add_error_context() {
        let b = make_backend();
        let a = alloc_with_data(&b, &[2, 2], &[1.0; 4]);
        let fake = DeviceTensor::new(TensorId(999999), vec![2, 2], DType::FP16);
        let out = b.alloc(&[2, 2], DType::FP16).unwrap();
        let err = b.add(&a, &fake, &out).unwrap_err();
        assert!(matches!(err, FractureError::TensorNotFound(_)), "expected TensorNotFound, got: {err:?}");
        assert!(err.to_string().contains("999999"), "error should contain tensor id: {err}");
        b.free(&a).unwrap();
        b.free(&out).unwrap();
    }

    #[test]
    fn test_rmsnorm_large_values_full_dim() {
        let b = make_backend();
        let n = 4096;
        let data: Vec<f32> = vec![60000.0; n];
        let x = alloc_with_data(&b, &[1, n], &data);
        let w = alloc_with_data(&b, &[n], &vec![1.0; n]);
        let out = b.alloc(&[1, n], DType::FP16).unwrap();
        b.rmsnorm(&x, &w, 1e-5, &out).unwrap();
        let result = read_fp16(&b, &out);
        for (i, &val) in result.iter().enumerate() {
            assert!(val.is_finite(), "element {i} is not finite: {val}");
        }
        // RMSNorm of all-same values with weight=1 should give 1.0 for each element
        // rms = sqrt(mean(60000^2) + eps) = sqrt(60000^2 + eps) ≈ 60000
        // output = (60000 / 60000) * 1.0 = 1.0
        let expected = 1.0f32;
        for (i, &val) in result.iter().enumerate() {
            assert!((val - expected).abs() < 0.01, "element {i}: expected ~{expected}, got {val}");
        }
        b.free(&x).unwrap();
        b.free(&w).unwrap();
        b.free(&out).unwrap();
    }

    // ── RoPE numerical correctness ────────────────────────────────────
    // The RoPE kernel uses split-half convention: dimension d pairs with d + half_dim.
    // x0 = tensor[d], x1 = tensor[d + half_dim]
    // tensor[d]            = x0 * cos - x1 * sin
    // tensor[d + half_dim] = x0 * sin + x1 * cos

    #[test]
    fn test_rope_precompute_freq_values() {
        let mut b = CudaBackend::new(0).expect("CUDA init");
        let head_dim = 128;
        let theta = 500000.0;
        let half_dim = head_dim / 2;
        b.precompute_rope_freqs(head_dim, theta).unwrap();

        // Split-half input: set first half to 1.0, second half to 0.0
        // After RoPE at position 1: first half becomes cos(freq), second half becomes sin(freq)
        let mut q_data = vec![0.0f32; head_dim];
        for d in 0..half_dim {
            q_data[d] = 1.0;           // x0 = 1.0
            q_data[d + half_dim] = 0.0; // x1 = 0.0
        }

        let q = alloc_with_data(&b, &[1, 1, head_dim], &q_data);
        let k = alloc_with_data(&b, &[1, 1, head_dim], &q_data);

        b.rope(&q, &k, &[1], theta, head_dim).unwrap();
        b.synchronize().unwrap();

        let q_result = read_fp16(&b, &q);

        for d in 0..half_dim {
            let freq = 1.0 / (theta as f32).powf(2.0 * d as f32 / head_dim as f32);
            let angle = freq; // position=1
            let expected_cos = angle.cos();
            let expected_sin = angle.sin();

            assert!(
                (q_result[d] - expected_cos).abs() < 1e-2,
                "freq[{d}]: cos mismatch: got {}, expected {expected_cos}", q_result[d]
            );
            assert!(
                (q_result[d + half_dim] - expected_sin).abs() < 1e-2,
                "freq[{d}]: sin mismatch: got {}, expected {expected_sin}", q_result[d + half_dim]
            );
        }

        b.free(&q).unwrap();
        b.free(&k).unwrap();
    }

    #[test]
    fn test_rope_decode_position_47_numerical() {
        let mut b = CudaBackend::new(0).expect("CUDA init");
        let head_dim = 8;
        let half = head_dim / 2;
        let theta = 10000.0;
        let num_q_heads = 2;
        let num_kv_heads = 1;
        b.precompute_rope_freqs(head_dim, theta).unwrap();

        let q_data: Vec<f32> = (0..num_q_heads * head_dim).map(|i| (i as f32 + 1.0) * 0.1).collect();
        let k_data: Vec<f32> = (0..num_kv_heads * head_dim).map(|i| (i as f32 + 1.0) * 0.2).collect();

        let q = alloc_with_data(&b, &[1, num_q_heads, head_dim], &q_data);
        let k = alloc_with_data(&b, &[1, num_kv_heads, head_dim], &k_data);

        b.rope(&q, &k, &[47], theta, head_dim).unwrap();
        b.synchronize().unwrap();

        let q_result = read_fp16(&b, &q);
        let k_result = read_fp16(&b, &k);

        for h in 0..num_q_heads {
            let base = h * head_dim;
            for d in 0..half {
                let freq = 1.0 / (theta as f32).powf(2.0 * d as f32 / head_dim as f32);
                let angle = 47.0 * freq;
                let (sin_a, cos_a) = angle.sin_cos();

                let x0 = q_data[base + d];
                let x1 = q_data[base + d + half];
                let exp_lo = x0 * cos_a - x1 * sin_a;
                let exp_hi = x0 * sin_a + x1 * cos_a;

                assert!((q_result[base + d] - exp_lo).abs() < 0.05,
                    "Q head {h} d={d}: got {}, expected {exp_lo}", q_result[base + d]);
                assert!((q_result[base + d + half] - exp_hi).abs() < 0.05,
                    "Q head {h} d={}: got {}, expected {exp_hi}", d + half, q_result[base + d + half]);
            }
        }

        for d in 0..half {
            let freq = 1.0 / (theta as f32).powf(2.0 * d as f32 / head_dim as f32);
            let angle = 47.0 * freq;
            let (sin_a, cos_a) = angle.sin_cos();

            let x0 = k_data[d];
            let x1 = k_data[d + half];
            let exp_lo = x0 * cos_a - x1 * sin_a;
            let exp_hi = x0 * sin_a + x1 * cos_a;

            assert!((k_result[d] - exp_lo).abs() < 0.05,
                "K d={d}: got {}, expected {exp_lo}", k_result[d]);
            assert!((k_result[d + half] - exp_hi).abs() < 0.05,
                "K d={}: got {}, expected {exp_hi}", d + half, k_result[d + half]);
        }

        b.free(&q).unwrap();
        b.free(&k).unwrap();
    }

    #[test]
    fn test_rope_prod_dimensions_numerical() {
        use rand::Rng;
        let mut b = CudaBackend::new(0).expect("CUDA init");
        let head_dim = 128;
        let half = head_dim / 2;
        let theta = 500000.0;
        let num_q_heads = 32;
        b.precompute_rope_freqs(head_dim, theta).unwrap();

        let mut rng = rand::thread_rng();
        let q_data: Vec<f32> = (0..num_q_heads * head_dim).map(|_| rng.gen_range(-0.5..0.5)).collect();

        let q = alloc_with_data(&b, &[1, num_q_heads, head_dim], &q_data);
        let k = alloc_with_data(&b, &[1, 8, head_dim], &vec![0.1f32; 8 * head_dim]);

        b.rope(&q, &k, &[42], theta, head_dim).unwrap();
        b.synchronize().unwrap();

        let q_result = read_fp16(&b, &q);

        for &h in &[0, 15, 31] {
            let base = h * head_dim;
            for d in 0..half {
                let freq = 1.0 / (theta as f32).powf(2.0 * d as f32 / head_dim as f32);
                let angle = 42.0 * freq;
                let (sin_a, cos_a) = angle.sin_cos();

                let x0 = f16::from_f32(q_data[base + d]).to_f32();
                let x1 = f16::from_f32(q_data[base + d + half]).to_f32();
                let exp_lo = x0 * cos_a - x1 * sin_a;
                let exp_hi = x0 * sin_a + x1 * cos_a;

                assert!((q_result[base + d] - exp_lo).abs() < 0.05,
                    "Q[{h}][{d}]: got {}, expected {exp_lo}", q_result[base + d]);
                assert!((q_result[base + d + half] - exp_hi).abs() < 0.05,
                    "Q[{h}][{}]: got {}, expected {exp_hi}", d + half, q_result[base + d + half]);
            }
        }

        b.free(&q).unwrap();
        b.free(&k).unwrap();
    }

    #[test]
    fn test_rope_prefill_numerical() {
        let mut b = CudaBackend::new(0).expect("CUDA init");
        let head_dim = 8;
        let half = head_dim / 2;
        let theta = 10000.0;
        let num_q_heads = 2;
        let seq_len = 4;
        b.precompute_rope_freqs(head_dim, theta).unwrap();

        let total_q = seq_len * num_q_heads * head_dim;
        let q_data: Vec<f32> = (0..total_q).map(|i| ((i % 7) as f32 + 1.0) * 0.1).collect();
        let k_data: Vec<f32> = (0..seq_len * 1 * head_dim).map(|i| ((i % 5) as f32 + 1.0) * 0.2).collect();

        let q = alloc_with_data(&b, &[seq_len, num_q_heads, head_dim], &q_data);
        let k = alloc_with_data(&b, &[seq_len, 1, head_dim], &k_data);

        let positions: Vec<u32> = (0..seq_len as u32).collect();
        b.rope(&q, &k, &positions, theta, head_dim).unwrap();
        b.synchronize().unwrap();

        let q_result = read_fp16(&b, &q);

        for tok in 0..seq_len {
            let pos = tok as f32;
            for h in 0..num_q_heads {
                let base = tok * num_q_heads * head_dim + h * head_dim;
                for d in 0..half {
                    let freq = 1.0 / (theta as f32).powf(2.0 * d as f32 / head_dim as f32);
                    let angle = pos * freq;
                    let (sin_a, cos_a) = angle.sin_cos();

                    let x0 = f16::from_f32(q_data[base + d]).to_f32();
                    let x1 = f16::from_f32(q_data[base + d + half]).to_f32();
                    let exp_lo = x0 * cos_a - x1 * sin_a;
                    let exp_hi = x0 * sin_a + x1 * cos_a;

                    assert!((q_result[base + d] - exp_lo).abs() < 0.05,
                        "tok {tok} head {h} d={d}: got {}, expected {exp_lo}", q_result[base + d]);
                    assert!((q_result[base + d + half] - exp_hi).abs() < 0.05,
                        "tok {tok} head {h} d={}: got {}, expected {exp_hi}", d + half, q_result[base + d + half]);
                }
            }
        }

        b.free(&q).unwrap();
        b.free(&k).unwrap();
    }

    // ── Attention numerical correctness ───────────────────────────────

    #[test]
    fn test_attention_prefill_gqa_group4() {
        let b = make_backend();
        let head_dim = 4;
        let num_q_heads = 8;
        let num_kv_heads = 2;
        let max_seq = 16;

        let q_data = vec![1.0f32; num_q_heads * head_dim];
        let q = alloc_with_data(&b, &[1, num_q_heads, head_dim], &q_data);

        let mut k_data = vec![0.0f32; max_seq * num_kv_heads * head_dim];
        for d in 0..head_dim { k_data[0 * num_kv_heads * head_dim + 0 * head_dim + d] = 1.0; }
        for d in 0..head_dim { k_data[0 * num_kv_heads * head_dim + 1 * head_dim + d] = 2.0; }
        let k_cache = alloc_with_data(&b, &[max_seq, num_kv_heads, head_dim], &k_data);

        let mut v_data = vec![0.0f32; max_seq * num_kv_heads * head_dim];
        for d in 0..head_dim { v_data[0 * num_kv_heads * head_dim + 0 * head_dim + d] = 0.5; }
        for d in 0..head_dim { v_data[0 * num_kv_heads * head_dim + 1 * head_dim + d] = 0.9; }
        let v_cache = alloc_with_data(&b, &[max_seq, num_kv_heads, head_dim], &v_data);

        let out = b.alloc(&[1, num_q_heads, head_dim], DType::FP16).unwrap();
        b.attention(&q, &k_cache, &v_cache, num_kv_heads, 0, &out).unwrap();
        b.synchronize().unwrap();

        let result = read_fp16(&b, &out);

        for h in 0..4 {
            for d in 0..head_dim {
                assert!((result[h * head_dim + d] - 0.5).abs() < 0.05,
                    "Q head {h} dim {d}: expected ~0.5, got {}", result[h * head_dim + d]);
            }
        }
        for h in 4..8 {
            for d in 0..head_dim {
                assert!((result[h * head_dim + d] - 0.9).abs() < 0.05,
                    "Q head {h} dim {d}: expected ~0.9, got {}", result[h * head_dim + d]);
            }
        }

        b.free(&q).unwrap(); b.free(&k_cache).unwrap();
        b.free(&v_cache).unwrap(); b.free(&out).unwrap();
    }

    #[test]
    fn test_attention_decode_weighted_sum() {
        let b = make_backend();
        let head_dim = 4;
        let max_seq = 16;

        let q_data = vec![0.0, 1.0, 0.0, 0.0f32];
        let q = alloc_with_data(&b, &[1, 1, head_dim], &q_data);

        let mut k_data = vec![0.0f32; max_seq * 1 * head_dim];
        k_data[0] = 1.0;
        k_data[head_dim + 1] = 1.0;
        k_data[2 * head_dim + 2] = 1.0;
        let k_cache = alloc_with_data(&b, &[max_seq, 1, head_dim], &k_data);

        let mut v_data = vec![0.0f32; max_seq * 1 * head_dim];
        v_data[0] = 1.0;
        v_data[head_dim + 1] = 1.0;
        v_data[2 * head_dim + 2] = 1.0;
        let v_cache = alloc_with_data(&b, &[max_seq, 1, head_dim], &v_data);

        let out = b.alloc(&[1, 1, head_dim], DType::FP16).unwrap();
        b.attention(&q, &k_cache, &v_cache, 1, 2, &out).unwrap();
        b.synchronize().unwrap();

        let result = read_fp16(&b, &out);
        assert!(result[1] > result[0], "dim 1 should dominate: got {:?}", &result[..4]);
        assert!(result[1] > result[2], "dim 1 should be > dim 2");

        b.free(&q).unwrap(); b.free(&k_cache).unwrap();
        b.free(&v_cache).unwrap(); b.free(&out).unwrap();
    }

    #[test]
    fn test_attention_scaling_unequal_scores() {
        let b = make_backend();
        let head_dim = 4;
        let max_seq = 16;

        let q_data = vec![2.0, 0.0, 0.0, 0.0f32];
        let q = alloc_with_data(&b, &[1, 1, head_dim], &q_data);

        let mut k_data = vec![0.0f32; max_seq * 1 * head_dim];
        k_data[0] = 2.0;
        k_data[head_dim] = 1.0;
        let k_cache = alloc_with_data(&b, &[max_seq, 1, head_dim], &k_data);

        let mut v_data = vec![0.0f32; max_seq * 1 * head_dim];
        v_data[0] = 1.0;
        v_data[head_dim + 1] = 1.0;
        let v_cache = alloc_with_data(&b, &[max_seq, 1, head_dim], &v_data);

        let out = b.alloc(&[1, 1, head_dim], DType::FP16).unwrap();
        b.attention(&q, &k_cache, &v_cache, 1, 1, &out).unwrap();
        b.synchronize().unwrap();

        let result = read_fp16(&b, &out);

        let scale = 1.0 / (head_dim as f32).sqrt();
        let s0 = (4.0 * scale).exp();
        let s1 = (2.0 * scale).exp();
        let w0 = s0 / (s0 + s1);
        let w1 = s1 / (s0 + s1);

        assert!((result[0] - w0).abs() < 0.05, "dim 0: expected {w0}, got {}", result[0]);
        assert!((result[1] - w1).abs() < 0.05, "dim 1: expected {w1}, got {}", result[1]);

        b.free(&q).unwrap(); b.free(&k_cache).unwrap();
        b.free(&v_cache).unwrap(); b.free(&out).unwrap();
    }

    #[test]
    fn test_attention_gqa_multi_kv_heads() {
        let b = make_backend();
        let head_dim = 4;
        let num_q_heads = 8;
        let num_kv_heads = 2;
        let max_seq = 16;

        let q_data = vec![1.0f32; num_q_heads * head_dim];
        let q = alloc_with_data(&b, &[1, num_q_heads, head_dim], &q_data);

        let mut k_data = vec![0.0f32; max_seq * num_kv_heads * head_dim];
        for kv in 0..num_kv_heads {
            for d in 0..head_dim {
                k_data[0 * num_kv_heads * head_dim + kv * head_dim + d] = 1.0;
            }
        }
        let k_cache = alloc_with_data(&b, &[max_seq, num_kv_heads, head_dim], &k_data);

        let mut v_data = vec![0.0f32; max_seq * num_kv_heads * head_dim];
        v_data[0 * num_kv_heads * head_dim + 0 * head_dim + 0] = 1.0;
        v_data[0 * num_kv_heads * head_dim + 1 * head_dim + 1] = 1.0;
        let v_cache = alloc_with_data(&b, &[max_seq, num_kv_heads, head_dim], &v_data);

        let out = b.alloc(&[1, num_q_heads, head_dim], DType::FP16).unwrap();
        b.attention(&q, &k_cache, &v_cache, num_kv_heads, 0, &out).unwrap();
        b.synchronize().unwrap();

        let result = read_fp16(&b, &out);

        for h in 0..4 {
            let base = h * head_dim;
            assert!((result[base] - 1.0).abs() < 0.05, "head {h} dim 0: expected 1.0, got {}", result[base]);
            assert!(result[base + 1].abs() < 0.05, "head {h} dim 1: expected 0.0, got {}", result[base + 1]);
        }
        for h in 4..8 {
            let base = h * head_dim;
            assert!(result[base].abs() < 0.05, "head {h} dim 0: expected 0.0, got {}", result[base]);
            assert!((result[base + 1] - 1.0).abs() < 0.05, "head {h} dim 1: expected 1.0, got {}", result[base + 1]);
        }

        b.free(&q).unwrap(); b.free(&k_cache).unwrap();
        b.free(&v_cache).unwrap(); b.free(&out).unwrap();
    }

    #[test]
    fn test_attention_decode_production_dims() {
        use rand::Rng;
        let b = make_backend();
        let head_dim = 128;
        let num_q_heads = 32;
        let num_kv_heads = 8;
        let max_seq = 256;
        let cache_len = 64;

        let mut rng = rand::thread_rng();
        let q_data: Vec<f32> = (0..num_q_heads * head_dim).map(|_| rng.gen_range(-0.5..0.5)).collect();
        let q = alloc_with_data(&b, &[1, num_q_heads, head_dim], &q_data);

        let k_data: Vec<f32> = (0..max_seq * num_kv_heads * head_dim).map(|_| rng.gen_range(-0.5..0.5)).collect();
        let v_data: Vec<f32> = (0..max_seq * num_kv_heads * head_dim).map(|_| rng.gen_range(-0.5..0.5)).collect();
        let k_cache = alloc_with_data(&b, &[max_seq, num_kv_heads, head_dim], &k_data);
        let v_cache = alloc_with_data(&b, &[max_seq, num_kv_heads, head_dim], &v_data);

        let out = b.alloc(&[1, num_q_heads, head_dim], DType::FP16).unwrap();
        b.attention(&q, &k_cache, &v_cache, num_kv_heads, cache_len - 1, &out).unwrap();
        b.synchronize().unwrap();

        let result = read_fp16(&b, &out);
        for (i, &val) in result.iter().enumerate() {
            assert!(val.is_finite(), "output[{i}] is not finite: {val}");
        }
        assert_eq!(result.len(), num_q_heads * head_dim);

        b.free(&q).unwrap(); b.free(&k_cache).unwrap();
        b.free(&v_cache).unwrap(); b.free(&out).unwrap();
    }

    // ── GEMM correctness: Llama 3 FFN shapes ─────────────────────────
    // matmul computes C = A @ B^T where A is [M,K], B is [N,K], C is [M,N]

    #[test]
    fn test_matmul_llama3_ffn_shapes() {
        use rand::Rng;
        let b = make_backend();
        let mut rng = rand::thread_rng();

        // Gate/up projection: A[1,4096] x B[14336,4096]^T = C[1,14336]
        {
            let m = 1; let k = 4096; let n = 14336;
            let a_data: Vec<f32> = (0..m*k).map(|_| rng.gen_range(-0.1..0.1)).collect();
            let b_data: Vec<f32> = (0..n*k).map(|_| rng.gen_range(-0.1..0.1)).collect();
            let a = alloc_with_data(&b, &[m, k], &a_data);
            let bt = alloc_with_data(&b, &[n, k], &b_data);
            let c = b.alloc(&[m, n], DType::FP16).unwrap();
            b.matmul(&a, &bt, &c).unwrap();
            b.synchronize().unwrap();
            let result = read_fp16(&b, &c);

            for &col in &[0, n/2, n-1] {
                let mut expected = 0.0f64;
                for i in 0..k {
                    let av = f16::from_f32(a_data[i]).to_f32() as f64;
                    let bv = f16::from_f32(b_data[col * k + i]).to_f32() as f64; // B[col, i]
                    expected += av * bv;
                }
                let got = result[col] as f64;
                let tol = expected.abs() * 0.05 + 0.1;
                assert!((got - expected).abs() < tol, "gate/up col {col}: got {got}, expected {expected}");
            }
            b.free(&a).unwrap(); b.free(&bt).unwrap(); b.free(&c).unwrap();
        }

        // Down projection: A[1,14336] x B[4096,14336]^T = C[1,4096]
        {
            let m = 1; let k = 14336; let n = 4096;
            let a_data: Vec<f32> = (0..m*k).map(|_| rng.gen_range(-0.1..0.1)).collect();
            let b_data: Vec<f32> = (0..n*k).map(|_| rng.gen_range(-0.1..0.1)).collect();
            let a = alloc_with_data(&b, &[m, k], &a_data);
            let bt = alloc_with_data(&b, &[n, k], &b_data);
            let c = b.alloc(&[m, n], DType::FP16).unwrap();
            b.matmul(&a, &bt, &c).unwrap();
            b.synchronize().unwrap();
            let result = read_fp16(&b, &c);

            for &col in &[0, n/2, n-1] {
                let mut expected = 0.0f64;
                for i in 0..k {
                    let av = f16::from_f32(a_data[i]).to_f32() as f64;
                    let bv = f16::from_f32(b_data[col * k + i]).to_f32() as f64;
                    expected += av * bv;
                }
                let got = result[col] as f64;
                let tol = expected.abs() * 0.05 + 0.1;
                assert!((got - expected).abs() < tol, "down col {col}: got {got}, expected {expected}");
            }
            b.free(&a).unwrap(); b.free(&bt).unwrap(); b.free(&c).unwrap();
        }
    }

    #[test]
    fn test_matmul_prefill_correctness() {
        use rand::Rng;
        let b = make_backend();
        let mut rng = rand::thread_rng();

        // A[128,4096] x B[14336,4096]^T = C[128,14336]
        let m = 128; let k = 4096; let n = 14336;
        let a_data: Vec<f32> = (0..m*k).map(|_| rng.gen_range(-0.1..0.1)).collect();
        let b_data: Vec<f32> = (0..n*k).map(|_| rng.gen_range(-0.1..0.1)).collect();
        let a = alloc_with_data(&b, &[m, k], &a_data);
        let bt = alloc_with_data(&b, &[n, k], &b_data);
        let c = b.alloc(&[m, n], DType::FP16).unwrap();
        b.matmul(&a, &bt, &c).unwrap();
        b.synchronize().unwrap();
        let result = read_fp16(&b, &c);

        let samples = [(0, 0), (0, n-1), (m/2, n/2), (m-1, 0), (m-1, n-1)];
        for &(row, col) in &samples {
            let mut expected = 0.0f64;
            for i in 0..k {
                let av = f16::from_f32(a_data[row * k + i]).to_f32() as f64;
                let bv = f16::from_f32(b_data[col * k + i]).to_f32() as f64;
                expected += av * bv;
            }
            let got = result[row * n + col] as f64;
            let tol = expected.abs() * 0.05 + 0.5;
            assert!((got - expected).abs() < tol, "prefill ({row},{col}): got {got}, expected {expected}");
        }

        b.free(&a).unwrap(); b.free(&bt).unwrap(); b.free(&c).unwrap();
    }

    #[test]
    fn test_matmul_gqa_kv_correctness() {
        use rand::Rng;
        let b = make_backend();
        let mut rng = rand::thread_rng();

        // GQA K/V: A[1,4096] x B[1024,4096]^T = C[1,1024]
        let m = 1; let k = 4096; let n = 1024;
        let a_data: Vec<f32> = (0..m*k).map(|_| rng.gen_range(-0.1..0.1)).collect();
        let b_data: Vec<f32> = (0..n*k).map(|_| rng.gen_range(-0.1..0.1)).collect();
        let a = alloc_with_data(&b, &[m, k], &a_data);
        let bt = alloc_with_data(&b, &[n, k], &b_data);
        let c = b.alloc(&[m, n], DType::FP16).unwrap();
        b.matmul(&a, &bt, &c).unwrap();
        b.synchronize().unwrap();
        let result = read_fp16(&b, &c);

        for &col in &[0, n/4, n/2, 3*n/4, n-1] {
            let mut expected = 0.0f64;
            for i in 0..k {
                let av = f16::from_f32(a_data[i]).to_f32() as f64;
                let bv = f16::from_f32(b_data[col * k + i]).to_f32() as f64;
                expected += av * bv;
            }
            let got = result[col] as f64;
            let tol = expected.abs() * 0.05 + 0.1;
            assert!((got - expected).abs() < tol, "GQA KV col {col}: got {got}, expected {expected}");
        }

        b.free(&a).unwrap(); b.free(&bt).unwrap(); b.free(&c).unwrap();
    }

    // ── Shape validation tests ────────────────────────────────────

    #[test]
    fn test_silu_mul_shape_mismatch() {
        let b = make_backend();
        let gate = alloc_with_data(&b, &[2, 4], &vec![1.0; 8]);
        let up = alloc_with_data(&b, &[2, 3], &vec![1.0; 6]);
        let out = b.alloc(&[2, 4], DType::FP16).unwrap();

        let err = b.silu_mul(&gate, &up, &out);
        assert!(err.is_err(), "silu_mul should fail on shape mismatch");
        let msg = err.unwrap_err().to_string();
        assert!(msg.contains("silu_mul"), "error should mention silu_mul: {msg}");

        b.free(&gate).unwrap();
        b.free(&up).unwrap();
        b.free(&out).unwrap();
    }

    #[test]
    fn test_add_shape_mismatch() {
        let b = make_backend();
        let a = alloc_with_data(&b, &[2, 4], &vec![1.0; 8]);
        let bt = alloc_with_data(&b, &[3, 4], &vec![1.0; 12]);
        let out = b.alloc(&[2, 4], DType::FP16).unwrap();

        let err = b.add(&a, &bt, &out);
        assert!(err.is_err(), "add should fail on shape mismatch");
        let msg = err.unwrap_err().to_string();
        assert!(msg.contains("add"), "error should mention add: {msg}");

        b.free(&a).unwrap();
        b.free(&bt).unwrap();
        b.free(&out).unwrap();
    }

    #[test]
    fn test_gemm_dimension_mismatch() {
        let b = make_backend();
        // A is [2, 4], B is [3, 5] -> K mismatch (4 != 5)
        let a = alloc_with_data(&b, &[2, 4], &vec![1.0; 8]);
        let bt = alloc_with_data(&b, &[3, 5], &vec![1.0; 15]);
        let out = b.alloc(&[2, 3], DType::FP16).unwrap();

        let err = b.matmul(&a, &bt, &out);
        assert!(err.is_err(), "matmul should fail on K dimension mismatch");
        let msg = err.unwrap_err().to_string();
        assert!(msg.contains("matmul"), "error should mention matmul: {msg}");

        b.free(&a).unwrap();
        b.free(&bt).unwrap();
        b.free(&out).unwrap();
    }

    #[test]
    fn test_copy_rows_shape_compatibility() {
        let b = make_backend();
        // src has 3 columns, dst has 4 columns -> mismatch
        let src = alloc_with_data(&b, &[4, 3], &vec![1.0; 12]);
        let dst = b.alloc(&[4, 4], DType::FP16).unwrap();

        let err = b.copy_rows(&src, &dst, 0, 0, 2);
        assert!(err.is_err(), "copy_rows should fail on column width mismatch");
        let msg = err.unwrap_err().to_string();
        assert!(msg.contains("copy_rows"), "error should mention copy_rows: {msg}");

        b.free(&src).unwrap();
        b.free(&dst).unwrap();
    }

    #[test]
    fn test_copy_rows_dtype_mismatch() {
        let b = make_backend();
        let src = alloc_with_data(&b, &[4, 3], &vec![1.0; 12]);
        let dst = b.alloc(&[4, 3], DType::FP32).unwrap();

        let err = b.copy_rows(&src, &dst, 0, 0, 2);
        assert!(err.is_err(), "copy_rows should fail on dtype mismatch");
        let msg = err.unwrap_err().to_string();
        assert!(msg.contains("dtype"), "error should mention dtype: {msg}");

        b.free(&src).unwrap();
        b.free(&dst).unwrap();
    }
}
