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
        nvtx::range_push("matmul");
        // A is [M, K], B is [K, N], C is [M, N] — all row-major.
        // cuBLAS expects column-major. The transpose trick:
        //   In column-major, row-major A is A^T, row-major B is B^T.
        //   We want C = A @ B in row-major.
        //   Column-major: C^T = B^T @ A^T
        //   So we call cublasGemmEx(N, N, N, M, ...) with B as first arg, A as second.
        let m = a.shape[0] as c_int; // rows of A / rows of C
        let k = a.shape[1] as c_int;
        let n = b.shape[1] as c_int; // cols of B / cols of C

        let a_ptr = self.get_ptr(a.id)?;
        let b_ptr = self.get_ptr(b.id)?;
        let c_ptr = self.get_ptr(out.id)?;

        let alpha: f32 = 1.0;
        let beta: f32 = 0.0;

        cublas_check!(cublasGemmEx(
            self.cublas_handle,
            cublasOperation_t::CUBLAS_OP_N, // B^T treated as non-transposed in col-major
            cublasOperation_t::CUBLAS_OP_N, // A^T treated as non-transposed in col-major
            n,                               // rows of op(B^T) in col-major = N
            m,                               // cols of op(A^T) in col-major = M
            k,                               // shared dimension
            &alpha as *const f32 as *const c_void,
            b_ptr as *const c_void,
            cudaDataType_t::CUDA_R_16F,
            n,                               // ldb = N (leading dim of B in col-major = cols of row-major B)
            a_ptr as *const c_void,
            cudaDataType_t::CUDA_R_16F,
            k,                               // lda = K (leading dim of A in col-major = cols of row-major A)
            &beta as *const f32 as *const c_void,
            c_ptr,
            cudaDataType_t::CUDA_R_16F,
            n,                               // ldc = N
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

    fn silu_mul(
        &self,
        gate: &DeviceTensor,
        up: &DeviceTensor,
        out: &DeviceTensor,
    ) -> Result<()> {
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
        // A = [2, 3], B = [3, 2], C = [2, 2]
        let a_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b_data: Vec<f32> = vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0];

        let a = alloc_with_data(&b, &[2, 3], &a_data);
        let bt = alloc_with_data(&b, &[3, 2], &b_data);
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
        // M=1 decode-like shape
        let b = make_backend();
        let a_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0]; // [1, 4]
        let b_data: Vec<f32> = vec![1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0]; // [4, 2]

        let a = alloc_with_data(&b, &[1, 4], &a_data);
        let bt = alloc_with_data(&b, &[4, 2], &b_data);
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
}
