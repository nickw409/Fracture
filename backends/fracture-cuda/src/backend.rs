use crate::ffi::*;
use crate::nvtx;
use crate::CudaBackend;
use fracture_core::{Backend, DType, DeviceTensor, DeviceTimer, FractureError, Result, TensorId};
use std::ffi::{c_int, c_void, CStr};

impl Backend for CudaBackend {
    fn alloc(&self, shape: &[usize], dtype: DType) -> Result<DeviceTensor> {
        let numel: usize = shape.iter().product();
        let size = if dtype.is_packed() {
            numel.div_ceil(2)
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
        let pos_size = std::mem::size_of_val(positions);
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
                q_ptr,
                k_ptr,
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
        let bt_size = std::mem::size_of_val(block_table);
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
        let ids_size = std::mem::size_of_val(token_ids);
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
