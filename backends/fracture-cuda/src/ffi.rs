//! Minimal raw FFI bindings for CUDA runtime and cuBLAS.
//!
//! We use raw bindings rather than a wrapper crate to keep dependencies minimal
//! and maintain full control over the CUDA API surface we use.

#![allow(non_camel_case_types, non_snake_case, dead_code)]

use std::ffi::{c_char, c_int, c_void};

// ── CUDA Runtime types ─────────────────────────────────────────────

pub type cudaError_t = c_int;
pub type cudaStream_t = *mut c_void;

pub const CUDA_SUCCESS: cudaError_t = 0;

/// Opaque representation of cudaDeviceProp.
/// The actual struct is 1032 bytes in CUDA 13.x.
/// We store it as raw bytes and extract fields at known offsets.
#[repr(C, align(8))]
pub struct cudaDeviceProp {
    data: [u8; 1032],
}

impl cudaDeviceProp {
    /// Device name (256 bytes at offset 0).
    pub fn name_ptr(&self) -> *const c_char {
        self.data.as_ptr() as *const c_char
    }

    /// Total global memory in bytes (usize at offset 288).
    pub fn total_global_mem(&self) -> usize {
        let ptr = unsafe { self.data.as_ptr().add(288) as *const usize };
        unsafe { *ptr }
    }
}

impl Default for cudaDeviceProp {
    fn default() -> Self {
        Self { data: [0u8; 1032] }
    }
}

// ── cuBLAS types ───────────────────────────────────────────────────

pub type cublasHandle_t = *mut c_void;
pub type cublasStatus_t = c_int;

pub const CUBLAS_STATUS_SUCCESS: cublasStatus_t = 0;

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub enum cublasOperation_t {
    CUBLAS_OP_N = 0,
    CUBLAS_OP_T = 1,
    CUBLAS_OP_C = 2,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub enum cudaDataType_t {
    CUDA_R_16F = 2,
    CUDA_R_32F = 0,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub enum cublasComputeType_t {
    CUBLAS_COMPUTE_32F = 68,
}

// ── CUDA Runtime API ───────────────────────────────────────────────

unsafe extern "C" {
    pub fn cudaSetDevice(device: c_int) -> cudaError_t;
    pub fn cudaGetDevice(device: *mut c_int) -> cudaError_t;
    pub fn cudaGetDeviceCount(count: *mut c_int) -> cudaError_t;
    pub fn cudaGetDeviceProperties(prop: *mut cudaDeviceProp, device: c_int) -> cudaError_t;
    pub fn cudaDeviceSynchronize() -> cudaError_t;

    pub fn cudaMalloc(devPtr: *mut *mut c_void, size: usize) -> cudaError_t;
    pub fn cudaFree(devPtr: *mut c_void) -> cudaError_t;
    pub fn cudaMemcpy(
        dst: *mut c_void,
        src: *const c_void,
        count: usize,
        kind: c_int,
    ) -> cudaError_t;
    pub fn cudaMemset(devPtr: *mut c_void, value: c_int, count: usize) -> cudaError_t;

    pub fn cudaStreamCreate(pStream: *mut cudaStream_t) -> cudaError_t;
    pub fn cudaStreamDestroy(stream: cudaStream_t) -> cudaError_t;
    pub fn cudaStreamSynchronize(stream: cudaStream_t) -> cudaError_t;

    pub fn cudaMemGetInfo(free: *mut usize, total: *mut usize) -> cudaError_t;

    pub fn cudaGetErrorString(error: cudaError_t) -> *const c_char;
}

// cudaMemcpyKind constants
pub const CUDA_MEMCPY_HOST_TO_HOST: c_int = 0;
pub const CUDA_MEMCPY_HOST_TO_DEVICE: c_int = 1;
pub const CUDA_MEMCPY_DEVICE_TO_HOST: c_int = 2;
pub const CUDA_MEMCPY_DEVICE_TO_DEVICE: c_int = 3;

// ── cuBLAS API ─────────────────────────────────────────────────────

unsafe extern "C" {
    pub fn cublasCreate_v2(handle: *mut cublasHandle_t) -> cublasStatus_t;
    pub fn cublasDestroy_v2(handle: cublasHandle_t) -> cublasStatus_t;
    pub fn cublasSetStream_v2(handle: cublasHandle_t, stream: cudaStream_t) -> cublasStatus_t;

    pub fn cublasGemmEx(
        handle: cublasHandle_t,
        transa: cublasOperation_t,
        transb: cublasOperation_t,
        m: c_int,
        n: c_int,
        k: c_int,
        alpha: *const c_void,
        A: *const c_void,
        Atype: cudaDataType_t,
        lda: c_int,
        B: *const c_void,
        Btype: cudaDataType_t,
        ldb: c_int,
        beta: *const c_void,
        C: *mut c_void,
        Ctype: cudaDataType_t,
        ldc: c_int,
        computeType: cublasComputeType_t,
        algo: c_int, // cublasGemmAlgo_t
    ) -> cublasStatus_t;
}

// cublasGemmAlgo_t
pub const CUBLAS_GEMM_DEFAULT: c_int = -1;

// ── Custom kernel FFI ──────────────────────────────────────────────

unsafe extern "C" {
    pub fn launch_rmsnorm(
        output: *mut c_void,
        input: *const c_void,
        weight: *const c_void,
        eps: f32,
        rows: c_int,
        cols: c_int,
        stream: cudaStream_t,
    ) -> cudaError_t;

    pub fn launch_rope(
        q: *mut c_void,
        k: *mut c_void,
        positions: *const u32,
        freq_table: *const f32,
        num_tokens: c_int,
        num_q_heads: c_int,
        num_kv_heads: c_int,
        head_dim: c_int,
        stream: cudaStream_t,
    ) -> cudaError_t;

    pub fn launch_silu_mul(
        output: *mut c_void,
        gate: *const c_void,
        up: *const c_void,
        n: c_int,
        stream: cudaStream_t,
    ) -> cudaError_t;

    pub fn launch_attention(
        output: *mut c_void,
        q: *const c_void,
        k_cache: *const c_void,
        v_cache: *const c_void,
        num_tokens: c_int,
        num_q_heads: c_int,
        num_kv_heads: c_int,
        head_dim: c_int,
        kv_len: c_int,
        start_pos: c_int,
        stream: cudaStream_t,
    ) -> cudaError_t;

    pub fn launch_paged_attention(
        output: *mut c_void,
        q: *const c_void,
        block_table: *const c_int,
        k_block_ptrs: *const *const c_void,
        v_block_ptrs: *const *const c_void,
        num_tokens: c_int,
        num_q_heads: c_int,
        num_kv_heads: c_int,
        head_dim: c_int,
        kv_len: c_int,
        start_pos: c_int,
        num_blocks_in_seq: c_int,
        stream: cudaStream_t,
    ) -> cudaError_t;

    pub fn launch_embedding(
        output: *mut c_void,
        table: *const c_void,
        token_ids: *const u32,
        num_tokens: c_int,
        hidden_dim: c_int,
        vocab_size: c_int,
        stream: cudaStream_t,
    ) -> cudaError_t;

    pub fn launch_add(
        output: *mut c_void,
        a: *const c_void,
        b: *const c_void,
        n: c_int,
        stream: cudaStream_t,
    ) -> cudaError_t;
}

// ── CUDA Event API ────────────────────────────────────────────────

pub type cudaEvent_t = *mut c_void;

unsafe extern "C" {
    pub fn cudaEventCreate(event: *mut cudaEvent_t) -> cudaError_t;
    pub fn cudaEventDestroy(event: cudaEvent_t) -> cudaError_t;
    pub fn cudaEventRecord(event: cudaEvent_t, stream: cudaStream_t) -> cudaError_t;
    pub fn cudaEventSynchronize(event: cudaEvent_t) -> cudaError_t;
    pub fn cudaEventElapsedTime(ms: *mut f32, start: cudaEvent_t, end: cudaEvent_t) -> cudaError_t;
}

// ── NVTX API ──────────────────────────────────────────────────────

unsafe extern "C" {
    pub fn nvtxRangePushA(message: *const c_char) -> c_int;
    pub fn nvtxRangePop() -> c_int;
}
