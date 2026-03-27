// CUDA backend for Fracture.
//
// Implements the Backend trait using CUDA kernels and cuBLAS.
// This crate is only compiled on Linux with NVIDIA GPUs.
// The engine never imports this crate directly — it's selected at the binary level.

// TODO: Implement CudaBackend struct
// TODO: Implement Backend trait for CudaBackend
// TODO: CUDA context and stream management
// TODO: cuBLAS handle management
// TODO: Kernel launch wrappers
