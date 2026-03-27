fn main() {
    // TODO: Compile CUDA kernels (.cu files) using cc crate
    // TODO: Link against CUDA runtime and cuBLAS
    println!("cargo:rerun-if-changed=kernels/");
}
