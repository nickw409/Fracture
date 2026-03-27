use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=kernels/");

    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let kernel_dir = PathBuf::from("kernels");

    let cuda_files = [
        "rmsnorm.cu",
        "rope.cu",
        "silu_mul.cu",
        "attention.cu",
        "embedding.cu",
        "add.cu",
    ];

    let mut objects = Vec::new();

    for cu_file in &cuda_files {
        let src = kernel_dir.join(cu_file);
        let obj = out_dir.join(cu_file.replace(".cu", ".o"));

        let status = Command::new("nvcc")
            .args([
                "-c",
                "-o",
                obj.to_str().unwrap(),
                src.to_str().unwrap(),
                "-O3",
                "--use_fast_math",
                "-Xcompiler",
                "-fPIC",
                "-gencode=arch=compute_86,code=sm_86", // RTX 3090 (Ampere)
            ])
            .status()
            .expect("failed to run nvcc — is the CUDA toolkit installed?");

        if !status.success() {
            panic!("nvcc failed to compile {}", cu_file);
        }

        objects.push(obj);
    }

    // Create a static library from all kernel objects
    let lib_path = out_dir.join("libfracture_kernels.a");
    let status = Command::new("ar")
        .args(["rcs", lib_path.to_str().unwrap()])
        .args(objects.iter().map(|o| o.to_str().unwrap()))
        .status()
        .expect("failed to run ar");

    if !status.success() {
        panic!("ar failed to create static library");
    }

    // Link our kernel library
    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static=fracture_kernels");

    // Link CUDA runtime and cuBLAS
    println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64");
    // WSL2: the CUDA driver (libcuda.so) lives in /usr/lib/wsl/lib/
    if std::path::Path::new("/usr/lib/wsl/lib").exists() {
        println!("cargo:rustc-link-search=native=/usr/lib/wsl/lib");
    }
    println!("cargo:rustc-link-lib=static=cudart_static");
    println!("cargo:rustc-link-lib=dylib=cublas");
    println!("cargo:rustc-link-lib=dylib=cublasLt");
    println!("cargo:rustc-link-lib=dylib=dl");
    println!("cargo:rustc-link-lib=dylib=rt");
    println!("cargo:rustc-link-lib=dylib=pthread");
    println!("cargo:rustc-link-lib=dylib=stdc++");
    println!("cargo:rustc-link-lib=dylib=nvToolsExt");
}
