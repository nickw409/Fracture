# Models

This directory contains model weights used by Fracture.
Files are gitignored due to size.

## Required for development/testing

- `llama-3.1-8b-instruct-f16.gguf` — Llama 3.1 8B Instruct, FP16
  - Source: Converted from meta-llama/Meta-Llama-3.1-8B-Instruct (HuggingFace)
  - Converted with: llama.cpp convert_hf_to_gguf.py --outtype f16
  - Size: ~15 GB

## Required for distributed testing (Phase 3)

- `llama-3.1-70b-instruct-q4_k_m.gguf` — Llama 3.1 70B Instruct, Q4_K_M
  - Source: bartowski/Meta-Llama-3.1-70B-Instruct-GGUF (HuggingFace)
  - Size: ~40 GB
