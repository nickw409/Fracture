# Models

This directory contains model weights used by Fracture.
Files are gitignored due to size.

## Required for development/testing

- `llama-3.1-8b-instruct-f16.gguf` — Llama 3.1 8B Instruct, FP16
  - Source: Converted from meta-llama/Meta-Llama-3.1-8B-Instruct (HuggingFace)
  - Converted with: llama.cpp convert_hf_to_gguf.py --outtype f16
  - Size: ~15 GB

- `tokenizer.json` — Llama 3.1 BPE tokenizer (required by HTTP server and coordinator)
  - Source: meta-llama/Meta-Llama-3.1-8B-Instruct on HuggingFace
  - Must be placed in this directory alongside the GGUF file
  - The server and coordinator look for `tokenizer.json` in the model file's directory
  - Can also be specified via `--tokenizer <path>` CLI flag

## Required for distributed testing (Phase 3)

- `llama-3.1-70b-instruct-q4_k_m.gguf` — Llama 3.1 70B Instruct, Q4_K_M
  - Source: bartowski/Meta-Llama-3.1-70B-Instruct-GGUF (HuggingFace)
  - Size: ~40 GB
