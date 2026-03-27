# Fracture Reference Tensors

Ground-truth intermediate tensors from a PyTorch forward pass, used to validate
every kernel and layer in the Fracture inference engine.

## Model

- **HuggingFace ID:** `meta-llama/Meta-Llama-3.1-8B-Instruct`
- **Precision:** FP16 model weights, all intermediate tensors stored as **float32**
- **Seed:** `torch.manual_seed(42)`

## Prompts

| Index | Prompt |
|-------|--------|
| 0 | `The capital of France is` |
| 1 | `Write a Python function that sorts a list using merge sort` |

## Directory Structure

```
tests/reference/
  prompt_0/                        # Prefill pass for prompt 0
    token_ids.bin                  # Tokenized input [seq_len] int32
    embeddings.bin                 # After embedding lookup [1, seq_len, 4096] f32
    layer_00/ ... layer_31/        # Per-layer intermediates
      input_hidden.bin             # Input to the layer
      post_attn_norm.bin           # After first RMSNorm
      q.bin, k.bin, v.bin          # Q/K/V projections (before RoPE)
      q_rope.bin, k_rope.bin       # After RoPE application
      attn_output.bin              # After attention + Wo projection
      post_attn_residual.bin       # After attention residual add
      post_ffn_norm.bin            # After second RMSNorm
      gate.bin, up.bin             # Gate and up projections
      silu_mul.bin                 # silu(gate) * up
      ffn_output.bin               # After down projection
      output_hidden.bin            # After FFN residual add
    final_norm.bin                 # After output RMSNorm
    logits.bin                     # Final logits [1, seq_len, 128256] f32
    greedy_token.bin               # argmax of last position logits
    metadata.json                  # Prompt, token IDs, model config, tensor index
  prompt_1/                        # Same structure for prompt 1
  decode_step_0/                   # One decode step after prompt 0 prefill
    position_index.bin             # RoPE position for this decode step
    token_ids.bin                  # Input token (greedy from prefill)
    (same per-layer structure)
    greedy_token.bin
    metadata.json
  generation_info.json             # Model ID, torch version, GPU, seed, timestamp

tests/golden/
  prompt_0_greedy_50.bin           # Full token sequence (prompt + 50 generated) int32
  prompt_0_greedy_50.txt           # Detokenized text
  prompt_0_greedy_50_meta.json     # Generation metadata
  prompt_1_greedy_50.bin
  prompt_1_greedy_50.txt
  prompt_1_greedy_50_meta.json
```

## Binary Format

Each `.bin` file uses this format:

```
[4 bytes] u32 LE — number of dimensions (ndim)
[4 × ndim bytes] u32 LE — shape per dimension
[4 bytes] u32 LE — dtype enum: 0=float16, 1=float32, 2=int32
[remaining] raw tensor data in little-endian byte order
```

## How to Generate

Requires: NVIDIA GPU, PyTorch with CUDA, HuggingFace `transformers`, access to
`meta-llama/Meta-Llama-3.1-8B-Instruct` weights.

```bash
pip install torch transformers numpy
python scripts/dump_reference.py
python scripts/verify_reference.py
```

The script accepts flags:
- `--model-path PATH` — local model directory or HuggingFace model ID
- `--layers 0,1,31` — dump only specific layers (default: all)
- `--generate-tokens N` — number of greedy tokens to generate (default: 50)

## Status

If this directory contains only this README and a `.gitkeep`, the reference data
has not yet been generated on this machine. Run `dump_reference.py` to populate it.
