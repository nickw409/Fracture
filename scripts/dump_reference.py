#!/usr/bin/env python3
"""
Dump reference tensors from Llama 3.1 8B Instruct for Fracture numerical validation.

Loads meta-llama/Meta-Llama-3.1-8B-Instruct in FP16 on CUDA, hooks into every layer
to capture intermediate activations, and writes tensor files to tests/reference/.

Also generates greedy decoding golden outputs to tests/golden/.

Binary format per tensor file:
    [4 bytes: ndim (u32 LE)]
    [4 bytes x ndim: shape dimensions (u32 LE each)]
    [4 bytes: dtype enum (0=f16, 1=f32, 2=i32)]
    [remaining: raw tensor data in little-endian byte order]

All intermediate tensors are stored as float32 for maximum reference precision.

Usage:
    python scripts/dump_reference.py
    python scripts/dump_reference.py --model-path /data/models/llama-3.1-8b-instruct
    python scripts/dump_reference.py --layers 0,1,31
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM
except ImportError:
    print("ERROR: transformers library required. Install: pip install transformers", file=sys.stderr)
    sys.exit(1)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"

TEST_PROMPTS = [
    "The capital of France is",
    "Write a Python function that sorts a list using merge sort",
]

DTYPE_ENUM = {
    np.dtype("float16"): 0,
    np.dtype("float32"): 1,
    np.dtype("int32"): 2,
}

DTYPE_ENUM_REVERSE = {v: k for k, v in DTYPE_ENUM.items()}

SEED = 42
GREEDY_GENERATE_TOKENS = 50


# ---------------------------------------------------------------------------
# Binary tensor I/O
# ---------------------------------------------------------------------------

def save_tensor(path: Path, tensor: torch.Tensor, *, store_as: str = "float32") -> dict[str, Any]:
    """Save a tensor in Fracture reference binary format.

    All intermediate tensors are stored as float32 for maximum precision.
    Returns metadata dict.
    """
    if store_as == "float32":
        t = tensor.detach().float().cpu()
    elif store_as == "float16":
        t = tensor.detach().half().cpu()
    elif store_as == "int32":
        t = tensor.detach().int().cpu()
    else:
        t = tensor.detach().cpu()
        if t.dtype == torch.bfloat16:
            t = t.float()

    arr = t.numpy()
    np_dtype = arr.dtype
    if np_dtype not in DTYPE_ENUM:
        raise ValueError(f"Unsupported dtype {np_dtype} for {path}")

    path.parent.mkdir(parents=True, exist_ok=True)

    ndim = len(arr.shape)
    with open(path, "wb") as f:
        f.write(struct.pack("<I", ndim))
        for dim in arr.shape:
            f.write(struct.pack("<I", dim))
        f.write(struct.pack("<I", DTYPE_ENUM[np_dtype]))
        f.write(arr.tobytes())

    return {
        "shape": list(arr.shape),
        "dtype": str(np_dtype),
        "bytes": arr.nbytes,
    }


# ---------------------------------------------------------------------------
# Activation capture via hooks
# ---------------------------------------------------------------------------

class ActivationCapture:
    """Captures intermediate tensors from a LlamaForCausalLM forward pass."""

    def __init__(self, model: LlamaForCausalLM, layer_indices: list[int] | None):
        self.model = model
        self.layer_indices = layer_indices
        self.tensors: dict[str, torch.Tensor] = {}
        self._hooks: list[torch.utils.hooks.RemovableHook] = []
        self._patched_mlps: list[tuple] = []  # (mlp, original_forward)

    def _store(self, name: str, tensor: torch.Tensor) -> None:
        self.tensors[name] = tensor.detach().clone()

    def _should_capture_layer(self, idx: int) -> bool:
        return self.layer_indices is None or idx in self.layer_indices

    def install_hooks(self) -> None:
        model = self.model
        llama = model.model  # LlamaModel

        # Embedding output
        def embed_hook(module, input, output):
            self._store("embeddings", output)
        self._hooks.append(llama.embed_tokens.register_forward_hook(embed_hook))

        # Per-layer hooks
        for layer_idx, layer in enumerate(llama.layers):
            if not self._should_capture_layer(layer_idx):
                continue
            self._install_layer_hooks(layer_idx, layer)

        # Final RMSNorm
        def final_norm_hook(module, input, output):
            self._store("final_norm", output)
        self._hooks.append(llama.norm.register_forward_hook(final_norm_hook))

        # LM head (logits)
        def lm_head_hook(module, input, output):
            self._store("logits", output)
        self._hooks.append(model.lm_head.register_forward_hook(lm_head_hook))

    def _install_layer_hooks(self, idx: int, layer) -> None:
        prefix = f"layer_{idx:02d}"

        # Input to layer (pre-hook captures input tuple)
        def layer_pre_hook(module, input, _p=prefix):
            if isinstance(input, tuple):
                self._store(f"{_p}/input_hidden", input[0])
            else:
                self._store(f"{_p}/input_hidden", input)
        self._hooks.append(layer.register_forward_pre_hook(layer_pre_hook))

        # First RMSNorm (attention input norm)
        def attn_norm_hook(module, input, output, _p=prefix):
            self._store(f"{_p}/post_attn_norm", output)
        self._hooks.append(layer.input_layernorm.register_forward_hook(attn_norm_hook))

        # Q, K, V projections (before RoPE)
        def q_hook(module, input, output, _p=prefix):
            self._store(f"{_p}/q", output)
        self._hooks.append(layer.self_attn.q_proj.register_forward_hook(q_hook))

        def k_hook(module, input, output, _p=prefix):
            self._store(f"{_p}/k", output)
        self._hooks.append(layer.self_attn.k_proj.register_forward_hook(k_hook))

        def v_hook(module, input, output, _p=prefix):
            self._store(f"{_p}/v", output)
        self._hooks.append(layer.self_attn.v_proj.register_forward_hook(v_hook))

        # Attention output (after Wo projection)
        def attn_output_hook(module, input, output, _p=prefix):
            out = output[0] if isinstance(output, tuple) else output
            self._store(f"{_p}/attn_output", out)
        self._hooks.append(layer.self_attn.register_forward_hook(attn_output_hook))

        # Post-attention residual: input to the second RMSNorm
        def post_attn_residual_hook(module, input, _p=prefix):
            x = input[0] if isinstance(input, tuple) else input
            self._store(f"{_p}/post_attn_residual", x)
        self._hooks.append(layer.post_attention_layernorm.register_forward_pre_hook(post_attn_residual_hook))

        # Second RMSNorm (FFN input norm)
        def ffn_norm_hook(module, input, output, _p=prefix):
            self._store(f"{_p}/post_ffn_norm", output)
        self._hooks.append(layer.post_attention_layernorm.register_forward_hook(ffn_norm_hook))

        # Gate and up projections (captured by MLP patch below too, but hooks are cleaner)
        def gate_hook(module, input, output, _p=prefix):
            self._store(f"{_p}/gate", output)
        self._hooks.append(layer.mlp.gate_proj.register_forward_hook(gate_hook))

        def up_hook(module, input, output, _p=prefix):
            self._store(f"{_p}/up", output)
        self._hooks.append(layer.mlp.up_proj.register_forward_hook(up_hook))

        # Patch MLP forward to capture silu_mul and ffn_output
        self._patch_mlp(idx, layer)

        # Layer output (after FFN residual add)
        def layer_output_hook(module, input, output, _p=prefix):
            out = output[0] if isinstance(output, tuple) else output
            self._store(f"{_p}/output_hidden", out)
        self._hooks.append(layer.register_forward_hook(layer_output_hook))

    def _patch_mlp(self, idx: int, layer) -> None:
        prefix = f"layer_{idx:02d}"
        mlp = layer.mlp
        original_forward = mlp.forward
        capture = self

        def patched_forward(x):
            gate = mlp.gate_proj(x)
            up = mlp.up_proj(x)
            silu_mul = torch.nn.functional.silu(gate) * up
            capture._store(f"{prefix}/silu_mul", silu_mul)
            ffn_out = mlp.down_proj(silu_mul)
            capture._store(f"{prefix}/ffn_output", ffn_out)
            return ffn_out

        mlp.forward = patched_forward
        self._patched_mlps.append((mlp, original_forward))

    def compute_rope_tensors(self, seq_len: int, position_ids: torch.Tensor | None = None) -> None:
        """Compute Q/K after RoPE from captured pre-RoPE Q/K projections.

        Replicates the exact RoPE computation from the model to avoid
        fragile monkey-patching of HuggingFace internals.
        """
        config = self.model.config
        head_dim = config.hidden_size // config.num_attention_heads
        num_kv_heads = config.num_key_value_heads
        num_q_heads = config.num_attention_heads
        device = self.model.device

        rotary_emb = self.model.model.layers[0].self_attn.rotary_emb

        if position_ids is None:
            position_ids = torch.arange(seq_len, device=device).unsqueeze(0)

        cos_sin = rotary_emb(
            torch.ones(1, device=device),
            position_ids,
        )
        if isinstance(cos_sin, tuple) and len(cos_sin) == 2:
            cos, sin = cos_sin
        else:
            raise RuntimeError(f"Unexpected rotary_emb output: {type(cos_sin)}")

        for key in list(self.tensors.keys()):
            if not key.endswith("/q") and not key.endswith("/k"):
                continue

            prefix = key.rsplit("/", 1)[0]
            is_q = key.endswith("/q")
            tensor = self.tensors[key]  # [batch, seq_len, proj_dim]
            n_heads = num_q_heads if is_q else num_kv_heads
            actual_seq_len = tensor.shape[1] if tensor.dim() == 3 else tensor.shape[0]

            if tensor.dim() == 3:
                reshaped = tensor.view(tensor.shape[0], actual_seq_len, n_heads, head_dim).transpose(1, 2)
            else:
                reshaped = tensor.view(actual_seq_len, n_heads, head_dim).unsqueeze(0).transpose(1, 2)

            rotated = self._apply_rotary_pos_emb(reshaped, cos, sin)
            suffix = "q_rope" if is_q else "k_rope"
            self._store(f"{prefix}/{suffix}", rotated.squeeze(0).transpose(0, 1))

    @staticmethod
    def _apply_rotary_pos_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """Apply rotary position embeddings, matching HuggingFace LlamaRotaryEmbedding."""
        if cos.dim() == 3:
            cos = cos.unsqueeze(1)
            sin = sin.unsqueeze(1)

        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        rotated = torch.cat((-x2, x1), dim=-1)
        return (x * cos) + (rotated * sin)

    def remove_hooks(self) -> None:
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()
        for mlp, original in self._patched_mlps:
            mlp.forward = original
        self._patched_mlps.clear()

    def clear(self) -> None:
        self.tensors.clear()


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(model_path: str) -> tuple[LlamaForCausalLM, Any]:
    print(f"Loading tokenizer from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    print(f"Loading model from {model_path} (FP16, CUDA)...")
    torch.manual_seed(SEED)
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.float16,
        device_map="cuda",
        trust_remote_code=True,
    )
    elapsed = time.time() - t0
    print(f"Model loaded in {elapsed:.1f}s")

    if not isinstance(model, LlamaForCausalLM):
        print(f"WARNING: Model is {type(model).__name__}, not LlamaForCausalLM. "
              f"Hooks assume Llama architecture.", file=sys.stderr)

    model.eval()
    return model, tokenizer


# ---------------------------------------------------------------------------
# Prefill dump: full forward pass with all intermediates
# ---------------------------------------------------------------------------

def dump_prefill(
    model: LlamaForCausalLM,
    tokenizer: Any,
    prompt: str,
    prompt_idx: int,
    output_dir: Path,
    layer_indices: list[int] | None,
) -> dict[str, Any]:
    """Run prefill forward pass, dump all intermediate tensors. Returns metadata."""
    prompt_dir = output_dir / f"prompt_{prompt_idx}"
    prompt_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Prompt {prompt_idx}: \"{prompt[:80]}{'...' if len(prompt) > 80 else ''}\"")
    print(f"Output: {prompt_dir}")
    print(f"{'='*60}")

    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(model.device)
    token_ids_list = input_ids[0].tolist()
    seq_len = input_ids.shape[1]
    print(f"  Tokens: {seq_len} ids: {token_ids_list[:10]}{'...' if seq_len > 10 else ''}")

    # Save token IDs as int32
    tensor_meta = {}
    token_tensor = torch.tensor(token_ids_list, dtype=torch.int32)
    tensor_meta["token_ids.bin"] = save_tensor(prompt_dir / "token_ids.bin", token_tensor, store_as="int32")

    # Install hooks and run forward pass
    capture = ActivationCapture(model, layer_indices)
    capture.install_hooks()

    print("  Running forward pass...")
    torch.manual_seed(SEED)
    t0 = time.time()
    with torch.no_grad():
        outputs = model(input_ids, use_cache=True)
    elapsed = time.time() - t0
    print(f"  Forward pass: {elapsed:.3f}s")

    # Compute RoPE-applied Q/K
    print("  Computing RoPE-applied Q/K...")
    capture.compute_rope_tensors(seq_len)

    capture.remove_hooks()

    # Greedy token from last position
    logits = outputs.logits
    greedy_token = logits[0, -1, :].argmax().item()
    print(f"  Greedy next token: {greedy_token} = \"{tokenizer.decode([greedy_token])}\"")

    # Save greedy token
    greedy_tensor = torch.tensor([greedy_token], dtype=torch.int32)
    tensor_meta["greedy_token.bin"] = save_tensor(prompt_dir / "greedy_token.bin", greedy_tensor, store_as="int32")

    # Save all captured tensors as float32
    print(f"  Saving {len(capture.tensors)} intermediate tensors as float32...")
    for name, tensor in sorted(capture.tensors.items()):
        if tensor.dtype in (torch.int32, torch.int64):
            store = "int32"
        else:
            store = "float32"

        rel_path = name.replace("/", os.sep) + ".bin"
        full_path = prompt_dir / rel_path
        meta = save_tensor(full_path, tensor, store_as=store)
        tensor_meta[rel_path] = meta
        shape_str = "x".join(str(d) for d in meta["shape"])
        print(f"    {rel_path:<45s} {shape_str:<25s} {meta['dtype']}")

    # Metadata
    config = model.config
    metadata = {
        "prompt": prompt,
        "prompt_index": prompt_idx,
        "token_ids": token_ids_list,
        "seq_len": seq_len,
        "greedy_token": greedy_token,
        "model_config": {
            "hidden_size": config.hidden_size,
            "num_attention_heads": config.num_attention_heads,
            "num_key_value_heads": config.num_key_value_heads,
            "intermediate_size": config.intermediate_size,
            "num_hidden_layers": config.num_hidden_layers,
            "vocab_size": config.vocab_size,
            "rms_norm_eps": config.rms_norm_eps,
            "rope_theta": getattr(config, "rope_theta", 500000.0),
            "head_dim": config.hidden_size // config.num_attention_heads,
        },
        "layers_dumped": sorted(layer_indices) if layer_indices is not None else list(range(config.num_hidden_layers)),
        "tensors": tensor_meta,
    }

    with open(prompt_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    capture.clear()
    return {"input_ids": input_ids, "past_key_values": outputs.past_key_values, "greedy_token": greedy_token, "seq_len": seq_len}


# ---------------------------------------------------------------------------
# Decode step dump
# ---------------------------------------------------------------------------

def dump_decode_step(
    model: LlamaForCausalLM,
    tokenizer: Any,
    prefill_result: dict,
    output_dir: Path,
    layer_indices: list[int] | None,
) -> None:
    """Run one decode step after prefill of prompt 0, dump intermediates."""
    decode_dir = output_dir / "decode_step_0"
    decode_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Decode step 0 (after prompt 0 prefill)")
    print(f"Output: {decode_dir}")
    print(f"{'='*60}")

    greedy_token = prefill_result["greedy_token"]
    past_kv = prefill_result["past_key_values"]
    position = prefill_result["seq_len"]  # next position index

    # The decode input is just the greedy token
    decode_ids = torch.tensor([[greedy_token]], device=model.device)
    position_ids = torch.tensor([[position]], device=model.device)

    # Save position index
    tensor_meta = {}
    pos_tensor = torch.tensor([position], dtype=torch.int32)
    tensor_meta["position_index.bin"] = save_tensor(decode_dir / "position_index.bin", pos_tensor, store_as="int32")

    token_tensor = torch.tensor([greedy_token], dtype=torch.int32)
    tensor_meta["token_ids.bin"] = save_tensor(decode_dir / "token_ids.bin", token_tensor, store_as="int32")

    # Install hooks
    capture = ActivationCapture(model, layer_indices)
    capture.install_hooks()

    print(f"  Decode token: {greedy_token}, position: {position}")
    print("  Running decode forward pass...")
    t0 = time.time()
    with torch.no_grad():
        outputs = model(
            decode_ids,
            past_key_values=past_kv,
            position_ids=position_ids,
            use_cache=True,
        )
    elapsed = time.time() - t0
    print(f"  Decode forward pass: {elapsed:.3f}s")

    # Compute RoPE for decode step (single position)
    capture.compute_rope_tensors(1, position_ids=position_ids)

    capture.remove_hooks()

    # Save tensors
    print(f"  Saving {len(capture.tensors)} intermediate tensors...")
    for name, tensor in sorted(capture.tensors.items()):
        if tensor.dtype in (torch.int32, torch.int64):
            store = "int32"
        else:
            store = "float32"

        rel_path = name.replace("/", os.sep) + ".bin"
        full_path = decode_dir / rel_path
        meta = save_tensor(full_path, tensor, store_as=store)
        tensor_meta[rel_path] = meta
        shape_str = "x".join(str(d) for d in meta["shape"])
        print(f"    {rel_path:<45s} {shape_str:<25s} {meta['dtype']}")

    # Greedy token
    logits = outputs.logits
    next_token = logits[0, -1, :].argmax().item()
    print(f"  Greedy next token: {next_token} = \"{tokenizer.decode([next_token])}\"")

    greedy_tensor = torch.tensor([next_token], dtype=torch.int32)
    tensor_meta["greedy_token.bin"] = save_tensor(decode_dir / "greedy_token.bin", greedy_tensor, store_as="int32")

    metadata = {
        "step": 0,
        "input_token": greedy_token,
        "position": position,
        "output_token": next_token,
        "tensors": tensor_meta,
    }
    with open(decode_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    capture.clear()


# ---------------------------------------------------------------------------
# Greedy generation (golden outputs)
# ---------------------------------------------------------------------------

def dump_greedy_generation(
    model: LlamaForCausalLM,
    tokenizer: Any,
    prompts: list[str],
    golden_dir: Path,
    num_tokens: int = GREEDY_GENERATE_TOKENS,
) -> None:
    """Generate tokens greedily and save golden outputs."""
    golden_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Greedy generation ({num_tokens} tokens per prompt)")
    print(f"Output: {golden_dir}")
    print(f"{'='*60}")

    for idx, prompt in enumerate(prompts):
        print(f"\n  Prompt {idx}: \"{prompt[:60]}...\"" if len(prompt) > 60 else f"\n  Prompt {idx}: \"{prompt}\"")

        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(model.device)
        prompt_len = input_ids.shape[1]

        torch.manual_seed(SEED)
        t0 = time.time()
        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                max_new_tokens=num_tokens,
                do_sample=False,  # greedy / temperature=0
                temperature=None,
                top_p=None,
            )
        elapsed = time.time() - t0

        full_sequence = output_ids[0].tolist()
        generated = full_sequence[prompt_len:]
        print(f"  Generated {len(generated)} tokens in {elapsed:.2f}s")

        # Save full token sequence as int32 (prompt + generated)
        seq_tensor = torch.tensor(full_sequence, dtype=torch.int32)
        save_tensor(golden_dir / f"prompt_{idx}_greedy_{num_tokens}.bin", seq_tensor, store_as="int32")

        # Save decoded text
        decoded = tokenizer.decode(full_sequence, skip_special_tokens=True)
        text_path = golden_dir / f"prompt_{idx}_greedy_{num_tokens}.txt"
        text_path.write_text(decoded, encoding="utf-8")
        print(f"  Text: {decoded[:100]}...")

        # Save metadata
        meta = {
            "prompt": prompt,
            "prompt_index": idx,
            "prompt_token_ids": full_sequence[:prompt_len],
            "generated_token_ids": generated,
            "full_sequence": full_sequence,
            "num_generated": len(generated),
            "decoded_text": decoded,
        }
        with open(golden_dir / f"prompt_{idx}_greedy_{num_tokens}_meta.json", "w") as f:
            json.dump(meta, f, indent=2)


# ---------------------------------------------------------------------------
# Layer index parsing
# ---------------------------------------------------------------------------

def parse_layers(layers_str: str, num_layers: int) -> list[int] | None:
    if not layers_str or layers_str.lower() == "all":
        return None

    indices = []
    for part in layers_str.split(","):
        part = part.strip()
        if "-" in part:
            start, end = part.split("-", 1)
            indices.extend(range(int(start), int(end) + 1))
        else:
            indices.append(int(part))

    for idx in indices:
        if idx < 0 or idx >= num_layers:
            raise ValueError(f"Layer index {idx} out of range [0, {num_layers})")

    return sorted(set(indices))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Dump reference tensors from Llama 3.1 8B Instruct for Fracture validation.",
    )
    parser.add_argument(
        "--model-path",
        default=MODEL_ID,
        help=f"HuggingFace model path (default: {MODEL_ID})",
    )
    parser.add_argument(
        "--output-dir",
        default="tests/reference",
        help="Output directory for reference tensors (default: tests/reference)",
    )
    parser.add_argument(
        "--golden-dir",
        default="tests/golden",
        help="Output directory for golden generation outputs (default: tests/golden)",
    )
    parser.add_argument(
        "--layers",
        default="all",
        help="Comma-separated layer indices or 'all' (default: all). Supports ranges: '0-3,31'.",
    )
    parser.add_argument(
        "--generate-tokens",
        type=int,
        default=GREEDY_GENERATE_TOKENS,
        help=f"Number of tokens to generate greedily (default: {GREEDY_GENERATE_TOKENS})",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    golden_dir = Path(args.golden_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    golden_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reference output: {output_dir.resolve()}")
    print(f"Golden output:    {golden_dir.resolve()}")
    print(f"Model:            {args.model_path}")
    print(f"Prompts:          {TEST_PROMPTS}")
    print(f"Seed:             {SEED}")

    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available. This script requires an NVIDIA GPU.", file=sys.stderr)
        sys.exit(1)
    print(f"CUDA device:      {torch.cuda.get_device_name(0)}")
    print(f"PyTorch version:  {torch.__version__}")

    torch.manual_seed(SEED)

    model, tokenizer = load_model(args.model_path)

    num_layers = model.config.num_hidden_layers
    try:
        layer_indices = parse_layers(args.layers, num_layers)
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    if layer_indices is not None:
        print(f"Dumping layers: {layer_indices}")
    else:
        print(f"Dumping all {num_layers} layers")

    # --- Prefill dumps for each prompt ---
    t_total = time.time()
    prefill_results = []
    for i, prompt in enumerate(TEST_PROMPTS):
        result = dump_prefill(model, tokenizer, prompt, i, output_dir, layer_indices)
        prefill_results.append(result)

    # --- Decode step (after prompt 0 prefill) ---
    dump_decode_step(model, tokenizer, prefill_results[0], output_dir, layer_indices)

    # --- Greedy generation golden outputs ---
    dump_greedy_generation(model, tokenizer, TEST_PROMPTS, golden_dir, args.generate_tokens)

    # --- Write generation info ---
    gen_info = {
        "model_id": args.model_path,
        "torch_version": torch.__version__,
        "cuda_device": torch.cuda.get_device_name(0),
        "seed": SEED,
        "prompts": TEST_PROMPTS,
        "generate_tokens": args.generate_tokens,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    with open(output_dir / "generation_info.json", "w") as f:
        json.dump(gen_info, f, indent=2)

    elapsed_total = time.time() - t_total
    print(f"\nDone. Total time: {elapsed_total:.1f}s")
    print(f"Reference tensors: {output_dir.resolve()}")
    print(f"Golden outputs:    {golden_dir.resolve()}")


if __name__ == "__main__":
    main()
