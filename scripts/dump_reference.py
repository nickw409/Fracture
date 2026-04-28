#!/usr/bin/env python3
"""
Dump reference tensors for Fracture numerical validation.

Two modes:
- `--mode hf`: load a HuggingFace Llama checkpoint (e.g. Llama 3.1 8B Instruct) in
  FP16 on CUDA and dump per-layer activations + greedy golden outputs.
- `--mode fixture`: build a PyTorch Llama from a tiny seeded GGUF fixture (same
  weights the Fracture runtime loads) so reference data is reproducible without
  needing a real model checkpoint.

Hooks every transformer layer to capture intermediate activations and writes
tensor files under `--output-dir`. Greedy generation goldens go under `--golden-dir`.

Binary format per tensor file:
    [4 bytes: ndim (u32 LE)]
    [4 bytes x ndim: shape dimensions (u32 LE each)]
    [4 bytes: dtype enum (0=f16, 1=f32, 2=i32)]
    [remaining: raw tensor data in little-endian byte order]

All intermediate tensors are stored as float32 for maximum reference precision.

Usage:
    # Real model (HuggingFace directory):
    python scripts/dump_reference.py \\
        --mode hf \\
        --model-path /data/models/llama-3.1-8b-instruct \\
        --output-dir tests/reference \\
        --golden-dir tests/golden \\
        --layers 0,8,16,24,last

    # Tiny fixture model (committed GGUF):
    python scripts/dump_reference.py \\
        --mode fixture \\
        --model-path tests/fixtures/tiny-llama.gguf \\
        --config-path tests/fixtures/tiny-llama.config.json \\
        --output-dir tests/reference-fixture \\
        --golden-dir tests/golden-fixture \\
        --layers all
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

        Computes RoPE frequencies in pure FP32 (matching the GPU kernel's
        precompute_rope_freqs), then applies the rotation in FP32 to the
        FP16 inputs. This avoids HuggingFace's FP16-stored inv_freq which
        introduces small angle errors that exceed the kernel test tolerances
        on the random fixture model.
        """
        config = self.model.config
        head_dim = config.hidden_size // config.num_attention_heads
        num_kv_heads = config.num_key_value_heads
        num_q_heads = config.num_attention_heads
        device = self.model.device
        rope_theta = float(getattr(config, "rope_theta", 500000.0))

        if position_ids is None:
            position_ids = torch.arange(seq_len, device=device).unsqueeze(0)

        # Match the GPU kernel: freq[i] = 1.0 / theta**(2i/head_dim) in pure FP32.
        half = head_dim // 2
        idx = torch.arange(half, device=device, dtype=torch.float64)
        inv_freq = (1.0 / (rope_theta ** (2.0 * idx / head_dim))).to(torch.float32)

        # angles[batch, pos, d] = position * inv_freq[d]; broadcast to head_dim by duplicating halves
        positions_f32 = position_ids.to(torch.float32)
        # [batch, seq_len, half_dim]
        angles = positions_f32.unsqueeze(-1) * inv_freq.view(1, 1, half)
        # Duplicate across halves so cos[..., d] == cos[..., d + half_dim] (HF convention)
        angles_full = torch.cat((angles, angles), dim=-1)
        cos = angles_full.cos()
        sin = angles_full.sin()

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
        """Apply rotary position embeddings in FP32, matching the GPU kernel.

        The GPU kernel converts FP16 inputs to FP32, applies (x*cos - x_rot*sin)
        in FP32, and writes back as FP16. Promote the FP16 input to FP32 here to
        produce a precision-matched reference value (still stored as FP32).
        """
        if cos.dim() == 3:
            cos = cos.unsqueeze(1)
            sin = sin.unsqueeze(1)

        x32 = x.to(torch.float32)
        x1 = x32[..., : x32.shape[-1] // 2]
        x2 = x32[..., x32.shape[-1] // 2 :]
        rotated = torch.cat((-x2, x1), dim=-1)
        return (x32 * cos) + (rotated * sin)

    def compute_attention_outputs(self, position_ids: torch.Tensor | None = None) -> None:
        """Recompute per-layer `attn_output` in FP32 from captured Q_rope, K_rope, V.

        The HuggingFace forward pass runs attention in FP16, which can produce
        per-element disagreement with the GPU kernel (which uses FP32 accumulators
        for QK^T, softmax, and attention*V). To match the GPU kernel's precision
        we recompute attn_output here using FP32 math, with FP16-precision inputs
        upcast to FP32 — exactly what the GPU does internally. The output is
        stored as FP32 (the kernel test uploads it as FP16 via the comparison helper).
        """
        config = self.model.config
        head_dim = config.hidden_size // config.num_attention_heads
        num_kv_heads = config.num_key_value_heads
        num_q_heads = config.num_attention_heads
        groups = num_q_heads // num_kv_heads

        for layer_idx, layer in enumerate(self.model.model.layers):
            if not self._should_capture_layer(layer_idx):
                continue
            prefix = f"layer_{layer_idx:02d}"
            q_rope_key = f"{prefix}/q_rope"
            k_rope_key = f"{prefix}/k_rope"
            v_key = f"{prefix}/v"
            if q_rope_key not in self.tensors or k_rope_key not in self.tensors or v_key not in self.tensors:
                continue

            # q_rope, k_rope, v are stored as FP32 in self.tensors but the GPU
            # kernel reads them as FP16. Round-trip through FP16 here so the
            # reference computation starts from the same input precision the
            # GPU sees. This is the dominant source of disagreement: the GPU
            # runs on FP16 inputs, and even tiny per-element rounding compounds
            # through softmax and matmul on the random fixture's wide value range.
            q = self.tensors[q_rope_key].to(torch.float16).to(torch.float32)  # [seq, n_q, hd]
            k = self.tensors[k_rope_key].to(torch.float16).to(torch.float32)  # [seq, n_kv, hd]
            v_flat = self.tensors[v_key].to(torch.float16).to(torch.float32)  # [batch, seq, n_kv*hd] or [seq, n_kv*hd]
            seq_len = q.shape[0]
            if v_flat.dim() == 3:
                v = v_flat.view(v_flat.shape[0], seq_len, num_kv_heads, head_dim).squeeze(0)
            else:
                v = v_flat.view(seq_len, num_kv_heads, head_dim)

            # GQA: each kv head serves `groups` query heads. Repeat k/v along head dim.
            if groups > 1:
                k = k.repeat_interleave(groups, dim=1)  # [seq, n_q, hd]
                v = v.repeat_interleave(groups, dim=1)

            # Reshape to [n_q, seq, hd] for batched matmul.
            q_t = q.transpose(0, 1)  # [n_q, seq, hd]
            k_t = k.transpose(0, 1)  # [n_q, seq, hd]
            v_t = v.transpose(0, 1)  # [n_q, seq, hd]

            scale = 1.0 / (head_dim ** 0.5)
            # [n_q, seq, seq]
            scores = torch.matmul(q_t, k_t.transpose(-1, -2)) * scale
            # Causal mask
            mask = torch.full_like(scores, float("-inf"))
            mask = torch.triu(mask, diagonal=1)
            scores = scores + mask
            attn = torch.softmax(scores, dim=-1)
            # [n_q, seq, hd]
            ctx = torch.matmul(attn, v_t)
            # [seq, n_q, hd] -> [seq, hidden]
            ctx_flat = ctx.transpose(0, 1).contiguous().view(seq_len, num_q_heads * head_dim)

            # The GPU kernel writes the raw attention output (pre-o_proj) as FP16
            # before applying o_proj as a separate matmul. Round-trip through FP16
            # here so the reference reflects that intermediate quantization step.
            ctx_flat = ctx_flat.to(torch.float16).to(torch.float32)

            # Apply o_proj in FP32
            o_proj_w = layer.self_attn.o_proj.weight.to(torch.float32)
            attn_out = ctx_flat @ o_proj_w.t()  # [seq, hidden]

            # Restore original shape: stored attn_output keys use [batch, seq, hidden] shape
            # if the original capture was 3D, else [seq, hidden].
            orig = self.tensors[f"{prefix}/attn_output"]
            if orig.dim() == 3:
                attn_out = attn_out.unsqueeze(0)
            self.tensors[f"{prefix}/attn_output"] = attn_out

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
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    elapsed = time.time() - t0
    print(f"Model loaded in {elapsed:.1f}s")

    if not isinstance(model, LlamaForCausalLM):
        print(f"WARNING: Model is {type(model).__name__}, not LlamaForCausalLM. "
              f"Hooks assume Llama architecture.", file=sys.stderr)

    model.eval()
    return model, tokenizer


def reverse_permute_qk_from_gguf(weight: torch.Tensor, num_heads: int, head_dim: int) -> torch.Tensor:
    """Reverse the GGUF interleave permutation to recover HF-layout Q or K weights.

    Mirrors fracture-gguf's reverse_qk_permutation. Required when loading a
    GGUF-permuted fixture into an HF model whose forward pass expects HF layout.

    Input shape: (num_heads * head_dim, hidden)
    """
    assert weight.shape[0] == num_heads * head_dim, (
        f"weight rows {weight.shape[0]} != num_heads*head_dim "
        f"{num_heads}*{head_dim}={num_heads * head_dim}"
    )
    half = head_dim // 2
    w = weight.reshape(num_heads, head_dim, -1)
    out = torch.empty_like(w)
    out[:, :half, :] = w[:, 0::2, :]
    out[:, half:, :] = w[:, 1::2, :]
    return out.reshape(weight.shape)


def load_fixture_model(gguf_path: str, config_path: str) -> tuple[torch.nn.Module, Any]:
    """Build a PyTorch Llama matching the fixture GGUF, load weights from GGUF."""
    import gguf as gguf_lib
    from transformers import LlamaConfig, LlamaForCausalLM as _LlamaForCausalLM

    print(f"Loading fixture config from {config_path}...")
    cfg = json.loads(Path(config_path).read_text())
    print(f"Building LlamaForCausalLM (FP16, CUDA) with fixture dimensions...")
    hf_cfg = LlamaConfig(
        vocab_size=cfg["vocab_size"],
        hidden_size=cfg["hidden_size"],
        intermediate_size=cfg["intermediate_size"],
        num_hidden_layers=cfg["num_layers"],
        num_attention_heads=cfg["num_q_heads"],
        num_key_value_heads=cfg["num_kv_heads"],
        max_position_embeddings=cfg["max_position_embeddings"],
        rms_norm_eps=cfg["rms_norm_eps"],
        rope_theta=cfg["rope_theta"],
        torch_dtype=torch.float16,
        tie_word_embeddings=False,
    )
    torch.manual_seed(SEED)
    model = _LlamaForCausalLM(hf_cfg).to(dtype=torch.float16, device="cuda")

    print(f"Reading GGUF tensors from {gguf_path}...")
    reader = gguf_lib.GGUFReader(gguf_path)
    tensor_map = {t.name: torch.from_numpy(t.data.copy()) for t in reader.tensors}
    print(f"  Found {len(tensor_map)} GGUF tensors")

    sd = model.state_dict()

    n_q = cfg["num_q_heads"]
    n_kv = cfg["num_kv_heads"]
    hd = cfg["head_dim"]

    def assign(hf_name: str, gg_name: str) -> None:
        if gg_name not in tensor_map:
            raise KeyError(f"GGUF tensor '{gg_name}' not found (needed for HF '{hf_name}')")
        src = tensor_map[gg_name]
        dst = sd[hf_name]
        if src.shape != dst.shape:
            # GGUF stores 2-D linear weights in [out, in] which usually matches HF nn.Linear.
            # If shapes are transposed, attempt a transpose recovery.
            if src.dim() == 2 and dst.dim() == 2 and src.shape == (dst.shape[1], dst.shape[0]):
                src = src.t().contiguous()
            else:
                raise ValueError(
                    f"Shape mismatch for {hf_name} <- {gg_name}: "
                    f"GGUF {tuple(src.shape)} vs HF {tuple(dst.shape)}"
                )
        src = src.to(dst.dtype).reshape(dst.shape)
        # Q/K weights in the fixture GGUF are stored in llama.cpp's interleaved layout.
        # Reverse the permutation so HF's forward pass (which expects HF layout) is correct.
        if hf_name.endswith("self_attn.q_proj.weight"):
            src = reverse_permute_qk_from_gguf(src, n_q, hd)
        elif hf_name.endswith("self_attn.k_proj.weight"):
            src = reverse_permute_qk_from_gguf(src, n_kv, hd)
        dst.copy_(src.reshape(dst.shape))

    top_pairs = [
        ("model.embed_tokens.weight", "token_embd.weight"),
        ("model.norm.weight", "output_norm.weight"),
        ("lm_head.weight", "output.weight"),
    ]
    for hf, gg in top_pairs:
        assign(hf, gg)
    for i in range(cfg["num_layers"]):
        layer_pairs = [
            (f"model.layers.{i}.input_layernorm.weight", f"blk.{i}.attn_norm.weight"),
            (f"model.layers.{i}.self_attn.q_proj.weight", f"blk.{i}.attn_q.weight"),
            (f"model.layers.{i}.self_attn.k_proj.weight", f"blk.{i}.attn_k.weight"),
            (f"model.layers.{i}.self_attn.v_proj.weight", f"blk.{i}.attn_v.weight"),
            (f"model.layers.{i}.self_attn.o_proj.weight", f"blk.{i}.attn_output.weight"),
            (f"model.layers.{i}.post_attention_layernorm.weight", f"blk.{i}.ffn_norm.weight"),
            (f"model.layers.{i}.mlp.gate_proj.weight", f"blk.{i}.ffn_gate.weight"),
            (f"model.layers.{i}.mlp.up_proj.weight", f"blk.{i}.ffn_up.weight"),
            (f"model.layers.{i}.mlp.down_proj.weight", f"blk.{i}.ffn_down.weight"),
        ]
        for hf, gg in layer_pairs:
            assign(hf, gg)

    model.load_state_dict(sd)
    model.eval()

    # Ensure pad_token_id is set so model.generate doesn't complain.
    if model.config.pad_token_id is None:
        model.config.pad_token_id = 0
    if model.generation_config is not None and model.generation_config.pad_token_id is None:
        model.generation_config.pad_token_id = 0

    class FixtureTokenizer:
        def __init__(self, vocab: int):
            self.vocab_size = vocab
            self.eos_token_id = 2
            self.pad_token_id = 0

        def __call__(self, prompt: str, return_tensors: str = "pt") -> dict[str, torch.Tensor]:
            ids = [3 + (ord(c) % (self.vocab_size - 3)) for c in prompt][:32] or [3]
            return {"input_ids": torch.tensor([ids], dtype=torch.long).cuda()}

        def decode(self, ids, skip_special_tokens: bool = True) -> str:
            try:
                n = len(ids)
            except TypeError:
                n = 1
            return f"<{n} tokens>"

    return model, FixtureTokenizer(cfg["vocab_size"])


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

    # Recompute attn_output in FP32 to match GPU kernel precision.
    print("  Recomputing FP32 attention outputs...")
    capture.compute_attention_outputs()

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

    # Recompute attn_output in FP32 (note: decode-step attention also reads
    # past KV from cache, which we don't recompute here — the kernel attention
    # tests only use prefill data, so this is best-effort for the decode dump).
    capture.compute_attention_outputs()

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
        description="Dump reference tensors for Fracture validation (HF model or fixture GGUF).",
    )
    parser.add_argument(
        "--mode",
        choices=["hf", "fixture"],
        required=True,
        help="Source of weights: 'hf' (HuggingFace dir) or 'fixture' (GGUF + JSON config).",
    )
    parser.add_argument(
        "--model-path",
        required=True,
        help="HuggingFace model dir (hf mode) or path to GGUF file (fixture mode).",
    )
    parser.add_argument(
        "--config-path",
        default=None,
        help="Path to fixture config JSON. Required when --mode fixture.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for reference tensors.",
    )
    parser.add_argument(
        "--golden-dir",
        required=True,
        help="Output directory for golden generation outputs.",
    )
    parser.add_argument(
        "--prompts",
        nargs="+",
        default=None,
        help="Prompt strings (default: built-in TEST_PROMPTS).",
    )
    parser.add_argument(
        "--layers",
        default="0,last",
        help="Comma-separated layer indices, 'all', or 'last' as a sentinel (default: '0,last'). Supports ranges: '0-3,31'.",
    )
    parser.add_argument(
        "--generate-tokens",
        type=int,
        default=GREEDY_GENERATE_TOKENS,
        help=f"Number of tokens to generate greedily (default: {GREEDY_GENERATE_TOKENS})",
    )
    args = parser.parse_args()

    if args.mode == "fixture" and not args.config_path:
        parser.error("--config-path is required when --mode fixture")

    prompts = args.prompts if args.prompts is not None else list(TEST_PROMPTS)

    output_dir = Path(args.output_dir)
    golden_dir = Path(args.golden_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    golden_dir.mkdir(parents=True, exist_ok=True)

    print(f"Mode:             {args.mode}")
    print(f"Reference output: {output_dir.resolve()}")
    print(f"Golden output:    {golden_dir.resolve()}")
    print(f"Model:            {args.model_path}")
    if args.mode == "fixture":
        print(f"Config:           {args.config_path}")
    print(f"Prompts:          {prompts}")
    print(f"Seed:             {SEED}")

    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available. This script requires an NVIDIA GPU.", file=sys.stderr)
        sys.exit(1)
    print(f"CUDA device:      {torch.cuda.get_device_name(0)}")
    print(f"PyTorch version:  {torch.__version__}")

    torch.manual_seed(SEED)

    if args.mode == "hf":
        model, tokenizer = load_model(args.model_path)
    else:
        model, tokenizer = load_fixture_model(args.model_path, args.config_path)

    num_layers = model.config.num_hidden_layers

    # Resolve 'last' sentinel in --layers before parsing (parse_layers accepts ints/'all').
    layers_arg = args.layers
    if layers_arg and layers_arg.lower() != "all":
        parts = []
        for part in layers_arg.split(","):
            p = part.strip()
            if p.lower() == "last":
                parts.append(str(num_layers - 1))
            else:
                parts.append(p)
        layers_arg = ",".join(parts)

    try:
        layer_indices = parse_layers(layers_arg, num_layers)
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
    for i, prompt in enumerate(prompts):
        result = dump_prefill(model, tokenizer, prompt, i, output_dir, layer_indices)
        prefill_results.append(result)

    # --- Decode step (after prompt 0 prefill) ---
    dump_decode_step(model, tokenizer, prefill_results[0], output_dir, layer_indices)

    # --- Greedy generation golden outputs ---
    dump_greedy_generation(model, tokenizer, prompts, golden_dir, args.generate_tokens)

    # --- Write generation info ---
    gen_info = {
        "mode": args.mode,
        "model_id": args.model_path,
        "config_path": args.config_path,
        "torch_version": torch.__version__,
        "cuda_device": torch.cuda.get_device_name(0),
        "seed": SEED,
        "prompts": prompts,
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
