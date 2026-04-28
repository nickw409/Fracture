#!/usr/bin/env python3
"""Build a tiny seeded Llama-architecture model and write it as GGUF.

Both Fracture (via fracture-gguf) and PyTorch (via dump_reference.py) load
the resulting file, so weights are bit-identical across runtimes.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from gguf import GGUFWriter


def permute_qk_for_gguf(weight: np.ndarray, num_heads: int, head_dim: int) -> np.ndarray:
    """Permute an HF-layout Q or K weight matrix into GGUF interleaved layout.

    Mirrors what llama.cpp's convert_hf_to_gguf.py applies to Q/K weights. The
    inverse is fracture-gguf's reverse_qk_permutation, applied at load time.

    Input shape: (num_heads * head_dim, hidden)
    """
    assert weight.shape[0] == num_heads * head_dim, (
        f"weight rows {weight.shape[0]} != num_heads*head_dim "
        f"{num_heads}*{head_dim}={num_heads * head_dim}"
    )
    half = head_dim // 2
    w = weight.reshape(num_heads, head_dim, -1)
    out = np.empty_like(w)
    out[:, 0::2, :] = w[:, :half, :]
    out[:, 1::2, :] = w[:, half:, :]
    return out.reshape(weight.shape)


def build(config_path: Path, output_path: Path) -> None:
    cfg = json.loads(config_path.read_text())
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])

    n_layers = cfg["num_layers"]
    hidden = cfg["hidden_size"]
    n_q = cfg["num_q_heads"]
    n_kv = cfg["num_kv_heads"]
    head_dim = cfg["head_dim"]
    intermediate = cfg["intermediate_size"]
    vocab = cfg["vocab_size"]
    kv_dim = n_kv * head_dim

    writer = GGUFWriter(str(output_path), arch="llama")
    writer.add_context_length(cfg["max_position_embeddings"])
    writer.add_embedding_length(hidden)
    writer.add_block_count(n_layers)
    writer.add_feed_forward_length(intermediate)
    writer.add_head_count(n_q)
    writer.add_head_count_kv(n_kv)
    writer.add_layer_norm_rms_eps(cfg["rms_norm_eps"])
    writer.add_rope_freq_base(cfg["rope_theta"])
    writer.add_rope_dimension_count(head_dim)
    writer.add_vocab_size(vocab)
    writer.add_file_type(1)  # MOSTLY_F16

    # Minimal tokenizer metadata so loaders accept the file.
    tokens = [f"<tok{i}>" for i in range(vocab)]
    writer.add_tokenizer_model("llama")
    writer.add_token_list(tokens)
    writer.add_token_scores([0.0] * vocab)
    writer.add_token_types([1] * vocab)
    writer.add_bos_token_id(1)
    writer.add_eos_token_id(2)

    def randn(*shape: int) -> np.ndarray:
        return torch.randn(*shape, dtype=torch.float32).numpy().astype(np.float16)

    def ones(*shape: int) -> np.ndarray:
        return np.ones(shape, dtype=np.float16)

    writer.add_tensor("token_embd.weight", randn(vocab, hidden))
    for i in range(n_layers):
        writer.add_tensor(f"blk.{i}.attn_norm.weight", ones(hidden))
        writer.add_tensor(
            f"blk.{i}.attn_q.weight",
            permute_qk_for_gguf(randn(hidden, hidden), n_q, head_dim),
        )
        writer.add_tensor(
            f"blk.{i}.attn_k.weight",
            permute_qk_for_gguf(randn(kv_dim, hidden), n_kv, head_dim),
        )
        writer.add_tensor(f"blk.{i}.attn_v.weight", randn(kv_dim, hidden))
        writer.add_tensor(f"blk.{i}.attn_output.weight", randn(hidden, hidden))
        writer.add_tensor(f"blk.{i}.ffn_norm.weight", ones(hidden))
        writer.add_tensor(f"blk.{i}.ffn_gate.weight", randn(intermediate, hidden))
        writer.add_tensor(f"blk.{i}.ffn_up.weight", randn(intermediate, hidden))
        writer.add_tensor(f"blk.{i}.ffn_down.weight", randn(hidden, intermediate))
    writer.add_tensor("output_norm.weight", ones(hidden))
    writer.add_tensor("output.weight", randn(vocab, hidden))

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"Wrote {output_path} ({output_path.stat().st_size} bytes)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    build(args.config, args.output)
