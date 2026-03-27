#!/usr/bin/env python3
"""
Verify reference tensor dumps from dump_reference.py.

Loads every .bin file in the reference and golden directories, prints a summary
of each tensor (shape, dtype, min, max, mean, NaN/Inf counts), and flags any
issues.

Usage:
    python scripts/verify_reference.py
    python scripts/verify_reference.py --reference-dir tests/reference --golden-dir tests/golden
"""

from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

import numpy as np

DTYPE_MAP = {
    0: ("float16", np.float16),
    1: ("float32", np.float32),
    2: ("int32", np.int32),
}


def load_tensor(path: Path) -> tuple[np.ndarray, str]:
    """Load a tensor from Fracture reference binary format.

    Returns (ndarray, dtype_name).
    """
    data = path.read_bytes()
    offset = 0

    ndim = struct.unpack_from("<I", data, offset)[0]
    offset += 4

    shape = []
    for _ in range(ndim):
        dim = struct.unpack_from("<I", data, offset)[0]
        offset += 4
        shape.append(dim)

    dtype_enum = struct.unpack_from("<I", data, offset)[0]
    offset += 4

    if dtype_enum not in DTYPE_MAP:
        raise ValueError(f"Unknown dtype enum {dtype_enum} in {path}")

    dtype_name, np_dtype = DTYPE_MAP[dtype_enum]

    expected_bytes = int(np.prod(shape)) * np.dtype(np_dtype).itemsize if shape else 0
    actual_bytes = len(data) - offset

    if actual_bytes != expected_bytes:
        raise ValueError(
            f"Size mismatch in {path}: expected {expected_bytes} bytes for shape {shape} "
            f"dtype {dtype_name}, got {actual_bytes}"
        )

    arr = np.frombuffer(data[offset:], dtype=np_dtype).reshape(shape) if shape else np.array([], dtype=np_dtype)
    return arr, dtype_name


def verify_file(path: Path, base_dir: Path) -> dict:
    """Verify a single tensor file. Returns a summary dict."""
    rel = path.relative_to(base_dir)
    result = {"path": str(rel), "ok": True, "issues": []}

    try:
        arr, dtype_name = load_tensor(path)
    except Exception as e:
        result["ok"] = False
        result["issues"].append(f"LOAD ERROR: {e}")
        return result

    result["shape"] = list(arr.shape)
    result["dtype"] = dtype_name
    result["elements"] = int(np.prod(arr.shape)) if arr.shape else 0

    if arr.size == 0:
        result["issues"].append("EMPTY tensor")
        return result

    if np.issubdtype(arr.dtype, np.floating):
        nan_count = int(np.isnan(arr).sum())
        inf_count = int(np.isinf(arr).sum())
        arr_finite = arr[np.isfinite(arr)]

        result["nan_count"] = nan_count
        result["inf_count"] = inf_count

        if nan_count > 0:
            result["ok"] = False
            result["issues"].append(f"NaN: {nan_count} elements")
        if inf_count > 0:
            result["ok"] = False
            result["issues"].append(f"Inf: {inf_count} elements")

        if arr_finite.size > 0:
            result["min"] = float(arr_finite.min())
            result["max"] = float(arr_finite.max())
            result["mean"] = float(arr_finite.mean())
        else:
            result["min"] = None
            result["max"] = None
            result["mean"] = None
    else:
        result["min"] = int(arr.min())
        result["max"] = int(arr.max())
        result["mean"] = float(arr.astype(np.float64).mean())
        result["nan_count"] = 0
        result["inf_count"] = 0

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify Fracture reference tensor dumps.")
    parser.add_argument("--reference-dir", default="tests/reference", help="Reference tensor directory")
    parser.add_argument("--golden-dir", default="tests/golden", help="Golden output directory")
    args = parser.parse_args()

    ref_dir = Path(args.reference_dir)
    golden_dir = Path(args.golden_dir)

    all_bins: list[tuple[Path, Path]] = []  # (file_path, base_dir)

    if ref_dir.exists():
        for f in sorted(ref_dir.rglob("*.bin")):
            all_bins.append((f, ref_dir))
    else:
        print(f"WARNING: Reference directory {ref_dir} does not exist.", file=sys.stderr)

    if golden_dir.exists():
        for f in sorted(golden_dir.rglob("*.bin")):
            all_bins.append((f, golden_dir))
    else:
        print(f"WARNING: Golden directory {golden_dir} does not exist.", file=sys.stderr)

    if not all_bins:
        print("No .bin files found. Run dump_reference.py first to generate reference data.")
        sys.exit(0)

    print(f"Verifying {len(all_bins)} tensor files...\n")

    ok_count = 0
    fail_count = 0
    total_bytes = 0

    header = f"{'File':<60s} {'Shape':<25s} {'DType':<8s} {'Min':>12s} {'Max':>12s} {'Mean':>12s} {'Status'}"
    print(header)
    print("-" * len(header))

    for path, base in all_bins:
        result = verify_file(path, base)

        shape_str = "x".join(str(d) for d in result.get("shape", []))
        dtype_str = result.get("dtype", "???")
        min_str = f"{result['min']:.5g}" if result.get("min") is not None else "N/A"
        max_str = f"{result['max']:.5g}" if result.get("max") is not None else "N/A"
        mean_str = f"{result['mean']:.5g}" if result.get("mean") is not None else "N/A"

        if result["ok"]:
            status = "OK"
            ok_count += 1
        else:
            status = "FAIL: " + "; ".join(result["issues"])
            fail_count += 1

        print(f"{result['path']:<60s} {shape_str:<25s} {dtype_str:<8s} {min_str:>12s} {max_str:>12s} {mean_str:>12s} {status}")
        total_bytes += path.stat().st_size

    print(f"\n{'='*60}")
    print(f"Total files:  {ok_count + fail_count}")
    print(f"OK:           {ok_count}")
    print(f"Failed:       {fail_count}")
    print(f"Total size:   {total_bytes / (1024*1024):.1f} MB")

    if fail_count > 0:
        print(f"\nERROR: {fail_count} file(s) have issues!", file=sys.stderr)
        sys.exit(1)
    else:
        print("\nAll tensor files verified successfully.")


if __name__ == "__main__":
    main()
