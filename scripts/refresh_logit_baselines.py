#!/usr/bin/env python3
"""Refresh tests/baselines/logit_baselines.json from a captured nextest log.

Usage:
    cargo nextest run -p fracture-model-validation --no-capture 2>&1 | tee /tmp/log.txt
    python scripts/refresh_logit_baselines.py --log /tmp/log.txt

The log must contain `BASELINE_CAPTURE <key>: max_abs_error=<value>` lines emitted
by the model-validation tests. Lines for keys not present in the JSON file are
ignored. Existing values are overwritten in place; tolerance_factor and notes are preserved.
"""
import argparse
import json
import re
from pathlib import Path

PATTERN = re.compile(
    r"BASELINE_CAPTURE\s+(?P<key>[\w/.\-]+):\s+max_abs_error=(?P<v>[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)"
)
BASELINES_PATH = Path(__file__).resolve().parent.parent / "tests/baselines/logit_baselines.json"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--log", required=True, type=Path,
                   help="Path to a nextest log containing BASELINE_CAPTURE lines")
    args = p.parse_args()

    if not BASELINES_PATH.exists():
        raise SystemExit(f"baselines file missing: {BASELINES_PATH}")
    if not args.log.exists():
        raise SystemExit(f"log file missing: {args.log}")

    data = json.loads(BASELINES_PATH.read_text())
    log = args.log.read_text()

    updated = 0
    skipped = 0
    for m in PATTERN.finditer(log):
        key = m.group("key")
        v = float(m.group("v"))
        if key in data["baselines"]:
            old = data["baselines"][key].get("max_abs_error", 0.0)
            data["baselines"][key]["max_abs_error"] = v
            print(f"updated {key}: {old} -> {v}")
            updated += 1
        else:
            print(f"skip unknown key: {key}")
            skipped += 1

    BASELINES_PATH.write_text(json.dumps(data, indent=2) + "\n")
    print(f"\n{updated} baseline(s) updated; {skipped} unknown key(s) ignored")


if __name__ == "__main__":
    main()
