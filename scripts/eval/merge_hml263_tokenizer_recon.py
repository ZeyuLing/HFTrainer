#!/usr/bin/env python3
"""Merge sharded HML263 tokenizer reconstruction outputs."""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Any

import numpy as np


def _summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"mean": None, "std": None, "num_samples": 0}
    arr = np.asarray(values, dtype=np.float64)
    return {"mean": float(arr.mean()), "std": float(arr.std(ddof=0)), "num_samples": int(arr.size)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    paths: list[str] = []
    for item in args.inputs:
        matches = sorted(glob.glob(item))
        paths.extend(matches if matches else [item])
    paths = sorted(dict.fromkeys(paths))
    if not paths:
        raise ValueError("no inputs")
    payloads = [json.loads(Path(path).read_text(encoding="utf-8")) for path in paths]
    first = payloads[0]
    codebook_size = int(first["summary"].get("codebook_size") or first.get("model_meta", {}).get("codebook_size") or 0)
    usage: list[set[int]] = []
    per_case: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    written = 0
    selected = 0
    for payload in payloads:
        per_case.extend(payload.get("per_case", []))
        failures.extend(payload.get("failures", []))
        written += int(payload.get("written", 0))
        selected += int(payload.get("selected_samples", 0))
        values = payload.get("code_usage_values_per_quantizer", [])
        while len(usage) < len(values):
            usage.append(set())
        for idx, vals in enumerate(values):
            usage[idx].update(int(v) for v in vals)
    l1 = [float(item["hml263_l1"]) for item in per_case]
    mse = [float(item["hml263_mse"]) for item in per_case]
    per_quant = [(len(items) / codebook_size * 100.0) if codebook_size else None for items in usage]
    cb_util = float(np.mean([x for x in per_quant if x is not None])) if per_quant else None
    merged = {
        "method": first.get("method"),
        "recon_root": first.get("recon_root"),
        "split": first.get("split"),
        "out_dir": str(Path(args.output).parent),
        "merged_from": paths,
        "selected_samples": selected,
        "written": written,
        "num_failures": len(failures),
        "model_meta": first.get("model_meta", {}),
        "summary": {
            "hml263_l1": _summary(l1),
            "hml263_mse": _summary(mse),
            "cb_util_percent": cb_util,
            "cb_util_percent_per_quantizer": per_quant,
            "codebook_size": codebook_size,
            "num_failures": len(failures),
        },
        "code_usage_values_per_quantizer": [sorted(items) for items in usage],
        "failures": failures,
        "per_case": sorted(per_case, key=lambda item: item["key"]),
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(merged, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(merged["summary"], indent=2, ensure_ascii=False))
    print(f"[merge-hml263-tokenizer-recon] wrote {out}")


if __name__ == "__main__":
    main()
