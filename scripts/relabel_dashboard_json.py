#!/usr/bin/env python3
"""Rewrite dashboard import JSON metadata without touching per-sample metrics.

Useful when upstream generators emit technical model/setting identifiers
(`uncond_local`, `D_strict_mask_...`) but the dashboard should show a cleaner
user-facing model name under a unified task setting.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("src")
    parser.add_argument("dst")
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--setting", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--rotation-space", default=None)
    parser.add_argument("--notes", default=None)
    args = parser.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    data = json.loads(src.read_text())
    data["model"] = args.model_name
    if args.setting is not None:
        data["setting"] = args.setting
    if args.checkpoint is not None:
        data["checkpoint"] = args.checkpoint
    if args.rotation_space is not None:
        data["rotation_space"] = args.rotation_space
    if args.notes is not None:
        data["_rewrite_notes"] = args.notes

    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    print(f"[done] wrote {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
