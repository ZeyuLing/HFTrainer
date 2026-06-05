#!/usr/bin/env python3
"""Build annotation-key caption map from official HumanML3D text files."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_first_full_caption(text_path: Path) -> str | None:
    if not text_path.exists():
        return None
    for line in text_path.read_text().splitlines():
        parts = line.strip().split("#")
        if len(parts) < 4:
            continue
        try:
            f_tag = float(parts[2])
            to_tag = float(parts[3])
        except ValueError:
            continue
        if f_tag == 0.0 and to_tag == 0.0 and parts[0].strip():
            return parts[0].strip()
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--anno-file", default="data/annotation/test_hml3d.json")
    ap.add_argument("--src-h3d272", default="ref_repo/MotionStreamer/MotionStreamer/humanml3d_272")
    ap.add_argument("--out-json", required=True)
    args = ap.parse_args()

    anno = json.loads(Path(args.anno_file).read_text())["data_list"]
    text_root = Path(args.src_h3d272) / "texts"
    out = {}
    missing = 0
    for name, entry in anno.items():
        cid = Path(str(entry.get("smplx_path") or "")).stem
        cap = read_first_full_caption(text_root / f"{cid}.txt") if cid else None
        if cap is None:
            missing += 1
            continue
        out[name] = cap
    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"[done] wrote={len(out)} missing={missing} out={out_path}", flush=True)


if __name__ == "__main__":
    main()
