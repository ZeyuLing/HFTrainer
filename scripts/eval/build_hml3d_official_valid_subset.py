#!/usr/bin/env python3
"""Build an annotation/caption subset aligned with official HumanML3D test ids."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from build_hml3d_official_caption_map import read_first_full_caption


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--anno-file", default="data/annotation/test_hml3d.json")
    ap.add_argument("--recon-root", default="work_dirs/h3d263_eval/h3d263_test_recon_fk")
    ap.add_argument("--src-h3d272", default="ref_repo/MotionStreamer/MotionStreamer/humanml3d_272")
    ap.add_argument("--out-anno", required=True)
    ap.add_argument("--out-captions", required=True)
    ap.add_argument("--max-samples", type=int, default=256)
    ap.add_argument("--min-len", type=int, default=40)
    args = ap.parse_args()

    anno = json.loads(Path(args.anno_file).read_text())["data_list"]
    by_cid = {}
    for name, entry in anno.items():
        cid = Path(str(entry.get("smplx_path") or "")).stem
        if cid and cid not in by_cid:
            by_cid[cid] = (name, entry)

    split_ids = [
        x.strip()
        for x in (Path(args.recon_root) / "test.txt").read_text().splitlines()
        if x.strip()
    ]
    text_root = Path(args.src_h3d272) / "texts"
    out_data = {}
    out_caps = {}
    skipped = {"no_anno": 0, "short": 0, "no_caption": 0}
    for cid in split_ids:
        if cid not in by_cid:
            skipped["no_anno"] += 1
            continue
        motion_path = Path(args.recon_root) / "new_joint_vecs" / f"{cid}.npy"
        if not motion_path.exists() or len(np.load(motion_path, mmap_mode="r")) < args.min_len:
            skipped["short"] += 1
            continue
        cap = read_first_full_caption(text_root / f"{cid}.txt")
        if not cap:
            skipped["no_caption"] += 1
            continue
        name, entry = by_cid[cid]
        out_data[name] = entry
        out_caps[name] = cap
        if args.max_samples and len(out_data) >= args.max_samples:
            break

    out_anno = {"data_list": out_data}
    Path(args.out_anno).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_anno).write_text(json.dumps(out_anno, indent=2))
    Path(args.out_captions).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_captions).write_text(json.dumps(out_caps, indent=2, ensure_ascii=False))
    print(
        f"[done] samples={len(out_data)} skipped={skipped} "
        f"anno={args.out_anno} captions={args.out_captions}",
        flush=True,
    )


if __name__ == "__main__":
    main()
