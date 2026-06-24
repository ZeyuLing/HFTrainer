#!/usr/bin/env python3
"""Build a MotionHub-style annotation from the official HumanML3D-272 test split.

The generated annotation is keyed by canonical HumanML3D ids (for example
``004822``) and uses the exact frame count from
``humanml3d_272/motion_data/<id>.npy``.  It is intended for T2M generation
protocols where every method must emit a clip with the same frame count as the
official HumanML3D-272 ground truth.
"""
from __future__ import annotations

import argparse
import ast
import json
import struct
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def npy_header_shape(path: Path) -> tuple[int, ...]:
    with path.open("rb") as f:
        magic = f.read(6)
        if magic != b"\x93NUMPY":
            raise ValueError(f"bad npy magic: {path}")
        major, _minor = f.read(2)
        if major == 1:
            hlen = struct.unpack("<H", f.read(2))[0]
        else:
            hlen = struct.unpack("<I", f.read(4))[0]
        meta = ast.literal_eval(f.read(hlen).decode("latin1"))
    return tuple(meta["shape"])


def read_first_full_caption(text_path: Path) -> str | None:
    if not text_path.exists():
        return None
    for line in text_path.read_text().splitlines():
        parts = line.strip().split("#")
        if len(parts) < 4:
            continue
        try:
            f_tag = float(parts[2]) if parts[2] != "nan" else 0.0
            t_tag = float(parts[3]) if parts[3] != "nan" else 0.0
        except ValueError:
            continue
        caption = parts[0].strip()
        if caption and f_tag == 0.0 and t_tag == 0.0:
            return caption
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--src-h3d272",
        default="ref_repo/MotionStreamer/MotionStreamer/humanml3d_272",
        help="Root containing motion_data/, texts/, split/test.txt.",
    )
    ap.add_argument("--out-anno", default="data/annotation/test_hml3d_official272_gtlen.json")
    ap.add_argument("--out-len-map", default="data/annotation/test_hml3d_official272_gtlen_lenmap.json")
    ap.add_argument(
        "--caption-dir",
        default="data/annotation/hml3d_official272_captions",
        help="Directory where one hierarchical-caption JSON per canonical id is written.",
    )
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--workers", type=int, default=32)
    args = ap.parse_args()

    src = Path(args.src_h3d272)
    motion_dir = src / "motion_data"
    text_dir = src / "texts"
    ids = [x.strip() for x in (src / "split" / "test.txt").read_text().splitlines() if x.strip()]

    caption_dir = Path(args.caption_dir)
    caption_dir.mkdir(parents=True, exist_ok=True)

    def build_one(cid: str):
        motion_path = motion_dir / f"{cid}.npy"
        text_path = text_dir / f"{cid}.txt"
        if not motion_path.exists():
            return cid, None, f"missing motion: {motion_path}"
        caption = read_first_full_caption(text_path)
        if not caption:
            return cid, None, f"missing full caption: {text_path}"
        length = int(npy_header_shape(motion_path)[0])
        cap_rel = caption_dir / f"{cid}.json"
        cap_rel.write_text(
            json.dumps(
                {"macro": [caption], "meso": [], "micro": []},
                ensure_ascii=False,
                indent=2,
            )
        )
        entry = {
            "motion_id": cid,
            "smplx_path": str(motion_path),
            "hierarchical_caption_path": str(cap_rel),
            "num_frames": length,
            "fps": int(args.fps),
            "duration": float(length) / float(args.fps),
            "source": "official_humanml3d_272_test",
        }
        return cid, entry, None

    data_list = {}
    len_map = {}
    errors = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        for cid, entry, err in ex.map(build_one, ids):
            if err:
                errors.append({"id": cid, "error": err})
                continue
            data_list[cid] = entry
            len_map[cid] = int(entry["num_frames"])

    out_anno = Path(args.out_anno)
    out_len = Path(args.out_len_map)
    out_anno.parent.mkdir(parents=True, exist_ok=True)
    out_len.parent.mkdir(parents=True, exist_ok=True)
    out_anno.write_text(
        json.dumps(
            {
                "meta": {
                    "source": str(src),
                    "split": "test",
                    "fps": int(args.fps),
                    "count": len(data_list),
                    "errors": errors,
                },
                "data_list": data_list,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    out_len.write_text(json.dumps(len_map, indent=2))
    print(
        f"[done] ids={len(ids)} written={len(data_list)} errors={len(errors)} "
        f"anno={out_anno} len_map={out_len}",
        flush=True,
    )
    if errors:
        print(f"[warn] first_errors={errors[:5]}", flush=True)


if __name__ == "__main__":
    main()
