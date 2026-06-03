"""Build a HumanML3D-test editing datalist for eval_m2m_v2_all_tasks.py.

Source motions are our 272->motion_135 conversions (scripts/eval/h3d_272_to_135.py);
captions are the first full-clip (f_tag==to_tag==0) HumanML3D English caption.

The SAME datalist is reused across editing tasks (E2 in-between, E3 keyframe,
E10 part, E5 trajectory) -- the editing MASK is decided by the eval SETTING, not
the datalist, which only provides the source motion + caption.

Pass it to the eval with ``--data-file-override eval_h3d_editing.json``.

Usage:
    python3 scripts/eval/build_h3d_editing_datalist.py \
        --src-npz-dir data/eval/h3d_editing/source_npz \
        --texts-dir ref_repo/MotionStreamer/MotionStreamer/humanml3d_272/texts \
        --out data/eval/m2m_v2/eval_h3d_editing.json
"""
from __future__ import annotations

import argparse
import codecs as cs
import json
import os

import numpy as np

REPO = "/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer"


def read_caption(texts_dir, cid):
    path = os.path.join(texts_dir, cid + ".txt")
    if not os.path.exists(path):
        return None
    with cs.open(path, "r") as f:
        for line in f.readlines():
            parts = line.strip().split("#")
            if len(parts) < 4:
                continue
            cap = parts[0].strip()
            try:
                f_tag = float(parts[2]); to_tag = float(parts[3])
            except ValueError:
                f_tag = to_tag = 0.0
            f_tag = 0.0 if np.isnan(f_tag) else f_tag
            to_tag = 0.0 if np.isnan(to_tag) else to_tag
            if f_tag == 0.0 and to_tag == 0.0 and cap:
                return cap
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-npz-dir", default=os.path.join(REPO, "data/eval/h3d_editing/source_npz"))
    ap.add_argument("--texts-dir", default=os.path.join(
        REPO, "ref_repo/MotionStreamer/MotionStreamer/humanml3d_272/texts"))
    ap.add_argument("--out", default=os.path.join(REPO, "data/eval/m2m_v2/eval_h3d_editing.json"))
    ap.add_argument("--min-frames", type=int, default=40,
                    help="skip clips shorter than this (30 fps)")
    args = ap.parse_args()

    src_dir = args.src_npz_dir if os.path.isabs(args.src_npz_dir) else os.path.join(REPO, args.src_npz_dir)
    texts_dir = args.texts_dir if os.path.isabs(args.texts_dir) else os.path.join(REPO, args.texts_dir)

    ids = sorted(f[:-4] for f in os.listdir(src_dir) if f.endswith(".npz"))
    data_list = []
    no_cap = short = 0
    for cid in ids:
        cap = read_caption(texts_dir, cid)
        if not cap:
            no_cap += 1
            continue
        npz_path = os.path.join(src_dir, cid + ".npz")
        try:
            T = int(np.load(npz_path)["motion_135"].shape[0])
        except Exception:
            continue
        if T < args.min_frames:
            short += 1
            continue
        data_list.append({
            "motion_path": npz_path,          # absolute -> eval loads directly
            "action_name": cap,
            "caption_en": cap,
            "category": "humanml3d",
            "num_frames": T,
            "fps": 30,
            "duration_sec": round(T / 30.0, 3),
            "source": "humanml3d_test",
            "source_id": cid,
        })

    out = {
        "meta": {
            "task_id": "EDIT_H3D",
            "task_name": "HumanML3D-test editing source (shared)",
            "description": ("HumanML3D test clips (272->motion_135 @30fps) as editing "
                            "source; mask decided by eval setting. For baseline-"
                            "comparable in-between/keyframe/spatial/trajectory eval."),
            "total_items": len(data_list),
            "source": "humanml3d_272 test split",
            "fps": 30,
        },
        "data_list": data_list,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, ensure_ascii=False)
    print(f"[done] {len(data_list)} items -> {args.out}  "
          f"(no_caption={no_cap}, too_short={short})")


if __name__ == "__main__":
    main()
