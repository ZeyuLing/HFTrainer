#!/usr/bin/env python3
"""Build an \\ours E16 style_edit datalist from MotionFix *TRAIN* pairs.

These are the exact editing pairs the smpl_caption_editfix_latest model was
trained on (subset == 'MotionFix-train' in the training annotation). We point
motion_path/source_motion_path directly at the on-disk SMPLX train npz
(poses[T,156], trans[T,3]); load_motion_135d takes the first 22 SMPL joints.
"""
import argparse
import json
import random
from pathlib import Path

ROOT = Path("/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer")
DATA_ROOT = ROOT / "data" / "hymotion_data"
ANNO = ROOT / "data/annotation/train_hymotion_400h_hq_permo_motionfix_editing_20260527.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "data/eval/m2m_v2/eval_motionfix_train_instruction.json"))
    args = ap.parse_args()

    print(f"[load] {ANNO}")
    d = json.load(open(ANNO))
    items = list(d["data_list"].values()) if isinstance(d["data_list"], dict) else d["data_list"]
    mf = [x for x in items if isinstance(x, dict) and x.get("subset") == "MotionFix-train"]
    print(f"[info] {len(mf)} MotionFix-train editing pairs")

    random.seed(args.seed)
    random.shuffle(mf)

    out_items = []
    for x in mf:
        if len(out_items) >= args.n:
            break
        tgt = DATA_ROOT / x["smplx_path"]
        src = DATA_ROOT / x["source_motion_path"]
        if not (tgt.exists() and src.exists()):
            continue
        cap = ""
        cap_path = DATA_ROOT / x.get("caption_path", "")
        if cap_path.exists():
            try:
                cj = json.load(open(cap_path))
                cap = cj["result"][0]["short_caption"]
            except Exception:
                cap = ""
        kid = Path(x["smplx_path"]).stem.replace("_target", "")
        out_items.append({
            "prompt_id": kid,
            "annotation_id": kid,
            "motion_path": str(tgt),
            "source_motion_path": str(src),
            "caption": cap,
            "caption_en": cap,
            "edit_setting": "style_edit",
            "edit_type": "instruction",
            "source": "MotionFix TRAIN (in-training editing pair)",
            "num_frames": int(x.get("num_frames", 120)),
            "fps": x.get("fps", 30.0),
            "duration_sec": round(int(x.get("num_frames", 120)) / 30.0, 3),
        })

    out = {
        "meta": {
            "task": "motionfix_train_instruction_edit",
            "source": "MotionFix TRAIN split (subset==MotionFix-train)",
            "n": len(out_items),
        },
        "data_list": out_items,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.out, "w"))
    print(f"[done] {len(out_items)} items -> {args.out}")
    for it in out_items[:3]:
        print("  ex:", it["prompt_id"], "|", it["caption"][:60])


if __name__ == "__main__":
    main()
