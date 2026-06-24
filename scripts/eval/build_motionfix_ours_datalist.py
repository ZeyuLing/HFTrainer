#!/usr/bin/env python3
"""Build an \\ours{} M2M evaluation datalist from the MotionFix test split.

MotionFix (Athanasiou et al.) edits a SOURCE motion with a natural-language
instruction to produce a TARGET motion. We reuse \\ours{}'s E16 ``style_edit``
setting (source motion through the editing/reactive channel + edit text -> full
regeneration) to perform the same task, then evaluate with the official
MotionFix TMR Generated-to-Target retrieval (scripts under data/MotionFix).

motionfix_test.pth.tar is a joblib dict keyed by MotionFix keyid (e.g. '000004'),
each value = {'motion_source': {'rots'[T,66] aa, 'trans'[T,3], ...},
              'motion_target': {...}, 'text': <edit instruction>}.

We dump per-keyid SMPL npz ({trans, poses}) for source & target -- load_motion_135d
converts poses(axis-angle 22*3) -> rot6d on the fly -- and a datalist JSON in the
eval_e16_semantic_style_edit.json schema. keyids are preserved so the generated
outputs can be matched to MotionFix targets by the TMR evaluator.

Usage:
    python3 scripts/eval/build_motionfix_ours_datalist.py \
        --pth data/MotionFix/motionfix_test.pth.tar \
        --out-npz-dir data/eval/motionfix/ours_smpl_npz \
        --out-datalist data/eval/m2m_v2/eval_motionfix_instruction.json \
        --fps 30 [--max-samples N]
"""
import argparse
import json
import os
from pathlib import Path

import joblib
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pth", default="data/MotionFix/motionfix_test.pth.tar")
    ap.add_argument("--out-npz-dir", default="data/eval/motionfix/ours_smpl_npz")
    ap.add_argument("--out-datalist",
                    default="data/eval/m2m_v2/eval_motionfix_instruction.json")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--max-samples", type=int, default=0,
                    help="0 = all test items")
    args = ap.parse_args()

    # Use the canonical CephFS mount (visible on both this box and Taiji
    # containers); avoid Path.resolve() which expands to the /apdcephfs/AILab_DHA
    # symlink target that does NOT exist inside Taiji containers.
    root = Path("/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer")
    if not root.exists():
        root = Path(__file__).resolve().parents[2]
    pth = (root / args.pth) if not os.path.isabs(args.pth) else Path(args.pth)
    npz_dir = (root / args.out_npz_dir) if not os.path.isabs(args.out_npz_dir) \
        else Path(args.out_npz_dir)
    src_dir = npz_dir / "source"
    tgt_dir = npz_dir / "target"
    src_dir.mkdir(parents=True, exist_ok=True)
    tgt_dir.mkdir(parents=True, exist_ok=True)
    dl_path = (root / args.out_datalist) if not os.path.isabs(args.out_datalist) \
        else Path(args.out_datalist)
    dl_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[load] {pth}")
    data = joblib.load(pth)
    keyids = sorted(data.keys())
    if args.max_samples and args.max_samples > 0:
        keyids = keyids[: args.max_samples]
    print(f"[info] {len(keyids)} MotionFix test items")

    def dump(motion: dict, path: Path):
        rots = np.asarray(motion["rots"], dtype=np.float32)   # (T, 66) aa, SMPL-22
        trans = np.asarray(motion["trans"], dtype=np.float32)  # (T, 3)
        T = min(rots.shape[0], trans.shape[0])
        rots = rots[:T]
        # process_smplx_pose expects SMPL-X 55-joint axis-angle ([T, 165]).
        # MotionFix gives SMPL-22 (66 = 22*3); the first 22 SMPL-X joints are the
        # body, so zero-pad joints 22..54 (jaw/eyes/hands).
        if rots.shape[1] == 66:
            pad = np.zeros((T, 55 * 3 - 66), dtype=np.float32)
            poses = np.concatenate([rots, pad], axis=-1)
        else:
            poses = rots
        np.savez(path, trans=trans[:T], poses=poses)
        return T

    items = []
    n_bad = 0
    for kid in keyids:
        it = data[kid]
        try:
            sp = src_dir / f"{kid}.npz"
            tp = tgt_dir / f"{kid}.npz"
            T_src = dump(it["motion_source"], sp)
            T_tgt = dump(it["motion_target"], tp)
        except Exception as e:  # noqa: BLE001
            n_bad += 1
            continue
        text = it.get("text", "")
        if isinstance(text, (list, tuple)):
            text = text[0] if text else ""
        items.append({
            "prompt_id": kid,
            "annotation_id": kid,
            "motion_path": str(tp),            # TARGET = GT metric reference
            "source_motion_path": str(sp),     # SOURCE = editing/reactive channel
            "caption": text,
            "caption_en": text,
            "edit_setting": "style_edit",
            "edit_type": "instruction",
            "source": "MotionFix test (instruction editing)",
            "num_frames": int(T_tgt),
            "fps": args.fps,
            "duration_sec": round(T_tgt / args.fps, 3),
        })

    out = {
        "meta": {
            "task": "motionfix_instruction_edit",
            "source": "MotionFix test split (motionfix_test.pth.tar)",
            "n": len(items),
            "n_skipped": n_bad,
            "note": "Reuses E16 style_edit setting; keyids preserved for TMR G2T.",
        },
        "data_list": items,
    }
    with open(dl_path, "w") as f:
        json.dump(out, f)
    print(f"[done] {len(items)} items ({n_bad} skipped) -> {dl_path}")
    print(f"[npz]  source/target SMPL npz -> {npz_dir}")


if __name__ == "__main__":
    main()
