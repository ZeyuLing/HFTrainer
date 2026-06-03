#!/usr/bin/env python3
"""Build a SOTA-aligned eval prompt set from the OFFICIAL HumanML3D test split.

T2M baselines (MDM/MoMask/MotionGPT/...) all report on the HumanML3D test split.
This script samples captions from that exact split (``humanml3d_272/split/test.txt``
+ ``texts/<id>.txt``) so PhysFlow visualizations and physical-realism metrics are
computed on the same prompt distribution the field benchmarks on.

Output schema matches PhysFlowPromptDataset / PromptSpec (id, prompt, category,
difficulty, duration_sec, split, source, tags), so the existing feature-extract +
viz pipeline can consume it unchanged.

NOTE on leakage: our online-adversarial *training* corpus (physflow_text_train,
11241 captions) was carved from a per-subdataset split that is NOT aligned to the
official HumanML3D train/test partition, so some official-test captions may also
appear in training. That only matters for *quantitative held-out tables* (handle by
retraining with the official split excluded); for qualitative visualization on the
standard prompt distribution it is fine.
"""
from __future__ import annotations
import argparse
import json
import random
from pathlib import Path

ROOT = Path("/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer")
SPLIT_DIR = ROOT / "ref_repo/MotionStreamer/MotionStreamer/humanml3d_272/split"
TEXTS_DIR = ROOT / "ref_repo/MotionStreamer/MotionStreamer/humanml3d_272/texts"

# locomotion keywords to guarantee the hard cases (walk/turn/sit) are represented.
LOCO = ("walk", "run", "jog", "step", "turn", "stagger", "march", "stairs", "climb")


def read_first_caption(mid: str) -> str | None:
    p = TEXTS_DIR / f"{mid}.txt"
    if not p.is_file():
        return None
    for line in p.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        # format: caption#tokenized#f_tag#to_tag ; full clip => f_tag==to_tag==0.0
        parts = line.split("#")
        if len(parts) < 4:
            cap = parts[0].strip()
            if cap:
                return cap
            continue
        cap, f_tag, to_tag = parts[0].strip(), parts[-2].strip(), parts[-1].strip()
        try:
            if float(f_tag) == 0.0 and float(to_tag) == 0.0 and cap:
                return cap
        except ValueError:
            if cap:
                return cap
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split-file", default=str(SPLIT_DIR / "test.txt"))
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--min-loco", type=int, default=12,
                    help="guarantee at least this many locomotion prompts")
    ap.add_argument("--seed", type=int, default=20260602)
    ap.add_argument("--duration-sec", type=float, default=4.0)
    ap.add_argument("--out", default="configs/experiments/physflow_kimodo_g1/physflow_bench_hml3d_test.jsonl")
    args = ap.parse_args()

    ids = [l.strip() for l in Path(args.split_file).read_text().splitlines() if l.strip()]
    # skip mirror ids (prefixed 'M') for cleaner captions
    ids = [i for i in ids if not i.startswith("M")]
    rng = random.Random(args.seed)
    rng.shuffle(ids)

    loco, other = [], []
    for mid in ids:
        cap = read_first_caption(mid)
        if not cap:
            continue
        rec = {"mid": mid, "cap": cap}
        (loco if any(k in cap.lower() for k in LOCO) else other).append(rec)
        if len(loco) >= args.min_loco and len(loco) + len(other) >= args.n * 3:
            break

    picked = loco[: args.min_loco] + other[: max(0, args.n - args.min_loco)]
    picked = picked[: args.n]

    out_path = ROOT / args.out
    with open(out_path, "w", encoding="utf-8") as f:
        for i, r in enumerate(picked):
            is_loco = any(k in r["cap"].lower() for k in LOCO)
            f.write(json.dumps({
                "id": f"hml3dtest_{i:04d}",
                "prompt": r["cap"],
                "category": "humanml3d_official_test",
                "difficulty": 0,
                "duration_sec": args.duration_sec,
                "split": "test",
                "source": "HumanML3D-test",
                "hml3d_id": r["mid"],
                "tags": ["locomotion"] if is_loco else ["other"],
            }, ensure_ascii=False) + "\n")
    print(f"[build] wrote {len(picked)} prompts ({sum(1 for r in picked if any(k in r['cap'].lower() for k in LOCO))} locomotion) -> {out_path}")


if __name__ == "__main__":
    main()
