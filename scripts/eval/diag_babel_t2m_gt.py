#!/usr/bin/env python3
"""Diagnostic: reproduce the MotionStreamer BABEL GT R-precision/FID using the
VALIDATED standard t2m protocol (E.build_items + crop_and_norm + encode_items)
on the per-clip babel_272 val set (each clip has its own action caption).

If this reproduces the paper GT (R@3~0.634, MM-D~17.5), our val_stream slicing is
the culprit. If it also lands ~0.47, the released HumanML3D-272 evaluator simply
scores BABEL at ~0.47 and the paper used a (BABEL-trained) evaluator.
"""
import argparse
import os
import sys

import numpy as np

REPO = "/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer"
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts/eval"))
import eval_motionstreamer_272 as E  # noqa: E402

BABEL = os.path.join(REPO, "data/babel/babel_272")
MS = os.path.join(REPO, "ref_repo/MotionStreamer/MotionStreamer")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4000, help="subsample N val ids (0=all)")
    ap.add_argument("--mean-std", default="babel", choices=["babel", "humanml"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-mirror", action="store_true", help="drop M-prefixed mirrored ids")
    ap.add_argument("--caption-template", default="{cap}",
                    help="reformat BABEL terse label into a HumanML3D-like sentence, "
                         "e.g. 'a person {cap}' or 'a person is {cap}'")
    ap.add_argument("--drop-transition", action="store_true",
                    help="drop clips whose caption is the action-agnostic 'transition'")
    args = ap.parse_args()

    import torch
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    if args.mean_std == "babel":
        ms_dir = os.path.join(BABEL, "t2m_babel_mean_std")
    else:
        ms_dir = os.path.join(MS, "humanml3d_272/mean_std")
    mean = np.load(os.path.join(ms_dir, "Mean.npy"))
    std = np.load(os.path.join(ms_dir, "Std.npy"))

    ids = [l.strip() for l in open(os.path.join(BABEL, "split/val.txt")) if l.strip()]
    if args.no_mirror:
        ids = [i for i in ids if not i.startswith("M")]
    rng = np.random.RandomState(args.seed)
    if args.n and args.n < len(ids):
        sel = rng.choice(len(ids), args.n, replace=False)
        ids = [ids[i] for i in sel]
    print(f"[setup] ids={len(ids)} mean_std={args.mean_std} dir={ms_dir}", flush=True)

    text_dir = os.path.join(BABEL, "texts")
    mot_dir = os.path.join(BABEL, "motion_data")

    def read_caption(cid):
        p = os.path.join(text_dir, cid + ".txt")
        if not os.path.isfile(p):
            return None
        for line in open(p):
            parts = line.strip().split("#")
            if len(parts) < 4:
                continue
            cap = parts[0].strip()
            if args.drop_transition and cap.lower() == "transition":
                return None
            return cap  # raw label; templates applied at encode time
        return None

    def motion_source(cid):
        p = os.path.join(mot_dir, cid + ".npy")
        return np.load(p).astype(np.float32) if os.path.isfile(p) else None

    # build_items uses its own read_caption (humanml texts); replicate inline here.
    from concurrent.futures import ThreadPoolExecutor

    def _fetch(cid):
        cap = read_caption(cid)
        if cap is None:
            return (cid, None, None)
        return (cid, cap, motion_source(cid))

    items = []
    skipped = 0
    crng = np.random.RandomState(args.seed)
    with ThreadPoolExecutor(max_workers=48) as ex:
        for i, (cid, cap, raw) in enumerate(ex.map(_fetch, ids)):
            if cap is None or raw is None:
                skipped += 1
            else:
                m, L = E.crop_and_norm(raw, mean, std, crng)
                if m is None:
                    skipped += 1
                else:
                    items.append((cap, m, L))
            if (i + 1) % 2000 == 0:
                print(f"  build {i+1}/{len(ids)} kept={len(items)} skip={skipped}", flush=True)
    print(f"[items] kept={len(items)} skipped={skipped}", flush=True)

    textenc, motionenc = E.load_evaluator(device)

    templates = [
        "{cap}",
        "a person {cap}",
        "a person is {cap}",
        "a person {cap}s",
        "a person is {cap}ing",
        "a man {cap}",
    ]
    if args.caption_template not in templates:
        templates.insert(0, args.caption_template)

    print("\n==== BABEL t2m GT (standard proto, caption-template sweep) ====")
    print("paper GT (BABEL Tab.2 subseq): R@3=0.634 MM-D=17.54 Div=24.91\n")
    print(f"{'template':<24} {'R@1':>6} {'R@2':>6} {'R@3':>6} {'MM-D':>7} {'Div':>7}")
    for tmpl in templates:
        items_t = [(tmpl.format(cap=c), m, L) for (c, m, L) in items]
        enc = E.encode_items(items_t, textenc, motionenc, device,
                             np.random.RandomState(args.seed))
        div = E.diversity_of(enc["em"], np.random.RandomState(args.seed + 100))
        print("%-24s %6.4f %6.4f %6.4f %7.3f %7.3f  (nb=%d)" % (
            tmpl, enc["R"][0], enc["R"][1], enc["R"][2], enc["matching"], div, enc["nb"]))


if __name__ == "__main__":
    main()
