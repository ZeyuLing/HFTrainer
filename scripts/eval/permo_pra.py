#!/usr/bin/env python3
"""PRA (Persona Recognition Accuracy) proxy for Table 10.

PersonaBooth's PRA uses *their* trained persona classifier (unavailable locally),
so we report a clearly-labelled PROXY: a style classifier trained on REAL PerMo
motions in the SAME Guo HumanML3D-263 motion-embedding space used for FID/R/Div,
then applied to \\ours{} generated motions. PRA = fraction of generated clips whose
predicted style matches the requested target style label. This mirrors the
"recognition accuracy" protocol common in motion style-transfer papers, but the
absolute value is NOT directly comparable to PersonaBooth's cited PRA (different
classifier); the table footnote states this.

Two stages:
  train : encode real PerMo (non-Neutral) motions -> embeddings -> train classifier
  eval  : encode \\ours{} generated npz -> predict style -> accuracy vs target label
"""
import argparse
import glob
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch

_REPO = Path("/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer")
sys.path.insert(0, str(_REPO / "ref_repo/Momask/momask-codes"))
sys.path.insert(0, str(_REPO))

import utils.motion_process  # noqa: F401,E402
from hftrainer.motion.representation.convert import (  # noqa: E402
    motion135_to_motion272, motion272_to_hml263,
)
from hftrainer.evaluation.evaluators.humanml3d_263 import HumanML263Evaluator  # noqa: E402

PERMO = _REPO / "data/hymotion_data/PerMo/PerMo/20260513/motions/train"
PAT = re.compile(r"^([A-Za-z]+)_(.+?)_(A\d+)_(\d+)\.npz$")
CACHE = _REPO / "output/evaluation/permo_style_ours_big/pra_real_embeddings.npz"


def to263(m135):
    m272 = motion135_to_motion272(np.asarray(m135, np.float32))
    out = motion272_to_hml263(m272, joints_from="smpl_fk")
    m = out[0] if isinstance(out, tuple) else out
    return np.asarray(m, np.float32)


@torch.no_grad()
def encode(ev, m263):
    L = min(len(m263), 196)
    if L < 40:
        return None
    t_eff = (L // ev.unit_length) * ev.unit_length
    m = ev._pad_norm(m263, t_eff)
    mt = torch.from_numpy(m).unsqueeze(0).float().to(ev.device)
    mov = ev._movement_enc(mt[..., :-4])
    e = ev._motion_enc(mov, torch.tensor([t_eff // ev.unit_length]))
    return e.cpu().numpy()[0]


def build_real(ev, per_style):
    if CACHE.exists():
        d = np.load(CACHE, allow_pickle=True)
        print(f"[cache] loaded {len(d['X'])} real embeddings")
        return d["X"], d["y"], list(d["labels"])
    from collections import defaultdict
    files = defaultdict(list)
    for p in sorted(PERMO.glob("*.npz")):
        m = PAT.match(p.name)
        if m and m.group(1) != "Neutral":
            files[m.group(1)].append(p)
    labels = sorted(files)
    X, y = [], []
    for li, style in enumerate(labels):
        n = 0
        for p in files[style]:
            try:
                e = encode(ev, to263(np.load(p, allow_pickle=True)["motion_135"]))
            except Exception:
                e = None
            if e is None:
                continue
            X.append(e); y.append(li); n += 1
            if n >= per_style:
                break
        print(f"  {style}: {n}")
    X = np.stack(X); y = np.array(y)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(CACHE, X=X, y=y, labels=np.array(labels))
    print(f"[cache] saved {len(X)} embeddings -> {CACHE}")
    return X, y, labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-style", type=int, default=120,
                    help="real motions per style for classifier training")
    ap.add_argument("--npz-dir", default=str(
        _REPO / "output/evaluation/permo_style_ours_big/smpl_caption_editfix_latest/E16_style_edit/npz"))
    ap.add_argument("--out", default=str(_REPO / "output/evaluation/permo_style_ours_big/pra.json"))
    args = ap.parse_args()

    ev = HumanML263Evaluator(device="cuda"); ev._ensure_loaded()
    X, y, labels = build_real(ev, args.per_style)
    lab2i = {l: i for i, l in enumerate(labels)}

    # standardise + logistic regression (held-out accuracy as a sanity check)
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    sc = StandardScaler().fit(X)
    Xs = sc.transform(X)
    Xtr, Xte, ytr, yte = train_test_split(Xs, y, test_size=0.2, random_state=0, stratify=y)
    clf = LogisticRegression(max_iter=2000, C=1.0, multi_class="multinomial")
    clf.fit(Xtr, ytr)
    real_acc = float((clf.predict(Xte) == yte).mean())
    print(f"[classifier] held-out real-PerMo style acc = {real_acc:.4f} over {len(labels)} styles")

    # evaluate PRA on ours-generated
    files = sorted(glob.glob(f"{args.npz_dir}/*.npz"))
    correct = total = 0
    per_style_hit = {}
    for f in files:
        d = np.load(f, allow_pickle=True)
        cap = str(d.get("caption", ""))
        m = re.search(r"in a (\w+) style", cap.lower())
        if not m:
            continue
        style = m.group(1)
        tgt = next((l for l in labels if l.lower() == style), None)
        if tgt is None:
            continue
        try:
            e = encode(ev, to263(d["motion_135"]))
        except Exception:
            e = None
        if e is None:
            continue
        pred = labels[int(clf.predict(sc.transform(e[None]))[0])]
        total += 1
        ok = int(pred == tgt)
        correct += ok
        per_style_hit.setdefault(tgt, [0, 0])
        per_style_hit[tgt][0] += ok; per_style_hit[tgt][1] += 1
    pra = correct / max(total, 1)
    summary = {
        "PRA": round(pra * 100, 2), "n_eval": total,
        "classifier_heldout_real_acc": round(real_acc * 100, 2),
        "n_styles": len(labels),
        "per_style": {k: round(100 * v[0] / v[1], 1) for k, v in sorted(per_style_hit.items())},
    }
    print("\n===== PRA (proxy) =====")
    print(json.dumps(summary, indent=2))
    json.dump(summary, open(args.out, "w"), indent=2)
    print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()
