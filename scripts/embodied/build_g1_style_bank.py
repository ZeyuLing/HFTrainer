#!/usr/bin/env python3
"""Build a qpos-level G1 style bank for PhysFlow style reward."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hftrainer.models.motion.physflow.g1_repr import encode_g1_motion, decode_g1_to_qpos
from hftrainer.models.motion.physflow.g1_style_reward import (
    G1StyleBank,
    categorize_style_text,
    qpos_style_feature,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--anno", default="data/annotation/train_g1_t2m_emb_minus_heldout.json")
    parser.add_argument("--g1-dir", default="data/g1")
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-items", type=int, default=20000)
    parser.add_argument("--stride", type=int, default=1)
    args = parser.parse_args()

    anno = json.loads(Path(args.anno).read_text())
    items = anno["items"] if isinstance(anno, dict) else anno
    if args.stride > 1:
        items = items[:: args.stride]
    if args.max_items and args.max_items > 0:
        items = items[: args.max_items]

    features, labels, paths = [], [], []
    g1_dir = Path(args.g1_dir)
    for idx, item in enumerate(items):
        rel = item["g1_path"]
        path = g1_dir / rel
        try:
            data = dict(np.load(path, allow_pickle=True))
            motion = encode_g1_motion(data)
            qpos = decode_g1_to_qpos(torch.from_numpy(motion)).numpy()
            features.append(qpos_style_feature(qpos, length=qpos.shape[0], fps=float(data.get("fps", [30.0])[0])))
            labels.append(categorize_style_text(item.get("caption_rel") or rel))
            paths.append(rel)
        except Exception as exc:  # noqa: BLE001
            print(f"[skip] {idx} {rel}: {type(exc).__name__}: {exc}", file=sys.stderr)

    if not features:
        raise RuntimeError("No valid G1 style features were extracted.")
    bank = G1StyleBank.from_features(np.stack(features, axis=0), labels=labels, paths=paths)
    bank.save(args.out)
    print(json.dumps({"out": args.out, "num_items": len(features), "feature_dim": int(bank.features.shape[1])}, indent=2))


if __name__ == "__main__":
    main()

