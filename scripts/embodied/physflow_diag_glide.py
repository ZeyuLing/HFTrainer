#!/usr/bin/env python3
"""Diagnose the locomotion-glide failure: compare BASE KIMODO-G1 weights vs the
co-evolution fine-tuned checkpoint on the same prompts. For each motion report
root displacement (m) and mean joint std (rad). A "glide" = big root_disp with
near-zero joint_std (frozen pose dragged along the floor).
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.embodied.physflow_periodic_eval import _build_bundle, _load_checkpoint, _generate_qpos


def stats(qpos, length):
    a = np.asarray(qpos)[:length]
    rd = float(np.linalg.norm(a[-1, :3] - a[0, :3]))
    js = float(np.std(a[:, 7:], axis=0).mean())
    return rd, js


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/physflow/physflow_online_adv_v2.py")
    ap.add_argument("--ckpt", default="work_dirs/physflow_coevolve/anchor/gen/r0/checkpoint-iter_120")
    ap.add_argument("--eval-corpus", default="configs/experiments/physflow_kimodo_g1/physflow_text_eval.jsonl")
    ap.add_argument("--num-prompts", type=int, default=12)
    ap.add_argument("--diffusion-steps", type=int, default=20)
    args = ap.parse_args()

    from mmengine.config import Config
    from hftrainer.models.motion.physflow.dataset import PhysFlowPromptDataset
    cfg = Config.fromfile(args.config)
    fd = cfg.train_dataloader["dataset"]["feature_dir"]
    ds = PhysFlowPromptDataset(corpus_file=args.eval_corpus, feature_dir=fd, split="test",
                               fps=30.0, min_frames=60, max_frames=150, max_samples=args.num_prompts)
    feats = [ds[i]["text_feat"] for i in range(len(ds))]
    lens = [int(ds[i]["num_frames"]) for i in range(len(ds))]
    prompts = [ds[i].get("prompt", "")[:48] for i in range(len(ds))]

    bundle = _build_bundle(cfg)

    print("\n==== BASE (no finetune load) ====", flush=True)
    bundle.denoiser.eval()
    base_q = _generate_qpos(bundle, feats, lens, args.diffusion_steps, 6)
    base = [stats(base_q[i], lens[i]) for i in range(len(ds))]

    print("\n==== FINE-TUNED (%s) ====" % args.ckpt, flush=True)
    _load_checkpoint(bundle, Path(args.ckpt))
    bundle.denoiser.eval()
    ft_q = _generate_qpos(bundle, feats, lens, args.diffusion_steps, 6)
    ft = [stats(ft_q[i], lens[i]) for i in range(len(ds))]

    print("\n%-50s | %-18s | %-18s" % ("prompt", "BASE rd/js", "FT rd/js"), flush=True)
    for i, p in enumerate(prompts):
        print("%-50s | %5.2f / %.4f      | %5.2f / %.4f" %
              (p, base[i][0], base[i][1], ft[i][0], ft[i][1]), flush=True)


if __name__ == "__main__":
    main()
