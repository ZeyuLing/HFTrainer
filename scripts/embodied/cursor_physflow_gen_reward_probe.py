#!/usr/bin/env python3
"""Probe: drive KIMODO-G1 generation in-process WITHOUT the 8B text encoder.

Validates the riskiest seam of the online-adversarial loop:
  load_model (TEXT_ENCODER=dummy, no 8B)  ->  feed a PRE-EXTRACTED text_feat
  directly to model._generate(...)  ->  motion_rep.inverse  ->  MujocoQposConverter
  -> G1 qpos CSV.

We deliberately pass ``text_feat`` to ``_generate`` so KIMODO's text encoder is
never invoked (a DummyTextEncoder placeholder is loaded, but skipped). The
cached embedding comes from data/kimodo_text_feature (see
cursor_extract_kimodo_text_feature.py). If the resulting CSV has the expected
[T, 36] G1 qpos layout with sane values, the in-process / no-8B generation path
is proven and we can wrap it into the PhysFlowBundle.

Run on the node (py3.10) with:
  HF_HOME=checkpoints/kimodo HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  CUDA_VISIBLE_DEVICES=0 python3 scripts/embodied/cursor_physflow_gen_reward_probe.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
KIMODO_ROOT = PROJECT_ROOT / "ref_repo" / "KIMODO" / "kimodo"
for _p in (PROJECT_ROOT, KIMODO_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

DEFAULT_CACHE = PROJECT_ROOT / "data" / "kimodo_text_feature" / "kimodo_g1_llm2vec_v1"


def load_cached_text_feat(prompt: str, cache_dir: Path):
    """Return [seq_len, 4096] float tensor for `prompt` from the disk cache."""
    manifest = cache_dir / "manifest.jsonl"
    key = None
    with open(manifest, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if rec["prompt"] == prompt:
                key = rec["key"]
                break
    if key is None:
        raise KeyError(f"prompt not found in manifest: {prompt!r}")
    arr = np.load(cache_dir / f"{key}.npy")  # [seq_len, 4096]
    return torch.from_numpy(arr).float()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default=str(DEFAULT_CACHE))
    ap.add_argument("--model", default="Kimodo-G1-RP-v1")
    ap.add_argument("--checkpoint-dir", default=str(PROJECT_ROOT / "checkpoints" / "kimodo" / "hub"))
    ap.add_argument("--duration", type=float, default=4.0)
    ap.add_argument("--num-samples", type=int, default=2, help="best-of-N batch of the SAME prompt")
    ap.add_argument("--diffusion-steps", type=int, default=30)
    ap.add_argument("--cfg-weight", type=float, nargs="*", default=[2.0, 2.0])
    ap.add_argument("--cfg-type", default="separated")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default=str(PROJECT_ROOT / "output" / "physflow_probe"))
    args = ap.parse_args()

    os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / "checkpoints" / "kimodo"))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.pop("HUGGINGFACE_CACHE_DIR", None)
    # Skip the 8B encoder entirely; we feed text_feat ourselves.
    os.environ["TEXT_ENCODER"] = "dummy"
    os.environ["TEXT_ENCODER_MODE"] = "local"
    os.environ["LOCAL_CACHE"] = "true"
    if args.checkpoint_dir:
        os.environ.setdefault("CHECKPOINT_DIR", args.checkpoint_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cache_dir = Path(args.cache_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Pick a prompt that exists in the cache.
    with open(cache_dir / "manifest.jsonl", "r", encoding="utf-8") as f:
        first = json.loads(f.readline())
    prompt = first["prompt"]
    print(f"[probe] prompt: {prompt!r}", flush=True)

    from kimodo.model.load_model import load_model
    from kimodo.tools import seed_everything

    print(f"[probe] loading KIMODO model '{args.model}' (TEXT_ENCODER=dummy, no 8B) ...", flush=True)
    model = load_model(args.model, device=device, eval_mode=True)
    fps = float(model.fps)
    n_frames = int(round(args.duration * fps))
    print(f"[probe] model ready. fps={fps} num_base_steps={model.diffusion.num_base_steps} "
          f"skeleton={type(model.skeleton).__name__} frames={n_frames}", flush=True)

    seed_everything(args.seed)

    B = args.num_samples
    base_feat = load_cached_text_feat(prompt, cache_dir).to(device)  # [seq, 4096]
    seq = base_feat.shape[0]
    text_feat = base_feat[None].repeat(B, 1, 1)            # [B, seq, 4096]
    text_pad_mask = torch.ones(B, seq, dtype=torch.bool, device=device)
    print(f"[probe] text_feat {tuple(text_feat.shape)} {text_feat.dtype}", flush=True)

    from kimodo.motion_rep.feature_utils import length_to_mask

    lengths = torch.tensor([n_frames] * B, device=device)
    pad_mask = length_to_mask(lengths)                     # [B, n_frames]
    first_heading = torch.zeros(B, device=device)

    print("[probe] running _generate with precomputed text_feat ...", flush=True)
    motion = model._generate(
        texts=[prompt] * B,           # ignored because text_feat is provided
        max_frames=int(lengths.max()),
        num_denoising_steps=args.diffusion_steps,
        pad_mask=pad_mask,
        first_heading_angle=first_heading,
        motion_mask=None,
        observed_motion=None,
        cfg_weight=args.cfg_weight,
        text_feat=text_feat,
        text_pad_mask=text_pad_mask,
        cfg_type=args.cfg_type,
    )
    print(f"[probe] latent motion {tuple(motion.shape)} {motion.dtype} "
          f"mean={motion.float().mean().item():.4f} std={motion.float().std().item():.4f}", flush=True)

    output = model.motion_rep.inverse(motion, is_normalized=True, return_numpy=False)
    for k, v in output.items():
        if torch.is_tensor(v):
            print(f"[probe]   output[{k}] {tuple(v.shape)} {v.dtype}", flush=True)

    # Decode to G1 qpos CSV (mirror kimodo.scripts.generate G1 branch).
    from kimodo.exports.mujoco import MujocoQposConverter

    converter = MujocoQposConverter(model.skeleton)
    qpos = converter.dict_to_qpos(output, device)
    print(f"[probe] qpos {tuple(qpos.shape) if torch.is_tensor(qpos) else type(qpos)}", flush=True)
    for i in range(B):
        csv_path = out_dir / f"probe_s{i:02d}.csv"
        sample_qpos = qpos[i] if qpos.ndim == 3 else qpos
        converter.save_csv(sample_qpos, str(csv_path))
        a = np.loadtxt(str(csv_path), delimiter=",")
        print(f"[probe]   sample {i}: csv {csv_path.name} shape={a.shape} "
              f"root_xyz0={a[0, :3].round(3).tolist()} quat0={a[0, 3:7].round(3).tolist()}", flush=True)

    print("[probe] DONE — in-process no-8B generation produced G1 qpos CSVs.", flush=True)


if __name__ == "__main__":
    main()
