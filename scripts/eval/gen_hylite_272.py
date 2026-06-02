#!/usr/bin/env python3
"""Generate HY-Motion-T2M-1.0-Lite predictions on the HumanML3D test split and
export them as MotionStreamer 272-dim features for the (validated) TMR evaluator.

    HY-Lite (HunyuanMotionMMDiT, 201-dim, 30 fps)
      -> latent_denorm[..., :135]   (trans3 + 22x6 rot6d, 30 fps)
      -> motion135_to_272           (canonical SMPL-X-272 skeleton FK -> 272)
      -> save <out>/<id>.npy        (272, 30 fps)

The <id>.npy plug straight into the native validated evaluator:

    python3 ref_repo/MotionStreamer/eval_with_motionstreamer_evaluator.py \
        --pred_dir <out> --data_root /dev/shm/ms272_data \
        --evaluator_ckpt /dev/shm/eval272_epoch99.ckpt \
        --out_json <out>/../eval_hylite.json --n_repeats 20

ids + GT lengths come from the 272 GT set (30 fps), so gen and GT share ids,
texts and lengths (the evaluator truncates each to (len//4)*4).
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

T2M_CONFIG = "configs/hymotion_t2m/hymotion_t2m_201dim_046b.py"


def _read_first_caption(txt: Path):
    if not txt.exists():
        return None
    for line in txt.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split("#")
        if len(parts) < 4:
            continue
        cap, ftag, ttag = parts[0], parts[2], parts[3]
        try:
            fv, tv = float(ftag), float(ttag)
        except ValueError:
            continue
        if (fv == 0.0 or fv != fv) and (tv == 0.0 or tv != tv):
            return cap
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default="/dev/shm/ms272_data",
                   help="272 GT root (motion_data/, texts/, split/test.txt).")
    p.add_argument("--out", required=True, help="Output dir for <id>.npy (272).")
    p.add_argument("--m135-dir", default=None,
                   help="Also dump raw motion_135 <id>.npy here (rotation-space "
                        "agnostic, so 272 can be re-derived without re-inference). "
                        "Defaults to '<out>/../m135'.")
    p.add_argument("--num-steps", type=int, default=50)
    p.add_argument("--cfg-scale", type=float, default=5.0,
                   help="HY-Lite native text_guidance_scale (config test_cfg=5.0).")
    p.add_argument("--rotation-space", choices=["local", "global"], default="local")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--min-len", type=int, default=60)
    p.add_argument("--max-len", type=int, default=300)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    import torch
    from copy import deepcopy
    from mmengine.config import Config
    from hftrainer.registry import MODEL_BUNDLES
    from hftrainer.utils.checkpoint_utils import load_checkpoint
    from hftrainer.pipelines.motion.hymotion_t2m_pipeline import HyMotionT2MPipeline
    from hftrainer.models.motion.hymotion_m2m.network.text_encoder import HYTextModel

    sys.path.insert(0, str(_REPO / "scripts" / "eval"))
    from motionstreamer_272_encoder import motion135_to_272

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda:0"

    data_root = Path(args.data_root)
    motion_dir = data_root / "motion_data"
    texts_dir = data_root / "texts"
    test_ids = [s.strip() for s in (data_root / "split" / "test.txt").read_text().splitlines() if s.strip()]

    jobs = []  # (id, caption, gen_len30)
    for sid in test_ids:
        m_file = motion_dir / f"{sid}.npy"
        if not m_file.exists():
            continue
        cap = _read_first_caption(texts_dir / f"{sid}.txt")
        if not cap:
            continue
        L = int(np.load(str(m_file), mmap_mode="r").shape[0])
        if L < args.min_len or L >= args.max_len:
            continue
        jobs.append((sid, cap, L))
    if args.max_samples:
        jobs = jobs[:args.max_samples]
    if args.num_shards > 1:
        jobs = jobs[args.shard_index::args.num_shards]
    print(f"[+] {len(jobs)} gen jobs (shard {args.shard_index}/{args.num_shards}, "
          f"cfg={args.cfg_scale}, rot={args.rotation_space})", flush=True)

    # ---- build + load HY-Lite ----
    cfg = Config.fromfile(T2M_CONFIG)
    bundle = MODEL_BUNDLES.build(cfg.model.to_dict())
    bundle._text_encoder_cfg = {
        "llm_type": "qwen3", "max_length_llm": 128,
        "sentence_emb_type": "clipl", "max_length_sentence_emb": 77,
        # MUST be True for BATCHED encoding: captions in a batch have different
        # token lengths, so without max_length padding the tokenizer cannot
        # stack them into one tensor (ValueError: expected sequence of length
        # N got M). Padding to max_length_llm is numerically irrelevant -- the
        # motion model attends via ctxt_length (right-padding + attn mask).
        "enable_llm_padding": True,
    }
    ckpt_path = cfg.load_from["path"] if isinstance(cfg.load_from, dict) else cfg.load_from
    assert os.path.exists(ckpt_path), f"checkpoint not found: {ckpt_path}"
    print(f"[+] loading {ckpt_path}", flush=True)
    sd = load_checkpoint(ckpt_path, map_location="cpu")
    bundle.load_state_dict_selective(sd)
    del sd
    bundle.eval().to(device)

    tcfg = deepcopy(bundle._text_encoder_cfg)
    tcfg["torch_dtype"] = torch.float16
    print("[+] building text encoder (fp16) on GPU ...", flush=True)
    bundle._text_encoder = HYTextModel(**tcfg).eval().to(device)

    pipeline = HyMotionT2MPipeline(
        bundle=bundle, num_steps=args.num_steps, text_guidance_scale=args.cfg_scale)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    m135_dir = Path(args.m135_dir) if args.m135_dir else out.parent / "m135"
    m135_dir.mkdir(parents=True, exist_ok=True)

    # sort by length to minimise padding waste; batch
    jobs.sort(key=lambda x: x[2])
    bs = args.batch_size
    done = 0
    for i in range(0, len(jobs), bs):
        chunk = jobs[i:i + bs]
        caps = [c for _, c, _ in chunk]
        lens = [L for _, _, L in chunk]
        batch = {"caption": caps, "tgt_length": lens}
        result = pipeline(batch)
        denorm = result["latent_denorm"].float().cpu().numpy()  # (B, Lmax, 201)
        for k, (sid, _cap, L) in enumerate(chunk):
            m135 = denorm[k, :L, :135].astype(np.float32)
            np.save(str(m135_dir / f"{sid}.npy"), m135)
            try:
                m272 = motion135_to_272(m135, rotation_space=args.rotation_space)
            except Exception as e:
                print(f"  [!] {sid}: {e}", flush=True)
                continue
            np.save(str(out / f"{sid}.npy"), m272.astype(np.float32))
        done += len(chunk)
        if (i // bs) % 10 == 0:
            print(f"  [{done}/{len(jobs)}]", flush=True)
    print(f"[+] done: {done} preds -> {out}", flush=True)


if __name__ == "__main__":
    main()
