#!/usr/bin/env python3
"""Compare MotionGPT generated code distribution against VQ-VAE GT codes."""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "eval"))

from motiongpt_infer_hml3d263 import (  # noqa: E402
    MOTIONGPT_ROOT,
    DummyHumanML3DDataModule,
    force_untied_t5_lm_head,
    load_cfg,
    load_jobs,
)

SRC_H3D272 = REPO / "ref_repo" / "MotionStreamer" / "MotionStreamer" / "humanml3d_272"


def _read_first_caption(text_file: Path) -> str | None:
    if not text_file.exists():
        return None
    for line in text_file.read_text().splitlines():
        parts = line.strip().split("#")
        if len(parts) < 4:
            continue
        try:
            f_tag = float(parts[2])
            to_tag = float(parts[3])
        except ValueError:
            continue
        if f_tag == 0.0 and to_tag == 0.0 and parts[0].strip():
            return parts[0].strip()
    return None


def _load_recon_jobs(recon_root: Path, src_h3d272: Path, max_samples: int) -> list[tuple[str, str, int]]:
    jobs: list[tuple[str, str, int]] = []
    for sid in (x.strip() for x in (recon_root / "test.txt").read_text().splitlines()):
        if not sid:
            continue
        src = recon_root / "new_joint_vecs" / f"{sid}.npy"
        if not src.exists():
            continue
        length = int(np.load(src, mmap_mode="r").shape[0])
        if length < 40 or length >= 200:
            continue
        caption = _read_first_caption(src_h3d272 / "texts" / f"{sid}.txt")
        if not caption:
            continue
        jobs.append((sid, caption, (length // 4) * 4))
        if max_samples and len(jobs) >= max_samples:
            break
    return jobs


def _summary(tokens: list[int], lens: list[int]) -> dict:
    counts = Counter(tokens)
    total = max(1, len(tokens))
    probs = [c / total for c in counts.values()]
    entropy = -sum(p * math.log(p + 1e-12) for p in probs)
    return {
        "n_sequences": len(lens),
        "n_tokens": len(tokens),
        "unique_codes": len(counts),
        "entropy": entropy,
        "top1_frac": max(counts.values()) / total if counts else 0.0,
        "top5_frac": sum(c for _, c in counts.most_common(5)) / total if counts else 0.0,
        "top20": counts.most_common(20),
        "length_min": min(lens) if lens else None,
        "length_median": float(np.median(lens)) if lens else None,
        "length_max": max(lens) if lens else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--anno-file", default=None)
    ap.add_argument("--caption-file", default=None)
    ap.add_argument("--recon-root", default="work_dirs/h3d263_eval/h3d263_test_recon_fk")
    ap.add_argument("--src-h3d272", default=str(SRC_H3D272))
    ap.add_argument("--out-json", default="outputs/evaluation/humanml3d/motiongpt_token_stats_0605.json")
    ap.add_argument("--out-dir", default="outputs/evaluation/humanml3d/motiongpt_token_stats_runtime_0605")
    ap.add_argument("--cfg", default=str(MOTIONGPT_ROOT / "configs" / "config_h3d_stage3.yaml"))
    ap.add_argument("--checkpoint", default=str(MOTIONGPT_ROOT / "checkpoints" / "MotionGPT-base" / "motiongpt_s3_h3d.tar"))
    ap.add_argument("--t5-path", default="google/flan-t5-base")
    ap.add_argument("--mean", default=str(MOTIONGPT_ROOT / "assets" / "meta" / "mean.npy"))
    ap.add_argument("--std", default=str(MOTIONGPT_ROOT / "assets" / "meta" / "std.npy"))
    ap.add_argument("--max-samples", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    force_untied_t5_lm_head()
    os.chdir(MOTIONGPT_ROOT)
    from mGPT.models.base import BaseModel  # noqa: WPS433
    BaseModel.configure_metrics = lambda self: setattr(self, "metrics", torch.nn.Module())
    from mGPT.models.build_model import build_model  # noqa: WPS433

    cfg = load_cfg(args)
    dm = DummyHumanML3DDataModule(Path(args.mean), Path(args.std))
    model = build_model(cfg, dm).eval()
    state = torch.load(cfg.TEST.CHECKPOINTS, map_location="cpu")["state_dict"]
    model.load_state_dict(state, strict=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    recon_root = (REPO / args.recon_root).resolve()
    if args.anno_file and args.caption_file:
        jobs = load_jobs(
            Path(args.anno_file),
            Path(args.caption_file),
            args.max_samples,
            1,
            0,
            20.0,
            20.0,
            40,
            196,
        )
    else:
        jobs = _load_recon_jobs(
            recon_root,
            Path(args.src_h3d272).resolve(),
            args.max_samples,
        )

    gt_tokens: list[int] = []
    gt_lens: list[int] = []
    gen_tokens: list[int] = []
    gen_lens: list[int] = []
    examples = []

    for start in tqdm(range(0, len(jobs), args.batch_size), ncols=80):
        chunk = jobs[start:start + args.batch_size]
        texts = [x[1] for x in chunk]
        lengths = [x[2] for x in chunk]
        with torch.no_grad():
            outputs = model.lm.generate_conditional(
                texts=texts,
                lengths=lengths,
                task="t2m",
                with_len=False,
                stage="test",
                tasks=None,
            )
        for sid, caption, _length in chunk:
            src = recon_root / "new_joint_vecs" / f"{sid}.npy"
            if not src.exists():
                continue
            raw = np.load(src).astype(np.float32)
            if len(raw) < 40:
                continue
            feat = torch.from_numpy(raw).to(device)[None]
            norm = (feat - dm.mean.to(device)) / dm.std.to(device)
            with torch.no_grad():
                toks, _ = model.vae.encode(norm)
            ids = [int(x) for x in toks[0].detach().cpu().tolist()]
            gt_tokens.extend(ids)
            gt_lens.append(len(ids))
        for (sid, caption, _length), out in zip(chunk, outputs):
            ids = [int(x) for x in torch.clamp(out, 0, model.hparams.codebook_size - 1).detach().cpu().tolist()]
            gen_tokens.extend(ids)
            gen_lens.append(len(ids))
            if len(examples) < 12:
                examples.append({"id": sid, "caption": caption, "gen_len": len(ids), "gen_head": ids[:24]})

    result = {
        "gt": _summary(gt_tokens, gt_lens),
        "generated": _summary(gen_tokens, gen_lens),
        "examples": examples,
    }
    out = (REPO / args.out_json).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
