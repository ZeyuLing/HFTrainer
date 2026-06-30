#!/usr/bin/env python3
"""Probe whether T2M model outputs respond to prompt changes.

The script generates several deliberately different prompts with the same
length and seed, then compares output distances against a same-prompt
different-seed baseline. It is meant for quick debugging of text conditioning,
not for leaderboard evaluation.
"""

from __future__ import annotations

import argparse
import json
import os
import gc
import re
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

os.environ.setdefault("HFTRAINER_SKIP_AUTOREGISTER", "1")

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


PROMPTS = [
    "a person walks forward",
    "a person jumps up and down",
    "a person kicks with the right leg",
    "a person sits down and stands up",
    "a person waves both arms",
    "a person crawls on the ground",
]


def _torch_dtype(dtype: str) -> torch.dtype:
    mapping = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }
    return mapping[dtype]


def _slug(text: str) -> str:
    out = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return out[:64] or "prompt"


def _pairwise_stats(motions: list[np.ndarray], names: list[str]) -> list[dict]:
    rows = []
    for i in range(len(motions)):
        for j in range(i + 1, len(motions)):
            a = np.asarray(motions[i], np.float32)
            b = np.asarray(motions[j], np.float32)
            length = min(len(a), len(b))
            a = a[:length]
            b = b[:length]
            diff = a - b
            rows.append(
                {
                    "a": names[i],
                    "b": names[j],
                    "frames": int(length),
                    "motion135_rmse": float(np.sqrt(np.mean(diff[:, :135] ** 2))),
                    "trans_rmse_m": float(np.sqrt(np.mean(diff[:, :3] ** 2))),
                    "mean_abs_trans_m": float(np.mean(np.abs(diff[:, :3]))),
                }
            )
    return rows


def _motion_summary(motion: np.ndarray) -> dict:
    motion = np.asarray(motion, np.float32)
    trans = motion[:, :3]
    if len(trans) < 2:
        path = 0.0
    else:
        path = float(np.linalg.norm(np.diff(trans, axis=0), axis=-1).sum())
    disp = trans[-1] - trans[0]
    return {
        "frames": int(len(motion)),
        "root_path_m": path,
        "root_disp_xyz_m": [float(x) for x in disp],
        "root_range_xyz_m": [float(x) for x in (trans.max(axis=0) - trans.min(axis=0))],
    }


def _embedding_pairwise_cos(emb: torch.Tensor) -> list[dict]:
    # Reduce variable-length token embeddings with a non-zero-token mean.
    rows = []
    vecs = []
    for i in range(emb.shape[0]):
        x = emb[i].float()
        mask = torch.linalg.norm(x, dim=-1) > 1e-8
        if mask.any():
            x = x[mask].mean(dim=0)
        else:
            x = x.mean(dim=0)
        vecs.append(torch.nn.functional.normalize(x, dim=0))
    for i in range(len(vecs)):
        for j in range(i + 1, len(vecs)):
            rows.append({"i": i, "j": j, "cosine": float((vecs[i] * vecs[j]).sum().item())})
    return rows


def _collate_contexts(
    contexts: list[torch.Tensor],
    *,
    min_len: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    max_len = max([min_len] + [int(x.shape[0]) for x in contexts])
    hidden = int(contexts[0].shape[-1])
    out = torch.zeros(len(contexts), max_len, hidden, device=device, dtype=dtype)
    for i, ctx in enumerate(contexts):
        ctx = ctx.to(device=device, dtype=dtype)
        out[i, : ctx.shape[0]] = ctx
    return out


def _precompute_vimogen_embeddings(args, prompts: list[str]) -> torch.Tensor:
    from hftrainer.models.motion.vimogen.bundle import _DEFAULT_CONTEXT_NULL
    from hftrainer.models.motion.vimogen.network.vimogen.models.transformer.wan.modules.t5 import (
        T5EncoderModel,
    )

    device = torch.device(args.device)
    dtype = _torch_dtype(args.dtype)
    wan_dir = Path(args.vimogen_wan_dir)
    context_null = torch.load(str(_DEFAULT_CONTEXT_NULL), map_location="cpu", weights_only=True)
    if context_null.ndim == 2:
        context_null = context_null.unsqueeze(0)

    encoder = T5EncoderModel(
        text_len=args.text_len,
        dtype=dtype,
        device=device,
        checkpoint_path=str(wan_dir / "models_t5_umt5-xxl-enc-bf16.pth"),
        tokenizer_path=str(wan_dir / "google" / "umt5-xxl"),
        shard_fn=None,
    )
    with torch.no_grad():
        contexts = encoder(list(prompts), device)
        prompt_emb = _collate_contexts(
            contexts,
            min_len=int(context_null.shape[1]),
            device=device,
            dtype=dtype,
        ).detach().cpu()
    del encoder
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return prompt_emb


def _save_npz(path: Path, motion: np.ndarray, caption: str, model: str, extra: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        motion_135=np.asarray(motion, np.float32),
        caption=caption,
        model=model,
        **extra,
    )


def run_vimogen(args, prompts: list[str], out_dir: Path) -> dict:
    from hftrainer.models.motion.vimogen import ViMoGenBundle
    from hftrainer.motion.representation.dart276 import dart276_to_motion135

    prompt_emb_all = _precompute_vimogen_embeddings(args, prompts)
    bundle = ViMoGenBundle.from_pretrained(
        args.vimogen_model,
        device=args.device,
        dtype=args.dtype,
        text_dtype=args.dtype,
        cfg_scale=args.cfg_scale,
        num_inference_steps=args.steps,
        denoising_strength=args.denoising_strength,
        load_text_encoder=False,
    )

    emb_stats = _embedding_pairwise_cos(prompt_emb_all)

    names = []
    motions135 = []
    for idx, prompt in enumerate(prompts):
        motion276 = bundle.generate_motion276_from_embeddings(
            prompt_emb=prompt_emb_all[idx : idx + 1],
            lengths=[args.length],
            seed=args.seed,
            cfg_scale=args.cfg_scale,
            num_inference_steps=args.steps,
            denoising_strength=args.denoising_strength,
        )[0]
        motion135 = dart276_to_motion135(motion276, rotation_convention="row")
        name = _slug(prompt)
        names.append(name)
        motions135.append(np.asarray(motion135, np.float32))
        _save_npz(
            out_dir / "vimogen" / f"{name}.npz",
            motion135,
            prompt,
            "vimogen",
            {"seed": np.int32(args.seed), "denoising_strength": np.float32(args.denoising_strength)},
        )

    prompt0_seed2_276 = bundle.generate_motion276_from_embeddings(
        prompt_emb=prompt_emb_all[0:1],
        lengths=[args.length],
        seed=args.seed + 1,
        cfg_scale=args.cfg_scale,
        num_inference_steps=args.steps,
        denoising_strength=args.denoising_strength,
    )[0]
    prompt0_seed2 = np.asarray(
        dart276_to_motion135(prompt0_seed2_276, rotation_convention="row"),
        np.float32,
    )
    same_prompt_seed_delta = _pairwise_stats(
        [motions135[0], prompt0_seed2],
        [f"{names[0]}_seed{args.seed}", f"{names[0]}_seed{args.seed + 1}"],
    )[0]

    return {
        "model": "vimogen",
        "settings": {
            "length20": args.length,
            "seed": args.seed,
            "cfg_scale": args.cfg_scale,
            "steps": args.steps,
            "denoising_strength": args.denoising_strength,
        },
        "embedding_pairwise_cosine": emb_stats,
        "motion_pairwise": _pairwise_stats(motions135, names),
        "same_prompt_different_seed": same_prompt_seed_delta,
        "motion_summary": {name: _motion_summary(motion) for name, motion in zip(names, motions135)},
    }


def run_dart(args, prompts: list[str], out_dir: Path) -> dict:
    from hftrainer.models.motion.dart import DARTBundle
    from hftrainer.pipelines.dart import DARTPipeline

    bundle = DARTBundle.from_pretrained(
        args.dart_model,
        device=args.device,
        guidance_param=args.dart_guidance,
        initial_transform=args.dart_initial_transform,
        coord_conversion=args.dart_coord_conversion,
        load_dataset=True,
    )
    pipe = DARTPipeline(bundle)

    names = []
    motions135 = []
    for prompt in prompts:
        motion135 = pipe.infer_t2m_motion135(
            [prompt],
            [args.length],
            seed=args.seed,
            guidance_param=args.dart_guidance,
        )[0]
        name = _slug(prompt)
        names.append(name)
        motions135.append(np.asarray(motion135, np.float32))
        _save_npz(
            out_dir / "dart" / f"{name}.npz",
            motion135,
            prompt,
            "dart",
            {"seed": np.int32(args.seed), "guidance_param": np.float32(args.dart_guidance)},
        )

    prompt0_seed2 = pipe.infer_t2m_motion135(
        [prompts[0]],
        [args.length],
        seed=args.seed + 1,
        guidance_param=args.dart_guidance,
    )[0]
    same_prompt_seed_delta = _pairwise_stats(
        [motions135[0], np.asarray(prompt0_seed2, np.float32)],
        [f"{names[0]}_seed{args.seed}", f"{names[0]}_seed{args.seed + 1}"],
    )[0]

    return {
        "model": "dart",
        "settings": {
            "length20": args.length,
            "seed": args.seed,
            "guidance_param": args.dart_guidance,
            "initial_transform": args.dart_initial_transform,
            "coord_conversion": args.dart_coord_conversion,
        },
        "motion_pairwise": _pairwise_stats(motions135, names),
        "same_prompt_different_seed": same_prompt_seed_delta,
        "motion_summary": {name: _motion_summary(motion) for name, motion in zip(names, motions135)},
    }


def _selected_prompts(path: str | None) -> list[str]:
    if not path:
        return PROMPTS
    p = Path(path)
    if p.exists():
        return [x.strip() for x in p.read_text().splitlines() if x.strip()]
    return [x.strip() for x in path.split("|") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default="vimogen,dart", help="comma-separated: vimogen,dart")
    parser.add_argument("--prompts", default=None, help="pipe-separated prompts or a text file")
    parser.add_argument("--out-dir", default=str(REPO / "outputs/diagnostics/t2m_text_condition_sensitivity_20260628"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--length", type=int, default=120, help="native 20fps frame length")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--cfg-scale", type=float, default=5.0)
    parser.add_argument("--denoising-strength", type=float, default=0.7)
    parser.add_argument("--vimogen-model", default=str(REPO / "checkpoints/vimogen/hftrainer_1_3b"))
    parser.add_argument("--vimogen-wan-dir", default=str(REPO / "checkpoints/Wan2.1-T2V-1.3B"))
    parser.add_argument("--text-len", type=int, default=512)
    parser.add_argument("--dart-model", default=str(REPO / "checkpoints/dart/hftrainer_hml3d"))
    parser.add_argument("--dart-guidance", type=float, default=5.0)
    parser.add_argument("--dart-initial-transform", default="official_flowmdm")
    parser.add_argument("--dart-coord-conversion", default="mbench")
    args = parser.parse_args()

    prompts = _selected_prompts(args.prompts)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "prompts.json").write_text(json.dumps(prompts, indent=2) + "\n")

    requested = {x.strip().lower() for x in args.models.split(",") if x.strip()}
    summaries = {}
    if "vimogen" in requested:
        summaries["vimogen"] = run_vimogen(args, prompts, out_dir)
    if "dart" in requested:
        summaries["dart"] = run_dart(args, prompts, out_dir)

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summaries, indent=2) + "\n")

    for model, summary in summaries.items():
        pair = summary["motion_pairwise"]
        same = summary["same_prompt_different_seed"]
        mean_trans = float(np.mean([x["trans_rmse_m"] for x in pair])) if pair else 0.0
        mean_motion = float(np.mean([x["motion135_rmse"] for x in pair])) if pair else 0.0
        print(
            f"[{model}] mean_prompt_pair trans_rmse={mean_trans:.4f}m "
            f"motion135_rmse={mean_motion:.4f}; "
            f"same_prompt_seed trans_rmse={same['trans_rmse_m']:.4f}m "
            f"motion135_rmse={same['motion135_rmse']:.4f}",
            flush=True,
        )
    print(f"[done] {summary_path}", flush=True)


if __name__ == "__main__":
    main()
