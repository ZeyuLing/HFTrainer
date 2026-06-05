#!/usr/bin/env python3
"""Run FlowMDM text-to-motion inference on the HumanML3D test split.

The upstream FlowMDM generation runner is designed for visualizing composed
multi-prompt sequences. For Table 2 we need per-caption HumanML3D-263 feature
files keyed by the standard test ids, so this wrapper uses the released model
and sampler while bypassing the upstream dataset and visualization stack.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm


REPO = Path(__file__).resolve().parents[2]
FLOWMDM_ROOT = REPO / "ref_repo" / "FlowMDM"
SRC_H3D272 = REPO / "ref_repo" / "MotionStreamer" / "MotionStreamer" / "humanml3d_272"
RECON_ROOT = REPO / "work_dirs" / "h3d263_eval" / "h3d263_test_recon_fk"
MODEL_PATH = FLOWMDM_ROOT / "results" / "humanml" / "FlowMDM" / "model000500000.pt"
FALLBACK_HML_STATS = REPO / "ref_repo" / "MotionLab" / "datasets" / "all"


class _UnusedRotation2XYZ(torch.nn.Module):
    """Stand in for FlowMDM's visualization-only SMPL converter.

    The HumanML3D checkpoint predicts 263-dim HML vectors directly. The SMPL
    converter is only needed by upstream visualization helpers, but the model
    constructor initializes it unconditionally and fails when SMPL assets are
    absent. Keeping this as a Module preserves FlowMDM's ``_apply``/``train``
    hooks without loading body model files.
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.smpl_model = torch.nn.Identity()


def _resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else (REPO / path)


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
        f_tag = 0.0 if np.isnan(f_tag) else f_tag
        to_tag = 0.0 if np.isnan(to_tag) else to_tag
        if f_tag == 0.0 and to_tag == 0.0 and parts[0].strip():
            return parts[0].strip()
    return None


def _make_jobs(recon_root: Path, src_h3d272: Path, max_samples: int | None,
               num_shards: int, shard_index: int):
    ids = [s.strip() for s in (recon_root / "test.txt").read_text().splitlines() if s.strip()]
    jobs = []
    for sid in ids:
        m_file = recon_root / "new_joint_vecs" / f"{sid}.npy"
        if not m_file.exists():
            continue
        length = int(np.load(m_file, mmap_mode="r").shape[0])
        length = min((length // 4) * 4, 196)
        if length < 60:
            continue
        caption = _read_first_caption(src_h3d272 / "texts" / f"{sid}.txt")
        if not caption:
            continue
        jobs.append((sid, caption, length))
        if max_samples and len(jobs) >= max_samples:
            break
    if num_shards > 1:
        jobs = jobs[shard_index::num_shards]
    return jobs


def _build_flowmdm_args(args: argparse.Namespace) -> Namespace:
    args_path = Path(args.model_path).with_name("args.json")
    with args_path.open("r") as fp:
        model_args = json.load(fp)

    model_args.update({
        "model_path": str(args.model_path),
        "device": args.device,
        "seed": args.seed,
        "guidance_param": args.guidance_param,
        "bpe_denoising_step": args.bpe_denoising_step,
        "use_chunked_att": args.use_chunked_att,
    })
    model_args.setdefault("dataset", "humanml")
    model_args.setdefault("unconstrained", False)
    model_args.setdefault("lambda_fc", 0.0)
    model_args.setdefault("lambda_rcxyz", 0.0)
    model_args.setdefault("lambda_vel", 0.0)
    model_args.setdefault("lambda_vel_rcxyz", 0.0)
    model_args.setdefault("sigma_small", True)
    return Namespace(**model_args)


def _patch_clip_download_root(download_root: str | None):
    if not download_root:
        return
    import clip  # noqa: WPS433

    original_load = clip.load

    def load_with_root(name, device="cpu", jit=False, download_root=None):
        return original_load(
            name,
            device=device,
            jit=jit,
            download_root=download_root or download_root_path,
        )

    download_root_path = str(Path(download_root).expanduser())
    Path(download_root_path).mkdir(parents=True, exist_ok=True)
    clip.load = load_with_root


def _load_sampler(args: argparse.Namespace):
    os.chdir(str(FLOWMDM_ROOT))
    sys.path.insert(0, str(FLOWMDM_ROOT))

    _patch_clip_download_root(args.clip_download_root)

    from diffusion.diffusion_wrappers import DiffusionWrapper_FlowMDM as DiffusionWrapper  # noqa: WPS433
    import model.FlowMDM as flowmdm_module  # noqa: WPS433
    from utils import dist_util  # noqa: WPS433
    from utils.fixseed import fixseed  # noqa: WPS433
    from utils.model_util import load_model  # noqa: WPS433

    flowmdm_module.Rotation2xyz = _UnusedRotation2XYZ
    flow_args = _build_flowmdm_args(args)
    fixseed(flow_args.seed)
    dist_util.setup_dist(flow_args.device)
    device = dist_util.dev()
    print(f"[+] FlowMDM device={device}")
    model, diffusion = load_model(flow_args, device)
    return DiffusionWrapper(flow_args, diffusion, model), device


def _resolve_stats(args: argparse.Namespace) -> tuple[Path, Path]:
    if args.mean_path and args.std_path:
        return _resolve(args.mean_path), _resolve(args.std_path)

    flow_mean = FLOWMDM_ROOT / "dataset" / "HML_Mean_Gen.npy"
    flow_std = FLOWMDM_ROOT / "dataset" / "HML_Std_Gen.npy"
    if flow_mean.exists() and flow_std.exists():
        return flow_mean, flow_std

    return FALLBACK_HML_STATS / "Mean.npy", FALLBACK_HML_STATS / "Std.npy"


def _sample_one(sampler, caption: str, length: int, device: torch.device) -> torch.Tensor:
    mask = torch.ones((1, length), device=device, dtype=torch.bool)
    model_kwargs = {
        "y": {
            "mask": mask,
            "lengths": torch.tensor([length], dtype=torch.long),
            "text": [caption],
            "tokens": [""],
        }
    }
    with torch.no_grad():
        sample = sampler.p_sample_loop(
            clip_denoised=False,
            model_kwargs=model_kwargs,
            progress=False,
        )
    return sample[0, :, 0, :length].permute(1, 0).contiguous()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=str(MODEL_PATH))
    parser.add_argument("--recon-root", default=str(RECON_ROOT))
    parser.add_argument("--src-h3d272", default=str(SRC_H3D272))
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--guidance-param", type=float, default=2.5)
    parser.add_argument("--bpe-denoising-step", type=int, default=60)
    parser.add_argument("--use-chunked-att", action="store_true", default=True)
    parser.add_argument("--clip-download-root", default=None)
    parser.add_argument(
        "--mean-path",
        default=None,
        help="Official HumanML3D mean used to denormalize FlowMDM samples. "
             "Defaults to FlowMDM's HML_Mean_Gen.npy when present, otherwise "
             "ref_repo/MotionLab/datasets/all/Mean.npy.",
    )
    parser.add_argument(
        "--std-path",
        default=None,
        help="Official HumanML3D std used to denormalize FlowMDM samples. "
             "Defaults to FlowMDM's HML_Std_Gen.npy when present, otherwise "
             "ref_repo/MotionLab/datasets/all/Std.npy.",
    )
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--load-only", action="store_true")
    parser.add_argument("--load-model-only", action="store_true")
    args = parser.parse_args()

    args.model_path = _resolve(args.model_path)
    recon_root = _resolve(args.recon_root)
    src_h3d272 = _resolve(args.src_h3d272)
    out_dir = _resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    jobs = _make_jobs(src_h3d272=src_h3d272, recon_root=recon_root,
                      max_samples=args.max_samples, num_shards=args.num_shards,
                      shard_index=args.shard_index)
    print(f"[+] jobs={len(jobs)} shard={args.shard_index}/{args.num_shards} out={out_dir}")
    if args.load_only:
        return

    sampler, device = _load_sampler(args)
    if args.load_model_only:
        print("[+] model load check complete")
        return
    mean_path, std_path = _resolve_stats(args)
    mean = torch.from_numpy(np.load(mean_path)).float().to(device)
    std = torch.from_numpy(np.load(std_path)).float().to(device)
    print(f"[+] denorm_stats mean={mean_path} std={std_path}")

    written = skipped = failed = 0
    for idx, (sid, caption, length) in enumerate(tqdm(jobs, ncols=80)):
        if args.skip_existing and (out_dir / f"{sid}.npy").exists():
            skipped += 1
            continue
        try:
            torch.manual_seed(args.seed + args.shard_index * 100000 + idx)
            pred_norm = _sample_one(sampler, caption, length, device)
            pred = (pred_norm * std + mean).detach().cpu().numpy().astype(np.float32)
            np.save(out_dir / f"{sid}.npy", pred)
            written += 1
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"[fail] {sid}: {type(exc).__name__}: {exc}", flush=True)
    print(f"[done] written={written} skipped={skipped} failed={failed}")


if __name__ == "__main__":
    main()
