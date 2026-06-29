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
import traceback
import types
from argparse import Namespace
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from tqdm import tqdm


REPO = Path(__file__).resolve().parents[2]
FLOWMDM_ROOT = REPO / "ref_repo" / "FlowMDM"
SRC_H3D272 = REPO / "ref_repo" / "MotionStreamer" / "MotionStreamer" / "humanml3d_272"
RECON_ROOT = REPO / "work_dirs" / "h3d263_eval" / "h3d263_test_recon_fk"
MODEL_PATH = FLOWMDM_ROOT / "results" / "humanml" / "FlowMDM" / "model000500000.pt"
OFFICIAL_HML_STATS = REPO / "ref_repo" / "MotionLab" / "datasets" / "all"


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


def _install_unused_rotation2xyz_module():
    """Avoid importing FlowMDM's visualization-only SMPL converter."""
    module = types.ModuleType("model.rotation2xyz")
    module.Rotation2xyz = _UnusedRotation2XYZ
    module.JOINTSTYPES = ["a2m", "a2mpl", "smpl", "vibe", "vertices", "smplx"]
    sys.modules["model.rotation2xyz"] = module


def _configure_stable_cuda_kernels() -> None:
    """Prefer deterministic math kernels over fragile SDPA/cuDNN fast paths."""
    torch.backends.cudnn.enabled = False
    for name in (
        "enable_flash_sdp",
        "enable_mem_efficient_sdp",
        "enable_cudnn_sdp",
    ):
        fn = getattr(torch.backends.cuda, name, None)
        if fn is not None:
            fn(False)
    enable_math_sdp = getattr(torch.backends.cuda, "enable_math_sdp", None)
    if enable_math_sdp is not None:
        enable_math_sdp(True)


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


def _iter_anno_entries(raw) -> Iterable[tuple[str, dict]]:
    data = raw.get("data_list", raw) if isinstance(raw, dict) else raw
    if isinstance(data, dict):
        yield from data.items()
        return
    for idx, entry in enumerate(data):
        yield str(entry.get("motion_id") or entry.get("id") or idx), entry


def _load_caption_map(caption_file: Path | None) -> dict | None:
    if caption_file is None:
        return None
    raw = json.loads(caption_file.read_text())
    return raw.get("data_list", raw) if isinstance(raw, dict) else raw


def _caption_from_map(caption_map: dict | None, name: str) -> str | None:
    if caption_map is None:
        return None
    caption = caption_map.get(str(name))
    if isinstance(caption, dict):
        caption = caption.get("caption") or caption.get("text")
    return caption.strip() if isinstance(caption, str) and caption.strip() else None


def _load_motionhub_caption(entry: dict, data_dir: Path) -> str | None:
    c_rel = entry.get("hierarchical_caption_path")
    if not c_rel:
        return None
    c_path = data_dir / c_rel
    if not c_path.exists():
        return None
    try:
        data = json.loads(c_path.read_text())
    except Exception:
        return None
    pool: list[str] = []
    if isinstance(data, dict) and all(k in data for k in ("macro", "meso", "micro")):
        for key in ("macro", "meso", "micro"):
            values = data.get(key)
            if isinstance(values, list):
                pool.extend(v.strip() for v in values if isinstance(v, str) and v.strip())
    if isinstance(data, dict) and isinstance(data.get("result"), list):
        for item in data["result"]:
            if not isinstance(item, dict):
                continue
            for key in ("short_caption_rewritten", "short caption_rewritten"):
                values = item.get(key)
                if isinstance(values, list):
                    pool.extend(v.strip() for v in values if isinstance(v, str) and v.strip())
                    break
            else:
                for key in ("short_caption", "short caption"):
                    value = item.get(key)
                    if isinstance(value, str) and value.strip():
                        pool.append(value.strip())
                        break
    return pool[0] if pool else None


def _safe_name(name: str) -> str:
    return str(name).replace("/", "__")


def _load_only_ids(value: str | None) -> set[str] | None:
    if not value:
        return None
    path = _resolve(value)
    if path.exists():
        ids = [line.strip() for line in path.read_text().splitlines()]
    else:
        ids = [part.strip() for part in value.split(",")]
    return {sid for sid in ids if sid}


def _make_jobs_from_anno(
    anno_file: Path,
    caption_file: Path | None,
    data_dir: Path,
    max_samples: int | None,
    num_shards: int,
    shard_index: int,
    gt_fps: float,
    model_fps: float,
    min_length: int,
    max_length: int,
    gt_hml263_dir: Path | None = None,
):
    raw = json.loads(anno_file.read_text())
    caption_map = _load_caption_map(caption_file)
    jobs = []
    eligible = 0
    for name, entry in _iter_anno_entries(raw):
        caption = _caption_from_map(caption_map, str(name))
        if caption is None:
            caption = _load_motionhub_caption(entry, data_dir)
        if caption is None:
            continue
        src_fps = float(entry.get("fps") or gt_fps)
        length_src = int(entry.get("num_frames") or round(float(entry.get("duration", 0.0)) * src_fps))
        if length_src <= 0:
            continue
        length = int(round(length_src * model_fps / src_fps))
        length = (length // 4) * 4
        length = max(min_length, min(max_length, length))
        gt_path = None
        if gt_hml263_dir is not None:
            stem = Path(str(entry.get("smplx_path") or "")).stem
            candidates = [gt_hml263_dir / f"{_safe_name(name)}.npy"]
            if stem:
                candidates.append(gt_hml263_dir / f"{stem}.npy")
            gt_path = next((p for p in candidates if p.exists()), None)
            if gt_path is None:
                continue
            gt_len = int(np.load(gt_path, mmap_mode="r").shape[0])
            length = min(length, (gt_len // 4) * 4)
            if length < min_length:
                continue
        if eligible % num_shards == shard_index:
            jobs.append((str(name), caption, length, str(gt_path) if gt_path is not None else None))
            if max_samples and len(jobs) >= max_samples:
                break
        eligible += 1
    return jobs


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
        jobs.append((sid, caption, length, str(m_file)))
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
    _install_unused_rotation2xyz_module()

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

    return OFFICIAL_HML_STATS / "Mean.npy", OFFICIAL_HML_STATS / "Std.npy"


def _adaptive_keyframes(gt_features: np.ndarray, length: int,
                        target_period: int = 30) -> list[int]:
    """Adaptive key-pose selection on HumanML3D-263 (matched to the \\ours
    adaptive protocol): endpoints + local speed extrema + heading turning
    points, with a fallback so the key density is ~1 key / ``target_period``.

    263 layout: [root_rot_vel(1), root_lin_vel(2), root_y(1), ric(63), rot(126),
    local_vel(66=22*3), foot(4)]. We use local_vel for per-frame speed and
    root_rot_vel for heading turning points.
    """
    T = min(length, gt_features.shape[0])
    if T <= 2:
        return list(range(T))
    feats = gt_features[:T].astype(np.float32)
    local_vel = feats[:, 4 + 63 + 126: 4 + 63 + 126 + 66].reshape(T, 22, 3)
    speed = np.linalg.norm(local_vel, axis=-1).mean(axis=-1)          # (T,)
    rot_vel = np.abs(feats[:, 0])                                     # heading rate

    def _norm(x):
        rng = x.max() - x.min()
        return (x - x.min()) / rng if rng > 1e-8 else np.zeros_like(x)

    # Saliency: deviation of speed from a local mean (peaks AND contacts) + turning.
    win = max(3, target_period // 3)
    kernel = np.ones(win) / win
    speed_sm = np.convolve(speed, kernel, mode="same")
    saliency = _norm(np.abs(speed - speed_sm)) + _norm(rot_vel)

    # Target density ~ one key per ``target_period`` frames; endpoints always.
    n_target = max(2, int(round(T / float(target_period))) + 1)
    min_gap = max(2, int(round(target_period * 0.6)))
    keys = [0, T - 1]
    order = np.argsort(-saliency)
    for t in order:
        t = int(t)
        if len(keys) >= n_target:
            break
        if all(abs(t - k) >= min_gap for k in keys):
            keys.append(t)
    return sorted(set(keys))


def _observed_indices(mode: str, length: int, gt_features: np.ndarray,
                      obs_frac: float, key_period: int) -> np.ndarray:
    """Return observed (preserved) frame indices for an inpainting mask mode."""
    n = max(1, int(round(length * obs_frac)))
    if mode == "prefix":          # prediction: observe a leading window
        idx = list(range(min(n, length)))
    elif mode == "suffix":        # backcast: observe a trailing window
        idx = list(range(max(0, length - n), length))
    elif mode == "clip":          # CondMDI-clip / mid: observe both ends
        idx = list(range(min(n, length))) + list(range(max(0, length - n), length))
    elif mode == "mib":           # minimal in-betweening: first + last frame
        idx = [0, length - 1]
    elif mode == "keyframe":      # adaptive sparse keyframes
        idx = _adaptive_keyframes(gt_features, length, key_period)
    else:
        raise ValueError(f"unknown mask-mode: {mode}")
    return np.unique(np.clip(np.asarray(idx, dtype=np.int64), 0, length - 1))


def _build_inpainting_frames(
    gt_features: np.ndarray,
    length: int,
    observed_idx: np.ndarray,
    mean: torch.Tensor,
    std: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Inpaint arbitrary observed frames (generalizes the prefix-only path)."""
    if gt_features.shape[0] < length:
        raise ValueError(f"GT clip too short: {gt_features.shape[0]} < {length}")
    gt = torch.from_numpy(gt_features[:length].astype(np.float32)).to(device)
    gt_norm = (gt - mean.to(gt)) / std.to(gt)
    inpainted = torch.zeros((1, gt_norm.shape[1], 1, length), device=device, dtype=torch.float32)
    mask = torch.zeros_like(inpainted, dtype=torch.bool)
    obs = torch.from_numpy(observed_idx).to(device)
    inpainted[0, :, 0, obs] = gt_norm[obs].transpose(0, 1)
    mask[0, :, 0, obs] = True
    return mask, inpainted


def _build_inpainting_prefix(
    gt_features: np.ndarray,
    length: int,
    condition_num_frames: int,
    mean: torch.Tensor,
    std: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    n_cond = min(condition_num_frames, length)
    return _build_inpainting_frames(
        gt_features, length, np.arange(n_cond, dtype=np.int64), mean, std, device)


def _clear_flowmdm_embedding_cache(sampler) -> None:
    """FlowMDM caches text embeddings without keying by sequence length."""
    model = getattr(sampler, "model", None)
    if model is None:
        return
    device = next(model.parameters()).device
    for attr in ("emb_hash", "emb_forcemask_hash"):
        if hasattr(model, attr):
            setattr(model, attr, torch.tensor(-1, device=device, dtype=torch.long))


def _precompute_clip_text_cpu(sampler, raw_text: str, device: torch.device) -> torch.Tensor:
    """Encode FlowMDM text with CLIP on CPU and pass the embedding to diffusion.

    On some Taiji hosts, concurrent GPU CLIP attention in PyTorch SDPA/cuDNN can
    fail with ``CUDNN_STATUS_NOT_INITIALIZED``. FlowMDM already accepts
    ``text_embeddings`` in ``model_kwargs['y']``, so we keep the motion model on
    GPU and move only the frozen CLIP text encoder to CPU for this one-time
    caption embedding.
    """
    model = getattr(sampler, "model", None)
    if model is None or not hasattr(model, "clip_model"):
        raise RuntimeError("FlowMDM sampler does not expose clip_model for CPU precompute")
    import clip  # noqa: WPS433

    clip_model = model.clip_model.to("cpu")
    clip_model.eval()
    max_text_len = 20 if getattr(model, "dataset", None) in ["humanml", "kit"] else None
    if max_text_len is not None:
        default_context_length = 77
        context_length = max_text_len + 2
        tokens = clip.tokenize([raw_text], context_length=context_length, truncate=True)
        zero_pad = torch.zeros(
            [tokens.shape[0], default_context_length - context_length],
            dtype=tokens.dtype,
        )
        tokens = torch.cat([tokens, zero_pad], dim=1)
    else:
        tokens = clip.tokenize([raw_text], truncate=True)
    with torch.no_grad():
        embedding = clip_model.encode_text(tokens).float()
    return embedding.to(device)


def _sample_one(
    sampler,
    caption: str,
    length: int,
    device: torch.device,
    inpainting: tuple[torch.Tensor, torch.Tensor] | None = None,
    text_embedding: torch.Tensor | None = None,
) -> torch.Tensor:
    mask = torch.ones((1, length), device=device, dtype=torch.bool)
    y = {
        "mask": mask,
        "lengths": torch.tensor([length], dtype=torch.long),
        "text": [caption],
        "tokens": [""],
    }
    if inpainting is not None:
        inpainting_mask, inpainted_motion = inpainting
        y["inpainting_mask"] = inpainting_mask
        y["inpainted_motion"] = inpainted_motion
    if text_embedding is not None:
        # DiffusionWrapper_FlowMDM converts even a single text prompt to
        # all_texts=[["caption"]], so the fast path expects (I, N, D).
        y["text_embeddings"] = (
            text_embedding.unsqueeze(0)
            if text_embedding.ndim == 2
            else text_embedding
        )
    model_kwargs = {
        "y": {
            **y,
        }
    }
    _clear_flowmdm_embedding_cache(sampler)
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
    parser.add_argument("--anno-file", default=None,
                        help="MotionHub-format annotation file. If set, jobs are built from this split instead of recon-root/test.txt.")
    parser.add_argument("--caption-file", default=None,
                        help="Optional {id: caption} override for generation, e.g. rewritten captions.")
    parser.add_argument("--data-dir", default="data/motionhub")
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
    parser.add_argument(
        "--use-chunked-att",
        action="store_true",
        default=False,
        help="Enable FlowMDM's long-sequence chunked attention. Keep disabled for single-clip TP2M inference.",
    )
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
    parser.add_argument(
        "--stable-cuda-kernels",
        action="store_true",
        help="Disable fragile GPU SDPA/cuDNN fast paths for Taiji inference stability.",
    )
    parser.add_argument(
        "--precompute-clip-text-cpu",
        action="store_true",
        help="Precompute frozen CLIP text embeddings on CPU and feed them to FlowMDM.",
    )
    parser.add_argument(
        "--only-ids",
        default=None,
        help="Optional comma-separated id list or text file. Useful for targeted reruns.",
    )
    parser.add_argument("--gt-fps", type=float, default=30.0)
    parser.add_argument("--model-fps", type=float, default=20.0)
    parser.add_argument("--min-length", type=int, default=40)
    parser.add_argument("--max-length", type=int, default=196)
    parser.add_argument(
        "--condition-num-frames",
        type=int,
        default=0,
        help="If >0, run prefix-pose-conditioned generation by inpainting this many GT HML263 frames.",
    )
    parser.add_argument(
        "--mask-mode",
        default=None,
        choices=["prefix", "suffix", "clip", "mib", "keyframe"],
        help="Imputation mode for editing eval (FlowMDM/MDM native inference-only "
             "inpainting). Requires GT HML263 (recon jobs provide it). 'mib'=first+last "
             "frame; 'prefix'=prediction; 'suffix'=backcast; 'clip'=both-ends interior "
             "completion; 'keyframe'=adaptive sparse keyframes. Overrides "
             "--condition-num-frames when set.",
    )
    parser.add_argument("--obs-frac", type=float, default=0.2,
                        help="Observed fraction for prefix/suffix/clip modes.")
    parser.add_argument("--key-period", type=int, default=30,
                        help="Max gap between adaptive keyframes (keyframe mode).")
    parser.add_argument(
        "--keyframe-frac-file", default=None,
        help="JSON {source_id: {'fracs': [..]}} of SHARED adaptive-keyframe "
             "temporal fractions (computed once from \\ours's detector on the GT). "
             "When set in --mask-mode keyframe, observed frames are forced to "
             "round(frac*(length-1)) so every baseline observes the IDENTICAL "
             "keyframes as \\ours, instead of FlowMDM's own 263-space detector.")
    parser.add_argument(
        "--gt-hml263-dir",
        default=None,
        help="Directory containing GT HML263 .npy files for prefix conditioning. "
             "Required with --condition-num-frames when --anno-file is set.",
    )
    args = parser.parse_args()
    if args.stable_cuda_kernels:
        _configure_stable_cuda_kernels()

    args.model_path = _resolve(args.model_path)
    recon_root = _resolve(args.recon_root)
    src_h3d272 = _resolve(args.src_h3d272)
    out_dir = _resolve(args.out_dir)
    gt_hml263_dir = _resolve(args.gt_hml263_dir) if args.gt_hml263_dir else None
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.condition_num_frames < 0:
        raise ValueError("--condition-num-frames must be >= 0")
    if args.condition_num_frames > 0 and args.anno_file and gt_hml263_dir is None:
        raise ValueError("--condition-num-frames with --anno-file requires --gt-hml263-dir")

    if args.anno_file:
        jobs = _make_jobs_from_anno(
            _resolve(args.anno_file),
            _resolve(args.caption_file) if args.caption_file else None,
            _resolve(args.data_dir),
            max_samples=args.max_samples,
            num_shards=args.num_shards,
            shard_index=args.shard_index,
            gt_fps=args.gt_fps,
            model_fps=args.model_fps,
            min_length=args.min_length,
            max_length=args.max_length,
            gt_hml263_dir=gt_hml263_dir,
        )
    else:
        jobs = _make_jobs(src_h3d272=src_h3d272, recon_root=recon_root,
                          max_samples=args.max_samples, num_shards=args.num_shards,
                          shard_index=args.shard_index)
    only_ids = _load_only_ids(args.only_ids)
    if only_ids is not None:
        jobs = [job for job in jobs if str(job[0]) in only_ids]

    # SHARED adaptive-keyframe fractions (Table 5): {sid: [frac,...]}; forces
    # FlowMDM to observe the exact same relative keyframes \ours observes.
    kf_fracs = None
    if args.keyframe_frac_file:
        import json as _json
        _raw = _json.loads(_resolve(args.keyframe_frac_file).read_text())
        kf_fracs = {str(k): list(v.get("fracs", v) if isinstance(v, dict) else v)
                    for k, v in _raw.items()}
        print(f"[+] shared keyframe fracs for {len(kf_fracs)} clips "
              f"(<- {args.keyframe_frac_file})")
        if only_ids is None:
            jobs = [job for job in jobs if str(job[0]) in kf_fracs]
            print(f"[+] restricted to {len(jobs)} clips present in keyframe-frac-file")
    print(f"[+] jobs={len(jobs)} shard={args.shard_index}/{args.num_shards} out={out_dir}")
    if args.load_only:
        return

    sampler, device = _load_sampler(args)
    if args.precompute_clip_text_cpu:
        sampler.model.clip_model.to("cpu")
        sampler.model.clip_model.eval()
    if args.load_model_only:
        print("[+] model load check complete")
        return
    mean_path, std_path = _resolve_stats(args)
    mean = torch.from_numpy(np.load(mean_path)).float().to(device)
    std = torch.from_numpy(np.load(std_path)).float().to(device)
    print(f"[+] denorm_stats mean={mean_path} std={std_path}")

    written = skipped = failed = 0
    for idx, (sid, caption, length, gt_path) in enumerate(tqdm(jobs, ncols=80)):
        if args.skip_existing and (out_dir / f"{_safe_name(sid)}.npy").exists():
            skipped += 1
            continue
        try:
            torch.manual_seed(args.seed + args.shard_index * 100000 + idx)
            inpainting = None
            if args.mask_mode is not None:
                if gt_path is None:
                    raise ValueError(f"missing GT HML263 path for masked sample {sid}")
                gt_features = np.load(gt_path)
                if args.mask_mode == "keyframe" and kf_fracs is not None \
                        and str(sid) in kf_fracs:
                    fr = np.asarray(kf_fracs[str(sid)], dtype=np.float64)
                    observed_idx = np.unique(np.clip(
                        np.round(fr * (length - 1)).astype(np.int64), 0, length - 1))
                else:
                    observed_idx = _observed_indices(
                        args.mask_mode, length, gt_features, args.obs_frac, args.key_period)
                inpainting = _build_inpainting_frames(
                    gt_features, length, observed_idx, mean, std, device)
            elif args.condition_num_frames > 0:
                if gt_path is None:
                    raise ValueError(f"missing GT HML263 path for prefix-conditioned sample {sid}")
                gt_features = np.load(gt_path)
                inpainting = _build_inpainting_prefix(
                    gt_features,
                    length,
                    args.condition_num_frames,
                    mean,
                    std,
                    device,
                )
            text_embedding = (
                _precompute_clip_text_cpu(sampler, caption, device)
                if args.precompute_clip_text_cpu
                else None
            )
            pred_norm = _sample_one(
                sampler,
                caption,
                length,
                device,
                inpainting=inpainting,
                text_embedding=text_embedding,
            )
            pred = (pred_norm * std + mean).detach().cpu().numpy().astype(np.float32)
            np.save(out_dir / f"{_safe_name(sid)}.npy", pred)
            written += 1
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"[fail] {sid}: {type(exc).__name__}: {exc}", flush=True)
            traceback.print_exc()
    print(f"[done] written={written} skipped={skipped} failed={failed}")


if __name__ == "__main__":
    main()
