#!/usr/bin/env python3
"""Run MotionLab text-to-motion inference on the HumanML3D test split.

This wrapper intentionally avoids MotionLab's Lightning ``test.py`` path:
the upstream text-only test step updates metrics but does not return generated
motions to ``test_epoch_end``, so ``SAVE_PREDICTIONS`` cannot export the files
we need for cross-protocol evaluation.

The implementation below loads only the released text encoder and denoiser,
then mirrors ``RFMOTION_SEPERATE.diffusion_reverse`` for the text condition.
Predictions are saved as un-normalized HumanML3D-263 features keyed by the
reconstructed HumanML3D test ids.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import tqdm


REPO = Path(__file__).resolve().parents[2]
MOTIONLAB_ROOT = REPO / "ref_repo" / "MotionLab"
SRC_H3D272 = REPO / "ref_repo" / "MotionStreamer" / "MotionStreamer" / "humanml3d_272"
RECON_ROOT = REPO / "work_dirs" / "h3d263_eval" / "h3d263_test_recon_fk"
OFFICIAL_HML_STATS = MOTIONLAB_ROOT / "datasets" / "all"


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
                pool.extend(
                    v.strip() for v in values if isinstance(v, str) and v.strip()
                )
    if isinstance(data, dict) and isinstance(data.get("result"), list):
        for item in data["result"]:
            if not isinstance(item, dict):
                continue
            for key in ("short_caption_rewritten", "short caption_rewritten"):
                values = item.get(key)
                if isinstance(values, list):
                    pool.extend(
                        v.strip() for v in values if isinstance(v, str) and v.strip()
                    )
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


def _load_cfg(args):
    os.chdir(str(MOTIONLAB_ROOT))
    if str(MOTIONLAB_ROOT) not in sys.path:
        sys.path.insert(0, str(MOTIONLAB_ROOT))

    from rfmotion.config import get_module_config  # noqa: WPS433

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    ckpt_cfg = None
    if isinstance(ckpt, dict):
        ckpt_cfg = ckpt.get("datamodule_hyper_parameters", {}).get("cfg")

    if args.cfg_from_checkpoint and ckpt_cfg is not None:
        cfg = OmegaConf.create(ckpt_cfg)
        cfg_assets = OmegaConf.load(args.cfg_assets)
        cfg = OmegaConf.merge(cfg, cfg_assets)
    else:
        cfg_base = OmegaConf.load("configs/base.yaml")
        cfg_exp = OmegaConf.merge(cfg_base, OmegaConf.load(args.cfg))
        cfg_model = get_module_config(cfg_exp.model, cfg_exp.model.target)
        cfg_assets = OmegaConf.load(args.cfg_assets)
        cfg = OmegaConf.merge(cfg_exp, cfg_model, cfg_assets)

    cfg.DEBUG = False
    cfg.ACCELERATOR = "gpu"
    cfg.DEVICE = [0]
    cfg.TRAIN.STAGE = "diffusion"
    cfg.TRAIN.ABLATION.VAE = False
    cfg.DATASET.NFEATS = 263
    cfg.DATASET.NJOINTS = 22
    cfg.model.denoiser.params.nfeats = 263
    cfg.METRIC.TYPE = []
    cfg.TEST.CHECKPOINTS = args.checkpoint
    if args.clip_path:
        cfg.model.clip_path = args.clip_path
        cfg.model.text_encoder.params.modelpath = args.clip_path
    return cfg


def _load_modules(cfg, device: torch.device):
    from rfmotion.config import instantiate_from_config  # noqa: WPS433
    from rfmotion.models.operator.scheduling_flow_match_euler_discrete import (  # noqa: WPS433
        FlowMatchEulerDiscreteScheduler,
    )

    text_encoder = instantiate_from_config(cfg.model.text_encoder).eval().to(device)
    denoiser = instantiate_from_config(cfg.model.denoiser).eval().to(device)
    scheduler = FlowMatchEulerDiscreteScheduler(
        num_train_timesteps=cfg.model.noise_scheduler.params.num_train_timesteps
    )

    ckpt = torch.load(cfg.TEST.CHECKPOINTS, map_location="cpu")
    state = (
        ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    )
    den_state = {
        k.replace("denoiser.", "", 1): v
        for k, v in state.items()
        if k.startswith("denoiser.")
    }
    missing, unexpected = denoiser.load_state_dict(den_state, strict=False)
    print(f"[+] denoiser loaded: missing={len(missing)} unexpected={len(unexpected)}")
    if unexpected:
        print("    unexpected:", unexpected[:10])
    return text_encoder, denoiser, scheduler


def _make_jobs(
    recon_root: Path,
    src_h3d272: Path,
    max_samples: int | None,
    num_shards: int,
    shard_index: int,
):
    ids = [
        s.strip()
        for s in (recon_root / "test.txt").read_text().splitlines()
        if s.strip()
    ]
    jobs = []
    for sid in ids:
        m_file = recon_root / "new_joint_vecs" / f"{sid}.npy"
        if not m_file.exists():
            continue
        length = int(np.load(m_file, mmap_mode="r").shape[0])
        if length < 40 or length >= 200:
            continue
        caption = _read_first_caption(src_h3d272 / "texts" / f"{sid}.txt")
        if not caption:
            continue
        jobs.append((sid, caption, (length // 4) * 4, str(m_file)))
        if max_samples and len(jobs) >= max_samples:
            break
    if num_shards > 1:
        jobs = jobs[shard_index::num_shards]
    return jobs


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
        length_src = int(
            entry.get("num_frames")
            or round(float(entry.get("duration", 0.0)) * src_fps)
        )
        if length_src <= 0:
            continue
        length = int(round(length_src * model_fps / src_fps))
        length = (length // 4) * 4
        length = max(min_length, min(max_length, length))
        gt_path = None
        if gt_hml263_dir is not None:
            stem = Path(str(entry.get("smplx_path") or "")).stem
            candidates = [gt_hml263_dir / f"{name}.npy"]
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
            jobs.append(
                (
                    str(name),
                    caption,
                    length,
                    str(gt_path) if gt_path is not None else None,
                )
            )
            if max_samples and len(jobs) >= max_samples:
                break
        eligible += 1
    return jobs


def _hml263_to_motionlab_joints(features: torch.Tensor) -> torch.Tensor:
    from rfmotion.data.humanml.scripts.motion_process import (
        recover_from_ric,
    )  # noqa: WPS433

    return recover_from_ric(features, 22)


def _build_prefix_hint(
    gt_features: list[np.ndarray],
    lengths: list[int],
    condition_num_frames: int,
    mean_motion: torch.Tensor,
    std_motion: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if condition_num_frames <= 0:
        raise ValueError("condition_num_frames must be positive for prefix hints")
    bsz = len(gt_features)
    max_len = max(lengths)
    hint = torch.zeros((bsz, max_len, 66), device=device, dtype=torch.float32)
    hint_lengths = torch.zeros((bsz, max_len), device=device, dtype=torch.bool)
    hint_masks = torch.zeros((bsz, max_len, 22, 3), device=device, dtype=torch.bool)
    for i, (arr, length) in enumerate(zip(gt_features, lengths)):
        clip = torch.from_numpy(arr[:length].astype(np.float32)).to(device).unsqueeze(0)
        joints = _hml263_to_motionlab_joints(clip)
        joints = (joints - mean_motion.to(joints)) / std_motion.to(joints)
        n_cond = min(condition_num_frames, length)
        hint[i, :n_cond] = joints[0, :n_cond].reshape(n_cond, 66)
        hint_lengths[i, :n_cond] = True
        hint_masks[i, :n_cond, :, :] = True
    return hint, hint_lengths, hint_masks


def _build_mib_hint(
    gt_features: list[np.ndarray],
    lengths: list[int],
    mean_motion: torch.Tensor,
    std_motion: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Motion in-betweening hint: observe ONLY the first and last frame."""
    bsz = len(gt_features)
    max_len = max(lengths)
    hint = torch.zeros((bsz, max_len, 66), device=device, dtype=torch.float32)
    hint_lengths = torch.zeros((bsz, max_len), device=device, dtype=torch.bool)
    hint_masks = torch.zeros((bsz, max_len, 22, 3), device=device, dtype=torch.bool)
    for i, (arr, length) in enumerate(zip(gt_features, lengths)):
        if length < 2:
            continue
        clip = torch.from_numpy(arr[:length].astype(np.float32)).to(device).unsqueeze(0)
        joints = _hml263_to_motionlab_joints(clip)
        joints = (joints - mean_motion.to(joints)) / std_motion.to(joints)
        for fidx in (0, length - 1):
            hint[i, fidx] = joints[0, fidx].reshape(66)
            hint_lengths[i, fidx] = True
            hint_masks[i, fidx, :, :] = True
    return hint, hint_lengths, hint_masks


def _protocol_observed_indices(
    protocol: str, length: int, obs_frac: float
) -> list[int]:
    """Observed (preserved) frame indices for the temporal-completion protocols.

    Frame counts use ``ceil(obs_frac*L)`` to match hftrainer
    ``build_inbetween_mask(keep_*_frac)`` used by \\ours, so the observed window
    is IDENTICAL to ours' protocol mask:
        pre20  -> leading  ceil(0.2L) frames (Prediction)
        post20 -> trailing ceil(0.2L) frames (Backcast)
        mid60  -> both-end ceil(0.2L) windows (CondMDI-clip)
    """
    import math

    n = max(1, min(int(math.ceil(length * obs_frac)), length))
    if protocol == "pre20":
        idx = list(range(n))
    elif protocol == "post20":
        idx = list(range(max(0, length - n), length))
    elif protocol == "mid60":
        idx = list(range(n)) + list(range(max(0, length - n), length))
    else:
        raise ValueError(f"unknown protocol: {protocol}")
    return sorted(set(int(i) for i in idx))


def _build_window_hint(
    gt_features: list[np.ndarray],
    lengths: list[int],
    protocol: str,
    obs_frac: float,
    mean_motion: torch.Tensor,
    std_motion: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Keyframe hint observing a contiguous 20% window (full poses) per protocol."""
    bsz = len(gt_features)
    max_len = max(lengths)
    hint = torch.zeros((bsz, max_len, 66), device=device, dtype=torch.float32)
    hint_lengths = torch.zeros((bsz, max_len), device=device, dtype=torch.bool)
    hint_masks = torch.zeros((bsz, max_len, 22, 3), device=device, dtype=torch.bool)
    for i, (arr, length) in enumerate(zip(gt_features, lengths)):
        clip = torch.from_numpy(arr[:length].astype(np.float32)).to(device).unsqueeze(0)
        joints = _hml263_to_motionlab_joints(clip)
        joints = (joints - mean_motion.to(joints)) / std_motion.to(joints)
        obs = _protocol_observed_indices(protocol, length, obs_frac)
        for fidx in obs:
            hint[i, fidx] = joints[0, fidx].reshape(66)
            hint_lengths[i, fidx] = True
            hint_masks[i, fidx, :, :] = True
    return hint, hint_lengths, hint_masks


def _build_keyframe_hint(
    gt_features: list[np.ndarray],
    lengths: list[int],
    keyframe_obs: list[np.ndarray],
    mean_motion: torch.Tensor,
    std_motion: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Adaptive sparse keyframe hint: observe full poses at the SHARED keyframes.

    ``keyframe_obs[i]`` is the per-clip list of observed frame indices already
    mapped into MotionLab's generation length space (round(frac*(L-1))), so the
    observed keyframes are IDENTICAL (in relative position) to those \\ours
    observes. The full 22-joint pose is pinned at every observed frame.
    """
    bsz = len(gt_features)
    max_len = max(lengths)
    hint = torch.zeros((bsz, max_len, 66), device=device, dtype=torch.float32)
    hint_lengths = torch.zeros((bsz, max_len), device=device, dtype=torch.bool)
    hint_masks = torch.zeros((bsz, max_len, 22, 3), device=device, dtype=torch.bool)
    for i, (arr, length) in enumerate(zip(gt_features, lengths)):
        clip = torch.from_numpy(arr[:length].astype(np.float32)).to(device).unsqueeze(0)
        joints = _hml263_to_motionlab_joints(clip)
        joints = (joints - mean_motion.to(joints)) / std_motion.to(joints)
        obs = sorted({int(f) for f in keyframe_obs[i] if 0 <= int(f) < length})
        if not obs:
            obs = [0, length - 1]
        for fidx in obs:
            hint[i, fidx] = joints[0, fidx].reshape(66)
            hint_lengths[i, fidx] = True
            hint_masks[i, fidx, :, :] = True
    return hint, hint_lengths, hint_masks


def _build_trajectory_hint(
    gt_features: list[np.ndarray],
    lengths: list[int],
    traj_obs: list,
    mean_motion: torch.Tensor,
    std_motion: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pelvis (root) trajectory hint -- MotionLab's native ``hint_type='pelvis'``.

    Observe ONLY joint 0 (pelvis) at the requested frames (dense = every frame;
    sparse = waypoint frames), zeroing all other joints in the 66-dim hint exactly
    like rfmotion.hint_mask (``specify_joints=[0]``). This is the faithful Table 7
    trajectory/waypoint control task (text + trajectory branch). MotionLab pins the
    full pelvis xyz position (the metric still scores XZ error), recorded as a
    caveat in the report. ``traj_obs[i]`` are frame indices already in MotionLab's
    generation length space (round(frac*(L-1))).
    """
    bsz = len(gt_features)
    max_len = max(lengths)
    hint = torch.zeros((bsz, max_len, 66), device=device, dtype=torch.float32)
    hint_lengths = torch.zeros((bsz, max_len), device=device, dtype=torch.bool)
    hint_masks = torch.zeros((bsz, max_len, 22, 3), device=device, dtype=torch.bool)
    for i, (arr, length) in enumerate(zip(gt_features, lengths)):
        clip = torch.from_numpy(arr[:length].astype(np.float32)).to(device).unsqueeze(0)
        joints = _hml263_to_motionlab_joints(clip)
        joints = (joints - mean_motion.to(joints)) / std_motion.to(joints)
        obs = sorted({int(f) for f in traj_obs[i] if 0 <= int(f) < length})
        if not obs:
            obs = list(range(length))
        for fidx in obs:
            row = torch.zeros(22, 3, device=device, dtype=torch.float32)
            row[0, :] = joints[0, fidx, 0, :]  # pelvis only (all 3 coords)
            hint[i, fidx] = row.reshape(66)
            hint_lengths[i, fidx] = True
            hint_masks[i, fidx, 0, :] = True
    return hint, hint_lengths, hint_masks


def _build_bodypart_hint(
    gt_features: list[np.ndarray],
    lengths: list[int],
    part_joints: list[int],
    mean_motion: torch.Tensor,
    std_motion: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Body-part joint-coordinate hint (Table-6 ExpB) -- MotionLab joint hint.

    Generalises the pelvis-only trajectory hint to an arbitrary joint subset
    (``part_joints``), observed on EVERY frame via their 3D positions. All other
    joints are zeroed in the 66-dim hint so the model regenerates them from text.
    This is the position-based analog of \\ours's rotation body-part control and
    runs on the same shared editing clip set as OmniControl / CondMDI. SOFT hint
    (text_hint branch, no test-time guidance) -> Ctrl.Err > 0 by design.
    """
    bsz = len(gt_features)
    max_len = max(lengths)
    hint = torch.zeros((bsz, max_len, 66), device=device, dtype=torch.float32)
    hint_lengths = torch.zeros((bsz, max_len), device=device, dtype=torch.bool)
    hint_masks = torch.zeros((bsz, max_len, 22, 3), device=device, dtype=torch.bool)
    jt = torch.as_tensor(sorted(set(int(j) for j in part_joints)),
                         dtype=torch.long, device=device)
    for i, (arr, length) in enumerate(zip(gt_features, lengths)):
        clip = torch.from_numpy(arr[:length].astype(np.float32)).to(device).unsqueeze(0)
        joints = _hml263_to_motionlab_joints(clip)  # (1, length, 22, 3)
        joints = (joints - mean_motion.to(joints)) / std_motion.to(joints)
        row = torch.zeros(length, 22, 3, device=device, dtype=torch.float32)
        row[:, jt, :] = joints[0, :length, jt, :]
        hint[i, :length] = row.reshape(length, 66)
        hint_lengths[i, :length] = True
        hint_masks[i, :length, jt, :] = True
    return hint, hint_lengths, hint_masks


def _sample_text_batch(
    cfg,
    text_encoder,
    denoiser,
    scheduler,
    captions,
    lengths,
    device,
    stage,
    num_steps=None,
    condition_num_frames=0,
    gt_features=None,
    mean_motion=None,
    std_motion=None,
    motionlab_condition_type="text_hint",
    mask_mode=None,
    protocol=None,
    obs_frac=0.20,
    keyframe_obs=None,
    part_joints=None,
):
    bsz = len(captions)
    max_len = max(lengths)
    target_motion = torch.zeros((bsz, max_len, 263), device=device, dtype=torch.float32)
    noisy_latents = torch.randn_like(target_motion)
    if num_steps is not None:
        steps = int(num_steps)
    elif stage == "demo":
        steps = int(cfg.model.scheduler.num_demo_steps)
    else:
        steps = int(cfg.model.scheduler.num_eval_steps)
    scheduler.set_timesteps(num_inference_steps=steps, device=device)

    def _encode_text(texts):
        if hasattr(text_encoder, "tokenizer") and hasattr(text_encoder, "text_model"):
            tokenizer = text_encoder.tokenizer
            max_length = getattr(text_encoder, "max_length", tokenizer.model_max_length)
            text_inputs = tokenizer(
                texts,
                padding="max_length",
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            input_ids = text_inputs.input_ids.to(device)
            text_model = text_encoder.text_model
            model_device = getattr(text_model, "device", device)
            input_ids = input_ids.to(model_device)
            name = getattr(text_encoder, "name", "")
            if hasattr(text_model, "get_text_features"):
                pooled = text_model.get_text_features(input_ids=input_ids)
                if not torch.is_tensor(pooled):
                    pooled = getattr(pooled, "pooler_output", None)
                if pooled is None:
                    raise TypeError(
                        "CLIP get_text_features did not return a tensor or pooler_output"
                    )
                pooled = pooled.to(device).unsqueeze(1)
                if name == "clip_hidden":
                    if hasattr(text_model, "text_model"):
                        hidden = text_model.text_model(input_ids).last_hidden_state
                    else:
                        hidden = text_model(input_ids).last_hidden_state
                    return (pooled, hidden.to(device))
                return (pooled,)
            encoded = text_model(input_ids)
            hidden = getattr(encoded, "last_hidden_state", None)
            pooler = getattr(encoded, "pooler_output", None)
            if hidden is not None and pooler is not None and name == "clip_hidden":
                return (pooler.to(device).unsqueeze(1), hidden.to(device))
            if hidden is not None:
                return (hidden.to(device),)

        encoded = text_encoder(texts)
        if isinstance(encoded, (tuple, list)):
            return tuple(encoded)
        last_hidden = getattr(encoded, "last_hidden_state", None)
        pooler = getattr(encoded, "pooler_output", None)
        if last_hidden is not None and pooler is not None:
            return (pooler.unsqueeze(1), last_hidden)
        if last_hidden is not None:
            return (last_hidden,)
        if torch.is_tensor(encoded):
            return (encoded.unsqueeze(1) if encoded.ndim == 2 else encoded,)
        raise TypeError(f"Unsupported text encoder output type: {type(encoded)!r}")

    model_type = str(getattr(cfg.model, "model_type", ""))
    is_mib = mask_mode == "mib"
    is_keyframe = mask_mode == "keyframe"
    is_trajectory = mask_mode == "trajectory"
    is_bodypart = mask_mode == "bodypart"
    is_window = protocol is not None
    is_prefix = (
        condition_num_frames > 0
        or is_mib
        or is_window
        or is_keyframe
        or is_trajectory
        or is_bodypart
    )
    if is_prefix:
        if gt_features is None or mean_motion is None or std_motion is None:
            raise ValueError(
                "hint-conditioned MotionLab inference requires GT HML263 features and joint stats"
            )
        if is_bodypart:
            # Table-6 ExpB: observe a body-part's joints (part_joints) on EVERY
            # frame via their 3D positions, regenerate the rest from text. This
            # is MotionLab's native joint-coordinate hint (text_hint branch),
            # generalising the pelvis-only trajectory hint to an arbitrary joint
            # subset. SOFT hint (no test-time guidance) -> Ctrl.Err > 0, reported
            # honestly as MotionLab's true capability.
            motionlab_condition_type = "text_hint"
            if not part_joints:
                raise ValueError("--mask-mode bodypart requires --part joints")
            cond_hint, cond_hint_lengths, hint_masks = _build_bodypart_hint(
                gt_features,
                list(lengths),
                list(part_joints),
                mean_motion,
                std_motion,
                device,
            )
        elif is_trajectory:
            # text + trajectory branch (pelvis-only hint); keyframe_obs carries the
            # observed frame indices (dense=all, sparse=waypoints).
            motionlab_condition_type = "text_hint"
            if keyframe_obs is None:
                raise ValueError(
                    "--mask-mode trajectory requires trajectory obs indices"
                )
            cond_hint, cond_hint_lengths, hint_masks = _build_trajectory_hint(
                gt_features,
                list(lengths),
                keyframe_obs,
                mean_motion,
                std_motion,
                device,
            )
        elif is_keyframe:
            motionlab_condition_type = "text_inbetween"
            if keyframe_obs is None:
                raise ValueError("--mask-mode keyframe requires keyframe_obs indices")
            cond_hint, cond_hint_lengths, hint_masks = _build_keyframe_hint(
                gt_features,
                list(lengths),
                keyframe_obs,
                mean_motion,
                std_motion,
                device,
            )
        elif is_window:
            motionlab_condition_type = "text_inbetween"
            cond_hint, cond_hint_lengths, hint_masks = _build_window_hint(
                gt_features,
                list(lengths),
                protocol,
                obs_frac,
                mean_motion,
                std_motion,
                device,
            )
        elif is_mib:
            motionlab_condition_type = "text_inbetween"
            cond_hint, cond_hint_lengths, hint_masks = _build_mib_hint(
                gt_features,
                list(lengths),
                mean_motion,
                std_motion,
                device,
            )
        else:
            if motionlab_condition_type not in {"text_hint", "text_inbetween"}:
                raise ValueError(
                    f"unsupported MotionLab prefix condition: {motionlab_condition_type}"
                )
            cond_hint, cond_hint_lengths, hint_masks = _build_prefix_hint(
                gt_features,
                list(lengths),
                condition_num_frames,
                mean_motion,
                std_motion,
                device,
            )
        hint = torch.cat([torch.zeros_like(cond_hint), cond_hint], dim=0)
        hint_lengths = torch.cat(
            [torch.zeros_like(cond_hint_lengths), cond_hint_lengths], dim=0
        )
        if motionlab_condition_type == "text_inbetween":
            instruction_prompt = "generate motion by given text and key frames."
            guidance_scale = float(
                getattr(
                    cfg.model,
                    "text_inbetween_guidance_scale",
                    getattr(cfg.model, "text_guidance_scale", 1.0),
                )
            )
        else:
            instruction_prompt = "generate motion by given text and trajectory."
            guidance_scale = float(
                getattr(
                    cfg.model,
                    "text_hint_guidance_scale",
                    getattr(cfg.model, "text_guidance_scale", 1.0),
                )
            )
    else:
        hint = None
        hint_lengths = None
        hint_masks = None
        instruction_prompt = "generate motion by given text."
        guidance_scale = float(getattr(cfg.model, "text_guidance_scale", 1.0))
    text_lengths = [0] * bsz + [77] * bsz
    if model_type == "rfmotion_seperate":
        instructions = None
    else:
        # Unified RFMOTION conditions its denoiser on task instructions in
        # addition to the text encoder states; mirror demo_text/eval_text and
        # demo_text_inbetween/eval_text_inbetween.
        uncond_instruction = _encode_text(["reconstruct given masked source motion."])[
            0
        ][0]
        text_instruction = _encode_text([instruction_prompt])[0][0]
        instructions = torch.cat(
            [
                uncond_instruction.repeat(bsz, 1),
                text_instruction.repeat(bsz, 1),
            ],
            dim=0,
        ).to(device)
    with torch.no_grad():
        text = _encode_text([""] * bsz + list(captions))
        for t in scheduler.timesteps.to(torch.int32):
            if int(t.item()) == 0:
                continue
            latent_model_input = torch.cat([noisy_latents] * 2)
            v_pred = denoiser(
                instructions=instructions,
                hidden_states=latent_model_input,
                timestep=t,
                text=text,
                text_lengths=text_lengths,
                hint=hint,
                hint_lengths=hint_lengths,
                style=None,
                style_lengths=None,
                content=None,
                content_lengths=None,
                source_motion=None,
                source_lengths=None,
                source_lengths_z=None,
                target_lengths=list(lengths) + list(lengths),
                target_lengths_z=list(lengths) + list(lengths),
                return_dict=False,
            )[0]
            v_uncond, v_cond = v_pred.chunk(2)
            v_pred = v_uncond + guidance_scale * (v_cond - v_uncond)
            noisy_latents = scheduler.step(v_pred, t, noisy_latents, return_dict=False)[
                0
            ]
    return noisy_latents


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", default="configs/config_rfmotion_text.yaml")
    parser.add_argument("--cfg-assets", default="configs/assets.yaml")
    parser.add_argument(
        "--checkpoint", default="checkpoints/motionflow/motionflow/motionflow.ckpt"
    )
    parser.add_argument(
        "--no-cfg-from-checkpoint",
        dest="cfg_from_checkpoint",
        action="store_false",
        help="Use --cfg instead of the training cfg saved in the checkpoint.",
    )
    parser.set_defaults(cfg_from_checkpoint=True)
    parser.add_argument("--clip-path", default="openai/clip-vit-large-patch14")
    parser.add_argument(
        "--anno-file",
        default=None,
        help="MotionHub-format annotation file. If set, jobs are built from this split instead of recon-root/test.txt.",
    )
    parser.add_argument(
        "--caption-file",
        default=None,
        help="Optional {id: caption} override for generation, e.g. rewritten captions.",
    )
    parser.add_argument("--data-dir", default="data/motionhub")
    parser.add_argument("--recon-root", default=str(RECON_ROOT))
    parser.add_argument("--src-h3d272", default=str(SRC_H3D272))
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--stats-root",
        default=str(OFFICIAL_HML_STATS),
        help="Directory containing the HumanML3D Mean.npy/Std.npy used to denormalize model outputs. "
        "Defaults to the reconstructed HumanML3D test statistics shared by the other HML263 baselines.",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument(
        "--num-steps",
        type=int,
        default=None,
        help="Override the official stage step count. "
        "MotionLab eval uses cfg.model.scheduler.num_eval_steps; "
        "demo uses cfg.model.scheduler.num_demo_steps.",
    )
    parser.add_argument(
        "--stage",
        choices=["demo", "eval"],
        default="demo",
        help="Official MotionLab sampling stage. Use demo for the released visual demo "
        "(usually 201 steps) and eval for the paper evaluator path "
        "(the released checkpoint stores 21 steps).",
    )
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--gt-fps", type=float, default=30.0)
    parser.add_argument("--model-fps", type=float, default=20.0)
    parser.add_argument("--min-length", type=int, default=40)
    parser.add_argument("--max-length", type=int, default=196)
    parser.add_argument(
        "--condition-num-frames",
        type=int,
        default=0,
        help="If >0, run prefix-pose-conditioned generation using this many initial GT HML263 frames.",
    )
    parser.add_argument(
        "--gt-hml263-dir",
        default=None,
        help="Directory containing GT HML263 .npy files for prefix conditioning. "
        "Files may be named by annotation key or smplx_path stem.",
    )
    parser.add_argument(
        "--motionlab-condition-type",
        choices=["text_hint", "text_inbetween"],
        default="text_hint",
        help="MotionLab conditioning branch used when --condition-num-frames > 0.",
    )
    parser.add_argument(
        "--mask-mode",
        choices=["mib", "keyframe", "trajectory", "bodypart"],
        default=None,
        help="If 'mib', run motion in-betweening: observe ONLY the first and last "
        "GT frame (text_inbetween branch). If 'keyframe', observe full poses "
        "at the SHARED adaptive keyframes from --keyframe-ctrl-file "
        "(text_inbetween branch). Requires GT HML263 (recon provides it).",
    )
    parser.add_argument(
        "--keyframe-ctrl-file",
        default=None,
        help="JSON {source_id: {'fracs': [..]}} of SHARED adaptive-keyframe "
        "temporal fractions (computed once from \\ours's detector on the GT). "
        "Used with --mask-mode keyframe: observed frames are forced to "
        "round(frac*(length-1)) so MotionLab observes the IDENTICAL keyframes "
        "as \\ours. Jobs are restricted to clips present in this file.",
    )
    parser.add_argument(
        "--protocol",
        choices=["pre20", "post20", "mid60"],
        default=None,
        help="Temporal-completion protocol (text_inbetween keyframe branch): "
        "pre20=Prediction (observe leading 20%%), post20=Backcast (observe "
        "trailing 20%%), mid60=CondMDI-clip (observe both-end 20%%). Observed "
        "frame counts = ceil(obs_frac*L), parity with \\ours. Requires GT HML263.",
    )
    parser.add_argument(
        "--obs-frac",
        type=float,
        default=0.20,
        help="Observed fraction per side for --protocol.",
    )
    parser.add_argument(
        "--part",
        default=None,
        help="Body-part key for --mask-mode bodypart (e.g. A_upper); joint set "
        "from scripts/eval/bodypart_pos_common.PART_JOINTS.",
    )
    parser.add_argument(
        "--source-id-file",
        default=None,
        help="JSON list / newline txt of HumanML3D source ids to restrict jobs to "
        "(shared editing clip set, parity with OmniControl / CondMDI bodypart).",
    )
    args = parser.parse_args()

    if args.condition_num_frames < 0:
        raise ValueError("--condition-num-frames must be >= 0")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir).resolve()
    recon_root = Path(args.recon_root).resolve()
    src_h3d272 = Path(args.src_h3d272).resolve()
    stats_root = Path(args.stats_root).resolve()
    anno_file = Path(args.anno_file).resolve() if args.anno_file else None
    caption_file = Path(args.caption_file).resolve() if args.caption_file else None
    data_dir = Path(args.data_dir).resolve()
    gt_hml263_dir = Path(args.gt_hml263_dir).resolve() if args.gt_hml263_dir else None
    if (
        args.condition_num_frames > 0
        and anno_file is not None
        and gt_hml263_dir is None
    ):
        raise ValueError(
            "--condition-num-frames with --anno-file requires --gt-hml263-dir"
        )
    if args.mask_mode in ("keyframe", "trajectory", "bodypart") and gt_hml263_dir is None:
        raise ValueError(
            f"--mask-mode {args.mask_mode} requires --gt-hml263-dir (GT HML263 for the hint)"
        )
    part_joints = None
    if args.mask_mode == "bodypart":
        if not args.part:
            raise ValueError("--mask-mode bodypart requires --part")
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from bodypart_pos_common import part_joints as _pj  # noqa: WPS433

        part_joints = _pj(args.part)
        print(f"[+] bodypart={args.part} obs joints={part_joints}")
    # Optional shared source-id restriction (parity with OmniControl / CondMDI).
    src_id_set = None
    if args.source_id_file:
        _sp = Path(args.source_id_file)
        _txt = _sp.read_text()
        try:
            src_id_set = {str(x) for x in json.loads(_txt)}
        except Exception:  # noqa: BLE001
            src_id_set = {s.strip() for s in _txt.splitlines() if s.strip()}
        print(f"[+] source-id-file: {len(src_id_set)} ids (<- {args.source_id_file})")
    # SHARED adaptive-keyframe fractions (Table 5): {sid: [frac,...]}; forces
    # MotionLab to observe the exact same relative keyframes \ours observes.
    kf_fracs = None
    if args.keyframe_ctrl_file:
        _raw = json.loads(Path(args.keyframe_ctrl_file).read_text())
        kf_fracs = {
            str(k): list(v.get("fracs", v) if isinstance(v, dict) else v)
            for k, v in (
                _raw.get("data_list", _raw) if isinstance(_raw, dict) else {}
            ).items()
        }
        print(
            f"[+] shared keyframe fracs for {len(kf_fracs)} clips (<- {args.keyframe_ctrl_file})"
        )
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = _load_cfg(args)
    print(
        "[+] cfg model_type={} condition_type={} text_guidance_scale={} "
        "num_eval_steps={} num_demo_steps={}".format(
            getattr(cfg.model, "model_type", None),
            getattr(cfg.model, "condition_type", None),
            getattr(cfg.model, "text_guidance_scale", None),
            getattr(cfg.model.scheduler, "num_eval_steps", None),
            getattr(cfg.model.scheduler, "num_demo_steps", None),
        )
    )
    text_encoder, denoiser, scheduler = _load_modules(cfg, device)

    if anno_file:
        jobs = _make_jobs_from_anno(
            anno_file,
            caption_file,
            data_dir,
            args.max_samples,
            args.num_shards,
            args.shard_index,
            args.gt_fps,
            args.model_fps,
            args.min_length,
            args.max_length,
            gt_hml263_dir,
        )
    else:
        jobs = _make_jobs(
            recon_root,
            src_h3d272,
            args.max_samples,
            args.num_shards,
            args.shard_index,
        )
    if kf_fracs is not None:
        jobs = [job for job in jobs if str(job[0]) in kf_fracs]
        print(f"[+] restricted to {len(jobs)} clips present in keyframe-ctrl-file")
    if src_id_set is not None:
        jobs = [job for job in jobs if str(job[0]) in src_id_set]
        print(f"[+] restricted to {len(jobs)} clips present in source-id-file")
    mean = torch.from_numpy(np.load(stats_root / "Mean.npy")).float().to(device)
    std = torch.from_numpy(np.load(stats_root / "Std.npy")).float().to(device)
    mean_motion = (
        torch.from_numpy(np.load(stats_root / "mean_motion.npy")).float().to(device)
    )
    std_motion = (
        torch.from_numpy(np.load(stats_root / "std_motion.npy")).float().to(device)
    )
    print(f"[+] denorm_stats={stats_root}")
    print(
        f"[+] jobs={len(jobs)} shard={args.shard_index}/{args.num_shards} "
        f"out={out_dir} device={device} stage={args.stage} "
        f"steps={args.num_steps or (cfg.model.scheduler.num_demo_steps if args.stage == 'demo' else cfg.model.scheduler.num_eval_steps)} "
        f"condition_num_frames={args.condition_num_frames}"
    )

    written = skipped = failed = 0
    for start in tqdm(range(0, len(jobs), args.batch_size), ncols=80):
        chunk = jobs[start : start + args.batch_size]
        todo = []
        for item in chunk:
            sid = item[0]
            if args.skip_existing and (out_dir / f"{_safe_name(sid)}.npy").exists():
                skipped += 1
            else:
                todo.append(item)
        if not todo:
            continue
        ids = [x[0] for x in todo]
        captions = [x[1] for x in todo]
        lengths = [int(x[2]) for x in todo]
        gt_features = None
        keyframe_obs = None
        use_hint = (
            args.condition_num_frames > 0
            or args.mask_mode == "mib"
            or args.mask_mode == "keyframe"
            or args.mask_mode == "trajectory"
            or args.mask_mode == "bodypart"
            or args.protocol is not None
        )
        if use_hint:
            gt_features = []
            for sid, _cap, length, gt_path in todo:
                if not gt_path:
                    raise ValueError(
                        f"missing GT HML263 path for hint-conditioned sample {sid}"
                    )
                arr = np.load(gt_path).astype(np.float32)
                gt_features.append(arr[:length])
        if args.mask_mode in ("keyframe", "trajectory"):
            keyframe_obs = []
            for sid, _cap, length, _gt_path in todo:
                fr = np.asarray(kf_fracs[str(sid)], dtype=np.float64)
                obs = np.clip(np.round(fr * (length - 1)).astype(int), 0, length - 1)
                keyframe_obs.append(sorted(set(obs.tolist())))
        try:
            pred_norm = _sample_text_batch(
                cfg,
                text_encoder,
                denoiser,
                scheduler,
                captions,
                lengths,
                device,
                args.stage,
                args.num_steps,
                args.condition_num_frames,
                gt_features,
                mean_motion,
                std_motion,
                args.motionlab_condition_type,
                args.mask_mode,
                args.protocol,
                args.obs_frac,
                keyframe_obs,
                part_joints,
            )
            pred = pred_norm * std + mean
            pred = pred.detach().cpu().numpy().astype(np.float32)
            for i, (sid, _cap, length, _gt_path) in enumerate(todo):
                np.save(out_dir / f"{_safe_name(sid)}.npy", pred[i, :length])
                written += 1
        except Exception as exc:  # noqa: BLE001
            failed += len(todo)
            print(f"[fail] batch={start}: {type(exc).__name__}: {exc}", flush=True)
            traceback.print_exc()
    print(f"[done] written={written} skipped={skipped} failed={failed}")


if __name__ == "__main__":
    main()
