#!/usr/bin/env python3
"""Re-evaluate VerMo motion-tokenizer reconstruction quality.

The VerMo tokenizer consumes the same motion tensor as the training pipeline:
``abs_rel`` translation (6D) + SMPL-22 local rotation6d (22 * 6D), using the
column-major 6D convention from ``LoadSmplx55(..., rot6d_convention='column')``.

This script intentionally does not reuse old reconstruction artifacts. It loads
an annotation split, runs the tokenizer round trip, restores absolute root
translation with ``SMPLPoseProcessor.inv_convert_transl()``, then computes FK
metrics with the configured SMPL body model.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from mmengine import Config
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import hftrainer.models.motion.vermo  # noqa: F401,E402
import hftrainer.models.motion.components.body_models.smplx_lite  # noqa: F401,E402
from hftrainer.datasets.motion.motionhub.transforms.load_smplx import LoadSmplx55  # noqa: E402
from hftrainer.evaluation.motion.m2m_eval_metrics import compute_pa_mpjpe  # noqa: E402
from hftrainer.models.motion.components.utils.geometry.rotation_convert import (  # noqa: E402
    rotation_6d_to_matrix,
)
from hftrainer.models.motion.vermo.vqvae_1d import VQVAEVermo1D  # noqa: E402
from hftrainer.registry import HF_MODELS, MODELS  # noqa: E402


def load_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def resolve_path(data_dir: str | Path, value: str | list[str]) -> str | list[str]:
    data_dir = Path(data_dir)
    if isinstance(value, list):
        return [str((data_dir / item).resolve()) if not os.path.isabs(item) else item for item in value]
    return str((data_dir / value).resolve()) if not os.path.isabs(value) else value


def motion_to_abs_and_pose(
    motion: torch.Tensor,
    smpl_processor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return absolute translation, pose6d, and compact 135D motion.

    ``motion`` is denormalized VerMo format ``[P,T,138]``:
    translation abs_rel ``[..., :6]`` plus 22-joint rot6d ``[..., 6:]``.
    """
    transl_abs = smpl_processor.inv_convert_transl(motion[..., :6], use_rollout=True)
    pose6d = motion[..., 6:]
    motion135 = torch.cat([transl_abs, pose6d], dim=-1)
    return transl_abs, pose6d, motion135


@torch.no_grad()
def fk_positions_and_rotmats(
    transl_abs: torch.Tensor,
    pose6d: torch.Tensor,
    smpl_processor,
    rot6d_convention: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute SMPL FK positions and local rotation matrices."""
    joints = smpl_processor.fk(transl_abs, pose6d, rot_type="rotation_6d")
    rot6d = pose6d.reshape(*pose6d.shape[:-1], 22, 6)
    rotmats = rotation_6d_to_matrix(rot6d, convention=rot6d_convention)
    return joints.detach().cpu().numpy(), rotmats.detach().cpu().numpy()


def geodesic_deg(pred_rot: np.ndarray, gt_rot: np.ndarray) -> float:
    rel = np.matmul(np.swapaxes(pred_rot, -1, -2), gt_rot)
    trace = np.trace(rel, axis1=-2, axis2=-1)
    cos = np.clip((trace - 1.0) * 0.5, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos)).mean())


def root_aware_position_metrics(pred_pos: np.ndarray, gt_pos: np.ndarray) -> dict[str, float]:
    """Position errors used by the reconstruction viewer and Table 3 reruns.

    ``mpjpe_mm`` is the root-frame pose error, matching the viewer's aligned
    pose comparison. ``raw_mpjpe_mm`` keeps the global trajectory error explicit
    so root drift is still visible and auditable.
    """
    raw = np.linalg.norm(pred_pos - gt_pos, axis=-1)
    root_delta = pred_pos[:, :, 0, :] - gt_pos[:, :, 0, :]
    root0_shift = gt_pos[:, 0:1, 0:1, :] - pred_pos[:, 0:1, 0:1, :]
    root0 = np.linalg.norm((pred_pos + root0_shift) - gt_pos, axis=-1)
    rootframe = np.linalg.norm(
        (pred_pos - pred_pos[:, :, 0:1, :]) - (gt_pos - gt_pos[:, :, 0:1, :]),
        axis=-1,
    )
    rootframe_mm = float(rootframe.mean() * 1000.0)
    return {
        "mpjpe_mm": rootframe_mm,
        "raw_mpjpe_mm": float(raw.mean() * 1000.0),
        "root0_mpjpe_mm": float(root0.mean() * 1000.0),
        "rootframe_mpjpe_mm": rootframe_mm,
        "root_mpjpe_mm": float(np.linalg.norm(root_delta, axis=-1).mean() * 1000.0),
    }


def load_motion_from_record(
    key: str,
    record: dict[str, Any],
    data_dir: str,
    loader: LoadSmplx55,
) -> torch.Tensor:
    motion_path = record.get("smplx_path")
    if motion_path is None:
        raise KeyError(f"{key}: missing smplx_path")
    results = dict(record)
    results["motion_path"] = resolve_path(data_dir, motion_path)
    out = loader(results)
    motion = out["motion"]
    if motion.ndim == 2:
        motion = motion.unsqueeze(0)
    if motion.ndim != 3:
        raise ValueError(f"{key}: expected motion [P,T,D], got {tuple(motion.shape)}")
    return motion


@torch.no_grad()
def tokenizer_roundtrip(
    motion_raw: torch.Tensor,
    vqvae,
    smpl_processor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return denormalized reconstruction and raw code indices."""
    device = next(vqvae.parameters()).device
    motion = motion_raw.to(device=device, dtype=torch.float32)

    motion_norm = smpl_processor.normalize(motion)
    if getattr(vqvae.config, "use_static", False):
        transl_abs, pose6d, _ = motion_to_abs_and_pose(motion, smpl_processor)
        static_joints = smpl_processor.get_static_joint_mask(
            smpl_processor.fk(transl_abs, pose6d)[..., [7, 10, 8, 11, 20, 21], :],
            vel_thr=0.15,
            repeat_last=True,
        )
        motion_norm = torch.cat([motion_norm, static_joints.to(motion_norm.device)], dim=-1)

    if isinstance(vqvae, VQVAEVermo1D):
        enc = vqvae.encode(motion_norm)
        recon_norm = vqvae.decode(enc.quant)
        indices = enc.indices
    else:
        if motion_norm.shape[-1] % 6 != 0:
            raise ValueError(f"2D tokenizer expects feature dim divisible by 6, got {motion_norm.shape[-1]}")
        motion_grid = motion_norm.reshape(*motion_norm.shape[:-1], motion_norm.shape[-1] // 6, 6)
        enc = vqvae.encode(motion_grid, flatten=False)
        recon_grid = vqvae.decode(enc.quant)
        recon_norm = recon_grid.reshape(*recon_grid.shape[:-2], recon_grid.shape[-2] * recon_grid.shape[-1])
        indices = enc.indices

    if getattr(vqvae.config, "use_static", False):
        recon_norm = recon_norm[..., : smpl_processor.mean.numel()]
    recon = smpl_processor.denormalize(recon_norm)
    return recon.detach().cpu(), indices.detach().cpu()


def update_code_usage(
    usage: list[set[int]],
    indices: torch.Tensor,
    codebook_size: int,
    split_last_dim_as_quantizers: bool,
) -> None:
    arr = indices.detach().cpu().numpy()
    if split_last_dim_as_quantizers and arr.ndim == 3:
        while len(usage) < arr.shape[-1]:
            usage.append(set())
        for qi in range(arr.shape[-1]):
            vals = arr[..., qi].reshape(-1)
            usage[qi].update(int(x) for x in vals)
    else:
        if not usage:
            usage.append(set())
        vals = arr.reshape(-1)
        usage[0].update(int(x) for x in vals)


def summarize(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"mean": None, "std": None, "num_samples": 0}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=0)),
        "num_samples": int(arr.size),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/vermo/vermo_pretrain_16k_llama1b_wavtokenizer.py")
    parser.add_argument("--tokenizer-path", default="", help="Override processor.motion_tokenizer.from_pretrained path.")
    parser.add_argument("--anno-file", required=True)
    parser.add_argument("--data-dir", default="data/motionhub")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--num-person", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--id-list",
        default="",
        help="Optional newline-separated key list. When set, only these annotation ids are evaluated.",
    )
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument(
        "--max-duration",
        type=float,
        default=0.0,
        help="Skip motions longer than this many seconds. 0 disables filtering.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--rot6d-convention", choices=["column", "row"], default="column")
    parser.add_argument("--save-recon-npz", action="store_true")
    args = parser.parse_args()
    if args.num_shards < 1:
        raise ValueError("--num-shards must be >= 1")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("--shard-index must satisfy 0 <= shard_index < num_shards")

    cfg = Config.fromfile(args.config)
    processor_cfg = cfg["model"]["processor"]
    smpl_cfg = processor_cfg["smpl_pose_processor"]
    tok_cfg = dict(processor_cfg["motion_tokenizer"])
    if args.tokenizer_path:
        tok_cfg["from_pretrained"] = {"pretrained_model_name_or_path": args.tokenizer_path}

    smpl_processor = MODELS.build(smpl_cfg).eval()
    vqvae = HF_MODELS.build(tok_cfg).eval()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    device = torch.device(args.device)
    smpl_processor = smpl_processor.to(device=device, dtype=torch.float32)
    vqvae = vqvae.to(device=device, dtype=torch.float32)

    loader = LoadSmplx55(
        key="motion",
        rot_type="rotation_6d",
        transl_type="abs_rel",
        smpl_type="smpl_22",
        rot6d_convention=args.rot6d_convention,
        transl_aug_prob=0.0,
    )

    annotations = load_json(args.anno_file)
    data_list = annotations["data_list"]
    allowed_ids: set[str] | None = None
    if args.id_list:
        allowed_ids = {
            line.strip()
            for line in Path(args.id_list).read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
    selected_all: list[tuple[str, dict[str, Any]]] = []
    skipped_person = 0
    skipped_duration = 0
    skipped_id_filter = 0
    for key, record in data_list.items():
        if allowed_ids is not None and key not in allowed_ids:
            skipped_id_filter += 1
            continue
        path = record.get("smplx_path")
        n_person = len(path) if isinstance(path, list) else 1
        if n_person != args.num_person:
            skipped_person += 1
            continue
        if args.max_duration > 0:
            fps = float(record.get("fps") or 30.0)
            num_frames = int(record.get("num_frames") or 0)
            duration = num_frames / fps if fps > 0 and num_frames > 0 else 0.0
            if duration > args.max_duration:
                skipped_duration += 1
                continue
        selected_all.append((key, record))
        if args.limit and len(selected_all) >= args.limit:
            break
    selected = [
        item for idx, item in enumerate(selected_all)
        if idx % args.num_shards == args.shard_index
    ]

    out_dir = Path(args.out_dir)
    recon_dir = out_dir / "recon_npz"
    if args.save_recon_npz:
        recon_dir.mkdir(parents=True, exist_ok=True)

    metric_values: dict[str, list[float]] = {
        "mpjpe_mm": [],
        "raw_mpjpe_mm": [],
        "root0_mpjpe_mm": [],
        "rootframe_mpjpe_mm": [],
        "root_mpjpe_mm": [],
        "pa_mpjpe_mm": [],
        "mpjre_deg": [],
    }
    frame_deltas: list[int] = []
    per_case: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    subset_counter: Counter[str] = Counter()
    code_usage: list[set[int]] = []
    codebook_size = int(getattr(vqvae, "codebook_size", 0))
    split_usage_by_quantizer = isinstance(vqvae, VQVAEVermo1D)

    for key, record in tqdm(selected, desc=f"recon {args.num_person}P"):
        try:
            subset_counter[str(record.get("subset", "?"))] += 1
            motion_raw = load_motion_from_record(key, record, args.data_dir, loader)
            recon_raw, indices = tokenizer_roundtrip(motion_raw, vqvae, smpl_processor)
            update_code_usage(
                code_usage,
                indices,
                codebook_size,
                split_last_dim_as_quantizers=split_usage_by_quantizer,
            )

            T = min(motion_raw.shape[1], recon_raw.shape[1])
            frame_delta = int(recon_raw.shape[1] - motion_raw.shape[1])
            frame_deltas.append(frame_delta)
            motion_eval = motion_raw[:, :T].to(device=device, dtype=torch.float32)
            recon_eval = recon_raw[:, :T].to(device=device, dtype=torch.float32)

            gt_transl, gt_pose, _ = motion_to_abs_and_pose(motion_eval, smpl_processor)
            pr_transl, pr_pose, _ = motion_to_abs_and_pose(recon_eval, smpl_processor)
            gt_pos, gt_rot = fk_positions_and_rotmats(
                gt_transl, gt_pose, smpl_processor, args.rot6d_convention
            )
            pr_pos, pr_rot = fk_positions_and_rotmats(
                pr_transl, pr_pose, smpl_processor, args.rot6d_convention
            )

            pos_metrics = root_aware_position_metrics(pr_pos, gt_pos)
            mpjpe = pos_metrics["mpjpe_mm"]
            pa_vals = []
            for person_idx in range(pr_pos.shape[0]):
                pa_vals.append(
                    compute_pa_mpjpe(pr_pos[person_idx], gt_pos[person_idx])["pa_mpjpe_mean"] * 1000.0
                )
            pa_mpjpe = float(np.mean(pa_vals))
            mpjre = geodesic_deg(pr_rot, gt_rot)

            metric_values["mpjpe_mm"].append(mpjpe)
            metric_values["raw_mpjpe_mm"].append(pos_metrics["raw_mpjpe_mm"])
            metric_values["root0_mpjpe_mm"].append(pos_metrics["root0_mpjpe_mm"])
            metric_values["rootframe_mpjpe_mm"].append(pos_metrics["rootframe_mpjpe_mm"])
            metric_values["root_mpjpe_mm"].append(pos_metrics["root_mpjpe_mm"])
            metric_values["pa_mpjpe_mm"].append(pa_mpjpe)
            metric_values["mpjre_deg"].append(mpjre)
            per_case.append(
                {
                    "key": key,
                    "subset": record.get("subset"),
                    "num_person": int(motion_raw.shape[0]),
                    "gt_frames": int(motion_raw.shape[1]),
                    "recon_frames": int(recon_raw.shape[1]),
                    "frame_delta": frame_delta,
                    "fps": float(record.get("fps") or 30.0),
                    "duration_sec": float(motion_raw.shape[1]) / float(record.get("fps") or 30.0),
                    "mpjpe_mm": mpjpe,
                    "raw_mpjpe_mm": pos_metrics["raw_mpjpe_mm"],
                    "root0_mpjpe_mm": pos_metrics["root0_mpjpe_mm"],
                    "rootframe_mpjpe_mm": pos_metrics["rootframe_mpjpe_mm"],
                    "root_mpjpe_mm": pos_metrics["root_mpjpe_mm"],
                    "pa_mpjpe_mm": pa_mpjpe,
                    "mpjre_deg": mpjre,
                }
            )

            if args.save_recon_npz:
                np.savez_compressed(
                    recon_dir / f"{key}.npz",
                    gt=motion_raw.numpy().astype(np.float32),
                    recon=recon_raw.numpy().astype(np.float32),
                    fps=np.asarray(record.get("fps", 30.0), dtype=np.float32),
                )
        except Exception as exc:  # keep full-run audit moving, record exact failures
            failures.append({"key": key, "error": repr(exc)})

    util_per_quantizer = [
        (len(items) / codebook_size * 100.0) if codebook_size else None
        for items in code_usage
    ]
    cb_util = None
    if util_per_quantizer:
        cb_util = float(np.mean([x for x in util_per_quantizer if x is not None]))

    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "config": args.config,
        "tokenizer_path": args.tokenizer_path or tok_cfg.get("from_pretrained", {}).get("pretrained_model_name_or_path"),
        "anno_file": args.anno_file,
        "id_list": args.id_list,
        "data_dir": args.data_dir,
        "num_person": args.num_person,
        "rot6d_convention": args.rot6d_convention,
        "selected_samples": len(selected),
        "selected_samples_before_shard": len(selected_all),
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "max_duration": args.max_duration,
        "skipped_person_mismatch": skipped_person,
        "skipped_duration": skipped_duration,
        "skipped_id_filter": skipped_id_filter,
        "subsets": dict(subset_counter),
        "summary": {
            "mpjpe_mm": summarize(metric_values["mpjpe_mm"]),
            "raw_mpjpe_mm": summarize(metric_values["raw_mpjpe_mm"]),
            "root0_mpjpe_mm": summarize(metric_values["root0_mpjpe_mm"]),
            "rootframe_mpjpe_mm": summarize(metric_values["rootframe_mpjpe_mm"]),
            "root_mpjpe_mm": summarize(metric_values["root_mpjpe_mm"]),
            "pa_mpjpe_mm": summarize(metric_values["pa_mpjpe_mm"]),
            "mpjre_deg": summarize(metric_values["mpjre_deg"]),
            "cb_util_percent": cb_util,
            "cb_util_percent_per_quantizer": util_per_quantizer,
            "codebook_size": codebook_size,
            "frame_delta_abs_mean": float(np.mean(np.abs(frame_deltas))) if frame_deltas else None,
            "frame_delta_abs_max": int(np.max(np.abs(frame_deltas))) if frame_deltas else None,
            "num_failures": len(failures),
        },
        "code_usage_values_per_quantizer": [sorted(items) for items in code_usage],
        "failures": failures,
        "per_case": per_case,
    }
    write_json(out_dir / "recon_metrics.json", payload)
    print(json.dumps(payload["summary"], indent=2, ensure_ascii=False))
    print(f"[vermo-tokenizer-recon] wrote {out_dir / 'recon_metrics.json'}")


if __name__ == "__main__":
    main()
