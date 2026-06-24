#!/usr/bin/env python3
"""Evaluate MotionStreamer Causal-TAE reconstruction on MotionHub SMPL clips.

The MotionStreamer tokenizer is a continuous TAE on the 272D representation:

    MotionHub SMPL npz -> 135D row-major local rotations -> MotionStreamer 272
      -> normalize with official 272 mean/std -> Causal_HumanTAE -> denormalize
      -> recover 272 stored joints/local rotations -> reconstruction metrics.

For 2P rows, the single-person TAE is applied independently to each person and
the metrics are averaged over both people, matching the "applied per person"
interpretation in ``tab_recons.tex``.
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
from tqdm import tqdm

REPO = Path(__file__).resolve().parents[2]
MS_ROOT = REPO / "ref_repo" / "MotionStreamer" / "MotionStreamer"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(MS_ROOT))

from hftrainer.datasets.motion.representation.humanml_repr import (  # noqa: E402
    recover_272_stored_positions,
    recover_local_rotations_and_root,
)
from hftrainer.evaluation.motion.m2m_eval_metrics import compute_pa_mpjpe  # noqa: E402
from hftrainer.models.motion.components.utils.geometry.rotation_convert import (  # noqa: E402
    axis_angle_to_matrix,
)
from hftrainer.pipelines.motion.differentiable_fk import rotmat_to_rot6d_row_major  # noqa: E402
from scripts.eval.motionstreamer_272_encoder import motion135_to_272  # noqa: E402


def load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def iter_annotation(path: Path):
    raw = load_json(path)
    data = raw.get("data_list", raw) if isinstance(raw, dict) else raw
    if isinstance(data, dict):
        yield from data.items()
        return
    for idx, entry in enumerate(data):
        key = entry.get("motion_id") or entry.get("id") or str(idx)
        yield str(key), entry


def resolve_path(data_dir: str | Path, value: str) -> Path:
    value_path = Path(value)
    if value_path.is_absolute():
        return value_path
    return Path(data_dir) / value_path


def load_smpl22_row135(path: Path) -> np.ndarray:
    z = np.load(str(path), allow_pickle=True)
    transl = np.asarray(z["transl"], dtype=np.float32)
    t = transl.shape[0]
    go = torch.from_numpy(np.asarray(z["global_orient"], dtype=np.float32)).reshape(t, 3)
    bp = torch.from_numpy(np.asarray(z["body_pose"], dtype=np.float32)).reshape(t, 21, 3)
    aa = torch.cat([go[:, None], bp], dim=1)
    rot6d = rotmat_to_rot6d_row_major(axis_angle_to_matrix(aa)).reshape(t, 132)
    return torch.cat([torch.from_numpy(transl), rot6d], dim=1).numpy().astype(np.float32)


def build_tae(device: torch.device, checkpoint: Path):
    import models.tae as tae  # noqa: WPS433

    net = tae.Causal_HumanTAE(
        hidden_size=1024,
        down_t=2,
        stride_t=2,
        depth=3,
        dilation_growth_rate=3,
        activation="relu",
        latent_dim=16,
        clip_range=[-30, 20],
    )
    ckpt = torch.load(str(checkpoint), map_location="cpu")
    net.load_state_dict(ckpt["net"], strict=True)
    return net.eval().to(device)


def summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"mean": None, "std": None, "num_samples": 0}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=0)),
        "num_samples": int(arr.size),
    }


def geodesic_deg(pred: np.ndarray, gt: np.ndarray) -> float:
    rel = np.matmul(np.swapaxes(pred, -1, -2), gt)
    trace = np.trace(rel, axis1=-2, axis2=-1)
    cos = np.clip((trace - 1.0) * 0.5, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos)).mean())


def root_aligned_mpjpe_mm(pred: np.ndarray, gt: np.ndarray) -> float:
    pred_ra = pred - pred[..., :1, :]
    gt_ra = gt - gt[..., :1, :]
    return float(np.linalg.norm(pred_ra - gt_ra, axis=-1).mean() * 1000.0)


def collect_selected(args: argparse.Namespace) -> tuple[list[tuple[str, dict[str, Any]]], dict[str, int]]:
    if args.source_motion272_dir:
        return collect_source_motion272(args)

    allowed_ids = None
    if args.id_list:
        allowed_ids = {
            line.strip()
            for line in Path(args.id_list).read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
    counters = Counter()
    selected: list[tuple[str, dict[str, Any]]] = []
    for key, record in iter_annotation(Path(args.anno_file)):
        if allowed_ids is not None and key not in allowed_ids:
            counters["skipped_id_filter"] += 1
            continue
        smplx_path = record.get("smplx_path")
        if smplx_path is None:
            counters["skipped_missing_path"] += 1
            continue
        n_person = len(smplx_path) if isinstance(smplx_path, list) else 1
        if n_person != args.num_person:
            counters["skipped_person_mismatch"] += 1
            continue
        if args.max_duration > 0:
            fps = float(record.get("fps") or args.src_fps)
            frames = int(record.get("num_frames") or 0)
            duration = frames / fps if fps > 0 and frames > 0 else 0.0
            if duration > args.max_duration:
                counters["skipped_duration"] += 1
                continue
        selected.append((key, record))
    if args.limit:
        selected = selected[: args.limit]
    return selected, dict(counters)


def collect_source_motion272(args: argparse.Namespace) -> tuple[list[tuple[str, dict[str, Any]]], dict[str, int]]:
    counters = Counter()
    if args.id_list:
        ids = [
            line.strip()
            for line in Path(args.id_list).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    else:
        ids = [
            line.strip()
            for line in Path(args.split).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    if args.limit:
        ids = ids[: args.limit]

    source_dir = Path(args.source_motion272_dir)
    selected: list[tuple[str, dict[str, Any]]] = []
    for sid in ids:
        path = source_dir / f"{sid}.npy"
        if not path.exists():
            counters["skipped_missing_motion272"] += 1
            continue
        selected.append(
            (
                sid,
                {
                    "motion272_path": str(path),
                    "data_source": "humanml3d_official272",
                    "num_person": 1,
                },
            )
        )
    return selected, dict(counters)


@torch.no_grad()
def tae_roundtrip(
    net,
    m272: np.ndarray,
    mean_t: torch.Tensor,
    std_t: torch.Tensor,
    device: torch.device,
) -> np.ndarray:
    x = torch.from_numpy(m272.astype(np.float32)).to(device)[None]
    x_norm = (x - mean_t) / std_t
    recon_norm, _mu, _logvar = net(x_norm)
    recon = recon_norm[:, : m272.shape[0]] * std_t + mean_t
    return recon.squeeze(0).detach().cpu().numpy().astype(np.float32)


def match_temporal_length_np(motion: np.ndarray, target_len: int) -> np.ndarray:
    """Clamp tokenizer reconstruction to the source clip length."""
    cur_len = int(len(motion))
    if cur_len == target_len:
        return motion
    if cur_len > target_len:
        return motion[:target_len]
    if cur_len <= 0:
        raise ValueError("tokenizer returned an empty motion")
    pad = np.repeat(motion[-1:], target_len - cur_len, axis=0)
    return np.concatenate([motion, pad], axis=0).astype(np.float32)


def paths_for_record(data_dir: str, record: dict[str, Any]) -> list[Path]:
    smplx_path = record["smplx_path"]
    if isinstance(smplx_path, list):
        return [resolve_path(data_dir, item) for item in smplx_path]
    return [resolve_path(data_dir, smplx_path)]


def motion272_for_record(args: argparse.Namespace, record: dict[str, Any]) -> list[np.ndarray]:
    if record.get("motion272_path"):
        return [np.load(record["motion272_path"]).astype(np.float32)]
    arrays = []
    for path in paths_for_record(args.data_dir, record):
        m135 = load_smpl22_row135(path)
        arrays.append(motion135_to_272(m135, rotation_space="local"))
    return arrays


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anno-file", default="")
    parser.add_argument("--data-dir", default="data/motionhub")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--id-list", default="")
    parser.add_argument(
        "--source-motion272-dir",
        default="",
        help="Read canonical MotionStreamer-272 .npy clips directly instead of SMPL annotation records.",
    )
    parser.add_argument(
        "--split",
        default=str(MS_ROOT / "humanml3d_272" / "split" / "test.txt"),
        help="Newline-separated ids used with --source-motion272-dir when --id-list is empty.",
    )
    parser.add_argument("--num-person", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--max-duration", type=float, default=0.0)
    parser.add_argument("--src-fps", type=float, default=30.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--checkpoint",
        default=str(MS_ROOT / "MotionStreamer_HF" / "Causal_TAE" / "net_last.pth"),
    )
    parser.add_argument(
        "--mean",
        default=str(MS_ROOT / "humanml3d_272" / "mean_std" / "Mean.npy"),
    )
    parser.add_argument(
        "--std",
        default=str(MS_ROOT / "humanml3d_272" / "mean_std" / "Std.npy"),
    )
    parser.add_argument("--save-recon", action="store_true")
    parser.add_argument("--save-flat-recon", action="store_true", help="Save <id>.npz with motion_272 directly in out-dir.")
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    if args.num_shards < 1:
        raise ValueError("--num-shards must be >= 1")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("--shard-index must satisfy 0 <= shard_index < num_shards")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    device = torch.device(args.device)

    selected_all, counters = collect_selected(args)
    selected = [item for idx, item in enumerate(selected_all) if idx % args.num_shards == args.shard_index]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.save_recon:
        (out_dir / "recon_272").mkdir(parents=True, exist_ok=True)

    mean = np.load(args.mean).astype(np.float32)
    std = np.load(args.std).astype(np.float32)
    mean_t = torch.from_numpy(mean).to(device)
    std_t = torch.from_numpy(std).to(device)
    net = build_tae(device, Path(args.checkpoint))

    values: dict[str, list[float]] = {
        "mpjpe_mm": [],
        "root_aligned_mpjpe_mm": [],
        "pa_mpjpe_mm": [],
        "mpjre_deg": [],
    }
    failures: list[dict[str, str]] = []
    per_case: list[dict[str, Any]] = []
    subsets = Counter()

    for key, record in tqdm(selected, desc="MotionStreamer TAE", ncols=90):
        try:
            flat_out = out_dir / f"{key}.npz"
            if args.skip_existing and args.save_flat_recon and flat_out.exists():
                continue
            gt_pos_people = []
            pred_pos_people = []
            gt_rot_people = []
            pred_rot_people = []
            motion272_people = motion272_for_record(args, record)
            for person_idx, m272 in enumerate(motion272_people):
                pred272 = tae_roundtrip(net, m272, mean_t, std_t, device)
                pred272 = match_temporal_length_np(pred272, len(m272))
                gt_pos_people.append(recover_272_stored_positions(m272))
                pred_pos_people.append(recover_272_stored_positions(pred272))
                gt_rot_people.append(recover_local_rotations_and_root(m272)[0])
                pred_rot_people.append(recover_local_rotations_and_root(pred272)[0])
                if args.save_recon:
                    suffix = f"_p{person_idx}" if args.num_person > 1 else ""
                    np.save(out_dir / "recon_272" / f"{key}{suffix}.npy", pred272.astype(np.float32))
                if args.save_flat_recon:
                    suffix = f"_p{person_idx}" if len(motion272_people) > 1 else ""
                    np.savez(out_dir / f"{key}{suffix}.npz", motion_272=pred272.astype(np.float32), source_id=key)

            gt_pos = np.stack(gt_pos_people, axis=0).astype(np.float32)
            pred_pos = np.stack(pred_pos_people, axis=0).astype(np.float32)
            gt_rot = np.stack(gt_rot_people, axis=0).astype(np.float32)
            pred_rot = np.stack(pred_rot_people, axis=0).astype(np.float32)
            mpjpe = float(np.linalg.norm(pred_pos - gt_pos, axis=-1).mean() * 1000.0)
            root_mpjpe = root_aligned_mpjpe_mm(pred_pos, gt_pos)
            pa = float(
                np.mean(
                    [
                        compute_pa_mpjpe(pred_pos[p], gt_pos[p])["pa_mpjpe_mean"] * 1000.0
                        for p in range(pred_pos.shape[0])
                    ]
                )
            )
            mpjre = geodesic_deg(pred_rot, gt_rot)
            values["mpjpe_mm"].append(mpjpe)
            values["root_aligned_mpjpe_mm"].append(root_mpjpe)
            values["pa_mpjpe_mm"].append(pa)
            values["mpjre_deg"].append(mpjre)
            subsets[str(record.get("data_source") or record.get("dataset") or "unknown")] += 1
            per_case.append(
                {
                    "key": key,
                    "num_person": args.num_person,
                    "frames": int(gt_pos.shape[1]),
                    "mpjpe_mm": mpjpe,
                    "root_aligned_mpjpe_mm": root_mpjpe,
                    "pa_mpjpe_mm": pa,
                    "mpjre_deg": mpjre,
                }
            )
        except Exception as exc:  # noqa: BLE001
            failures.append({"key": key, "error": f"{type(exc).__name__}: {exc}"})
            if len(failures) <= 10:
                print(f"[fail] {key}: {type(exc).__name__}: {exc}", flush=True)

    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "method": "MotionStreamer Causal_TAE",
        "checkpoint": str(args.checkpoint),
        "mean": str(args.mean),
        "std": str(args.std),
        "anno_file": args.anno_file,
        "id_list": args.id_list,
        "source_motion272_dir": args.source_motion272_dir,
        "split": args.split,
        "data_dir": args.data_dir,
        "num_person": args.num_person,
        "max_duration": args.max_duration,
        "selected_samples_before_shard": len(selected_all),
        "selected_samples": len(selected),
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        **counters,
        "subsets": dict(subsets),
        "summary": {
            "mpjpe_mm": summary(values["mpjpe_mm"]),
            "root_aligned_mpjpe_mm": summary(values["root_aligned_mpjpe_mm"]),
            "pa_mpjpe_mm": summary(values["pa_mpjpe_mm"]),
            "mpjre_deg": summary(values["mpjre_deg"]),
            "cb_util_percent": None,
            "num_failures": len(failures),
        },
        "failures": failures,
        "per_case": per_case,
    }
    write_json(out_dir / "recon_metrics.json", payload)
    if args.save_flat_recon or args.source_motion272_dir:
        write_json(out_dir / "metrics" / "recon_metrics.json", payload)
    print(json.dumps(payload["summary"], indent=2, ensure_ascii=False), flush=True)
    print(f"[motionstreamer-tae-recon] wrote {out_dir / 'recon_metrics.json'}", flush=True)


if __name__ == "__main__":
    main()
