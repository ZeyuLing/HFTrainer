#!/usr/bin/env python3
"""Generate body-part position baselines on the official 4,012 clips.

The runner uses the released Motius wrappers for CondMDI, MaskControl,
OmniControl, and ProjFlow and writes physical HumanML3D-263 predictions at 20 fps. It
validates the native control contract before loading a model:

* CondMDI supports XYZ position atoms for arbitrary joints and frames.
* OmniControl supports arbitrary joints, axes, and sparse/dense frame support.
* ProjFlow supports arbitrary Cartesian joint, axis, and frame masks.
* Released MaskControl weights support only pelvis, feet, head, and wrists, and
  expose a joint-level (XYZ) mask rather than an axis-level mask.

Unsupported settings fail loudly instead of silently receiving a weaker or more
informative condition than the paper protocol.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Iterable

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("HFTRAINER_SKIP_AUTOREGISTER", "1")
DEFAULT_MOTIUS = ROOT.parent / "Motius"
DEFAULT_DATA = ROOT / "data/eval/m2m_v2/eval_hml3d_official_control_4012.json"
DEFAULT_GT_HML263 = ROOT / "ref_repo/CondMDI/dataset/HumanML3D/new_joint_vecs"
DEFAULT_PROJFLOW_REPO = DEFAULT_MOTIUS / "ref_repo/ProjFlow"
DEFAULT_PROJFLOW_ARTIFACT = (
    DEFAULT_MOTIUS / "outputs/checkpoints/projflow-official"
)

MASKCONTROL_JOINTS = frozenset((0, 10, 11, 15, 20, 21))


def parse_setting(setting: str) -> tuple[list[int], bool, str, str]:
    """Return joint ids, dense flag, axes, and target name for an E17 setting."""
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from scripts.eval.eval_table6_local_position import TARGET_JOINTS
    from hftrainer.evaluation.motion.m2m_eval_metrics import JOINT_NAME_TO_IDX

    name = setting.removeprefix("E17_")
    axes = name.rsplit("_", 1)[-1]
    if axes not in {"xz", "xyz"}:
        raise ValueError(f"setting must end in _xz or _xyz: {setting}")
    stem = name.removesuffix(f"_{axes}")
    if stem.endswith("_dense"):
        dense = True
        target = stem.removesuffix("_dense")
    elif stem.endswith("_sparse"):
        dense = False
        target = stem.removesuffix("_sparse")
    else:
        raise ValueError(f"setting must contain _sparse or _dense: {setting}")
    if target not in TARGET_JOINTS:
        raise ValueError(f"unknown body-part target {target!r}")
    joints = [int(JOINT_NAME_TO_IDX[name]) for name in TARGET_JOINTS[target]]
    return joints, dense, axes, target


def protocol_status(
    method: str,
    setting: str,
    *,
    allow_maskcontrol_extra_axes: bool = False,
) -> dict:
    joints, dense, axes, target = parse_setting(setting)
    if method in {"condmdi", "omnicontrol", "projflow"}:
        if method == "condmdi" and axes != "xyz":
            raise ValueError(
                "CondMDI exposes joint-level XYZ position atoms and cannot "
                f"represent the native {axes.upper()}-only protocol"
            )
        return {
            "status": "native",
            "requested_axes": axes,
            "provided_axes": axes,
            "joints": joints,
            "target": target,
            "density": "dense" if dense else "sparse",
        }
    if method != "maskcontrol":
        raise ValueError(f"unknown method {method!r}")
    unsupported = sorted(set(joints) - MASKCONTROL_JOINTS)
    if unsupported:
        raise ValueError(
            "released MaskControl weights do not support the requested exact "
            f"joint set; unsupported joint ids={unsupported}, supported="
            f"{sorted(MASKCONTROL_JOINTS)}"
        )
    if axes != "xyz" and not allow_maskcontrol_extra_axes:
        raise ValueError(
            "MaskControl exposes a joint-level XYZ mask and cannot represent "
            f"the native {axes.upper()}-only protocol. Pass "
            "--allow-maskcontrol-extra-axes only for an explicitly footnoted "
            "extra-evidence diagnostic."
        )
    return {
        "status": "native" if axes == "xyz" else "extra_axis_evidence",
        "requested_axes": axes,
        "provided_axes": "xyz",
        "joints": joints,
        "target": target,
        "density": "dense" if dense else "sparse",
    }


def sparse_frames(length: int, fps: int = 20) -> list[int]:
    """One-second support matching every 30 frames in the 30-fps paper grid."""
    if length <= 0:
        return []
    frames = list(range(0, length, fps))
    if frames[-1] != length - 1:
        frames.append(length - 1)
    return frames


def read_records(path: Path) -> list[tuple[str, dict]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    records = raw.get("data_list", raw) if isinstance(raw, dict) else raw
    if isinstance(records, dict):
        return [(str(key), value) for key, value in records.items()]
    output = []
    for index, value in enumerate(records):
        motion_id = str(value.get("motion_id") or value.get("id") or index)
        output.append((motion_id, value))
    return output


def caption(record: dict) -> str:
    for key in ("caption_en", "caption", "selected_caption", "text"):
        value = record.get(key)
        if isinstance(value, str):
            return value
    return ""


def recover_joints(motion: np.ndarray) -> np.ndarray:
    from motius.motion.representation.humanml import recover_from_ric

    value = torch.from_numpy(np.asarray(motion, dtype=np.float32))[None]
    return recover_from_ric(value, 22)[0].cpu().numpy().astype(np.float32)


def validate_prediction(
    motion_id: str,
    value: np.ndarray,
    length: int,
    *,
    exact_length: bool = True,
) -> np.ndarray:
    value = np.asarray(value, dtype=np.float32)[:length]
    expected = (length, 263)
    if value.ndim != 2 or value.shape[1] != 263:
        raise RuntimeError(f"{motion_id}: invalid prediction shape {value.shape}")
    if exact_length and value.shape != expected:
        raise RuntimeError(
            f"{motion_id}: invalid prediction shape {value.shape}, expected {expected}"
        )
    if not np.isfinite(value).all():
        raise RuntimeError(f"{motion_id}: prediction contains non-finite values")
    return value


def batched(values: list, batch_size: int) -> Iterable[list]:
    for start in range(0, len(values), batch_size):
        yield values[start : start + batch_size]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--method",
        required=True,
        choices=("condmdi", "maskcontrol", "omnicontrol", "projflow"),
    )
    parser.add_argument("--setting", required=True)
    parser.add_argument("--data-file", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--gt-hml263-dir", type=Path, default=DEFAULT_GT_HML263)
    parser.add_argument("--motius-root", type=Path, default=DEFAULT_MOTIUS)
    parser.add_argument("--artifact")
    parser.add_argument("--projflow-repo", type=Path, default=DEFAULT_PROJFLOW_REPO)
    parser.add_argument("--projflow-num-steps", type=int, default=100)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-maskcontrol-extra-axes", action="store_true")
    parser.add_argument(
        "--maskcontrol-profile",
        choices=("paper", "fast"),
        default="paper",
        help="paper=100 optimization iterations per step plus 600 final",
    )
    args = parser.parse_args()
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        parser.error("require 0 <= shard-index < num-shards")
    if args.projflow_num_steps < 1:
        parser.error("--projflow-num-steps must be positive")
    return args


def main() -> None:
    args = parse_args()
    data_file = args.data_file.resolve()
    gt_dir = args.gt_hml263_dir.resolve()
    motius_root = args.motius_root.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    contract = protocol_status(
        args.method,
        args.setting,
        allow_maskcontrol_extra_axes=args.allow_maskcontrol_extra_axes,
    )
    joints = contract["joints"]
    dense = contract["density"] == "dense"

    records = read_records(data_file)
    records = [(motion_id, record) for motion_id, record in records if (gt_dir / f"{motion_id}.npy").is_file()]
    if args.max_samples:
        records = records[: args.max_samples]
    total_records = len(records)
    records = records[args.shard_index :: args.num_shards]
    print(
        f"[setup] method={args.method} setting={args.setting} protocol={contract['status']} "
        f"shard={args.shard_index}/{args.num_shards} cases={len(records)}/{total_records}",
        flush=True,
    )
    if args.dry_run:
        print(json.dumps(contract, indent=2), flush=True)
        return

    if str(motius_root) not in sys.path:
        sys.path.insert(0, str(motius_root))
    if args.method == "condmdi":
        from motius.pipelines.condmdi import CondMDIPipeline

        artifact = args.artifact or str(
            motius_root / "checkpoints/condmdi/motius_humanml3d"
        )
        local_clip = motius_root / "checkpoints/condmdi/clip/ViT-B-32.pt"
        if local_clip.is_file():
            os.environ.setdefault("MOTIUS_CLIP_PATH", str(local_clip))
        pipeline = CondMDIPipeline.from_pretrained(
            artifact,
            bundle_kwargs={"guidance_param": 2.5, "respacing": "ddim100"},
            device=args.device,
        )
        batch_size = args.batch_size or 16
        each_iterations = final_iterations = None
    elif args.method == "maskcontrol":
        from motius.pipelines.maskcontrol import MaskControlPipeline

        artifact = args.artifact or str(motius_root / "outputs/checkpoints/maskcontrol-humanml3d")
        pipeline = MaskControlPipeline.from_pretrained(
            artifact, bundle_kwargs={"device": args.device}, device=args.device
        )
        batch_size = args.batch_size or 1
        each_iterations, final_iterations = (
            (100, 600) if args.maskcontrol_profile == "paper" else (0, 100)
        )
    elif args.method == "omnicontrol":
        from motius.pipelines.omnicontrol import OmniControlPipeline

        artifact = args.artifact or str(
            motius_root / "checkpoints/omnicontrol/extracted/omnicontrol_ckpt/model_humanml3d.pt"
        )
        pipeline = OmniControlPipeline.from_pretrained(artifact, device=args.device)
        batch_size = args.batch_size or 8
        each_iterations = final_iterations = None
    else:
        from motius.pipelines.projflow import ProjFlowPipeline

        artifact = args.artifact or str(DEFAULT_PROJFLOW_ARTIFACT)
        pipeline = ProjFlowPipeline.from_pretrained(
            artifact,
            bundle_kwargs={"repo_path": str(args.projflow_repo.resolve())},
            device=args.device,
        )
        batch_size = args.batch_size or 4
        each_iterations = final_iterations = None

    written = skipped = failed = 0
    started = time.time()
    for chunk in batched(records, max(1, batch_size)):
        todo = [
            (motion_id, record)
            for motion_id, record in chunk
            if not (args.skip_existing and (out_dir / f"{motion_id}.npy").is_file())
        ]
        skipped += len(chunk) - len(todo)
        if not todo:
            continue
        ids = [item[0] for item in todo]
        motions = [np.load(gt_dir / f"{motion_id}.npy").astype(np.float32) for motion_id in ids]
        lengths = [min(len(motion), 196) for motion in motions]
        motions = [motion[:length] for motion, length in zip(motions, lengths)]
        prompts = [caption(record) for _, record in todo]
        try:
            batch_seed = args.seed + sum(ord(ch) for ch in ids[0])
            if args.method == "condmdi":
                control_lengths = [pipeline.clamp_length(length) for length in lengths]
                control_motions = [
                    motion[:control_length]
                    for motion, control_length in zip(motions, control_lengths)
                ]
                keys = None if dense else [
                    sparse_frames(length) for length in control_lengths
                ]
                predictions = pipeline.infer_control(
                    prompts,
                    control_motions,
                    lengths=control_lengths,
                    control_mode="joints",
                    feature_mode="pos",
                    joint_indices=joints,
                    keyframe_indices=keys,
                    seed=batch_seed,
                )
            elif args.method == "omnicontrol":
                keys = None if dense else [sparse_frames(length) for length in lengths]
                predictions = pipeline.infer_control(
                    prompts,
                    motions,
                    lengths=lengths,
                    control_mode="dense" if dense else "keyframes",
                    joint_indices=joints,
                    axes=contract["provided_axes"],
                    keyframe_indices=keys,
                    seed=batch_seed,
                )
            elif args.method == "projflow":
                max_length = max(lengths)
                position_mask = np.zeros(
                    (len(todo), max_length, 22, 3), dtype=bool
                )
                axis_ids = [
                    {"x": 0, "y": 1, "z": 2}[axis]
                    for axis in contract["provided_axes"]
                ]
                for index, length in enumerate(lengths):
                    frames = list(range(length)) if dense else sparse_frames(length)
                    position_mask[index][np.ix_(frames, joints, axis_ids)] = True
                predictions = pipeline.infer_control(
                    prompts,
                    motions,
                    lengths=lengths,
                    position_mask=position_mask,
                    num_steps=args.projflow_num_steps,
                    seed=batch_seed,
                    return_format="joints",
                )
            else:
                max_length = max(lengths)
                targets = np.zeros((len(todo), max_length, 22, 3), dtype=np.float32)
                target_mask = np.zeros((len(todo), max_length, 22), dtype=bool)
                for index, (motion, length) in enumerate(zip(motions, lengths)):
                    target_joints = recover_joints(motion)
                    frames = list(range(length)) if dense else sparse_frames(length)
                    targets[index, :length] = target_joints[:length]
                    target_mask[index][np.ix_(frames, joints)] = True
                predictions = pipeline.infer_control(
                    prompts,
                    lengths,
                    targets,
                    target_mask,
                    seed=batch_seed,
                    each_iterations=each_iterations,
                    final_iterations=final_iterations,
                )
            for motion_id, length, prediction in zip(ids, lengths, predictions):
                if args.method == "projflow":
                    value = np.asarray(prediction, dtype=np.float32)[:length]
                    if value.shape != (length, 22, 3) or not np.isfinite(value).all():
                        raise RuntimeError(
                            f"{motion_id}: invalid 22-joint prediction {value.shape}"
                        )
                else:
                    value = validate_prediction(
                        motion_id,
                        prediction,
                        length,
                        exact_length=args.method != "condmdi",
                    )
                np.save(
                    out_dir / f"{motion_id}.npy",
                    value,
                )
                written += 1
        except Exception as exc:  # noqa: BLE001
            failed += len(todo)
            print(f"[failed] ids={ids} error={type(exc).__name__}: {exc}", flush=True)
            traceback.print_exc()
        print(
            f"[progress] written={written} skipped={skipped} failed={failed} "
            f"elapsed={time.time() - started:.1f}s",
            flush=True,
        )

    summary = {
        "method": args.method,
        "setting": args.setting,
        "protocol": contract,
        "artifact": str(artifact),
        "data_file": str(data_file),
        "gt_hml263_dir": str(gt_dir),
        "official_total": total_records,
        "shard": args.shard_index,
        "num_shards": args.num_shards,
        "assigned": len(records),
        "written": written,
        "skipped": skipped,
        "failed": failed,
        "elapsed_seconds": time.time() - started,
    }
    summary_dir = out_dir / "_generation"
    summary_dir.mkdir(exist_ok=True)
    (summary_dir / f"shard_{args.shard_index:03d}.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
