#!/usr/bin/env python3
"""KIMODO-SMPLX prefix-conditioned generation on HumanML3D.

This is the TP2M counterpart of ``gen_kimodo_t2m_positions.py``.  It uses the
SMPL-X KIMODO checkpoint, builds full-body prefix constraints from official
HumanML3D-272 ground truth converted to ``motion_135``, and writes native
KIMODO debug NPZs.  Downstream evaluation can then reuse
``kimodo_smplx_to_motion135.py`` and the existing MotionStreamer-272 bridge.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch


REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(REPO / "scripts/eval") not in sys.path:
    sys.path.insert(0, str(REPO / "scripts/eval"))

from h3d_272_to_135 import humanml272_to_motion135  # noqa: E402
from hftrainer.motion.skeleton.fk import rot6d_to_rotmat_row_major  # noqa: E402
from scripts.eval.gen_kimodo_t2m_positions import (  # noqa: E402
    CacheOnlyTextEncoder,
    _fit_length,
    _load_corpus_jobs,
    _load_humanml3d_jobs,
    _load_kimodo_model,
    _resample_nearest,
    _resample_positions,
    _select_shard,
    _split_num_frames,
    _to_numpy_sequence,
)


DEFAULT_KIMODO_ARTIFACT = REPO / "checkpoints/kimodo/hftrainer_smplx_rp"
DEFAULT_KIMODO_MODEL_NAME = "Kimodo-SMPLX-RP-v1"
DIFFUSION_STEPS = 100


def _rotmat_to_rot6d_row_major_np(rotmat: np.ndarray) -> np.ndarray:
    rotmat = np.asarray(rotmat, dtype=np.float32)
    col6d = np.concatenate([rotmat[..., 0:3, 0], rotmat[..., 0:3, 1]], axis=-1)
    return col6d[..., [0, 3, 1, 4, 2, 5]]


def _motion135_to_local_root(motion_135: np.ndarray, device: torch.device):
    motion = torch.from_numpy(np.asarray(motion_135, dtype=np.float32)).to(device)
    root = motion[:, :3]
    rot6d = motion[:, 3:135].reshape(len(motion), 22, 6)
    local = rot6d_to_rotmat_row_major(rot6d)
    return local, root


def _make_prefix_constraint(model, gt_motion_135: np.ndarray, cond_frames: int):
    from hftrainer.models.motion.kimodo.network.constraints import FullBodyConstraintSet

    device = torch.device(model.device)
    skeleton = model.skeleton.to(device)
    local, root = _motion135_to_local_root(gt_motion_135, device)
    global_rots, positions, _ = skeleton.fk(local, root)

    k = max(1, min(int(cond_frames), int(len(gt_motion_135))))
    frame_idx = torch.arange(k, device=device, dtype=torch.long)
    smooth_root_2d = positions[frame_idx, skeleton.root_idx, :][:, [0, 2]]
    constraint = FullBodyConstraintSet(
        skeleton,
        frame_indices=frame_idx,
        global_joints_positions=positions[frame_idx],
        global_joints_rots=global_rots[frame_idx],
        smooth_root_2d=smooth_root_2d,
        to_crop=False,
    )
    return [constraint], local.detach().cpu().numpy(), root.detach().cpu().numpy()


def _recompute_debug_arrays(model, local_np: np.ndarray, root_np: np.ndarray):
    device = torch.device(model.device)
    skeleton = model.skeleton.to(device)
    local = torch.from_numpy(np.asarray(local_np, dtype=np.float32)).to(device)
    root = torch.from_numpy(np.asarray(root_np, dtype=np.float32)).to(device)
    global_rots, posed, _ = skeleton.fk(local, root)
    return {
        "local_rot_mats": local.detach().cpu().numpy().astype(np.float32),
        "global_rot_mats": global_rots.detach().cpu().numpy().astype(np.float32),
        "root_positions": root.detach().cpu().numpy().astype(np.float32),
        "posed_joints": posed.detach().cpu().numpy().astype(np.float32),
    }


def _run_one(
    model,
    caption: str,
    gt_motion_135: np.ndarray,
    target_fps: float,
    cond_frames: int,
    postprocess: bool,
    *,
    force_clean_prefix: bool = True,
    force_single_segment: bool = False,
    max_segment_frames: int | None = None,
):
    num_frames_30 = int(len(gt_motion_135))
    model_fps = float(model.fps)
    model_frames = max(10, int(round(num_frames_30 * model_fps / target_fps)))

    constraints, gt_local, gt_root = _make_prefix_constraint(model, gt_motion_135, cond_frames)
    seg_lens = (
        [model_frames]
        if force_single_segment
        else _split_num_frames(model_frames, safe_len=max_segment_frames)
    )
    is_multi = len(seg_lens) > 1
    prompts = [caption] * len(seg_lens)
    constraint_arg = constraints if is_multi else [constraints]

    output = model(
        prompts,
        seg_lens,
        num_denoising_steps=DIFFUSION_STEPS,
        cfg_weight=[2.0, 2.0],
        num_samples=1,
        return_numpy=True,
        multi_prompt=is_multi,
        constraint_lst=constraint_arg,
        post_processing=postprocess,
    )
    local = _to_numpy_sequence(output.get("local_rot_mats"))
    global_rot = _to_numpy_sequence(output.get("global_rot_mats"))
    root = _to_numpy_sequence(output.get("root_positions"))
    posed = _to_numpy_sequence(output.get("posed_joints"))
    if local is None:
        raise KeyError("KIMODO output has no local_rot_mats")
    if root is None:
        if posed is None:
            raise KeyError("KIMODO output has neither root_positions nor posed_joints")
        root = posed[:, 0]

    if abs(model_fps - target_fps) > 1e-6:
        local = _resample_nearest(local, num_frames_30)
        root = _resample_positions(root[:, None, :], num_frames_30)[:, 0]
        if global_rot is not None:
            global_rot = _resample_nearest(global_rot, num_frames_30)
        if posed is not None:
            posed = _resample_positions(posed, num_frames_30)

    local = _fit_length(local, num_frames_30)
    root = _fit_length(root, num_frames_30)

    k = max(1, min(int(cond_frames), num_frames_30))
    if force_clean_prefix:
        local[:k] = gt_local[:k]
        root[:k] = gt_root[:k]
        payload = _recompute_debug_arrays(model, local, root)
    else:
        payload = {
            "local_rot_mats": local.astype(np.float32),
            "root_positions": root.astype(np.float32),
        }
        if global_rot is not None:
            payload["global_rot_mats"] = _fit_length(global_rot, num_frames_30).astype(np.float32)
        if posed is not None:
            payload["posed_joints"] = _fit_length(posed, num_frames_30).astype(np.float32)
        else:
            payload.update(_recompute_debug_arrays(model, local, root))

    payload["motion_135"] = np.concatenate(
        [
            payload["root_positions"],
            _rotmat_to_rot6d_row_major_np(payload["local_rot_mats"]).reshape(num_frames_30, 132),
        ],
        axis=1,
    ).astype(np.float32)
    return payload


def _load_gt_motion(gt_dir: Path, sid: str) -> np.ndarray:
    path = gt_dir / f"{sid}.npy"
    if not path.exists():
        raise FileNotFoundError(path)
    return humanml272_to_motion135(np.load(str(path)).astype(np.float32))


def _content_hash(*parts: str) -> str:
    h = hashlib.sha1()
    for part in parts:
        h.update(part.encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()[:12]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--humanml3d-272", default="ref_repo/MotionStreamer/MotionStreamer/humanml3d_272")
    parser.add_argument("--gt-dir", default=None)
    parser.add_argument("--model-path", default=str(DEFAULT_KIMODO_ARTIFACT))
    parser.add_argument("--model-name", default=DEFAULT_KIMODO_MODEL_NAME)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--debug-npz-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--condition-frames", type=int, required=True)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--min-len", type=int, default=1)
    parser.add_argument("--max-len", type=int, default=100000)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--diffusion-steps", type=int, default=100)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--postprocess", action="store_true")
    parser.add_argument("--no-force-clean-prefix", action="store_true")
    parser.add_argument("--corpus", default=None, help="Optional JSONL prompt bank with id/prompt/length.")
    parser.add_argument("--write-corpus", default=None)
    parser.add_argument("--force-single-segment", action="store_true")
    parser.add_argument("--max-segment-frames", type=int, default=None)
    parser.add_argument("--text-feature-cache-dir", default=None)
    parser.add_argument("--text-feature-namespace", default=None)
    parser.add_argument("--text-feature-encoder-id", default="LLM2VecEncoder")
    args = parser.parse_args()

    if not (0 <= args.shard_index < args.num_shards):
        raise ValueError("--shard-index must be in [0, num_shards)")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    h3d = Path(args.humanml3d_272)
    gt_dir = Path(args.gt_dir) if args.gt_dir else h3d / "motion_data"
    out = Path(args.out_dir)
    debug = Path(args.debug_npz_dir)
    out.mkdir(parents=True, exist_ok=True)
    debug.mkdir(parents=True, exist_ok=True)

    if args.corpus:
        all_jobs = _load_corpus_jobs(Path(args.corpus), args.min_len, args.max_len, args.max_samples)
    else:
        all_jobs = _load_humanml3d_jobs(h3d, args.min_len, args.max_len, args.max_samples)
    if args.write_corpus:
        Path(args.write_corpus).parent.mkdir(parents=True, exist_ok=True)
        with Path(args.write_corpus).open("w", encoding="utf-8") as f:
            for sid, caption, length in all_jobs:
                f.write(json.dumps({"id": sid, "prompt": caption, "length": length}, ensure_ascii=False) + "\n")
    jobs: List[Tuple[str, str, int]] = _select_shard(all_jobs, args.num_shards, args.shard_index)
    print(
        f"[setup] total={len(all_jobs)} shard={args.shard_index}/{args.num_shards} "
        f"jobs={len(jobs)} cond={args.condition_frames}",
        flush=True,
    )

    use_feature_cache = bool(args.text_feature_cache_dir and args.text_feature_namespace)
    if use_feature_cache:
        os.environ["TEXT_ENCODER"] = "dummy"
        os.environ["TEXT_ENCODER_MODE"] = "local"

    global DIFFUSION_STEPS
    DIFFUSION_STEPS = int(args.diffusion_steps)
    model, bundle, kimodo_file = _load_kimodo_model(args, use_feature_cache=use_feature_cache)
    if use_feature_cache:
        model.text_encoder = CacheOnlyTextEncoder(
            namespace=args.text_feature_namespace,
            cache_dir=args.text_feature_cache_dir,
            encoder_id=args.text_feature_encoder_id,
            device=args.device,
        )
        print(
            "[setup] using cached text features "
            f"{Path(args.text_feature_cache_dir) / args.text_feature_namespace}",
            flush=True,
        )
    skeleton_type = type(model.skeleton).__name__
    print(
        f"[setup] KIMODO loaded fps={model.fps} skeleton={skeleton_type} "
        f"source={args.model_path or args.model_name}",
        flush=True,
    )
    if "SMPL" not in skeleton_type.upper():
        raise RuntimeError(f"expected SMPLX KIMODO skeleton, got {skeleton_type}")
    print(f"[setup] kimodo_import={kimodo_file}", flush=True)

    manifest = out / f"manifest_cond{args.condition_frames}_shard{args.shard_index}of{args.num_shards}.jsonl"
    ok = skipped = failed = 0
    with manifest.open("w", encoding="utf-8") as mf:
        for i, (sid, caption, length) in enumerate(jobs):
            out_file = out / f"{sid}.npy"
            debug_file = debug / f"{sid}.npz"
            if args.skip_existing and out_file.exists() and debug_file.exists():
                skipped += 1
                continue
            try:
                gt_motion_135 = _load_gt_motion(gt_dir, sid)
                if int(len(gt_motion_135)) != int(length):
                    raise ValueError(f"GT length mismatch corpus={length} gt={len(gt_motion_135)}")
                payload = _run_one(
                    model,
                    caption,
                    gt_motion_135,
                    args.fps,
                    args.condition_frames,
                    args.postprocess,
                    force_clean_prefix=not args.no_force_clean_prefix,
                    force_single_segment=args.force_single_segment,
                    max_segment_frames=args.max_segment_frames,
                )
                pos = payload["posed_joints"][:, :22]
                if not np.isfinite(pos).all() or pos.shape != (length, 22, 3):
                    raise ValueError(f"bad posed_joints shape/range: {pos.shape}")
                np.save(str(out_file), pos.astype(np.float32))
                np.savez_compressed(
                    str(debug_file),
                    **payload,
                    caption=np.array(caption, dtype=object),
                    sample_id=np.array(sid, dtype=object),
                    target_length=np.array(length, dtype=np.int32),
                    condition_frames=np.array(args.condition_frames, dtype=np.int32),
                    forced_clean_prefix=np.array(not args.no_force_clean_prefix, dtype=np.bool_),
                )
                mf.write(json.dumps({
                    "sample_id": sid,
                    "caption": caption,
                    "caption_hash": _content_hash(sid, caption),
                    "target_length": int(length),
                    "condition_frames": int(args.condition_frames),
                    "path": str(out_file),
                    "debug_npz": str(debug_file),
                    "forced_clean_prefix": not args.no_force_clean_prefix,
                    "text_feature_namespace": args.text_feature_namespace,
                    "model_path": args.model_path,
                    "model_name": getattr(bundle, "resolved_model_name", args.model_name),
                    "skeleton_type": skeleton_type,
                    "kimodo_import": kimodo_file,
                }, ensure_ascii=False) + "\n")
                mf.flush()
                ok += 1
            except Exception as exc:  # noqa: BLE001
                failed += 1
                print(f"[fail] {sid}: {type(exc).__name__}: {exc}", flush=True)
            if (i + 1) % 25 == 0 or (i + 1) == len(jobs):
                print(f"[progress] {i+1}/{len(jobs)} ok={ok} skipped={skipped} failed={failed}", flush=True)

    summary = {
        "all_jobs": len(all_jobs),
        "jobs": len(jobs),
        "ok": ok,
        "skipped": skipped,
        "failed": failed,
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "condition_frames": args.condition_frames,
        "model_path": args.model_path,
        "model_name": getattr(bundle, "resolved_model_name", args.model_name),
        "skeleton_type": skeleton_type,
        "kimodo_import": kimodo_file,
        "forced_clean_prefix": not args.no_force_clean_prefix,
    }
    (out / f"summary_cond{args.condition_frames}_shard{args.shard_index}of{args.num_shards}.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print("[done] " + json.dumps(summary), flush=True)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
