#!/usr/bin/env python3
"""Debug PRISM first-frame orientation failures on BABEL sequential T2M.

The runner is intentionally narrow: it reruns one BABEL episode with optional
GT prefix conditioning and dumps orientation diagnostics before and after
PRISM post-processing.  It is meant to distinguish protocol/prefix problems
from 6D rotation decoding convention bugs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from einops import rearrange

SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(SCRIPT_DIR))

from babel_caption import rewrite_caption  # noqa: E402
from eval_prism_kafs_ablation import load_prism_bundle, save_smplx_npz  # noqa: E402
from hftrainer.datasets.motion.representation.humanml_repr import fk_smplh_joints  # noqa: E402
from hftrainer.motion.representation.rotation import (  # noqa: E402
    axis_angle_to_matrix,
    matrix_to_axis_angle,
    rotation_6d_to_axis_angle,
    rotation_6d_to_matrix,
)
from hftrainer.pipelines.motion.prism_pipeline import PrismPipeline  # noqa: E402


DEFAULT_ROOT = (
    REPO / "outputs/evaluation/babel/official_val/msstyle_30fps_gt"
)


def _read_manifest(path: Path) -> dict[str, dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return {str(row["id"]): row for row in rows}


def _motion135_to_smplx_npz(motion_135: np.ndarray, out_path: Path) -> Path:
    motion_135 = np.asarray(motion_135, dtype=np.float32)
    transl = motion_135[:, :3].astype(np.float32)
    rot6d = motion_135[:, 3:].reshape(len(motion_135), 22, 6)
    rot = rotation_6d_to_matrix(torch.from_numpy(rot6d), convention="row")
    aa = matrix_to_axis_angle(rot).numpy().astype(np.float32).reshape(len(motion_135), 22, 3)
    go = aa[:, 0]
    bp = aa[:, 1:].reshape(len(motion_135), 63)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        transl=transl,
        trans=transl,
        global_orient=go,
        body_pose=bp,
        poses=np.concatenate(
            [
                go,
                bp,
                np.zeros((len(motion_135), 99), dtype=np.float32),
            ],
            axis=1,
        ),
        jaw_pose=np.zeros((len(motion_135), 3), dtype=np.float32),
        leye_pose=np.zeros((len(motion_135), 3), dtype=np.float32),
        reye_pose=np.zeros((len(motion_135), 3), dtype=np.float32),
        left_hand_pose=np.zeros((len(motion_135), 45), dtype=np.float32),
        right_hand_pose=np.zeros((len(motion_135), 45), dtype=np.float32),
        betas=np.zeros((10,), dtype=np.float32),
        expression=np.zeros((len(motion_135), 10), dtype=np.float32),
        gender=np.array("neutral"),
        mocap_framerate=np.array(30.0, dtype=np.float32),
    )
    return out_path


def _joints_from_axis_angle(global_orient: np.ndarray, body_pose: np.ndarray, transl: np.ndarray) -> np.ndarray:
    poses = np.concatenate([global_orient, body_pose], axis=-1).reshape(len(transl), 22, 3)
    rot = axis_angle_to_matrix(torch.from_numpy(poses.astype(np.float32)).reshape(-1, 3))
    rot = rot.reshape(len(transl), 22, 3, 3).numpy()
    return fk_smplh_joints(rot, transl.astype(np.float32))


def _metrics_from_smplx_dict(smplx_dict: dict[str, Any]) -> dict[str, float]:
    transl = np.asarray(smplx_dict.get("transl", smplx_dict.get("trans")), dtype=np.float32)
    go = np.asarray(smplx_dict["global_orient"], dtype=np.float32)
    bp = np.asarray(smplx_dict["body_pose"], dtype=np.float32)
    joints = _joints_from_axis_angle(go[:1], bp[:1], transl[:1])[0]
    across = (joints[2] - joints[1]) + (joints[17] - joints[16])
    forward = np.cross(np.array([0.0, 1.0, 0.0], dtype=np.float32), across)
    yaw = float(np.degrees(np.arctan2(forward[0], forward[2])))
    return {
        "root_y": float(joints[0, 1]),
        "head_y": float(joints[15, 1]),
        "head_root_y": float(joints[15, 1] - joints[0, 1]),
        "min_y": float(joints[:, 1].min()),
        "max_y": float(joints[:, 1].max()),
        "body_yaw_deg": yaw,
    }


def _metrics_from_npz(path: Path) -> dict[str, float]:
    z = dict(np.load(path, allow_pickle=True))
    return _metrics_from_smplx_dict(z)


def _decode_metrics(pipe, x_dec: torch.Tensor, convention: str, use_rollout_trans: bool) -> dict[str, float]:
    smp = pipe.backend.smpl_processor
    flat = rearrange(x_dec, "b t j d -> b t (j d)")
    flat = smp.denormalize(flat)
    transl_abs_rel = flat[..., :6]
    transl = smp.inv_convert_transl(transl_abs_rel, use_rollout=use_rollout_trans)
    poses6 = rearrange(flat[..., 6:], "b t (j d) -> (b t) j d", d=6)
    poses_aa = rotation_6d_to_axis_angle(poses6, convention=convention)
    poses_aa = rearrange(poses_aa, "(b t) j d -> b t (j d)", b=1)
    smplx = smp.transl_pose_to_smplx_dict(
        transl.squeeze(0),
        poses_aa.squeeze(0),
        mocap_framerate=30.0,
        gender="neutral",
        rot_type="axis_angle",
    )
    return _metrics_from_smplx_dict(smplx)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sid", default="val_3665")
    ap.add_argument("--root", default=str(DEFAULT_ROOT))
    ap.add_argument("--manifest", default="")
    ap.add_argument("--config", default="configs/prism/prism_1b_tp2m_multiframe_kt_spectral_unified.py")
    ap.add_argument("--checkpoint", default="work_dirs/prism_1b_tp2m_multiframe_kt_spectral_unified_t5cached/checkpoint-epoch_43")
    ap.add_argument("--out-dir", default=".codex_tmp/prism_babel_orientation_debug")
    ap.add_argument("--conditions", default="none,gt1,gt5")
    ap.add_argument("--num-inference-steps", type=int, default=50)
    ap.add_argument("--guidance-scale", type=float, default=5.0)
    ap.add_argument("--ar-cond-frames", type=int, default=5)
    ap.add_argument("--use-rollout-trans", action="store_true", default=True)
    ap.add_argument("--absolute-trans", dest="use_rollout_trans", action="store_false")
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    manifest = Path(args.manifest) if args.manifest else root / "manifest.jsonl"
    rows = _read_manifest(manifest)
    if args.sid not in rows:
        raise KeyError(f"{args.sid} not found in {manifest}")
    rec = rows[args.sid]

    out_root = Path(args.out_dir) / args.sid
    out_root.mkdir(parents=True, exist_ok=True)

    gt_mesh = root / "mesh135_evalcanon" / "GT" / f"{args.sid}.npz"
    prefix_npz = out_root / f"{args.sid}_gt_prefix_from_mesh135.npz"
    if gt_mesh.exists():
        gt_motion = np.load(gt_mesh)["motion_135"]
        _motion135_to_smplx_npz(gt_motion, prefix_npz)
    else:
        raise FileNotFoundError(f"missing GT mesh135 prefix source: {gt_mesh}")

    prompts = []
    seg_lens = []
    for seg in rec.get("segments", []):
        cap = str(seg.get("caption") or "").strip()
        prompts.append(rewrite_caption(cap))
        seg_lens.append(max(1, int(seg["end"]) - int(seg["start"])))

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bundle = load_prism_bundle(args.config, args.checkpoint, device)
    pipe = PrismPipeline(bundle=bundle)
    pipe.backend.set_kafs_alpha(mode="none")

    summary: dict[str, Any] = {
        "sid": args.sid,
        "manifest": str(manifest),
        "config": args.config,
        "checkpoint": args.checkpoint,
        "seed": int(args.seed),
        "prompts": prompts,
        "segment_lengths": seg_lens,
        "gt_prefix_npz": str(prefix_npz),
        "existing": {},
        "runs": {},
    }
    for name, path in {
        "gt_prefix": prefix_npz,
        "existing_prism": root / "prism_gen_segfix" / f"{args.sid}.npz",
    }.items():
        if path.exists():
            summary["existing"][name] = _metrics_from_npz(path)

    orig_post = pipe.backend.post_process_motion
    current_mode = {"name": ""}

    def wrapped_post_process(x_dec: torch.Tensor, *a, **kw):
        mode = current_mode["name"]
        diag = {
            "decoded_as_column_before_normalize": _decode_metrics(
                pipe, x_dec, "column", bool(kw.get("use_rollout_trans", args.use_rollout_trans))
            ),
            "decoded_as_row_before_normalize": _decode_metrics(
                pipe, x_dec, "row", bool(kw.get("use_rollout_trans", args.use_rollout_trans))
            ),
        }
        (out_root / f"{mode}_decode_diag.json").write_text(
            json.dumps(diag, indent=2, ensure_ascii=False) + "\n"
        )
        return orig_post(x_dec, *a, **kw)

    pipe.backend.post_process_motion = wrapped_post_process

    for cond in [c.strip() for c in args.conditions.split(",") if c.strip()]:
        if cond == "none":
            first_frame_motion_path = None
            condition_num_frames = 1
        elif cond.startswith("gt"):
            first_frame_motion_path = str(prefix_npz)
            condition_num_frames = int(cond[2:] or "1")
        else:
            raise ValueError(f"unknown condition: {cond}")

        current_mode["name"] = cond
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        smplx = pipe(
            prompts=prompts,
            first_frame_motion_path=first_frame_motion_path,
            condition_num_frames=condition_num_frames,
            num_frames_per_segment=seg_lens,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            ar_condition_frames=args.ar_cond_frames,
            use_blend=False,
            use_rollout_trans=args.use_rollout_trans,
            preserve_segment_lengths=True,
        )
        out_npz = out_root / f"{cond}.npz"
        save_smplx_npz(str(out_npz), smplx)
        summary["runs"][cond] = {
            "first_frame_motion_path": first_frame_motion_path,
            "condition_num_frames": condition_num_frames,
            "out_npz": str(out_npz),
            "postprocess_output": _metrics_from_smplx_dict(smplx),
            "decode_diag": json.loads((out_root / f"{cond}_decode_diag.json").read_text()),
        }
        print(f"[{cond}] {json.dumps(summary['runs'][cond]['postprocess_output'], sort_keys=True)}", flush=True)

    summary_path = out_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(f"[done] {summary_path}", flush=True)


if __name__ == "__main__":
    main()
