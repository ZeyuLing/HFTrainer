#!/usr/bin/env python3
"""Diagnose KIMODO keyframe imputation continuity without post-processing.

This script reruns one full-body keyframe sample with the same constraints under
different sampling clamp modes. It intentionally keeps ``post_processing=False``
so the measurement isolates the denoising/imputation path.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
KIMODO_ROOT = PROJECT_ROOT / "ref_repo" / "KIMODO" / "kimodo"
DEFAULT_ARTIFACT = PROJECT_ROOT / "checkpoints" / "kimodo" / "hftrainer_soma_rp"
os.environ.setdefault("HFTRAINER_SKIP_AUTOREGISTER", "1")
os.environ.setdefault("TRANSFORMERS_NO_LIBROSA", "1")
os.environ.setdefault("TRANSFORMERS_NO_AUDIO", "1")
os.environ.setdefault("NUMBA_CACHE_DIR", str(PROJECT_ROOT / "outputs" / ".numba_cache"))
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(KIMODO_ROOT))

from scripts.kimodo.run_kimodo_all_tasks import (  # noqa: E402
    KIMODO_MODEL,
    SMPLX22_TO_SOMA30,
    _make_fullbody_with_rot_constraint_set,
    kimodo_apply_canon,
    kimodo_compute_canon_transform,
    kimodo_invert_canon_positions,
    smpl22_to_soma30_retarget,
    soma77_to_smpl22,
)


def _patch_llm2vec_adapter_configs(text_encoders_dir: str) -> None:
    root = Path(text_encoders_dir)
    base_model = root / "meta-llama" / "Meta-Llama-3-8B-Instruct"
    if not base_model.exists():
        return
    for rel in (
        "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp/adapter_config.json",
        "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised/adapter_config.json",
    ):
        cfg_path = root / rel
        if not cfg_path.exists():
            continue
        cfg = json.loads(cfg_path.read_text())
        cfg["base_model_name_or_path"] = str(base_model.resolve())
        cfg_path.write_text(json.dumps(cfg, indent=2))


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _edge_stats(pred22: np.ndarray, target22: np.ndarray, keyframes: list[int]) -> dict:
    rows = []
    max_root_ratio = 0.0
    max_mean_ratio = 0.0
    for frame in keyframes:
        for a, b in ((frame - 1, frame), (frame, frame + 1)):
            if a < 0 or b >= len(pred22) or b >= len(target22):
                continue
            pred_root = float(np.linalg.norm(pred22[b, 0] - pred22[a, 0]))
            pred_mean = float(np.linalg.norm(pred22[b] - pred22[a], axis=-1).mean())
            target_root = float(np.linalg.norm(target22[b, 0] - target22[a, 0]))
            target_mean = float(np.linalg.norm(target22[b] - target22[a], axis=-1).mean())
            root_ratio = pred_root / max(target_root, 1e-8)
            mean_ratio = pred_mean / max(target_mean, 1e-8)
            max_root_ratio = max(max_root_ratio, root_ratio)
            max_mean_ratio = max(max_mean_ratio, mean_ratio)
            rows.append(
                {
                    "edge": [int(a), int(b)],
                    "keyframe": int(frame),
                    "pred_root": pred_root,
                    "target_root": target_root,
                    "root_ratio": float(root_ratio),
                    "pred_joint_mean": pred_mean,
                    "target_joint_mean": target_mean,
                    "joint_mean_ratio": float(mean_ratio),
                }
            )
    return {
        "max_root_ratio": float(max_root_ratio),
        "max_joint_mean_ratio": float(max_mean_ratio),
        "edges": rows,
    }


def _keyframe_error(pred22: np.ndarray, target22: np.ndarray, keyframes: list[int]) -> dict:
    rows = []
    for frame in keyframes:
        err = np.linalg.norm(pred22[frame] - target22[frame], axis=-1)
        rows.append(
            {
                "frame": int(frame),
                "mean": float(err.mean()),
                "max": float(err.max()),
                "root": float(np.linalg.norm(pred22[frame, 0] - target22[frame, 0])),
            }
        )
    return {
        "mean": float(np.mean([r["mean"] for r in rows])) if rows else 0.0,
        "max": float(max((r["max"] for r in rows), default=0.0)),
        "root_max": float(max((r["root"] for r in rows), default=0.0)),
        "frames": rows,
    }


def _invert_soma77_output(output: dict, r_yaw: torch.Tensor, t_xz: torch.Tensor) -> dict:
    posed = output["posed_joints"]
    if isinstance(posed, torch.Tensor):
        posed = posed.detach().cpu().float()
    else:
        posed = torch.from_numpy(np.asarray(posed)).float()
    if posed.ndim == 4:
        posed = posed[0]
    posed_world = kimodo_invert_canon_positions(posed, r_yaw.cpu(), t_xz.cpu()).numpy()

    result = {"posed_joints": posed_world.astype(np.float32)}
    if "global_rot_mats" in output:
        rots = output["global_rot_mats"]
        if isinstance(rots, torch.Tensor):
            rots = rots.detach().cpu().float()
        else:
            rots = torch.from_numpy(np.asarray(rots)).float()
        if rots.ndim == 5:
            rots = rots[0]
        r_inv = r_yaw.transpose(-1, -2).cpu()
        result["global_rot_mats"] = torch.einsum("ij,tnjk->tnik", r_inv, rots).numpy().astype(np.float32)
    return result


def _run_variant(
    model,
    prompts,
    num_frames: int,
    constraints,
    *,
    seed: int,
    repaint: bool,
    final_paste: bool,
    steps: int,
    cfg_weight,
):
    old_repaint = os.environ.get("KIMODO_REPAINT_CONDITION")
    old_paste = os.environ.get("KIMODO_FINAL_HARD_PASTE")
    os.environ["KIMODO_REPAINT_CONDITION"] = "1" if repaint else "0"
    os.environ["KIMODO_FINAL_HARD_PASTE"] = "1" if final_paste else "0"
    try:
        _set_seed(seed)
        with torch.inference_mode():
            return model(
                prompts,
                num_frames,
                constraint_lst=[constraints],
                num_denoising_steps=steps,
                cfg_weight=cfg_weight,
                num_samples=1,
                return_numpy=False,
                multi_prompt=False,
                post_processing=False,
            )
    finally:
        if old_repaint is None:
            os.environ.pop("KIMODO_REPAINT_CONDITION", None)
        else:
            os.environ["KIMODO_REPAINT_CONDITION"] = old_repaint
        if old_paste is None:
            os.environ.pop("KIMODO_FINAL_HARD_PASTE", None)
        else:
            os.environ["KIMODO_FINAL_HARD_PASTE"] = old_paste


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--npz",
        default=str(PROJECT_ROOT / "output/evaluation/keyframe_viewer/kimodo/E3_adaptive/npz/000000.npz"),
    )
    parser.add_argument("--out-dir", default=str(PROJECT_ROOT / "outputs/debug/kimodo_keyframe_repaint_000000"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--cfg-weight", type=float, nargs="+", default=[2.0, 2.0])
    parser.add_argument(
        "--checkpoint-dir",
        default=str(DEFAULT_ARTIFACT / "kimodo_checkpoint"),
        help="Directory containing Kimodo-SOMA-RP-v1.",
    )
    parser.add_argument(
        "--text-encoder",
        choices=["dummy", "llm2vec"],
        default="dummy",
        help="Use dummy for continuity debugging without loading LLM2Vec.",
    )
    parser.add_argument(
        "--text-encoders-dir",
        default=str(DEFAULT_ARTIFACT / "text_encoders"),
        help="Directory containing local LLM2Vec text encoder components.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = np.load(args.npz, allow_pickle=True)
    gt_motion = np.asarray(data["gt_motion_135"], dtype=np.float32)
    keyframes = [int(x) for x in np.asarray(data["keyframe_indices"]).reshape(-1)]
    caption = str(data["caption"].item() if np.asarray(data["caption"]).shape == () else data["caption"])
    num_frames = int(gt_motion.shape[0])

    bone_offsets = torch.load(PROJECT_ROOT / "data/hymotion_m2m_data/bone_offsets_22.pt", map_location="cpu").numpy()
    soma30_rots, soma30_pos = smpl22_to_soma30_retarget(gt_motion, bone_offsets)

    os.environ.setdefault("LOCAL_CACHE", "true")
    os.environ.setdefault("TEXT_ENCODER_MODE", "local")
    os.environ["TEXT_ENCODER"] = args.text_encoder
    os.environ["CHECKPOINT_DIR"] = str(Path(args.checkpoint_dir).resolve())
    if args.text_encoder == "llm2vec":
        os.environ["TEXT_ENCODERS_DIR"] = str(Path(args.text_encoders_dir).resolve())
        _patch_llm2vec_adapter_configs(os.environ["TEXT_ENCODERS_DIR"])

    from kimodo import load_model

    model = load_model(KIMODO_MODEL, device=args.device)
    skeleton = model.skeleton
    r_yaw, t_xz, heading0 = kimodo_compute_canon_transform(soma30_pos, skeleton, anchor_frame=0)
    soma30_rots_c, soma30_pos_c = kimodo_apply_canon(soma30_rots, soma30_pos, r_yaw, t_xz)

    fullbody_cls = _make_fullbody_with_rot_constraint_set()
    frame_idx_cpu = torch.tensor(keyframes, dtype=torch.long)
    frame_idx = frame_idx_cpu.to(args.device)
    constraints = [
        fullbody_cls(
            skeleton,
            frame_indices=frame_idx,
            global_joints_positions=soma30_pos_c[frame_idx_cpu].to(args.device),
            global_joints_rots=soma30_rots_c[frame_idx_cpu].to(args.device),
            to_crop=False,
        )
    ]

    target22 = soma30_pos[:, SMPLX22_TO_SOMA30, :].detach().cpu().numpy()
    variants = {
        "baseline_final_paste": {"repaint": False, "final_paste": True},
        "repaint_final_paste": {"repaint": True, "final_paste": True},
        "soft_no_final_paste": {"repaint": False, "final_paste": False},
    }
    summary = {
        "npz": str(args.npz),
        "caption": caption,
        "num_frames": num_frames,
        "keyframes": keyframes,
        "seed": args.seed,
        "steps": args.steps,
        "cfg_weight": args.cfg_weight,
        "heading0": float(heading0.detach().cpu()),
        "variants": {},
    }
    for name, flags in variants.items():
        output = _run_variant(
            model,
            [caption],
            num_frames,
            constraints,
            seed=args.seed,
            steps=args.steps,
            cfg_weight=args.cfg_weight,
            **flags,
        )
        world = _invert_soma77_output(output, r_yaw, t_xz)
        pred22 = soma77_to_smpl22(world["posed_joints"])
        summary["variants"][name] = {
            "keyframe_error": _keyframe_error(pred22, target22, keyframes),
            "edge_stats": _edge_stats(pred22, target22, keyframes),
        }
        np.savez_compressed(out_dir / f"{name}.npz", **world)

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
