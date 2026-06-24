#!/usr/bin/env python3
"""Online adversarial PhysFlow loop for HYMotion T2M Lite and G1 tracking.

This is the first robot-native PhysFlow runner:

  T2M prompt -> HYMotion motion_135 -> PyRoki retarget -> Unitree G1
  -> G1 ONNX tracker in MuJoCo -> adversarial score

The loop uses the G1 tracker as an online adversary:
  - hard/failing prompts drive generator updates with category-matched GT SFT;
  - high-quality retargeted motions are collected for G1 tracker fine-tuning.

The implementation is intentionally conservative. It records every command and
score before launching expensive RL jobs, so the experiment can be audited.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from physflow_tracker_bundle_paths import PROTOMOTIONS_ROOT

DEFAULT_TRACKER_EXPERIMENT = (
    PROTOMOTIONS_ROOT / "examples" / "experiments" / "mimic" / "physflow_g1_xy_offset.py"
)
DEFAULT_TRACKER_CKPT = (
    PROJECT_ROOT / "output" / "physflow_kimodo_g1" / "checkpoints" / "g1_xyvel_partial_warmstart.ckpt"
)

from scripts.embodied.physflow_curriculum import PhysFlowCurriculum
from scripts.embodied.physflow_trainer import (
    GTMotionLoader,
    PhysFlowTrainer,
    load_bundle,
    motion_135_to_201,
)
from scripts.embodied.run_g1_rl_tracker_export import (
    DEFAULT_MJCF,
    DEFAULT_ONNX,
    DEFAULT_URDF,
    parse_body_mesh_mapping,
    retarget_npz_to_motion,
    simulate_and_export,
)
from scripts.embodied.physflow_g1_scoring import (
    DEFAULT_G1_HARD_PROMPT_MIN_SCORE,
    compute_g1_adversarial_score,
    config_from_args,
    is_hard_adversarial_case,
    is_good_tracker_motion,
    tracker_pool_config_from_args,
)

log = logging.getLogger("physflow_online_adv_g1")

DEFAULT_POSITION_AWARE_ONNX = (
    PROTOMOTIONS_ROOT
    / "results"
    / "physflow_g1_xyvel_stable_isaacgym_train_v1"
    / "compiled_models"
    / "unified_pipeline.onnx"
)


@dataclass
class CandidateRecord:
    round_idx: int
    candidate_idx: int
    prompt: str
    category: str
    num_frames: int
    npz_path: str
    motion_path: Optional[str] = None
    robot_json_path: Optional[str] = None
    g1_onnx_path: Optional[str] = None
    g1_onnx_md5: Optional[str] = None
    g1_yaml_path: Optional[str] = None
    g1_yaml_md5: Optional[str] = None
    status: str = "pending"
    completion_ratio: float = 0.0
    max_joint_error_rad: float = 0.0
    fall_detected: bool = False
    root_height_final: float = 0.0
    root_displacement_ref_m: float = 0.0
    root_displacement_track_m: float = 0.0
    root_displacement_error_m: float = 0.0
    root_trajectory_error_mean_m: float = 0.0
    root_trajectory_error_final_m: float = 0.0
    root_metrics_available: bool = False
    adversarial_score: float = 0.0
    error: Optional[str] = None


def slugify(text: str, max_len: int = 48) -> str:
    keep = []
    for ch in text.lower():
        if ch.isalnum():
            keep.append(ch)
        elif ch in {" ", "-", "_"}:
            keep.append("_")
    slug = "".join(keep).strip("_")
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug[:max_len] or "motion"


def append_jsonl(path: Path, item: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(item, default=str) + "\n")


def write_json(path: Path, item: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(item, f, indent=2, default=str)


def file_md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tracker_artifact_metadata(onnx_path: str | Path) -> Dict[str, object]:
    onnx = resolve_project_path(onnx_path)
    yaml_path = onnx.with_suffix(".yaml")
    return {
        "g1_onnx_path": str(onnx),
        "g1_onnx_exists": onnx.is_file(),
        "g1_onnx_md5": file_md5(onnx) if onnx.is_file() else None,
        "g1_yaml_path": str(yaml_path),
        "g1_yaml_exists": yaml_path.is_file(),
        "g1_yaml_md5": file_md5(yaml_path) if yaml_path.is_file() else None,
    }


def resolve_project_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def check_import(module: str) -> Tuple[bool, str]:
    try:
        mod = importlib.import_module(module)
        return True, str(getattr(mod, "__version__", "OK"))
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def run_preflight(args: argparse.Namespace) -> Dict[str, object]:
    pose_only_tracker = Path(args.g1_onnx).resolve() == Path(DEFAULT_ONNX).resolve()
    required_files = {
        "t2m_config": args.t2m_config,
        "t2m_ckpt": args.t2m_ckpt,
        "text_cache": args.text_cache,
        "g1_onnx": args.g1_onnx,
        "g1_mjcf": args.g1_mjcf,
        "g1_urdf": args.g1_urdf,
        "gt_annotation": args.gt_annotation,
    }
    file_status = {
        name: {"path": path, "exists": bool(path and Path(path).exists())}
        for name, path in required_files.items()
    }
    modules = [
        "torch",
        "numpy",
        "mujoco",
        "onnxruntime",
        "yaml",
        "smplx",
        "jax",
        "jaxlib",
        "pyroki",
        "jax_dataclasses",
        "jaxlie",
        "jaxls",
        "yourdfpy",
    ]
    module_status = {
        name: {"ok": ok, "detail": detail}
        for name in modules
        for ok, detail in [check_import(name)]
    }
    report = {
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "cwd": str(PROJECT_ROOT),
        "files": file_status,
        "modules": module_status,
        "ready_for_pyroki_retarget": all(
            module_status[m]["ok"]
            for m in ["jax", "jaxlib", "pyroki", "jax_dataclasses", "jaxlie", "jaxls", "yourdfpy"]
        ),
        "ready_for_g1_mujoco": all(
            module_status[m]["ok"] for m in ["mujoco", "onnxruntime", "yaml"]
        )
        and file_status["g1_onnx"]["exists"]
        and file_status["g1_mjcf"]["exists"],
        "pose_only_tracker": pose_only_tracker,
        "position_aware_required": not args.allow_pose_only_tracker,
    }
    if pose_only_tracker and not args.allow_pose_only_tracker:
        report["ready_for_g1_mujoco"] = False
    out_path = Path(args.output_dir) / "preflight.json"
    write_json(out_path, report)
    print(json.dumps(report, indent=2))
    print(f"[preflight] saved: {out_path}")
    return report


def save_motion_npz(path: Path, motion_135: np.ndarray, fps: int = 30) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, motion_135=motion_135.astype(np.float32), fps=fps)


def categorize_prompt(prompt: str) -> str:
    p = prompt.lower()
    if any(k in p for k in ["kick", "jump", "squat", "spin", "hop", "lunge", "crouch"]):
        return "dynamic"
    if any(k in p for k in ["wave", "arm", "hand", "clap", "point", "reach", "gesture"]):
        return "upper_body"
    if any(k in p for k in ["walk", "step", "turn", "pace", "sidestep", "march", "jog"]):
        return "walking"
    return "standing"


def sample_prompts(curriculum: PhysFlowCurriculum, count: int) -> List[Tuple[str, int, str]]:
    prompts = []
    for i in range(count):
        if i % 2 == 0:
            prompt = curriculum.get_prompt()
            frames = curriculum.get_num_frames()
        else:
            prompt = curriculum.get_diverse_prompt()
            frames = curriculum.get_diverse_num_frames()
        prompts.append((prompt, frames, categorize_prompt(prompt)))
    return prompts


def build_trainer(args: argparse.Namespace, device: torch.device) -> PhysFlowTrainer:
    bundle = load_bundle(args.t2m_config, args.t2m_ckpt, device)
    curriculum = PhysFlowCurriculum(
        seed=args.seed,
        min_locomotion_ratio=args.min_locomotion_ratio,
    )
    trainer = PhysFlowTrainer(
        bundle=bundle,
        physics_oracle=None,
        curriculum=curriculum,
        device=device,
        lr=args.lr,
        num_ode_steps=args.num_ode_steps,
        text_guidance_scale=args.text_guidance_scale,
        train_last_n_blocks=args.train_last_n_blocks,
        grad_accum=args.grad_accum,
        kl_weight=args.kl_weight,
        target_blend=1.0,
        output_dir=args.output_dir,
    )
    if args.text_cache_on_cpu:
        cache = torch.load(args.text_cache, map_location="cpu")
        trainer._text_cache = {
            prompt: {
                "text_vec_raw": feats["text_vec_raw"],
                "text_ctxt_raw": feats["text_ctxt_raw"],
                "text_ctxt_raw_length": feats["text_ctxt_raw_length"],
            }
            for prompt, feats in cache.items()
        }
        log.info("[text-cache] loaded %d prompts on CPU", len(trainer._text_cache))

        def _get_text_feats_cpu(prompt: str) -> Dict[str, torch.Tensor]:
            if prompt not in trainer._text_cache:
                raise KeyError(
                    f"Prompt is not in CPU text cache: {prompt}. "
                    "Use pre-cached curriculum prompts or rebuild the cache."
                )
            feats = trainer._text_cache[prompt]
            return {k: v.to(trainer.device) for k, v in feats.items()}

        trainer._get_text_feats = _get_text_feats_cpu  # type: ignore[method-assign]
    else:
        trainer.precompute_text_embeddings(cache_path=args.text_cache)
    return trainer


def generate_candidates(
    trainer: PhysFlowTrainer,
    args: argparse.Namespace,
    round_idx: int,
    round_dir: Path,
) -> List[CandidateRecord]:
    prompts = sample_prompts(trainer.curriculum, args.candidates_per_round)
    records: List[CandidateRecord] = []
    cand_dir = round_dir / "candidates_npz"
    trainer.bundle.motion_transformer.eval()
    for idx, (prompt, frames, category) in enumerate(prompts):
        name = f"r{round_idx:02d}_{idx:04d}_{slugify(prompt)}"
        npz_path = cand_dir / f"{name}.npz"
        log.info("[gen] r=%d i=%d prompt=%s frames=%d", round_idx, idx, prompt, frames)
        with torch.no_grad():
            motion_135 = trainer.generate_motion(prompt, frames)
        save_motion_npz(npz_path, motion_135, fps=30)
        records.append(
            CandidateRecord(
                round_idx=round_idx,
                candidate_idx=idx,
                prompt=prompt,
                category=category,
                num_frames=frames,
                npz_path=str(npz_path),
            )
        )
    return records


def load_npz_candidates(args: argparse.Namespace, round_idx: int) -> List[CandidateRecord]:
    npz_paths = sorted(Path(args.npz_dir).glob("*.npz"))
    if args.max_score_motions > 0:
        npz_paths = npz_paths[: args.max_score_motions]
    records = []
    for idx, path in enumerate(npz_paths):
        prompt = path.stem
        records.append(
            CandidateRecord(
                round_idx=round_idx,
                candidate_idx=idx,
                prompt=prompt,
                category=categorize_prompt(prompt),
                num_frames=0,
                npz_path=str(path),
            )
        )
    return records


def expected_g1_frames(motion_path: Path, onnx_path: str) -> int:
    from deployment.motion_utils import MotionPlayer

    with open(str(onnx_path).replace(".onnx", ".yaml")) as f:
        meta = yaml.safe_load(f)
    control_dt = meta["timing"]["control_dt"]
    return int(MotionPlayer(str(motion_path), control_dt=control_dt).total_frames)


def score_candidate(record: CandidateRecord, args: argparse.Namespace, round_dir: Path) -> CandidateRecord:
    npz_path = Path(record.npz_path)
    robot_json_dir = round_dir / "g1_tracker_json"
    robot_json_path = robot_json_dir / f"{npz_path.stem}.json"
    body_mesh_mapping = parse_body_mesh_mapping(Path(args.g1_mjcf))
    tracker_meta = tracker_artifact_metadata(args.g1_onnx)
    record.g1_onnx_path = str(tracker_meta["g1_onnx_path"])
    record.g1_onnx_md5 = tracker_meta["g1_onnx_md5"]  # type: ignore[assignment]
    record.g1_yaml_path = str(tracker_meta["g1_yaml_path"])
    record.g1_yaml_md5 = tracker_meta["g1_yaml_md5"]  # type: ignore[assignment]

    try:
        motion_path = retarget_npz_to_motion(
            npz_path=npz_path,
            output_dir=round_dir / "retarget",
            smpl_model_path=args.smpl_model_path,
            urdf_path=args.g1_urdf,
            fps=30,
            pyroki_max_iterations=args.pyroki_max_iterations,
            subsample_factor=args.retarget_subsample_factor,
            target_raw_frames=args.retarget_target_raw_frames,
        )
        total_frames = expected_g1_frames(motion_path, args.g1_onnx)
        stats = simulate_and_export(
            onnx_path=args.g1_onnx,
            motion_file=str(motion_path),
            output_json_path=str(robot_json_path),
            mjcf_path=args.g1_mjcf,
            body_mesh_mapping=body_mesh_mapping,
            subsample_factor=args.robot_json_subsample,
        )
        completion = float(stats["total_steps"] / max(total_frames, 1))
        max_err = float(stats.get("max_joint_error_rad", 0.0))
        fell = bool(stats.get("fall_detected", False))
        root_traj_err = float(stats.get("root_trajectory_error_mean_m", 0.0))
        root_disp_err = float(stats.get("root_displacement_error_m", 0.0))
        root_metrics_available = (
            "root_trajectory_error_mean_m" in stats
            or "root_displacement_error_m" in stats
            or "root_displacement_track_m" in stats
        )
        score = compute_g1_adversarial_score(
            completion=completion,
            max_joint_error_rad=max_err,
            root_trajectory_error_mean_m=root_traj_err,
            root_displacement_error_m=root_disp_err,
            fall_detected=fell,
            config=config_from_args(args),
        )

        record.motion_path = str(motion_path)
        record.robot_json_path = str(robot_json_path)
        record.status = "scored"
        record.completion_ratio = completion
        record.max_joint_error_rad = max_err
        record.fall_detected = fell
        record.root_height_final = float(stats.get("root_height_final", 0.0))
        record.root_displacement_ref_m = float(stats.get("root_displacement_ref_m", 0.0))
        record.root_displacement_track_m = float(stats.get("root_displacement_track_m", 0.0))
        record.root_displacement_error_m = root_disp_err
        record.root_trajectory_error_mean_m = root_traj_err
        record.root_trajectory_error_final_m = float(stats.get("root_trajectory_error_final_m", 0.0))
        record.root_metrics_available = root_metrics_available
        record.adversarial_score = float(score)
    except Exception as exc:
        record.status = "error"
        record.error = f"{type(exc).__name__}: {exc}"
        record.adversarial_score = args.retarget_fail_penalty
        log.exception("[score] failed: %s", record.npz_path)
    return record


def score_candidates(
    records: List[CandidateRecord],
    args: argparse.Namespace,
    round_dir: Path,
) -> List[CandidateRecord]:
    scored = []
    score_log = round_dir / "candidate_scores.jsonl"
    for record in records:
        scored_record = score_candidate(record, args, round_dir)
        scored.append(scored_record)
        append_jsonl(score_log, asdict(scored_record))
        log.info(
            "[score] i=%d status=%s adv=%.3f comp=%.2f err=%.3f fall=%s",
            scored_record.candidate_idx,
            scored_record.status,
            scored_record.adversarial_score,
            scored_record.completion_ratio,
            scored_record.max_joint_error_rad,
            scored_record.fall_detected,
        )
    return scored


def update_generator_from_hard_cases(
    trainer: PhysFlowTrainer,
    hard_cases: List[CandidateRecord],
    gt_loader: GTMotionLoader,
    args: argparse.Namespace,
    round_dir: Path,
) -> List[dict]:
    updates = []
    if args.dry_run_updates or args.gen_update_steps <= 0:
        return updates
    if not hard_cases:
        skip = {
            "status": "skipped",
            "reason": "no_hard_cases_above_threshold",
            "hard_min_score_for_generator": args.hard_min_score_for_generator,
            "requested_steps": args.gen_update_steps,
        }
        append_jsonl(round_dir / "generator_updates.jsonl", skip)
        write_json(round_dir / "generator_update_skipped.json", skip)
        log.info("[gen-update] skipped: no hard cases above threshold")
        return updates

    categories = [c.category for c in hard_cases]
    for step in range(args.gen_update_steps):
        category = categories[step % len(categories)]
        prompt = hard_cases[step % len(hard_cases)].prompt
        motion_135_gt, _caption = gt_loader.sample(category=category)
        motion_201_gt = motion_135_to_201(motion_135_gt, trainer.bundle.body_model, trainer.device)
        result = trainer.train_step(motion_201_gt, prompt)
        trainer.total_iterations += 1
        trainer.loss_history.append(result["loss"])
        update = {
            "step": step,
            "category": category,
            "prompt": prompt,
            "loss": result["loss"],
            "loss_velocity": result["loss_velocity"],
            "loss_kl": result.get("loss_kl", 0.0),
            "did_optimizer_step": result.get("did_optimizer_step", False),
        }
        updates.append(update)
        append_jsonl(round_dir / "generator_updates.jsonl", update)
    ckpt_path = round_dir / "t2m_after_round.pt"
    torch.save(
        {
            "round": hard_cases[0].round_idx,
            "model_state_dict": trainer.bundle.motion_transformer.state_dict(),
            "optimizer_state_dict": trainer.optimizer.state_dict(),
            "loss_history": trainer.loss_history[-500:],
            "mode": "online_adversarial_g1_gt_sft",
        },
        ckpt_path,
    )
    log.info("[gen-update] saved %s", ckpt_path)
    return updates


def stage_tracker_pool(good_cases: List[CandidateRecord], round_dir: Path) -> Path:
    pool_dir = round_dir / "tracker_motion_pool"
    if pool_dir.exists():
        shutil.rmtree(pool_dir)
    pool_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    seen_hashes = set()
    for case in good_cases:
        if not case.motion_path:
            continue
        src = Path(case.motion_path)
        if not src.is_absolute():
            src = PROJECT_ROOT / src
        if not src.is_file():
            continue
        motion_hash = file_md5(src)
        if motion_hash in seen_hashes:
            continue
        seen_hashes.add(motion_hash)
        dst = pool_dir / src.name
        if src.resolve() != dst.resolve():
            shutil.copy2(src, dst)
        manifest.append(
            {
                "motion_path": str(dst),
                "source_motion_path": str(src),
                "source_motion_md5": motion_hash,
                "prompt": case.prompt,
                "category": case.category,
                "adversarial_score": case.adversarial_score,
                "completion_ratio": case.completion_ratio,
                "max_joint_error_rad": case.max_joint_error_rad,
                "root_trajectory_error_mean_m": case.root_trajectory_error_mean_m,
                "root_displacement_error_m": case.root_displacement_error_m,
            }
        )
    write_json(pool_dir / "manifest.json", manifest)
    return pool_dir


def tracker_experiment_name(round_idx: int) -> str:
    return f"physflow_online_g1_r{round_idx:02d}"


def expected_tracker_checkpoint_path(round_idx: int) -> Path:
    return PROTOMOTIONS_ROOT / "results" / tracker_experiment_name(round_idx) / "last.ckpt"


def expected_tracker_export_dir(round_idx: int) -> Path:
    return PROTOMOTIONS_ROOT / "results" / tracker_experiment_name(round_idx) / "compiled_models"


def build_tracker_train_command(pool_dir: Path, args: argparse.Namespace, round_idx: int) -> List[str]:
    python_cmd = args.tracker_python or sys.executable
    tracker_experiment = resolve_project_path(args.tracker_experiment)
    tracker_ckpt = resolve_project_path(args.g1_tracker_ckpt)
    cmd = [
        python_cmd,
        str(PROTOMOTIONS_ROOT / "protomotions" / "train_agent.py"),
        "--robot-name",
        "g1",
        "--simulator",
        args.tracker_simulator,
        "--experiment-path",
        str(tracker_experiment),
        "--experiment-name",
        tracker_experiment_name(round_idx),
        "--motion-file",
        str(pool_dir.resolve()),
        "--checkpoint",
        str(tracker_ckpt),
        "--num-envs",
        str(args.tracker_num_envs),
        "--batch-size",
        str(args.tracker_batch_size),
        "--training-max-steps",
        str(args.tracker_steps),
        "--headless",
        "True",
        "--skip-initial-eval",
    ]
    if args.tracker_save_every > 0:
        cmd.extend(["--overrides", f"agent.save_last_checkpoint_every={args.tracker_save_every}"])
    return cmd


def build_tracker_export_command(checkpoint_path: Path, export_dir: Path, args: argparse.Namespace) -> List[str]:
    python_cmd = args.tracker_export_python or args.tracker_python or sys.executable
    export_script = resolve_project_path(args.tracker_export_script)
    cmd = [
        python_cmd,
        str(export_script),
        "--checkpoint",
        str(checkpoint_path),
        "--output",
        str(export_dir),
    ]
    if not args.validate_tracker_export:
        cmd.append("--no-validate")
    return cmd


def tracker_subprocess_env() -> Dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{PROTOMOTIONS_ROOT}:{env.get('PYTHONPATH', '')}"
    env.setdefault("ACCEPT_EULA", "Y")
    env.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    env.setdefault("WANDB_SILENT", "true")
    env.setdefault("WANDB_DISABLE_SENTRY", "true")
    env.setdefault("ISAACGYM_GRAPHICS_DEVICE_ID", "-1")

    for version in range(14, 8, -1):
        root = Path(f"/opt/rh/gcc-toolset-{version}/root/usr")
        bin_dir = root / "bin"
        lib64_dir = root / "lib64"
        if bin_dir.exists():
            env["PATH"] = f"{bin_dir}:{env.get('PATH', '')}"
            env["CC"] = str(bin_dir / "gcc")
            env["CXX"] = str(bin_dir / "g++")
            if lib64_dir.exists():
                env["LD_LIBRARY_PATH"] = f"{lib64_dir}:{env.get('LD_LIBRARY_PATH', '')}"
            env["PHYSFLOW_GCC_TOOLSET"] = str(root)
            break
    return env


def maybe_update_tracker(good_cases: List[CandidateRecord], args: argparse.Namespace, round_dir: Path, round_idx: int) -> dict:
    pool_dir = stage_tracker_pool(good_cases, round_dir)
    cmd = build_tracker_train_command(pool_dir, args, round_idx)
    manifest_path = pool_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else []
    min_motions = int(getattr(args, "min_tracker_motions_for_update", 1))
    checkpoint_path = expected_tracker_checkpoint_path(round_idx)
    export_dir = expected_tracker_export_dir(round_idx)
    exported_onnx = export_dir / "unified_pipeline.onnx"
    exported_yaml = export_dir / "unified_pipeline.yaml"
    export_cmd = build_tracker_export_command(checkpoint_path, export_dir, args)
    info = {
        "pool_dir": str(pool_dir),
        "manifest_path": str(manifest_path),
        "num_motions": len(manifest),
        "num_input_good_cases": len(good_cases),
        "num_duplicate_motions_skipped": max(len(good_cases) - len(manifest), 0),
        "min_tracker_motions_for_update": min_motions,
        "run_tracker_update_requested": bool(args.run_tracker_update),
        "should_run": bool(args.run_tracker_update and len(manifest) >= min_motions),
        "skip_reason": None,
        "cmd": cmd,
        "expected_checkpoint_path": str(checkpoint_path),
        "export_tracker_after_update": bool(args.export_tracker_after_update),
        "tracker_export_cmd": export_cmd,
        "tracker_export_dir": str(export_dir),
        "exported_tracker_onnx": str(exported_onnx),
        "exported_tracker_yaml": str(exported_yaml),
        "tracker_export_status": "not_requested",
        "activate_exported_tracker": bool(args.activate_exported_tracker),
        "activated_g1_onnx": None,
    }
    if not args.run_tracker_update:
        info["skip_reason"] = "run_tracker_update_not_requested"
    elif len(manifest) == 0:
        info["skip_reason"] = "no_good_tracker_motions"
    elif len(manifest) < min_motions:
        info["skip_reason"] = "below_min_tracker_motions_for_update"
    write_json(round_dir / "tracker_train_command.json", info)
    if not info["should_run"]:
        return info

    log.info("[tracker-update] running: %s", " ".join(cmd))
    env = tracker_subprocess_env()
    result = subprocess.run(cmd, cwd=str(PROTOMOTIONS_ROOT), env=env)
    info["returncode"] = result.returncode
    write_json(round_dir / "tracker_train_command.json", info)
    if result.returncode != 0:
        raise RuntimeError(f"tracker update failed with return code {result.returncode}")
    if not args.export_tracker_after_update:
        return info

    if not checkpoint_path.is_file():
        info["tracker_export_status"] = "skipped_missing_checkpoint"
        write_json(round_dir / "tracker_train_command.json", info)
        log.warning("[tracker-export] skipped: checkpoint not found at %s", checkpoint_path)
        return info

    log.info("[tracker-export] running: %s", " ".join(export_cmd))
    export_result = subprocess.run(export_cmd, cwd=str(PROTOMOTIONS_ROOT), env=env)
    info["tracker_export_returncode"] = export_result.returncode
    info["tracker_export_status"] = "completed" if export_result.returncode == 0 else "failed"
    info["exported_tracker_exists"] = exported_onnx.is_file()
    info["exported_tracker_yaml_exists"] = exported_yaml.is_file()
    if export_result.returncode != 0:
        write_json(round_dir / "tracker_train_command.json", info)
        raise RuntimeError(f"tracker export failed with return code {export_result.returncode}")
    if args.activate_exported_tracker and exported_onnx.is_file() and exported_yaml.is_file():
        args.g1_onnx = str(exported_onnx)
        info["activated_g1_onnx"] = str(exported_onnx)
    write_json(round_dir / "tracker_train_command.json", info)
    return info


def summarize_round(scored: List[CandidateRecord], args: argparse.Namespace, round_dir: Path, round_idx: int) -> Tuple[List[CandidateRecord], List[CandidateRecord]]:
    ranked_hard = [
        record
        for record in sorted(scored, key=lambda r: r.adversarial_score, reverse=True)
        if record.status == "scored"
        and is_hard_adversarial_case(asdict(record), args.hard_min_score_for_generator)
    ]
    excluded_error_hard = [
        record
        for record in scored
        if record.status == "error"
        and is_hard_adversarial_case(asdict(record), args.hard_min_score_for_generator)
    ]
    tracker_pool_config = tracker_pool_config_from_args(args)
    ranked_good = sorted(
        [r for r in scored if is_good_tracker_motion(asdict(r), tracker_pool_config)],
        key=lambda r: r.adversarial_score,
    )
    hard_cases = ranked_hard[: args.hard_cases_for_generator]
    good_cases = ranked_good[: args.good_cases_for_tracker]
    summary = {
        "round": round_idx,
        "active_g1_tracker": tracker_artifact_metadata(args.g1_onnx),
        "num_candidates": len(scored),
        "num_scored": sum(1 for r in scored if r.status == "scored"),
        "num_errors": sum(1 for r in scored if r.status == "error"),
        "num_falls": sum(1 for r in scored if r.fall_detected),
        "mean_completion": float(np.mean([r.completion_ratio for r in scored if r.status == "scored"]) if any(r.status == "scored" for r in scored) else 0.0),
        "mean_joint_error": float(np.mean([r.max_joint_error_rad for r in scored if r.status == "scored"]) if any(r.status == "scored" for r in scored) else 0.0),
        "mean_root_displacement_ref_m": float(np.mean([r.root_displacement_ref_m for r in scored if r.status == "scored"]) if any(r.status == "scored" for r in scored) else 0.0),
        "mean_root_displacement_track_m": float(np.mean([r.root_displacement_track_m for r in scored if r.status == "scored"]) if any(r.status == "scored" for r in scored) else 0.0),
        "mean_root_trajectory_error_m": float(np.mean([r.root_trajectory_error_mean_m for r in scored if r.status == "scored"]) if any(r.status == "scored" for r in scored) else 0.0),
        "hard_min_score_for_generator": args.hard_min_score_for_generator,
        "num_hard_eligible": len(ranked_hard),
        "num_error_hard_excluded": len(excluded_error_hard),
        "error_hard_excluded": [asdict(r) for r in excluded_error_hard[:5]],
        "tracker_pool_thresholds": tracker_pool_config.to_dict(),
        "hard_cases": [asdict(r) for r in hard_cases],
        "good_cases": [asdict(r) for r in good_cases],
    }
    write_json(round_dir / "round_summary.json", summary)
    return hard_cases, good_cases


def run_loop(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(output_dir / "online_adv_g1.log"),
        ],
    )
    write_json(output_dir / "args.json", vars(args))

    if args.mode == "preflight":
        run_preflight(args)
        return

    report = run_preflight(args)
    if args.require_pyroki and not report["ready_for_pyroki_retarget"]:
        raise RuntimeError("PyRoki/JAX dependencies are missing; cannot retarget to G1.")

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    trainer = None
    gt_loader = None
    if args.mode == "loop":
        trainer = build_trainer(args, device)
        gt_loader = GTMotionLoader(args.gt_annotation, max_frames=150, min_frames=30, seed=args.seed)

    for round_idx in range(args.num_rounds):
        round_dir = output_dir / f"round_{round_idx:02d}"
        round_dir.mkdir(parents=True, exist_ok=True)

        if args.npz_dir:
            records = load_npz_candidates(args, round_idx)
        else:
            if trainer is None:
                trainer = build_trainer(args, device)
            records = generate_candidates(trainer, args, round_idx, round_dir)

        scored = score_candidates(records, args, round_dir)
        hard_cases, good_cases = summarize_round(scored, args, round_dir, round_idx)

        if args.mode == "loop":
            assert trainer is not None and gt_loader is not None
            update_generator_from_hard_cases(trainer, hard_cases, gt_loader, args, round_dir)
            maybe_update_tracker(good_cases, args, round_dir, round_idx)

    log.info("[done] output=%s", output_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("PhysFlow online adversarial G1 runner")
    parser.add_argument("--mode", choices=["preflight", "score-npz-dir", "loop"], default="preflight")
    parser.add_argument("--output-dir", default="output/physflow_online_adv/g1_v0")
    parser.add_argument("--npz-dir", default=None, help="Score existing motion_135 NPZ files instead of generating.")
    parser.add_argument("--max-score-motions", type=int, default=0)
    parser.add_argument("--num-rounds", type=int, default=1)
    parser.add_argument("--candidates-per-round", type=int, default=4)
    parser.add_argument("--hard-cases-for-generator", type=int, default=2)
    parser.add_argument("--hard-min-score-for-generator", type=float, default=DEFAULT_G1_HARD_PROMPT_MIN_SCORE)
    parser.add_argument("--good-cases-for-tracker", type=int, default=2)
    parser.add_argument("--good-min-completion", type=float, default=0.95)
    parser.add_argument("--good-max-joint-error", type=float, default=0.7)
    parser.add_argument("--good-max-root-trajectory-error", type=float, default=0.25)
    parser.add_argument("--good-max-root-displacement-error", type=float, default=0.35)
    parser.add_argument(
        "--allow-tracker-pool-without-root-metrics",
        action="store_true",
        help="Allow tracker fine-tune pool entries from pose-only or legacy scoring runs.",
    )

    parser.add_argument("--t2m-config", default="configs/hymotion_t2m/hymotion_t2m_201dim_046b.py")
    parser.add_argument("--t2m-ckpt", default="checkpoints/HY-Motion-1.0/HY-Motion-1.0-Lite/latest.ckpt")
    parser.add_argument("--text-cache", default="output/physflow_v2_test/text_embeddings.pt")
    parser.add_argument("--text-cache-on-cpu", action="store_true")
    parser.add_argument("--gt-annotation", default="data/annotation/train_hymotion_400h.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")

    parser.add_argument("--num-ode-steps", type=int, default=25)
    parser.add_argument("--text-guidance-scale", type=float, default=4.5)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--train-last-n-blocks", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--kl-weight", type=float, default=0.05)
    parser.add_argument("--min-locomotion-ratio", type=float, default=0.3)
    parser.add_argument("--gen-update-steps", type=int, default=2)
    parser.add_argument("--dry-run-updates", action="store_true")

    parser.add_argument("--g1-onnx", default=str(DEFAULT_POSITION_AWARE_ONNX))
    parser.add_argument(
        "--allow-pose-only-tracker",
        action="store_true",
        help="Allow the released g1-bones-deploy tracker as an explicit pose-only baseline.",
    )
    parser.add_argument("--g1-mjcf", default=str(DEFAULT_MJCF))
    parser.add_argument("--g1-urdf", default=str(DEFAULT_URDF))
    parser.add_argument("--smpl-model-path", default=str(PROJECT_ROOT / "checkpoints/smpl_models"))
    parser.add_argument("--robot-json-subsample", type=int, default=10)
    parser.add_argument("--pyroki-max-iterations", type=int, default=80)
    parser.add_argument("--retarget-subsample-factor", type=int, default=2)
    parser.add_argument("--retarget-target-raw-frames", type=int, default=150)
    parser.add_argument("--joint-error-scale", type=float, default=1.0)
    parser.add_argument(
        "--root-trajectory-error-scale",
        type=float,
        default=0.5,
        help="Meters corresponding to one adversarial score unit for mean root trajectory error.",
    )
    parser.add_argument(
        "--root-displacement-error-scale",
        type=float,
        default=0.5,
        help="Meters corresponding to one adversarial score unit for final root displacement error.",
    )
    parser.add_argument("--root-trajectory-error-weight", type=float, default=1.0)
    parser.add_argument("--root-displacement-error-weight", type=float, default=0.5)
    parser.add_argument("--score-component-cap", type=float, default=2.0)
    parser.add_argument("--fall-penalty", type=float, default=2.0)
    parser.add_argument("--retarget-fail-penalty", type=float, default=5.0)
    parser.add_argument("--require-pyroki", action="store_true")

    parser.add_argument("--run-tracker-update", action="store_true")
    parser.add_argument(
        "--min-tracker-motions-for-update",
        type=int,
        default=2,
        help="Minimum unique good .motion files required before online tracker fine-tuning is launched.",
    )
    parser.add_argument("--tracker-python", default=None)
    parser.add_argument("--tracker-simulator", default="isaacgym")
    parser.add_argument("--tracker-experiment", default=str(DEFAULT_TRACKER_EXPERIMENT))
    parser.add_argument("--g1-tracker-ckpt", default=str(DEFAULT_TRACKER_CKPT))
    parser.add_argument("--tracker-num-envs", type=int, default=1024)
    parser.add_argument("--tracker-batch-size", type=int, default=4096)
    parser.add_argument("--tracker-steps", type=int, default=500)
    parser.add_argument("--tracker-save-every", type=int, default=5)
    parser.add_argument("--export-tracker-after-update", action="store_true")
    parser.add_argument(
        "--activate-exported-tracker",
        action="store_true",
        help="Use the exported tracker ONNX for subsequent rounds after a successful tracker update.",
    )
    parser.add_argument("--tracker-export-python", default=None)
    parser.add_argument(
        "--tracker-export-script",
        default=str(PROTOMOTIONS_ROOT / "deployment" / "export_bm_tracker_onnx.py"),
    )
    parser.add_argument("--validate-tracker-export", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run_loop(parse_args())
