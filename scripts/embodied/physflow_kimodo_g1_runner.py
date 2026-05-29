#!/usr/bin/env python3
"""PhysFlow KIMODO-G1 runner.

This runner starts the robot-native branch of PhysFlow:

  prompt -> KIMODO-G1 qpos CSV -> ProtoMotions .motion -> G1 ONNX tracker score

It intentionally keeps generation, conversion, and scoring as auditable stages.
The first goal is infrastructure correctness before launching long tracker
fine-tuning jobs.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import logging
import os
import shlex
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
KIMODO_ROOT = PROJECT_ROOT / "ref_repo" / "KIMODO" / "kimodo"
PROTOMOTIONS_ROOT = PROJECT_ROOT / "ref_repo" / "ProtoMotions"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(KIMODO_ROOT) not in sys.path:
    sys.path.insert(0, str(KIMODO_ROOT))
if str(PROTOMOTIONS_ROOT) not in sys.path:
    sys.path.insert(0, str(PROTOMOTIONS_ROOT))

from scripts.embodied.run_g1_rl_tracker_export import (
    DEFAULT_MJCF,
    DEFAULT_ONNX,
    parse_body_mesh_mapping,
    simulate_and_export,
)
from scripts.embodied.physflow_g1_scoring import (
    DEFAULT_G1_SCORE_CONFIG,
    DEFAULT_G1_HARD_PROMPT_MIN_SCORE,
    compute_g1_adversarial_score,
    config_from_args,
    is_hard_adversarial_case,
    is_good_tracker_motion,
    tracker_pool_config_from_args,
)

log = logging.getLogger("physflow_kimodo_g1")

DEFAULT_POSITION_AWARE_ONNX = (
    PROTOMOTIONS_ROOT
    / "results"
    / "physflow_g1_xyvel_stable_isaacgym_train_v1"
    / "compiled_models"
    / "unified_pipeline.onnx"
)


@dataclass
class PromptSpec:
    id: str
    prompt: str
    category: str
    difficulty: int
    duration_sec: float
    split: str
    source: str
    tags: List[str]


@dataclass
class KimodoRecord:
    prompt_id: str
    prompt: str
    category: str
    difficulty: int
    duration_sec: float
    split: str
    seed: int
    sample_idx: int
    output_stem: str
    npz_path: Optional[str] = None
    csv_path: Optional[str] = None
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
    out = []
    for ch in text.lower():
        if ch.isalnum():
            out.append(ch)
        elif ch in {" ", "-", "_"}:
            out.append("_")
    slug = "".join(out).strip("_")
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug[:max_len] or "motion"


def write_json(path: Path, item: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(item, f, indent=2, default=str)


def append_jsonl(path: Path, item: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        json.dump(item, f, default=str)
        f.write("\n")


def check_import(module: str) -> Dict[str, object]:
    try:
        mod = importlib.import_module(module)
        return {"ok": True, "detail": str(getattr(mod, "__version__", "OK"))}
    except Exception as exc:
        return {"ok": False, "detail": f"{type(exc).__name__}: {exc}"}


def load_prompt_bank(path: Path) -> List[PromptSpec]:
    prompts: List[PromptSpec] = []
    fields = PromptSpec.__dataclass_fields__
    with path.open() as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            data.setdefault("source", path.stem)
            data.setdefault("tags", [])
            filtered = {key: data[key] for key in fields if key in data}
            try:
                prompts.append(PromptSpec(**filtered))
            except TypeError as exc:
                raise ValueError(f"Bad prompt spec at {path}:{line_no}: {exc}") from exc
    return prompts


def select_prompts(args: argparse.Namespace) -> List[PromptSpec]:
    prompts = load_prompt_bank(Path(args.prompt_bank))
    if args.prompt_split != "all":
        prompts = [p for p in prompts if p.split == args.prompt_split]
    if args.prompt_category:
        wanted = {c.strip() for c in args.prompt_category.split(",") if c.strip()}
        prompts = [p for p in prompts if p.category in wanted]
    if args.max_difficulty > 0:
        prompts = [p for p in prompts if p.difficulty <= args.max_difficulty]
    if args.max_prompts > 0:
        prompts = prompts[: args.max_prompts]
    if not prompts:
        raise ValueError("No prompts selected. Check --prompt-split/category/difficulty.")
    return prompts


def command_env(args: argparse.Namespace) -> Dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{KIMODO_ROOT}:{PROTOMOTIONS_ROOT}:{PROJECT_ROOT}:{env.get('PYTHONPATH', '')}"
    env.setdefault("HF_HOME", str(PROJECT_ROOT / "checkpoints" / "kimodo"))
    env.setdefault("TRANSFORMERS_CACHE", str(PROJECT_ROOT / "checkpoints" / "kimodo"))
    if args.text_encoder:
        env["TEXT_ENCODER"] = args.text_encoder
        env["TEXT_ENCODER_MODE"] = "local"
    if args.checkpoint_dir:
        env["CHECKPOINT_DIR"] = str(Path(args.checkpoint_dir).resolve())
    if args.local_cache:
        env["LOCAL_CACHE"] = "true"
    if args.cuda_visible_devices:
        env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    return env


def run_command(cmd: List[str], cwd: Path, env: Dict[str, str], log_path: Path) -> None:
    log.info("[cmd] %s", " ".join(cmd))
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a") as f:
        f.write("\n$ " + " ".join(cmd) + "\n")
        f.flush()
        result = subprocess.run(cmd, cwd=str(cwd), env=env, stdout=f, stderr=subprocess.STDOUT)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed with return code {result.returncode}: {' '.join(cmd)}")


def run_preflight(args: argparse.Namespace) -> Dict[str, object]:
    pose_only_tracker = Path(args.g1_onnx).resolve() == Path(DEFAULT_ONNX).resolve()
    files = {
        "prompt_bank": args.prompt_bank,
        "kimodo_root": str(KIMODO_ROOT),
        "protomotions_root": str(PROTOMOTIONS_ROOT),
        "g1_onnx": args.g1_onnx,
        "g1_mjcf": args.g1_mjcf,
    }
    modules = [
        "torch",
        "numpy",
        "scipy",
        "typer",
        "mujoco",
        "onnxruntime",
        "yaml",
        "kimodo",
        "protomotions",
    ]
    report = {
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "cwd": str(PROJECT_ROOT),
        "files": {k: {"path": v, "exists": Path(v).exists()} for k, v in files.items()},
        "modules": {m: check_import(m) for m in modules},
        "selected_prompts": [asdict(p) for p in select_prompts(args)],
        "pose_only_tracker": pose_only_tracker,
        "position_aware_required": not args.allow_pose_only_tracker,
    }
    report["ready_for_generation"] = report["files"]["prompt_bank"]["exists"] and report["modules"]["kimodo"]["ok"]
    report["ready_for_scoring"] = (
        report["files"]["g1_onnx"]["exists"]
        and report["files"]["g1_mjcf"]["exists"]
        and report["modules"]["mujoco"]["ok"]
        and report["modules"]["onnxruntime"]["ok"]
        and (args.allow_pose_only_tracker or not pose_only_tracker)
    )
    out_path = Path(args.output_dir) / "preflight.json"
    write_json(out_path, report)
    print(json.dumps(report, indent=2, default=str))
    print(f"[preflight] saved: {out_path}")
    return report


def expected_g1_frames(motion_path: Path, onnx_path: str) -> int:
    from deployment.motion_utils import MotionPlayer

    with open(str(onnx_path).replace(".onnx", ".yaml")) as f:
        meta = yaml.safe_load(f)
    control_dt = meta["timing"]["control_dt"]
    return int(MotionPlayer(str(motion_path), control_dt=control_dt).total_frames)


def make_record(prompt: PromptSpec, args: argparse.Namespace, prompt_idx: int, sample_idx: int) -> KimodoRecord:
    stem = f"{prompt.id}_s{sample_idx:02d}_{slugify(prompt.prompt)}"
    return KimodoRecord(
        prompt_id=prompt.id,
        prompt=prompt.prompt,
        category=prompt.category,
        difficulty=prompt.difficulty,
        duration_sec=float(prompt.duration_sec),
        split=prompt.split,
        seed=args.seed + prompt_idx * max(args.samples_per_prompt, 1) + sample_idx,
        sample_idx=sample_idx,
        output_stem=stem,
    )


def make_records(prompts: List[PromptSpec], args: argparse.Namespace) -> List[KimodoRecord]:
    return [
        make_record(prompt, args, prompt_idx, sample_idx)
        for prompt_idx, prompt in enumerate(prompts)
        for sample_idx in range(max(args.samples_per_prompt, 1))
    ]


def generate_record(record: KimodoRecord, args: argparse.Namespace, out_dir: Path) -> KimodoRecord:
    gen_dir = out_dir / "kimodo_raw"
    output_base = gen_dir / record.output_stem
    csv_path = output_base.with_suffix(".csv")
    npz_path = output_base.with_suffix(".npz")
    if csv_path.exists() and npz_path.exists() and not args.force:
        record.csv_path = str(csv_path)
        record.npz_path = str(npz_path)
        return record

    cmd = [
        sys.executable,
        "-m",
        "kimodo.scripts.generate",
        record.prompt,
        "--model",
        args.kimodo_model,
        "--duration",
        str(record.duration_sec),
        "--num_samples",
        "1",
        "--diffusion_steps",
        str(args.diffusion_steps),
        "--seed",
        str(record.seed),
        "--output",
        str(output_base),
    ]
    if args.cfg_type:
        cmd += ["--cfg_type", args.cfg_type]
    if args.cfg_weight:
        cmd += ["--cfg_weight"] + [str(w) for w in args.cfg_weight]
    run_command(cmd, cwd=PROJECT_ROOT, env=command_env(args), log_path=out_dir / "kimodo_generate.log")
    if not csv_path.exists():
        raise FileNotFoundError(f"KIMODO did not create expected G1 CSV: {csv_path}")
    record.csv_path = str(csv_path)
    record.npz_path = str(npz_path) if npz_path.exists() else None
    return record


def convert_csvs_to_proto(csv_dir: Path, proto_dir: Path, args: argparse.Namespace) -> None:
    csv_dir = csv_dir.resolve()
    proto_dir = proto_dir.resolve()
    cmd = [
        sys.executable,
        "data/scripts/convert_g1_csv_to_proto.py",
        "--input-dir",
        str(csv_dir),
        "--output-dir",
        str(proto_dir),
        "--input-fps",
        "30",
        "--output-fps",
        "30",
        "--pos-units",
        "m",
        "--rot-format",
        "quat_wxyz",
        "--joint-units",
        "rad",
        "--no-has-header",
        "--no-has-frame-column",
        "--force-remake" if args.force else "",
    ]
    cmd = [c for c in cmd if c]
    env = command_env(args)
    env["MUJOCO_GL"] = "disable"
    run_command(cmd, cwd=PROTOMOTIONS_ROOT, env=env, log_path=proto_dir.parent / "convert_g1_csv_to_proto.log")


def attach_motion_paths(records: List[KimodoRecord], proto_dir: Path) -> List[KimodoRecord]:
    for record in records:
        candidates = sorted(proto_dir.glob(f"{record.output_stem}*.motion"))
        if candidates:
            record.motion_path = str(candidates[0])
    return records


def compute_adversarial_score(
    completion: float,
    max_joint_error_rad: float,
    root_trajectory_error_mean_m: float,
    root_displacement_error_m: float,
    fall_detected: bool,
    args: argparse.Namespace,
) -> float:
    return compute_g1_adversarial_score(
        completion=completion,
        max_joint_error_rad=max_joint_error_rad,
        root_trajectory_error_mean_m=root_trajectory_error_mean_m,
        root_displacement_error_m=root_displacement_error_m,
        fall_detected=fall_detected,
        config=config_from_args(args),
    )


def score_record(record: KimodoRecord, args: argparse.Namespace, out_dir: Path) -> KimodoRecord:
    attach_tracker_artifact_metadata(record, args)
    if not record.motion_path:
        record.status = "error"
        record.error = "missing .motion file"
        record.adversarial_score = args.error_penalty
        return record

    robot_json_dir = out_dir / "g1_tracker_json"
    robot_json_path = robot_json_dir / f"{record.output_stem}.json"
    body_mesh_mapping = parse_body_mesh_mapping(Path(args.g1_mjcf))
    total_frames = expected_g1_frames(Path(record.motion_path), args.g1_onnx)
    stats = simulate_and_export(
        onnx_path=args.g1_onnx,
        motion_file=record.motion_path,
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
    score = compute_adversarial_score(completion, max_err, root_traj_err, root_disp_err, fell, args)

    record.robot_json_path = str(robot_json_path)
    record.status = "scored"
    record.completion_ratio = completion
    record.max_joint_error_rad = max_err
    record.fall_detected = fell
    record.root_height_final = float(stats.get("root_height_final", 0.0))
    record.root_displacement_ref_m = float(stats.get("root_displacement_ref_m", 0.0))
    record.root_displacement_track_m = float(stats.get("root_displacement_track_m", 0.0))
    record.root_displacement_error_m = float(stats.get("root_displacement_error_m", 0.0))
    record.root_trajectory_error_mean_m = float(stats.get("root_trajectory_error_mean_m", 0.0))
    record.root_trajectory_error_final_m = float(stats.get("root_trajectory_error_final_m", 0.0))
    record.root_metrics_available = root_metrics_available
    record.adversarial_score = float(score)
    return record


def summarize(records: List[KimodoRecord], out_dir: Path) -> Dict[str, object]:
    scored = [r for r in records if r.status == "scored"]
    best_by_prompt: Dict[str, KimodoRecord] = {}
    for record in scored:
        previous = best_by_prompt.get(record.prompt_id)
        if previous is None or record.adversarial_score < previous.adversarial_score:
            best_by_prompt[record.prompt_id] = record

    summary = {
        "num_records": len(records),
        "num_scored": len(scored),
        "num_errors": sum(1 for r in records if r.status == "error"),
        "num_falls": sum(1 for r in scored if r.fall_detected),
        "mean_completion": float(np.mean([r.completion_ratio for r in scored])) if scored else 0.0,
        "mean_joint_error": float(np.mean([r.max_joint_error_rad for r in scored])) if scored else 0.0,
        "mean_root_displacement_ref_m": float(np.mean([r.root_displacement_ref_m for r in scored])) if scored else 0.0,
        "mean_root_displacement_track_m": float(np.mean([r.root_displacement_track_m for r in scored])) if scored else 0.0,
        "mean_root_trajectory_error_m": float(np.mean([r.root_trajectory_error_mean_m for r in scored])) if scored else 0.0,
        "best_records_by_prompt": [asdict(r) for r in sorted(best_by_prompt.values(), key=lambda r: r.prompt_id)],
        "records": [asdict(r) for r in records],
    }
    write_json(out_dir / "summary.json", summary)
    return summary


def copy_artifact(src_value: Optional[str], target_dir: Path) -> Optional[str]:
    if not src_value:
        return None
    src = Path(src_value)
    if not src.is_absolute():
        src = PROJECT_ROOT / src
    if not src.is_file():
        return src_value
    target_dir.mkdir(parents=True, exist_ok=True)
    dst = target_dir / src.name
    if src.resolve() != dst.resolve():
        shutil.copy2(src, dst)
    return str(dst)


def file_md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tracker_artifact_metadata(onnx_path: str | Path) -> Dict[str, object]:
    onnx = Path(onnx_path)
    if not onnx.is_absolute():
        onnx = PROJECT_ROOT / onnx
    yaml_path = onnx.with_suffix(".yaml")
    return {
        "g1_onnx_path": str(onnx),
        "g1_onnx_exists": onnx.is_file(),
        "g1_onnx_md5": file_md5(onnx) if onnx.is_file() else None,
        "g1_yaml_path": str(yaml_path),
        "g1_yaml_exists": yaml_path.is_file(),
        "g1_yaml_md5": file_md5(yaml_path) if yaml_path.is_file() else None,
    }


def attach_tracker_artifact_metadata(record: KimodoRecord, args: argparse.Namespace) -> None:
    if not getattr(args, "g1_onnx", None):
        return
    tracker_meta = tracker_artifact_metadata(args.g1_onnx)
    record.g1_onnx_path = str(tracker_meta["g1_onnx_path"])
    record.g1_onnx_md5 = str(tracker_meta["g1_onnx_md5"]) if tracker_meta["g1_onnx_md5"] else None
    record.g1_yaml_path = str(tracker_meta["g1_yaml_path"])
    record.g1_yaml_md5 = str(tracker_meta["g1_yaml_md5"]) if tracker_meta["g1_yaml_md5"] else None


def write_adversarial_outputs(records: List[KimodoRecord], args: argparse.Namespace, out_dir: Path) -> Dict[str, object]:
    scored = [r for r in records if r.status == "scored"]
    for record in scored:
        attach_tracker_artifact_metadata(record, args)
        record.adversarial_score = compute_adversarial_score(
            record.completion_ratio,
            record.max_joint_error_rad,
            record.root_trajectory_error_mean_m,
            record.root_displacement_error_m,
            record.fall_detected,
            args,
        )
    best_by_prompt: Dict[str, KimodoRecord] = {}
    for record in scored:
        previous = best_by_prompt.get(record.prompt_id)
        if previous is None or record.adversarial_score < previous.adversarial_score:
            best_by_prompt[record.prompt_id] = record

    ranked_scored = sorted(scored, key=lambda r: r.adversarial_score, reverse=True)
    hard_candidates = [
        record
        for record in ranked_scored
        if is_hard_adversarial_case(asdict(record), args.hard_min_score)
    ]
    hard_cases = hard_candidates[: args.hard_cases]
    hard_prompts_by_text: Dict[str, KimodoRecord] = {}
    for record in hard_candidates:
        key = record.prompt.strip().lower()
        if key not in hard_prompts_by_text:
            hard_prompts_by_text[key] = record
        if len(hard_prompts_by_text) >= args.hard_cases:
            break
    hard_records = [
        {
            "id": f"{record.prompt_id}_hard_s{record.sample_idx:02d}",
            "prompt": record.prompt,
            "category": record.category,
            "difficulty": record.difficulty,
            "duration_sec": record.duration_sec,
            "split": "adversarial_hard",
            "source": "physflow_kimodo_g1_adv_sweep",
            "tags": [
                "adversarial",
                "hard",
                f"seed_{record.seed}",
                f"score_{record.adversarial_score:.3f}",
            ],
        }
        for record in hard_prompts_by_text.values()
    ]
    tracker_pool_config = tracker_pool_config_from_args(args)
    good_pool = sorted(
        [r for r in scored if is_good_tracker_motion(asdict(r), tracker_pool_config)],
        key=lambda r: r.adversarial_score,
    )[: args.good_cases]

    hard_path = out_dir / "hard_prompt_bank.jsonl"
    if hard_path.exists():
        hard_path.unlink()
    hard_path.parent.mkdir(parents=True, exist_ok=True)
    hard_path.touch()
    for prompt_record in hard_records:
        append_jsonl(hard_path, prompt_record)

    tracker_pool_dir = out_dir / "tracker_motion_pool"
    if tracker_pool_dir.exists():
        shutil.rmtree(tracker_pool_dir)
    staged_good = []
    tracker_pool_manifest = []
    seen_motion_hashes = set()
    for record in good_pool:
        copied = asdict(record)
        src_motion = Path(record.motion_path) if record.motion_path else None
        if src_motion and not src_motion.is_absolute():
            src_motion = PROJECT_ROOT / src_motion
        if not src_motion or not src_motion.is_file():
            staged_good.append(copied)
            continue
        motion_hash = file_md5(src_motion)
        if motion_hash in seen_motion_hashes:
            copied["tracker_pool_status"] = "duplicate_skipped"
            copied["source_motion_md5"] = motion_hash
            staged_good.append(copied)
            continue
        seen_motion_hashes.add(motion_hash)
        copied_motion = copy_artifact(str(src_motion), tracker_pool_dir)
        if copied_motion:
            copied["motion_path"] = copied_motion
        copied["tracker_pool_status"] = "staged"
        copied["source_motion_md5"] = motion_hash
        staged_good.append(copied)
        tracker_pool_manifest.append(
            {
                "motion_path": copied.get("motion_path"),
                "source_motion_path": str(src_motion),
                "source_motion_md5": motion_hash,
                "prompt_id": record.prompt_id,
                "prompt": record.prompt,
                "adversarial_score": record.adversarial_score,
                "completion_ratio": record.completion_ratio,
                "max_joint_error_rad": record.max_joint_error_rad,
                "root_trajectory_error_mean_m": record.root_trajectory_error_mean_m,
                "root_displacement_error_m": record.root_displacement_error_m,
            }
        )
    tracker_pool_dir.mkdir(parents=True, exist_ok=True)
    write_json(tracker_pool_dir / "manifest.json", tracker_pool_manifest)

    best_dir = out_dir / "best_by_prompt"
    if best_dir.exists():
        shutil.rmtree(best_dir)
    copied_best = []
    for record in sorted(best_by_prompt.values(), key=lambda r: r.prompt_id):
        copied = asdict(record)
        copied["npz_path"] = copy_artifact(record.npz_path, best_dir / "kimodo_raw") or record.npz_path
        copied["csv_path"] = copy_artifact(record.csv_path, best_dir / "kimodo_raw") or record.csv_path
        copied["motion_path"] = copy_artifact(record.motion_path, best_dir / "proto") or record.motion_path
        copied["robot_json_path"] = copy_artifact(record.robot_json_path, best_dir / "g1_tracker_json") or record.robot_json_path
        copied_best.append(copied)

    next_hard_out = out_dir / "next_hard_adv_sweep"
    next_hard_cmd = None
    if hard_records:
        next_hard_cmd = (
            f"PHYSFLOW_MODE=adv-sweep "
            f"PHYSFLOW_PROMPT_BANK={shlex.quote(str(hard_path))} "
            f"PHYSFLOW_PROMPT_SPLIT=adversarial_hard "
            f"PHYSFLOW_MAX_PROMPTS={len(hard_records)} "
            f"PHYSFLOW_SAMPLES_PER_PROMPT={max(args.samples_per_prompt, 1)} "
            f"PHYSFLOW_HARD_MIN_SCORE={args.hard_min_score:.6g} "
            f"PHYSFLOW_PYTHON_CMD=${{PHYSFLOW_PYTHON_CMD:-/usr/local/bin/python3}} "
            f"bash scripts/embodied/launch_lzy_kimodo_g1_job.sh "
            f"{shlex.quote(str(next_hard_out))} 0 {args.seed + 1000}"
        )
    tracker_cmd = (
        f"PHYSFLOW_MOTION_FILE={shlex.quote(str(tracker_pool_dir))} "
        f"PHYSFLOW_EXPERIMENT_NAME=physflow_g1_xyvel_from_{shlex.quote(out_dir.name)} "
        f"bash scripts/embodied/launch_position_aware_g1_tracker_train.sh"
    )
    next_commands = {
        "continue_t2m_hard_prompt_adv_sweep": next_hard_cmd,
        "train_position_aware_tracker_on_good_pool": tracker_cmd,
    }

    command_script = out_dir / "next_round_commands.sh"
    hard_command_line = (
        f"{next_hard_cmd}\n"
        if next_hard_cmd
        else "# skipped: no hard prompts met PHYSFLOW_HARD_MIN_SCORE in this run\n"
    )
    command_script.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n\n"
        "# Continue adversarial T2M sampling on the hard prompts found in this run.\n"
        f"{hard_command_line}\n"
        "# Fine-tune the position-aware G1 tracker on the good motion pool.\n"
        f"# {tracker_cmd}\n"
    )
    command_script.chmod(0o755)

    selection = {
        "selection_method": "position_aware_adversarial_score",
        "score_terms": config_from_args(args).to_dict(),
        "hard_prompt_min_score": args.hard_min_score,
        "active_g1_tracker": tracker_artifact_metadata(args.g1_onnx) if getattr(args, "g1_onnx", None) else None,
        "tracker_pool_thresholds": tracker_pool_config.to_dict(),
        "num_scored": len(scored),
        "num_hard_candidates": len(hard_candidates),
        "num_below_hard_threshold": len(scored) - len(hard_candidates),
        "top_scored_cases": [asdict(r) for r in ranked_scored[: args.hard_cases]],
        "hard_cases": [asdict(r) for r in hard_cases],
        "hard_prompt_records": hard_records,
        "good_tracker_pool": staged_good,
        "tracker_motion_pool_manifest": str(tracker_pool_dir / "manifest.json"),
        "num_tracker_motion_pool_unique": len(tracker_pool_manifest),
        "num_tracker_motion_pool_duplicates": sum(
            1 for record in staged_good if record.get("tracker_pool_status") == "duplicate_skipped"
        ),
        "best_by_prompt": copied_best,
        "hard_prompt_bank": str(hard_path),
        "tracker_motion_pool": str(tracker_pool_dir),
        "best_by_prompt_dir": str(best_dir),
        "next_commands": next_commands,
        "next_round_commands_script": str(command_script),
    }
    write_json(out_dir / "adversarial_selection.json", selection)
    return selection


def run(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(out_dir / "kimodo_g1.log")],
    )
    write_json(out_dir / "args.json", vars(args))

    if args.mode == "preflight":
        run_preflight(args)
        return

    report = run_preflight(args)
    if args.require_ready:
        if args.mode in {"generate", "loop-smoke", "adv-sweep"} and not report["ready_for_generation"]:
            raise RuntimeError("KIMODO generation dependencies are not ready.")
        if args.mode in {"convert", "score", "loop-smoke", "adv-sweep"} and not report["ready_for_scoring"]:
            raise RuntimeError("G1 scoring dependencies are not ready.")

    prompts = select_prompts(args)
    records = make_records(prompts, args)

    if args.mode in {"generate", "loop-smoke", "adv-sweep"}:
        generated = []
        for record in records:
            try:
                generated.append(generate_record(record, args, out_dir))
            except Exception as exc:
                record.status = "error"
                record.error = f"{type(exc).__name__}: {exc}"
                record.adversarial_score = args.error_penalty
                generated.append(record)
                log.exception("[generate] failed: %s", record.prompt_id)
        records = generated

    if args.mode in {"convert", "score", "loop-smoke", "adv-sweep"}:
        csv_dir = Path(args.csv_dir) if args.csv_dir else out_dir / "kimodo_raw"
        proto_dir = out_dir / "proto"
        convert_csvs_to_proto(csv_dir, proto_dir, args)
        records = attach_motion_paths(records, proto_dir)

    if args.mode in {"score", "loop-smoke", "adv-sweep"}:
        score_log = out_dir / "candidate_scores.jsonl"
        if score_log.exists() and args.force:
            score_log.unlink()
        for record in records:
            if record.status == "error":
                append_jsonl(score_log, asdict(record))
                continue
            try:
                record = score_record(record, args, out_dir)
            except Exception as exc:
                record.status = "error"
                record.error = f"{type(exc).__name__}: {exc}"
                record.adversarial_score = args.error_penalty
                log.exception("[score] failed: %s", record.prompt_id)
            append_jsonl(score_log, asdict(record))
            log.info(
                "[score] %s status=%s adv=%.3f comp=%.2f err=%.3f fall=%s",
                record.prompt_id,
                record.status,
                record.adversarial_score,
                record.completion_ratio,
                record.max_joint_error_rad,
                record.fall_detected,
            )

    summary = summarize(records, out_dir)
    if args.mode == "adv-sweep":
        summary["adversarial_selection"] = write_adversarial_outputs(records, args, out_dir)
        write_json(out_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, default=str))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("PhysFlow KIMODO-G1 runner")
    parser.add_argument("--mode", choices=["preflight", "generate", "convert", "score", "loop-smoke", "adv-sweep"], default="preflight")
    parser.add_argument("--output-dir", default="output/physflow_kimodo_g1/v0")
    parser.add_argument("--prompt-bank", default="configs/experiments/physflow_kimodo_g1/prompt_bank_v0.jsonl")
    parser.add_argument("--prompt-split", choices=["smoke", "train", "eval", "adversarial_hard", "all"], default="smoke")
    parser.add_argument("--prompt-category", default=None, help="Comma-separated category filter.")
    parser.add_argument("--max-difficulty", type=int, default=0)
    parser.add_argument("--max-prompts", type=int, default=2)
    parser.add_argument(
        "--samples-per-prompt",
        type=int,
        default=1,
        help="Number of KIMODO seeds/candidates to generate per selected prompt.",
    )
    parser.add_argument("--hard-cases", type=int, default=8)
    parser.add_argument("--hard-min-score", type=float, default=DEFAULT_G1_HARD_PROMPT_MIN_SCORE)
    parser.add_argument("--good-cases", type=int, default=8)
    parser.add_argument("--good-min-completion", type=float, default=0.95)
    parser.add_argument("--good-max-joint-error", type=float, default=0.7)
    parser.add_argument("--good-max-root-trajectory-error", type=float, default=0.25)
    parser.add_argument("--good-max-root-displacement-error", type=float, default=0.35)
    parser.add_argument(
        "--allow-tracker-pool-without-root-metrics",
        action="store_true",
        help="Allow tracker fine-tune pool entries from pose-only or legacy scoring runs.",
    )
    parser.add_argument("--csv-dir", default=None, help="Existing KIMODO-G1 CSV directory for convert/score modes.")
    parser.add_argument("--kimodo-model", default="Kimodo-G1-RP-v1")
    parser.add_argument("--diffusion-steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cfg-type", default="separated")
    parser.add_argument("--cfg-weight", type=float, nargs="*", default=[2.0, 2.0])
    parser.add_argument("--text-encoder", default=None, help="KIMODO TEXT_ENCODER override, e.g. dummy for infra smoke.")
    parser.add_argument("--checkpoint-dir", default=None, help="KIMODO CHECKPOINT_DIR override.")
    parser.add_argument("--local-cache", action="store_true", help="Resolve HuggingFace models from local cache only when possible.")
    parser.add_argument("--cuda-visible-devices", default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--require-ready", action="store_true")

    parser.add_argument("--g1-onnx", default=str(DEFAULT_POSITION_AWARE_ONNX))
    parser.add_argument(
        "--allow-pose-only-tracker",
        action="store_true",
        help="Allow the released g1-bones-deploy tracker as an explicit pose-only baseline.",
    )
    parser.add_argument("--g1-mjcf", default=str(DEFAULT_MJCF))
    parser.add_argument("--robot-json-subsample", type=int, default=1)
    parser.add_argument("--joint-error-scale", type=float, default=DEFAULT_G1_SCORE_CONFIG.joint_error_scale)
    parser.add_argument(
        "--root-trajectory-error-scale",
        type=float,
        default=DEFAULT_G1_SCORE_CONFIG.root_trajectory_error_scale,
        help="Meters corresponding to one adversarial score unit for mean root trajectory error.",
    )
    parser.add_argument(
        "--root-displacement-error-scale",
        type=float,
        default=DEFAULT_G1_SCORE_CONFIG.root_displacement_error_scale,
        help="Meters corresponding to one adversarial score unit for final root displacement error.",
    )
    parser.add_argument(
        "--root-trajectory-error-weight",
        type=float,
        default=DEFAULT_G1_SCORE_CONFIG.root_trajectory_error_weight,
    )
    parser.add_argument(
        "--root-displacement-error-weight",
        type=float,
        default=DEFAULT_G1_SCORE_CONFIG.root_displacement_error_weight,
    )
    parser.add_argument("--score-component-cap", type=float, default=DEFAULT_G1_SCORE_CONFIG.score_component_cap)
    parser.add_argument("--fall-penalty", type=float, default=DEFAULT_G1_SCORE_CONFIG.fall_penalty)
    parser.add_argument("--error-penalty", type=float, default=5.0)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
