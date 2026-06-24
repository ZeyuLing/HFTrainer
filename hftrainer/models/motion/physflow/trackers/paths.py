"""Canonical paths for packaged PhysFlow tracker baselines."""

from __future__ import annotations

from pathlib import Path

TRACKERS_ROOT = Path(__file__).resolve().parent

PROTOMOTIONS_ROOT = TRACKERS_ROOT / "protomotions" / "vendor"
PROTOMOTIONS_G1_TRACKER_ROOT = (
    PROTOMOTIONS_ROOT
    / "data"
    / "pretrained_models"
    / "motion_tracker"
    / "g1-bones-deploy"
)
PROTOMOTIONS_G1_ONNX = PROTOMOTIONS_G1_TRACKER_ROOT / "compiled_models" / "unified_pipeline.onnx"
PROTOMOTIONS_G1_YAML = PROTOMOTIONS_G1_TRACKER_ROOT / "compiled_models" / "unified_pipeline.yaml"
PROTOMOTIONS_G1_CKPT = PROTOMOTIONS_G1_TRACKER_ROOT / "last.ckpt"
PROTOMOTIONS_G1_MJCF = PROTOMOTIONS_ROOT / "protomotions" / "data" / "assets" / "mjcf" / "g1_holo_compat.xml"
PROTOMOTIONS_G1_URDF = (
    PROTOMOTIONS_ROOT
    / "protomotions"
    / "data"
    / "assets"
    / "urdf"
    / "for_retargeting"
    / "g1.urdf"
)
PROTOMOTIONS_G1_MESH_DIR = PROTOMOTIONS_ROOT / "protomotions" / "data" / "assets" / "mesh" / "G1"

ANY2TRACK_ROOT = TRACKERS_ROOT / "any2track"
ANY2TRACK_ONNX = (
    ANY2TRACK_ROOT
    / "storage"
    / "logs"
    / "dagger"
    / "general_tracker_lafan1_v2"
    / "checkpoints"
    / "model.onnx"
)
ANY2TRACK_CONFIG = (
    ANY2TRACK_ROOT
    / "storage"
    / "logs"
    / "dagger"
    / "general_tracker_lafan1_v2"
    / "config.json"
)
ANY2TRACK_CHECKPOINT_CONFIG = ANY2TRACK_ONNX.with_name("config.json")
ANY2TRACK_G1_MJCF = ANY2TRACK_ROOT / "storage" / "assets" / "unitree_g1" / "scene_mjx_flat_terrain.xml"

HUMANOID_GPT_ROOT = TRACKERS_ROOT / "humanoid_gpt"
HUMANOID_GPT_ONNX = HUMANOID_GPT_ROOT / "storage" / "ckpts" / "pns_wo_priv216.onnx"
HUMANOID_GPT_VENV_PYTHON = HUMANOID_GPT_ROOT / ".venv311" / "bin" / "python"


def resolve_project_path(path: str | Path | None, default: Path, project_root: Path) -> Path:
    """Resolve user/config paths against the project root, with bundled default."""
    p = Path(path) if path else default
    if not p.is_absolute():
        p = project_root / p
    return p.resolve()
