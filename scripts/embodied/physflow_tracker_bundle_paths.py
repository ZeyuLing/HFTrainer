"""Dependency-light access to bundled PhysFlow tracker paths.

Some tracker scripts run in the IsaacGym py3.8 environment, which intentionally
does not import the full hftrainer stack. Loading the canonical path module by
filename avoids triggering ``hftrainer.__init__`` while keeping a single source
of truth for the filesystem layout.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
_PATHS_FILE = PROJECT_ROOT / "hftrainer" / "models" / "motion" / "physflow" / "trackers" / "paths.py"
_SPEC = importlib.util.spec_from_file_location("physflow_tracker_paths", _PATHS_FILE)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"cannot load tracker paths from {_PATHS_FILE}")
_MOD = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MOD)

TRACKERS_ROOT = _MOD.TRACKERS_ROOT
PROTOMOTIONS_ROOT = _MOD.PROTOMOTIONS_ROOT
PROTOMOTIONS_G1_TRACKER_ROOT = _MOD.PROTOMOTIONS_G1_TRACKER_ROOT
PROTOMOTIONS_G1_ONNX = _MOD.PROTOMOTIONS_G1_ONNX
PROTOMOTIONS_G1_YAML = _MOD.PROTOMOTIONS_G1_YAML
PROTOMOTIONS_G1_CKPT = _MOD.PROTOMOTIONS_G1_CKPT
PROTOMOTIONS_G1_MJCF = _MOD.PROTOMOTIONS_G1_MJCF
PROTOMOTIONS_G1_URDF = _MOD.PROTOMOTIONS_G1_URDF
PROTOMOTIONS_G1_MESH_DIR = _MOD.PROTOMOTIONS_G1_MESH_DIR
ANY2TRACK_ROOT = _MOD.ANY2TRACK_ROOT
ANY2TRACK_ONNX = _MOD.ANY2TRACK_ONNX
ANY2TRACK_CONFIG = _MOD.ANY2TRACK_CONFIG
ANY2TRACK_CHECKPOINT_CONFIG = _MOD.ANY2TRACK_CHECKPOINT_CONFIG
ANY2TRACK_G1_MJCF = _MOD.ANY2TRACK_G1_MJCF
HUMANOID_GPT_ROOT = _MOD.HUMANOID_GPT_ROOT
HUMANOID_GPT_ONNX = _MOD.HUMANOID_GPT_ONNX
HUMANOID_GPT_VENV_PYTHON = _MOD.HUMANOID_GPT_VENV_PYTHON
resolve_project_path = _MOD.resolve_project_path
