#!/usr/bin/env python3
"""Stage hard G1 motions for OpenTrack adversarial training.

The OpenTrack DAgger implementation expects motion names under
``storage/data/mocap/lafan1/UnitreeG1``.  This helper stages any selected
OpenTrack-format ``.npz`` motions into that directory with a stable prefix and
writes both JSON and text manifests containing the staged trajectory names.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import shutil
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OPENTRACK_MOTION_DIR = (
    PROJECT_ROOT / "ref_repo/OpenTrack/storage/data/mocap/lafan1/UnitreeG1"
)
DEFAULT_OPENTRACK_G1_XML = (
    PROJECT_ROOT / "ref_repo/OpenTrack/storage/assets/unitree_g1/scene_mjx_flat_terrain.xml"
)

INFO_KEYS = {"joint_names", "frequency", "body_names", "site_names", "metadata"}
MODEL_KEYS = {
    "njnt",
    "jnt_type",
    "nbody",
    "body_rootid",
    "body_weldid",
    "body_mocapid",
    "body_pos",
    "body_quat",
    "body_ipos",
    "body_iquat",
    "nsite",
    "site_bodyid",
    "site_pos",
    "site_quat",
}
DATA_KEYS = {
    "qpos",
    "qvel",
    "xpos",
    "xquat",
    "cvel",
    "subtree_com",
    "site_xpos",
    "site_xmat",
    "split_points",
}


def safe_name(path: Path, prefix: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", path.stem).strip("_")
    if prefix and not stem.startswith(prefix):
        stem = f"{prefix}{stem}"
    return stem


def stage_file(src: Path, dst: Path, mode: str, force: bool) -> None:
    if dst.exists() or dst.is_symlink():
        if not force:
            return
        dst.unlink()
    dst.parent.mkdir(parents=True, exist_ok=True)
    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "symlink":
        dst.symlink_to(src.resolve())
    else:
        raise ValueError(f"Unsupported stage mode: {mode}")


def stage_opentrack_file(src: Path, dst: Path, mode: str, force: bool, normalize_metadata: bool) -> bool:
    """Stage an OpenTrack trajectory and make per-file info concat-safe."""
    if not normalize_metadata:
        stage_file(src, dst, mode, force)
        return False

    if dst.exists() or dst.is_symlink():
        if not force:
            return False
        dst.unlink()
    dst.parent.mkdir(parents=True, exist_ok=True)
    with np.load(src, allow_pickle=True) as raw:
        item = {key: raw[key] for key in raw.files}
    item["metadata"] = _none_object()
    np.savez(str(dst), **item)
    return True


def npz_keys(path: Path) -> set[str]:
    with np.load(path, allow_pickle=True) as data:
        return set(data.files)


def is_opentrack_trajectory(path: Path) -> bool:
    keys = npz_keys(path)
    required = {"qpos", "frequency", "njnt", "jnt_type", "joint_names", "split_points"}
    return required.issubset(keys)


def is_qpos_only(path: Path) -> bool:
    keys = npz_keys(path)
    return "qpos" in keys and not {"njnt", "jnt_type", "joint_names"}.issubset(keys)


def _none_object() -> np.ndarray:
    return np.array(None, dtype=object)


def _joint_names_from_model(model: Any, mujoco_mod: Any) -> list[str]:
    return [
        mujoco_mod.mj_id2name(model, mujoco_mod.mjtObj.mjOBJ_JOINT, i)
        for i in range(model.njnt)
    ]


def _names_from_model(model: Any, mujoco_mod: Any, obj: Any, count: int, prefix: str) -> list[str]:
    names: list[str] = []
    for i in range(count):
        name = mujoco_mod.mj_id2name(model, obj, i)
        names.append(name if name is not None else f"{prefix}_{i}")
    return names


def _kinematics_from_qpos(qpos: np.ndarray, frequency: float, model: Any, mujoco_mod: Any) -> dict[str, np.ndarray]:
    qpos64 = np.asarray(qpos, dtype=np.float64)
    qvel64 = np.zeros((qpos.shape[0], model.nv), dtype=np.float64)
    if qpos.shape[0] > 1:
        dt = 1.0 / max(float(frequency), 1e-6)
        for i in range(qpos.shape[0] - 1):
            mujoco_mod.mj_differentiatePos(model, qvel64[i], dt, qpos64[i], qpos64[i + 1])
        qvel64[-1] = qvel64[-2]

    data = mujoco_mod.MjData(model)
    xpos = np.empty((qpos.shape[0], model.nbody, 3), dtype=np.float32)
    xquat = np.empty((qpos.shape[0], model.nbody, 4), dtype=np.float32)
    cvel = np.empty((qpos.shape[0], model.nbody, 6), dtype=np.float32)
    subtree_com = np.empty((qpos.shape[0], model.nbody, 3), dtype=np.float32)
    site_xpos = np.empty((qpos.shape[0], model.nsite, 3), dtype=np.float32)
    site_xmat = np.empty((qpos.shape[0], model.nsite, 9), dtype=np.float32)
    for i, pose in enumerate(qpos64):
        data.qpos[:] = pose
        data.qvel[:] = qvel64[i]
        mujoco_mod.mj_forward(model, data)
        xpos[i] = data.xpos
        xquat[i] = data.xquat
        cvel[i] = data.cvel
        subtree_com[i] = data.subtree_com
        site_xpos[i] = data.site_xpos
        site_xmat[i] = data.site_xmat
    return {
        "qvel": qvel64.astype(np.float32),
        "xpos": xpos,
        "xquat": xquat,
        "cvel": cvel,
        "subtree_com": subtree_com,
        "site_xpos": site_xpos,
        "site_xmat": site_xmat,
    }


def _valid_existing_opentrack(path: Path) -> bool:
    try:
        return path.exists() and is_opentrack_trajectory(path)
    except Exception:
        return False


def select_files(files: list[Path], max_files: int, strategy: str, seed: int) -> list[Path]:
    if max_files <= 0 or len(files) <= max_files:
        return files
    if strategy == "first":
        return files[:max_files]
    if strategy == "evenly":
        if max_files == 1:
            return [files[0]]
        last = len(files) - 1
        indices = sorted({round(i * last / (max_files - 1)) for i in range(max_files)})
        return [files[i] for i in indices]
    if strategy == "random":
        rng = random.Random(seed)
        return sorted(rng.sample(files, max_files))
    raise ValueError(f"Unsupported selection strategy: {strategy}")


def repair_qpos_only_to_opentrack(
    src: Path,
    dst: Path,
    model_xml: Path,
    force: bool,
    model: Any | None = None,
    mujoco_mod: Any | None = None,
    joint_names: list[str] | None = None,
) -> None:
    if dst.exists() or dst.is_symlink():
        if not force and _valid_existing_opentrack(dst) and not dst.is_symlink():
            return
        dst.unlink()

    if mujoco_mod is None:
        try:
            import mujoco as mujoco_mod
        except Exception as exc:  # pragma: no cover - depends on runtime env
            raise RuntimeError(
                "qpos-only PhysFlow motions require mujoco to build OpenTrack metadata"
            ) from exc

    with np.load(src, allow_pickle=True) as raw:
        qpos = np.asarray(raw["qpos"], dtype=np.float32)
        frequency = float(np.asarray(raw["frequency"]).item()) if "frequency" in raw.files else 30.0

    if qpos.ndim != 2:
        raise ValueError(f"{src} qpos must be rank-2, got shape={qpos.shape}")

    if model is None:
        model = mujoco_mod.MjModel.from_xml_path(str(model_xml))
    if joint_names is None:
        joint_names = _joint_names_from_model(model, mujoco_mod)
    if qpos.shape[1] != model.nq:
        raise ValueError(
            f"{src} qpos dim {qpos.shape[1]} does not match OpenTrack G1 model.nq={model.nq}; "
            "do a real retarget/projection before staging."
        )
    kinematics = _kinematics_from_qpos(qpos, frequency, model, mujoco_mod)

    item = {
        "qpos": qpos,
        "qvel": kinematics["qvel"],
        "xpos": kinematics["xpos"],
        "xquat": kinematics["xquat"],
        "cvel": kinematics["cvel"],
        "subtree_com": kinematics["subtree_com"],
        "site_xpos": kinematics["site_xpos"],
        "site_xmat": kinematics["site_xmat"],
        "split_points": np.array([0, qpos.shape[0]], dtype=np.int32),
        "joint_names": np.asarray(joint_names, dtype=str),
        "frequency": np.asarray(frequency, dtype=np.float32),
        "body_names": np.asarray(
            _names_from_model(model, mujoco_mod, mujoco_mod.mjtObj.mjOBJ_BODY, model.nbody, "body"),
            dtype=str,
        ),
        "site_names": np.asarray(
            _names_from_model(model, mujoco_mod, mujoco_mod.mjtObj.mjOBJ_SITE, model.nsite, "site"),
            dtype=str,
        ),
        "metadata": _none_object(),
        "njnt": np.asarray(model.njnt, dtype=np.int32),
        "jnt_type": np.asarray(model.jnt_type, dtype=np.int32),
        "nbody": np.asarray(model.nbody, dtype=np.int32),
        "body_rootid": np.asarray(model.body_rootid, dtype=np.int32),
        "body_weldid": np.asarray(model.body_weldid, dtype=np.int32),
        "body_mocapid": np.asarray(model.body_mocapid, dtype=np.int32),
        "body_pos": np.asarray(model.body_pos, dtype=np.float32),
        "body_quat": np.asarray(model.body_quat, dtype=np.float32),
        "body_ipos": np.asarray(model.body_ipos, dtype=np.float32),
        "body_iquat": np.asarray(model.body_iquat, dtype=np.float32),
        "nsite": np.asarray(model.nsite, dtype=np.int32),
        "site_bodyid": np.asarray(model.site_bodyid, dtype=np.int32),
        "site_pos": np.asarray(model.site_pos, dtype=np.float32),
        "site_quat": np.asarray(model.site_quat, dtype=np.float32),
    }
    missing = (INFO_KEYS | MODEL_KEYS | DATA_KEYS) - set(item)
    if missing:
        raise AssertionError(f"internal staging error, missing keys: {sorted(missing)}")

    dst.parent.mkdir(parents=True, exist_ok=True)
    np.savez(str(dst), **item)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OPENTRACK_MOTION_DIR)
    parser.add_argument("--manifest-json", type=Path, required=True)
    parser.add_argument("--manifest-txt", type=Path, default=None)
    parser.add_argument("--prefix", default="physflow_adv_")
    parser.add_argument("--keywords", default="", help="Comma-separated filename filters.")
    parser.add_argument("--max-files", type=int, default=0)
    parser.add_argument("--selection-strategy", choices=["first", "evenly", "random"], default="first")
    parser.add_argument("--selection-seed", type=int, default=0)
    parser.add_argument("--mode", choices=["symlink", "copy"], default="symlink")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--normalize-metadata", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--repair-qpos-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--model-xml", type=Path, default=DEFAULT_OPENTRACK_G1_XML)
    args = parser.parse_args()

    keywords = [x.strip().lower() for x in args.keywords.split(",") if x.strip()]
    files: list[Path] = []
    for input_dir in args.input_dir:
        files.extend(sorted(input_dir.rglob("*.npz")))
    if keywords:
        files = [p for p in files if any(k in p.name.lower() for k in keywords)]
    files = select_files(files, args.max_files, args.selection_strategy, args.selection_seed)
    if not files:
        raise SystemExit("No input .npz files selected.")

    staged: list[dict[str, str]] = []
    used_names: set[str] = set()
    repair_model: Any | None = None
    repair_mujoco: Any | None = None
    repair_joint_names: list[str] | None = None
    for src in files:
        name = safe_name(src, args.prefix)
        base = name
        suffix = 1
        while name in used_names:
            name = f"{base}_{suffix:02d}"
            suffix += 1
        used_names.add(name)
        dst = args.output_dir / f"{name}.npz"
        repaired = False
        if args.repair_qpos_only and is_qpos_only(src) and not is_opentrack_trajectory(src):
            if repair_model is None or repair_mujoco is None or repair_joint_names is None:
                try:
                    import mujoco as repair_mujoco
                except Exception as exc:  # pragma: no cover - depends on runtime env
                    raise RuntimeError(
                        "qpos-only PhysFlow motions require mujoco to build OpenTrack metadata"
                    ) from exc
                repair_model = repair_mujoco.MjModel.from_xml_path(str(args.model_xml))
                repair_joint_names = _joint_names_from_model(repair_model, repair_mujoco)
            repair_qpos_only_to_opentrack(
                src,
                dst,
                args.model_xml,
                args.force,
                model=repair_model,
                mujoco_mod=repair_mujoco,
                joint_names=repair_joint_names,
            )
            repaired = True
        else:
            repaired = stage_opentrack_file(src, dst, args.mode, args.force, args.normalize_metadata)
        staged.append({"name": name, "source": str(src), "path": str(dst), "repaired": str(repaired).lower()})

    args.manifest_json.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_json.write_text(json.dumps({"motions": [x["name"] for x in staged], "items": staged}, indent=2) + "\n")
    if args.manifest_txt:
        args.manifest_txt.parent.mkdir(parents=True, exist_ok=True)
        args.manifest_txt.write_text("\n".join(x["name"] for x in staged) + "\n")
    print(f"staged={len(staged)} output_dir={args.output_dir}")
    print(args.manifest_json)


if __name__ == "__main__":
    main()
