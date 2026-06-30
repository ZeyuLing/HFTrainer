"""Reusable MBench Body_Penetration evaluator.

Body_Penetration is a self-collision score on SMPL mesh vertices.  Lower values
are better.  MBench's official implementation uses ``torch-mesh-isect`` via
``mesh_intersection.bvh_search_tree.BVH`` and reports the percentage of
self-colliding triangles averaged over frames and clips.

This module exposes the metric as a reusable API and keeps the collision backend
explicit.  ``backend="official"`` requires ``mesh_intersection``.  The
``backend="winding"`` fallback mirrors the previous repository script and uses
libigl winding numbers; it is a geometry proxy, not byte-for-byte official
MBench.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from scipy.spatial.transform import Rotation as R

from hftrainer.datasets.motion.representation.humanml_repr import (
    recover_local_rotations_and_root,
)
from hftrainer.evaluation.motion.mbench_physics import (
    MotionMode,
    list_metric_files,
    load_manifest,
)

BODY_PENETRATION_KEY = "BodyPenet"
BODY_PENETRATION_DIRECTION: Literal["lower"] = "lower"
BODY_PENETRATION_LOWER_IS_BETTER = True
CollisionBackend = Literal["auto", "official", "winding"]

_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SMPL_DIR = _ROOT / "ref_repo" / "ViMoGen" / "data" / "body_models" / "smpl"
_VIMOGEN = _ROOT / "ref_repo" / "ViMoGen"
_WORKER_CFG = None
_WORKER_SMPL_MODEL = None


@dataclass(frozen=True)
class BodyPenetrationConfig:
    """Configuration for body-penetration evaluation."""

    backend: CollisionBackend = "auto"
    frame_step: int = 2
    winding_eps: float = 0.001
    smpl_dir: str | os.PathLike[str] = DEFAULT_SMPL_DIR
    device: str = "cpu"


class MissingBodyPenetrationDependency(RuntimeError):
    """Raised when the requested collision backend is unavailable."""


def _ensure_vimogen_on_path() -> None:
    path = str(_VIMOGEN)
    if path not in sys.path:
        sys.path.insert(0, path)


def _rot6d_to_rotmat_rowmajor(d6: np.ndarray) -> np.ndarray:
    x = np.asarray(d6, dtype=np.float32).reshape(*d6.shape[:-1], 3, 2)
    a1, a2 = x[..., 0], x[..., 1]
    b1 = a1 / (np.linalg.norm(a1, axis=-1, keepdims=True) + 1e-8)
    a2p = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = a2p / (np.linalg.norm(a2p, axis=-1, keepdims=True) + 1e-8)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=-1)


def _axis_angle_from_rotmat(rotmat: np.ndarray) -> np.ndarray:
    rot = np.asarray(rotmat, dtype=np.float32)
    return R.from_matrix(rot.reshape(-1, 3, 3)).as_rotvec().reshape(rot.shape[0], 21, 3).astype(np.float32)


def body_axis_angle_from_motion135(motion_135: np.ndarray) -> np.ndarray | None:
    """Return SMPL body-joint axis-angle ``(T,21,3)`` from ``motion_135``."""

    motion = np.asarray(motion_135, dtype=np.float32)
    if motion.ndim != 2 or motion.shape[1] < 135:
        return None
    rot6d = motion[:, 3:135].reshape(-1, 22, 6)[:, 1:22]
    rotmat = _rot6d_to_rotmat_rowmajor(rot6d)
    return _axis_angle_from_rotmat(rotmat)


def body_axis_angle_from_smpl_npz(data: np.lib.npyio.NpzFile) -> np.ndarray | None:
    """Return body axis-angle from an SMPL-param NPZ."""

    if "body_pose" in data:
        body = np.asarray(data["body_pose"], dtype=np.float32)
        if body.ndim == 2 and body.shape[1] == 63:
            return body.reshape(-1, 21, 3)
        if body.ndim == 3 and body.shape[1:] == (21, 3):
            return body
        return None
    if "poses" in data:
        poses = np.asarray(data["poses"], dtype=np.float32)
        if poses.ndim == 2 and poses.shape[1] >= 66:
            return poses[:, 3:66].reshape(-1, 21, 3).astype(np.float32)
    return None


def body_axis_angle_from_272(motion_272: np.ndarray) -> np.ndarray | None:
    """Return SMPL body-joint axis-angle ``(T,21,3)`` from native 272 motion."""

    motion = np.asarray(motion_272, dtype=np.float32)
    if motion.ndim != 2 or motion.shape[1] != 272:
        return None
    rot, _ = recover_local_rotations_and_root(motion)
    return _axis_angle_from_rotmat(np.asarray(rot, dtype=np.float32)[:, 1:22])


def body_axis_angle_for_file(path: str | os.PathLike[str], mode: MotionMode) -> np.ndarray | None:
    """Load one supported motion file and return SMPL body axis-angle."""

    p = str(path)
    try:
        if mode == "m135":
            data = np.load(p, allow_pickle=True)
            if "motion_135" in data:
                return body_axis_angle_from_motion135(np.asarray(data["motion_135"], dtype=np.float32))
            return body_axis_angle_from_smpl_npz(data)
        if mode == "gt272":
            if p.endswith(".npz"):
                data = np.load(p, allow_pickle=True)
                if "motion_272" not in data:
                    return None
                motion_272 = np.asarray(data["motion_272"], dtype=np.float32)
            else:
                motion_272 = np.asarray(np.load(p), dtype=np.float32)
            return body_axis_angle_from_272(motion_272)
    except Exception:
        return None
    raise ValueError(f"Unknown mode {mode!r}")


def load_smpl_model(
    smpl_dir: str | os.PathLike[str] = DEFAULT_SMPL_DIR,
    batch_size: int = 1,
    device: str = "cpu",
):
    """Load the neutral SMPL model used for Body_Penetration vertices."""

    # Some legacy SMPL pickles import chumpy, which still references removed
    # NumPy scalar aliases under modern NumPy.
    for name, value in {
        "bool": bool,
        "int": int,
        "float": float,
        "complex": complex,
        "object": object,
        "str": str,
        "unicode": str,
    }.items():
        if name not in np.__dict__:
            setattr(np, name, value)

    import smplx

    model = smplx.SMPL(
        model_path=str(smpl_dir),
        gender="neutral",
        batch_size=batch_size,
        use_pca=False,
    ).to(device)
    model.eval()
    return model


def vertices_from_body_axis_angle(
    body_axis_angle: np.ndarray,
    smpl_model,
    device: str = "cpu",
) -> np.ndarray:
    """Convert ``(T,21,3)`` body pose to SMPL vertices with zero root/translation."""

    import torch

    aa = np.asarray(body_axis_angle, dtype=np.float32)
    if aa.ndim != 3 or aa.shape[1:] != (21, 3):
        raise ValueError(f"Expected body axis-angle shape (T,21,3), got {aa.shape}")
    T = aa.shape[0]
    body = np.zeros((T, 69), dtype=np.float32)
    body[:, :63] = aa.reshape(T, 63)
    with torch.no_grad():
        out = smpl_model(
            global_orient=torch.zeros(T, 3, device=device),
            body_pose=torch.from_numpy(body).to(device),
            betas=torch.zeros(T, 10, device=device),
            transl=torch.zeros(T, 3, device=device),
        )
    return out.vertices.detach().cpu().numpy().astype(np.float64)


def compute_body_penetration_from_vertices_official(
    vertices: np.ndarray,
    faces: np.ndarray,
    device: str = "cpu",
) -> float:
    """Compute official MBench BVH triangle-collision percentage."""

    try:
        import torch
        from mesh_intersection.bvh_search_tree import BVH
    except Exception as exc:  # noqa: BLE001
        raise MissingBodyPenetrationDependency(
            "backend='official' requires mesh_intersection / torch-mesh-isect"
        ) from exc

    verts = np.asarray(vertices, dtype=np.float32)
    faces_t = torch.as_tensor(np.asarray(faces, dtype=np.int64), device=device, dtype=torch.long)
    bvh = BVH(max_collisions=8)
    scores = []
    for frame in range(verts.shape[0]):
        v = torch.as_tensor(verts[frame], device=device, dtype=torch.float32).unsqueeze(0)
        triangles = v[:, faces_t]
        outputs = bvh(triangles).detach().cpu().numpy().squeeze()
        collisions = outputs[outputs[:, 0] >= 0, :]
        scores.append(collisions.shape[0] / float(triangles.shape[1]) * 100.0)
    return float(np.mean(scores)) if scores else 0.0


def compute_body_penetration_from_vertices_winding(
    vertices: np.ndarray,
    faces: np.ndarray,
    frame_step: int = 2,
    eps: float = 0.001,
) -> float:
    """Compute libigl winding-number proxy body penetration percentage."""

    try:
        import igl
    except Exception as exc:  # noqa: BLE001
        raise MissingBodyPenetrationDependency("backend='winding' requires igl / libigl") from exc

    verts = np.asarray(vertices, dtype=np.float64)
    faces_i = np.asarray(faces, dtype=np.int32)
    scores = []
    for frame in range(0, verts.shape[0], max(1, int(frame_step))):
        vt = verts[frame]
        normals = igl.per_vertex_normals(vt, faces_i)
        query = vt + eps * normals
        wn = igl.fast_winding_number_for_meshes(vt, faces_i, query)
        scores.append(float((wn > 0.5).mean()) * 100.0)
    return float(np.mean(scores)) if scores else 0.0


def compute_body_penetration_from_vertices(
    vertices: np.ndarray,
    faces: np.ndarray,
    cfg: BodyPenetrationConfig = BodyPenetrationConfig(),
) -> float:
    """Compute Body_Penetration from vertices using the configured backend."""

    official_error = None
    if cfg.backend in ("auto", "official"):
        try:
            return compute_body_penetration_from_vertices_official(vertices, faces, device=cfg.device)
        except MissingBodyPenetrationDependency as exc:
            if cfg.backend == "official":
                raise
            official_error = exc
    try:
        return compute_body_penetration_from_vertices_winding(
            vertices,
            faces,
            frame_step=cfg.frame_step,
            eps=cfg.winding_eps,
        )
    except MissingBodyPenetrationDependency as winding_error:
        if official_error is None:
            raise
        raise MissingBodyPenetrationDependency(
            "backend='auto' could not find an available Body_Penetration backend: "
            f"{official_error}; {winding_error}"
        ) from winding_error


def compute_body_penetration_for_file(
    path: str | os.PathLike[str],
    mode: MotionMode,
    cfg: BodyPenetrationConfig = BodyPenetrationConfig(),
    smpl_model=None,
) -> float | None:
    """Compute Body_Penetration for one motion file."""

    aa = body_axis_angle_for_file(path, mode)
    if aa is None or aa.shape[0] < 1:
        return None
    model = smpl_model if smpl_model is not None else load_smpl_model(cfg.smpl_dir, batch_size=aa.shape[0], device=cfg.device)
    vertices = vertices_from_body_axis_angle(aa, model, device=cfg.device)
    faces = np.asarray(model.faces, dtype=np.int64)
    return compute_body_penetration_from_vertices(vertices, faces, cfg)


def aggregate_body_penetration_scores(scores: list[float | None]) -> dict[str, float]:
    """Average per-clip Body_Penetration scores."""

    valid = [float(v) for v in scores if v is not None]
    return {
        "n": len(valid),
        BODY_PENETRATION_KEY: float(np.mean(valid)) if valid else 0.0,
    }


def evaluate_body_penetration_dir(
    src: str | os.PathLike[str],
    mode: MotionMode,
    limit: int = 150,
    seed: int = 0,
    cfg: BodyPenetrationConfig = BodyPenetrationConfig(),
    workers: int = 1,
) -> dict[str, float]:
    """Evaluate Body_Penetration over all metric files in one directory."""

    files = list_metric_files(src, mode, limit=limit, seed=seed)
    if workers <= 1:
        scores = [compute_body_penetration_for_file(path, mode, cfg=cfg) for path in files]
        return aggregate_body_penetration_scores(scores)

    with mp.Pool(workers, initializer=_init_worker, initargs=(cfg,)) as pool:
        scores = list(pool.imap_unordered(_one, [(path, mode) for path in files], chunksize=2))
    return aggregate_body_penetration_scores(scores)


def _init_worker(cfg: BodyPenetrationConfig) -> None:
    global _WORKER_CFG, _WORKER_SMPL_MODEL
    _WORKER_CFG = cfg
    _WORKER_SMPL_MODEL = load_smpl_model(cfg.smpl_dir, batch_size=1, device=cfg.device)


def _one(task: tuple[str, MotionMode]) -> float | None:
    path, mode = task
    try:
        return compute_body_penetration_for_file(
            path,
            mode,
            cfg=_WORKER_CFG,
            smpl_model=_WORKER_SMPL_MODEL,
        )
    except MissingBodyPenetrationDependency:
        raise
    except Exception:
        return None


def dump_results_json(results: dict[str, dict[str, float]], out_json: str | os.PathLike[str]) -> None:
    """Write Body_Penetration results to JSON, creating parent directories."""

    out_path = Path(out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=1))


__all__ = [
    "BODY_PENETRATION_DIRECTION",
    "BODY_PENETRATION_KEY",
    "BODY_PENETRATION_LOWER_IS_BETTER",
    "BodyPenetrationConfig",
    "MissingBodyPenetrationDependency",
    "aggregate_body_penetration_scores",
    "body_axis_angle_for_file",
    "body_axis_angle_from_272",
    "body_axis_angle_from_motion135",
    "body_axis_angle_from_smpl_npz",
    "compute_body_penetration_for_file",
    "compute_body_penetration_from_vertices",
    "compute_body_penetration_from_vertices_official",
    "compute_body_penetration_from_vertices_winding",
    "dump_results_json",
    "evaluate_body_penetration_dir",
    "list_metric_files",
    "load_manifest",
    "load_smpl_model",
    "vertices_from_body_axis_angle",
]
