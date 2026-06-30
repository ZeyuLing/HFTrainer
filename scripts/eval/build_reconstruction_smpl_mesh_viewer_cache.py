#!/usr/bin/env python3
"""Build a Three.js SMPL-mesh cache for HumanML3D reconstruction outputs."""
from __future__ import annotations

import argparse
import base64
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from hftrainer.datasets.motion.representation.humanml_repr import (  # noqa: E402
    recover_local_rotations_and_root,
)
from hftrainer.motion.representation.rotation import (  # noqa: E402
    matrix_to_axis_angle,
    rotation_6d_to_matrix,
)


ANNO = REPO / (
    "outputs/evaluation/t2m/humanml3d_official_test/captions/"
    "gt_motionclip_selected_20260622/"
    "test_hml3d_official272_gtlen_motionclip_selected_caption.json"
)
RECON_ROOT = REPO / "outputs/evaluation/reconstruction/humanml3d_official_test"
GT_MOTION135 = REPO / "outputs/evaluation/t2m/humanml3d_official_test/motion135/gt_0beta"
SMPL_MODEL_DIR = REPO / "checkpoints/baselines/body_models"
OUT_DIR = RECON_ROOT / "viewers/smpl_mesh_20260630"

METHOD_SPECS: dict[str, dict[str, Any]] = {
    "gt_0beta": {
        "label": "GT",
        "kind": "motion135",
        "path": GT_MOTION135,
        "color": "#cbd5e1",
    },
    "t2mgpt": {
        "label": "T2M-GPT",
        "kind": "motion135",
        "path": RECON_ROOT / "motion135/t2mgpt",
        "color": "#60a5fa",
    },
    "momask": {
        "label": "MoMask",
        "kind": "motion135",
        "path": RECON_ROOT / "motion135/momask",
        "color": "#34d399",
    },
    "mld": {
        "label": "MLD",
        "kind": "motion135",
        "path": RECON_ROOT / "motion135/mld",
        "color": "#fbbf24",
    },
    "mogents": {
        "label": "MoGenTS",
        "kind": "motion135",
        "path": RECON_ROOT / "motion135/mogents",
        "color": "#f472b6",
    },
    "motiongpt3": {
        "label": "MotionGPT3",
        "kind": "motion135",
        "path": RECON_ROOT / "motion135/motiongpt3",
        "color": "#a78bfa",
    },
    "motionstreamer": {
        "label": "MotionStreamer",
        "kind": "ms272",
        "path": RECON_ROOT / "ms272/motionstreamer",
        "color": "#fb7185",
    },
    "prism": {
        "label": "PRISM",
        "kind": "motion135",
        "path": RECON_ROOT / "motion135/prism",
        "color": "#22c55e",
    },
    "vermo": {
        "label": "VerMo",
        "kind": "motion135",
        "path": RECON_ROOT / "motion135/vermo",
        "color": "#38bdf8",
    },
    "t2mgpt_parent": {
        "label": "T2M-GPT parent-fit",
        "kind": "motion135",
        "path": RECON_ROOT / "_debug_motion135/t2mgpt_parent",
        "color": "#93c5fd",
    },
    "momask_parent": {
        "label": "MoMask parent-fit",
        "kind": "motion135",
        "path": RECON_ROOT / "_debug_motion135/momask_parent",
        "color": "#6ee7b7",
    },
}


def _b64(arr: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(arr).tobytes()).decode("ascii")


def _official_rows() -> dict[str, dict[str, Any]]:
    raw = json.loads(ANNO.read_text(encoding="utf-8"))
    data = raw.get("data_list")
    if not isinstance(data, dict):
        raise RuntimeError(f"bad annotation format: {ANNO}")
    return data


def _caption(row: dict[str, Any]) -> str:
    cpath = row.get("hierarchical_caption_path")
    if not cpath:
        return ""
    path = REPO / cpath
    if not path.exists():
        return ""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return ""
    for key in ("macro", "meso", "micro"):
        vals = data.get(key) or []
        if vals:
            return str(vals[0])
    return ""


def _path_for(root: Path, sid: str, kind: str) -> Path | None:
    suffixes = (".npz", ".npy") if kind == "ms272" else (".npz",)
    alts = [sid]
    if sid.startswith("M") and sid[1:].isdigit():
        alts.append(sid[1:])
    elif sid and sid[0].isdigit():
        alts.append("M" + sid)
    for aid in alts:
        for suffix in suffixes:
            path = root / f"{aid}{suffix}"
            if path.exists():
                return path
    return None


def _selected_frame_indices(total: int, max_frames: int) -> np.ndarray:
    if total <= max_frames:
        return np.arange(total, dtype=np.int64)
    return np.unique(np.round(np.linspace(0, total - 1, max_frames)).astype(np.int64))


def _load_motion_272(path: Path) -> np.ndarray:
    if path.suffix == ".npy":
        return np.asarray(np.load(path), dtype=np.float32)
    with np.load(path, allow_pickle=True) as data:
        if "motion_272" not in data:
            raise KeyError(f"{path} does not contain motion_272")
        return np.asarray(data["motion_272"], dtype=np.float32)


def _params_from_272(motion_272: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rot, root = recover_local_rotations_and_root(motion_272)
    aa = matrix_to_axis_angle(rot.reshape(-1, 3, 3)).reshape(rot.shape[0], rot.shape[1], 3)
    return (
        np.asarray(aa[:, 0], dtype=np.float32),
        np.asarray(aa[:, 1:22].reshape(len(aa), 63), dtype=np.float32),
        np.asarray(root, dtype=np.float32),
    )


def _params_from_motion135(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=True) as data:
        if {"global_orient", "body_pose", "transl"}.issubset(data.files):
            return (
                np.asarray(data["global_orient"], dtype=np.float32),
                np.asarray(data["body_pose"], dtype=np.float32).reshape(-1, 63),
                np.asarray(data["transl"], dtype=np.float32),
            )
        if "motion_272" in data.files:
            return _params_from_272(np.asarray(data["motion_272"], dtype=np.float32))
        if "motion_135" not in data.files:
            raise KeyError(f"{path} does not contain SMPL params or motion_135")
        motion_135 = np.asarray(data["motion_135"], dtype=np.float32)
    rot6d = motion_135[:, 3:135].reshape(len(motion_135), 22, 6)
    rotmat = rotation_6d_to_matrix(rot6d, convention="row")
    aa = matrix_to_axis_angle(rotmat.reshape(-1, 3, 3)).reshape(len(motion_135), 22, 3)
    return (
        np.asarray(aa[:, 0], dtype=np.float32),
        np.asarray(aa[:, 1:22].reshape(len(aa), 63), dtype=np.float32),
        np.asarray(motion_135[:, :3], dtype=np.float32),
    )


def _load_params(path: Path, kind: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if kind == "motion135":
        return _params_from_motion135(path)
    if kind == "ms272":
        return _params_from_272(_load_motion_272(path))
    raise ValueError(f"unsupported kind: {kind}")


def _floor_center(verts: np.ndarray) -> np.ndarray:
    out = np.asarray(verts, dtype=np.float32).copy()
    out[..., 1] -= float(np.percentile(out[..., 1], 0.5))
    center = out[0].mean(axis=0)
    out[..., 0] -= float(center[0])
    out[..., 2] -= float(center[2])
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def _save_verts(out_dir: Path, sid: str, method: str, verts: np.ndarray) -> str:
    safe_method = method.replace("/", "_")
    fname = f"{sid}__{safe_method}.f32"
    verts = np.round(np.asarray(verts, dtype=np.float32), 4)
    vdir = out_dir / "verts"
    vdir.mkdir(parents=True, exist_ok=True)
    (vdir / fname).write_bytes(np.ascontiguousarray(verts).tobytes())
    return fname


class SMPLMesh:
    def __init__(self, model_dir: Path, device: str):
        from hftrainer.motion.retarget.smpl_soma import (
            _import_smplx,
            _resolve_smpl_model_dir,
        )
        import torch

        self.torch = torch
        self.device = torch.device(device)
        smplx = _import_smplx()
        resolved = _resolve_smpl_model_dir(model_dir)
        self.model = smplx.create(
            str(resolved),
            model_type="smpl",
            gender="neutral",
            ext="pkl",
            batch_size=1,
        ).to(self.device)
        self.model.eval()
        self.faces = np.asarray(self.model.faces, dtype=np.int32)

    def vertices(
        self,
        global_orient: np.ndarray,
        body_pose63: np.ndarray,
        transl: np.ndarray,
        batch_size: int,
    ) -> np.ndarray:
        torch = self.torch
        chunks = []
        n = int(len(global_orient))
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            bsz = end - start
            body69 = np.zeros((bsz, 69), dtype=np.float32)
            body69[:, :63] = body_pose63[start:end]
            with torch.no_grad():
                out = self.model(
                    betas=torch.zeros(bsz, 10, device=self.device),
                    global_orient=torch.from_numpy(np.asarray(global_orient[start:end], np.float32)).to(self.device),
                    body_pose=torch.from_numpy(body69).to(self.device),
                    transl=torch.from_numpy(np.asarray(transl[start:end], np.float32)).to(self.device),
                )
            chunks.append(out.vertices.detach().cpu().numpy().astype(np.float32))
        return np.concatenate(chunks, axis=0)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    ap.add_argument(
        "--methods",
        default="gt_0beta,t2mgpt,momask,mld,mogents,motiongpt3,motionstreamer,prism,vermo",
        help="comma-separated method ids to include",
    )
    ap.add_argument("--ids", default="", help="comma-separated HumanML3D ids; default selects from official test")
    ap.add_argument("--num-cases", type=int, default=8)
    ap.add_argument("--max-frames", type=int, default=120)
    ap.add_argument("--device", default="")
    ap.add_argument("--batch-size", type=int, default=96)
    ap.add_argument("--smpl-model-dir", default=str(SMPL_MODEL_DIR))
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    import torch

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    unknown = [m for m in methods if m not in METHOD_SPECS]
    if unknown:
        raise ValueError(f"unknown methods: {unknown}")

    rows = _official_rows()
    if args.ids:
        ids = [s.strip() for s in args.ids.split(",") if s.strip()]
    else:
        ids = []
        for sid in sorted(rows):
            if all(_path_for(METHOD_SPECS[m]["path"], sid, METHOD_SPECS[m]["kind"]) for m in methods):
                ids.append(sid)
            if len(ids) >= args.num_cases:
                break
    ids = ids[: args.num_cases]
    if not ids:
        raise RuntimeError("no cases selected")

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "verts").mkdir(parents=True, exist_ok=True)

    mesh = SMPLMesh(Path(args.smpl_model_dir), device=device)
    faces_b64 = _b64(mesh.faces.astype(np.int32))
    index_cases = []

    for idx, sid in enumerate(ids, 1):
        print(f"[case] {idx}/{len(ids)} {sid}", flush=True)
        case_methods = []
        for method in methods:
            spec = METHOD_SPECS[method]
            path = _path_for(spec["path"], sid, spec["kind"])
            if path is None:
                print(f"  [missing] {method} {sid}", flush=True)
                continue
            global_orient, body_pose, transl = _load_params(path, spec["kind"])
            total = int(len(global_orient))
            keep = _selected_frame_indices(total, args.max_frames)
            verts = mesh.vertices(
                global_orient[keep],
                body_pose[keep],
                transl[keep],
                batch_size=args.batch_size,
            )
            verts = _floor_center(verts)
            verts_file = _save_verts(out_dir, sid, method, verts)
            fps = 30.0 * (float(len(keep)) / float(total)) if total else 30.0
            method_payload = {
                "id": method,
                "label": spec["label"],
                "kind": spec["kind"],
                "color": spec["color"],
                "source_path": str(path.relative_to(REPO)),
                "verts_file": verts_file,
                "num_frames": int(verts.shape[0]),
                "num_verts": int(verts.shape[1]),
                "total_frames": total,
                "frame_indices": keep.astype(int).tolist(),
                "fps": fps,
            }
            if path.suffix == ".npz":
                with np.load(path, allow_pickle=True) as data:
                    if "fit_mpjpe_mm" in data.files:
                        fit = np.asarray(data["fit_mpjpe_mm"], dtype=np.float32)
                        method_payload["fit_mpjpe_mm"] = float(fit.mean())
            case_methods.append(method_payload)

        row = rows.get(sid, {})
        case = {
            "id": sid,
            "caption": _caption(row),
            "meta": {
                "num_frames": int(row.get("num_frames", 0) or 0),
                "duration": float(row.get("duration", 0.0) or 0.0),
                "source": row.get("source", "official_humanml3d_272_test"),
            },
            "methods": case_methods,
        }
        (out_dir / f"case_{sid}.json").write_text(json.dumps(case, indent=2, ensure_ascii=False), encoding="utf-8")
        index_cases.append({
            "id": sid,
            "caption": case["caption"],
            "num_methods": len(case_methods),
            "num_frames": case["meta"]["num_frames"],
            "duration": case["meta"]["duration"],
        })

    index = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "task": "reconstruction",
        "test_dataset": "humanml3d_official_test",
        "viewer": "threejs_smpl_mesh",
        "faces_b64": faces_b64,
        "num_faces": int(mesh.faces.shape[0]),
        "methods": [
            {"id": m, "label": METHOD_SPECS[m]["label"], "color": METHOD_SPECS[m]["color"]}
            for m in methods
        ],
        "cases": index_cases,
    }
    (out_dir / "index.json").write_text(json.dumps(index, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[done] wrote {len(index_cases)} cases to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
