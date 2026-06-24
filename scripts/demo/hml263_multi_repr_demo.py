#!/usr/bin/env python3
"""Demo: convert a few HumanML3D-263 clips through SMPL / SOMA / G1 and dump
viewer data for the web visualizer at ``motion_annot_web/repr_convert_demo``.

Per clip we produce 5 views that exercise the public ``hftrainer.motion``
conversion APIs.  Everything except HumanML3D-263 is rendered as a **mesh**:

    1. ``hml263``  skeleton : recover_from_ric              (convert.hml263_to_joints)
    2. ``smpl``    MESH     : HML263 -> SMPL motion_135 (IK) -> SMPL LBS verts
    3. ``soma``    MESH     : SMPL motion_135 -> SOMA30 -> SOMA77 -> skin LBS verts
    4. ``smpl_from_soma`` MESH : SOMA30 -> SMPL (round trip) -> SMPL LBS verts
    5. ``g1``      ROBOT    : SMPL motion_135 -> SMPL-X -> GMR mink IK -> MuJoCo FK
                             link poses + per-link STL meshes (Unitree G1, 29-DOF)

The Unitree G1 retarget uses the **GMR** (General Motion Retargeting, mink IK)
pipeline via ``scripts/embodied/gmr_retarget_headless.py`` -- NOT the old
analytic/MuJoCo decomposition.

Mesh vertices are written as raw float32 ``(T*V*3)`` binaries under ``data/verts/``
and streamed on demand by the Flask app; faces are base64-embedded in the case
JSON.  G1 STL link meshes are served from the ProtoMotions asset dir.

Usage::

    HFTRAINER_SKIP_AUTOREGISTER=1 python3 scripts/demo/hml263_multi_repr_demo.py \
        --num-cases 3 --device cuda

Outputs ``motion_annot_web/repr_convert_demo/data/{index.json, case_<id>.json,
verts/*.f32}``.
"""
from __future__ import annotations

import argparse
import base64
import importlib.util
import json
import os
import sys
import types
from pathlib import Path

os.environ.setdefault("HFTRAINER_SKIP_AUTOREGISTER", "1")

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

DEFAULT_IN_DIR = REPO / "ref_repo/CondMDI/dataset/HumanML3D/new_joint_vecs"
DEFAULT_TEXT_DIR = REPO / "ref_repo/CondMDI/dataset/HumanML3D/texts"
DEFAULT_OUT_DIR = REPO / "motion_annot_web/repr_convert_demo/data"
SMPL_MODEL_DIR = REPO / "ref_repo/MDM/body_models"
SOMA_SKIN = REPO / "ref_repo/KIMODO/kimodo/kimodo/assets/skeletons/somaskel77/skin_standard.npz"
G1_MJCF = REPO / "ref_repo/ProtoMotions/protomotions/data/assets/mjcf/g1_holo_compat.xml"
G1_STL_DIR = REPO / "ref_repo/ProtoMotions/protomotions/data/assets/mesh/G1"


# --------------------------------------------------------------------------- #
# KIMODO skeleton bootstrap (avoid importing the broken kimodo package root)
# --------------------------------------------------------------------------- #
def _bootstrap_kimodo_skeleton() -> None:
    """Load ``kimodo.skeleton.*`` without triggering ``kimodo/__init__`` (which
    pulls in model code that fails on this env). Mirrors
    ``scripts/kimodo/append_kimodo_context_soma77.py``."""
    if "kimodo.skeleton.definitions" in sys.modules:
        return
    kimodo_pkg_path = REPO / "ref_repo" / "KIMODO" / "kimodo"
    if "kimodo" not in sys.modules:
        pkg = types.ModuleType("kimodo")
        pkg.__path__ = [str(kimodo_pkg_path / "kimodo")]
        sys.modules["kimodo"] = pkg
    if "kimodo.assets" not in sys.modules:
        assets = types.ModuleType("kimodo.assets")
        skel_root = str(kimodo_pkg_path / "kimodo" / "assets" / "skeletons")
        assets.skeleton_asset_path = lambda name: Path(skel_root) / name
        assets.SKELETONS_ROOT = skel_root
        sys.modules["kimodo.assets"] = assets
    if "kimodo.skeleton" not in sys.modules:
        sk_pkg = types.ModuleType("kimodo.skeleton")
        sk_pkg.__path__ = [str(kimodo_pkg_path / "kimodo" / "skeleton")]
        sys.modules["kimodo.skeleton"] = sk_pkg

    def _load(name, relpath):
        spec = importlib.util.spec_from_file_location(name, str(kimodo_pkg_path / relpath))
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        return mod

    _load("kimodo.skeleton.kinematics", "kimodo/skeleton/kinematics.py")
    _load("kimodo.skeleton.transforms", "kimodo/skeleton/transforms.py")
    _load("kimodo.skeleton.base", "kimodo/skeleton/base.py")
    _load("kimodo.skeleton.definitions", "kimodo/skeleton/definitions.py")


def _load_demo_module(name: str, relpath: str):
    spec = importlib.util.spec_from_file_location(name, str(REPO / relpath))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
def parents_to_edges(parents) -> list[list[int]]:
    return [[int(p), int(j)] for j, p in enumerate(parents) if int(p) >= 0]


def _floor_center(pos: np.ndarray, ref_xz: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Floor-align (min y -> 0) and recenter XZ on ``ref_xz`` (or frame-0 mean)."""
    pos = np.asarray(pos, dtype=np.float32).copy()
    pos[..., 1] -= float(pos[..., 1].min())
    if ref_xz is None:
        ref_xz = pos[0].reshape(-1, 3).mean(axis=0)[[0, 2]]
    pos[..., 0] -= float(ref_xz[0])
    pos[..., 2] -= float(ref_xz[1])
    return pos, np.asarray(ref_xz, dtype=np.float32)


def _b64(arr: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(arr).tobytes()).decode("ascii")


def _save_verts(out_dir: Path, sid: str, rep: str, verts: np.ndarray) -> str:
    fname = f"{sid}__{rep}.f32"
    verts = np.round(np.asarray(verts, dtype=np.float32), 4)
    (out_dir / "verts").mkdir(parents=True, exist_ok=True)
    (out_dir / "verts" / fname).write_bytes(np.ascontiguousarray(verts).tobytes())
    return fname


# --------------------------------------------------------------------------- #
# SMPL mesh (server-side LBS forward)
# --------------------------------------------------------------------------- #
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
        mdir = _resolve_smpl_model_dir(model_dir)
        self.model = smplx.create(
            str(mdir), model_type="smpl", gender="neutral", ext="pkl", batch_size=1,
        ).to(self.device)
        self.model.eval()
        self.faces = np.asarray(self.model.faces, dtype=np.int32)

    def vertices(self, global_orient, body_pose63, transl, batch: int = 128) -> np.ndarray:
        torch = self.torch
        n = len(global_orient)
        chunks = []
        for s in range(0, n, batch):
            e = min(s + batch, n)
            b = e - s
            body69 = np.zeros((b, 69), dtype=np.float32)
            body69[:, :63] = body_pose63[s:e]
            with torch.no_grad():
                out = self.model(
                    betas=torch.zeros(b, 10, device=self.device),
                    body_pose=torch.from_numpy(body69).to(self.device),
                    global_orient=torch.from_numpy(np.asarray(global_orient[s:e], np.float32)).to(self.device),
                    transl=torch.from_numpy(np.asarray(transl[s:e], np.float32)).to(self.device),
                )
            chunks.append(out.vertices.detach().cpu().numpy().astype(np.float32))
        return np.concatenate(chunks, axis=0)


# --------------------------------------------------------------------------- #
# SOMA mesh (SMPL -> SOMA30 -> SOMA77 -> skin LBS)
# --------------------------------------------------------------------------- #
class SOMAMesh:
    def __init__(self, skin_path: Path, device: str):
        import torch
        from kimodo.skeleton.definitions import SOMASkeleton30

        self.torch = torch
        self.device = torch.device(device)
        self.soma30 = SOMASkeleton30()
        d = np.load(str(skin_path))
        bind_rig = np.asarray(d["bind_rig_transform"], dtype=np.float32)
        self.skin = {
            "bind_rig_transform_inv": torch.from_numpy(np.linalg.inv(bind_rig)).to(self.device),
            "bind_vertices": torch.tensor(d["bind_vertices"], device=self.device, dtype=torch.float),
            "lbs_indices": torch.tensor(d["lbs_indices"], device=self.device, dtype=torch.long),
            "lbs_weights": torch.tensor(d["lbs_weights"], device=self.device, dtype=torch.float),
        }
        self.faces = np.asarray(d["faces"], dtype=np.int32)

    def _soma_lbs(self, posed_transform):
        skin = self.skin
        bind_rig_inv = skin["bind_rig_transform_inv"]
        bind_verts = skin["bind_vertices"]
        lbs_weights = skin["lbs_weights"]
        lbs_indices = skin["lbs_indices"]
        for _ in range(posed_transform.dim() - 3):
            bind_rig_inv = bind_rig_inv.unsqueeze(0)
            bind_verts = bind_verts.unsqueeze(0)
            lbs_weights = lbs_weights.unsqueeze(0)
        affine = (posed_transform @ bind_rig_inv)[..., :3, :]  # (T,J,3,4)
        torch = self.torch
        vs = (
            affine[..., lbs_indices, :, :]
            @ torch.cat([bind_verts, torch.ones_like(bind_verts[..., 0:1])], dim=-1)[..., None, :, None]
        )  # (T,V,W,3,1)
        ws = lbs_weights[..., None, None]
        return (vs * ws).sum(dim=-3).squeeze(-1)  # (T,V,3)

    def vertices(self, soma30_global_rots: np.ndarray, soma30_root_pos: np.ndarray) -> np.ndarray:
        """SOMA-30 global rots (T,30,3,3) + root pos (T,3) -> SOMA-77 LBS verts."""
        import torch
        from kimodo.skeleton.transforms import global_rots_to_local_rots

        gr = torch.from_numpy(np.asarray(soma30_global_rots, np.float32)).to(self.device)
        root = torch.from_numpy(np.asarray(soma30_root_pos, np.float32)).to(self.device)
        soma30_local = global_rots_to_local_rots(gr, self.soma30)
        soma77_local = self.soma30.to_SOMASkeleton77(soma30_local)
        soma77 = self.soma30.somaskel77
        g77, pj77, _ = soma77.fk(soma77_local, root)
        # enforce SOMA77 root == SOMA30 root frame-by-frame
        root_delta = root - pj77[:, 0, :]
        pj77 = pj77 + root_delta[:, None, :]
        T, J = g77.shape[:2]
        fk = torch.eye(4, device=self.device).reshape(1, 1, 4, 4).expand(T, J, 4, 4).contiguous()
        fk[..., :3, :3] = g77
        fk[..., :3, 3] = pj77
        with torch.no_grad():
            verts = self._soma_lbs(fk)
        return verts.detach().cpu().numpy().astype(np.float32)


# --------------------------------------------------------------------------- #
# G1 robot via GMR
# --------------------------------------------------------------------------- #
_QF_ZUP2YUP = np.array([np.sqrt(0.5), -np.sqrt(0.5), 0.0, 0.0])  # Rx(-90) wxyz
# MuJoCo robot forward is +X; after the Z-up->Y-up map that lands on viewer +X
# (camera-right) while the SMPL/SOMA panels face the camera (+Z). Apply a global
# yaw Ry(-90deg) so the robot faces the camera like the other panels: +X -> +Z.
_QY_FACE_CAM = np.array([np.sqrt(0.5), 0.0, -np.sqrt(0.5), 0.0])  # Ry(-90) wxyz


def _quat_mul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], dtype=np.float64)


class G1GMR:
    def __init__(self, mjcf: Path, workdir: Path):
        self.demo = _load_demo_module("smpl_g1_compare_demo", "scripts/embodied/smpl_g1_compare_demo.py")
        self.model, self.bodies = self.demo.load_g1_model(mjcf)
        self.workdir = workdir
        self.workdir.mkdir(parents=True, exist_ok=True)

    def robot_frames(self, global_orient, body_pose63, transl, fps, stem) -> dict:
        poses = np.concatenate(
            [np.asarray(global_orient, np.float32), np.asarray(body_pose63, np.float32)], axis=1
        )  # (T,66)
        betas = np.zeros((16,), np.float32)
        qpos = self.demo.gmr_retarget_to_qpos(
            poses, np.asarray(transl, np.float32), betas, "neutral", int(fps), self.workdir, stem,
        )
        rf = self.demo.qpos_to_robot_frames(self.model, self.bodies, qpos, int(fps))  # Z-up
        # Z-up -> Y-up for the unified viewer.
        frames = []
        all_y = []
        for fr in rf["frames"]:
            bp = np.asarray(fr["body_pos"], np.float64)
            bq = np.asarray(fr["body_quat"], np.float64)
            bp_y = np.stack([bp[:, 0], bp[:, 2], -bp[:, 1]], axis=1)
            bq_y = np.stack([_quat_mul(_QF_ZUP2YUP, q) for q in bq], axis=0)
            # global yaw so the robot faces the camera (+X -> +Z): p' = Ry(-90)@p
            bp_y = np.stack([-bp_y[:, 2], bp_y[:, 1], bp_y[:, 0]], axis=1)
            bq_y = np.stack([_quat_mul(_QY_FACE_CAM, q) for q in bq_y], axis=0)
            frames.append({"_pos": bp_y, "_quat": bq_y})
            all_y.append(bp_y)
        all_y = np.stack(all_y, axis=0)  # (T,nb,3)
        y_off = float(all_y[..., 1].min())
        ref = all_y[0, 0, [0, 2]].copy()
        out = []
        for fr in frames:
            p = fr["_pos"].copy()
            p[:, 1] -= y_off
            p[:, 0] -= ref[0]
            p[:, 2] -= ref[1]
            out.append({
                "body_pos": np.round(p, 4).tolist(),
                "body_quat": np.round(fr["_quat"], 5).tolist(),
            })
        return {
            "bodies": rf["bodies"],
            "frames": out,
            "num_frames": len(out),
            "num_bodies": rf["num_bodies"],
        }


# --------------------------------------------------------------------------- #
# build a single case
# --------------------------------------------------------------------------- #
def build_case(feats, text, sid, device, out_dir, smpl_mesh, soma_mesh, g1, refine_iters=0) -> dict:
    from hftrainer.motion.representation import convert
    from hftrainer.motion.retarget.hml263_smpl import retarget_hml263_clip
    from hftrainer.motion.retarget import smpl_soma30_roundtrip, SMPL22_PARENTS
    from hftrainer.motion.retarget import SMPL_JOINT_NAMES

    reps = []
    smpl_faces_b64 = _b64(smpl_mesh.faces.astype(np.int32))
    soma_faces_b64 = _b64(soma_mesh.faces.astype(np.int32))

    # 1. HML263 -> joints (skeleton, 20 fps input space)
    hml_joints = convert.hml263_to_joints(feats, 22)
    hml_joints, _ = _floor_center(hml_joints)
    reps.append({
        "name": "hml263", "type": "skeleton",
        "label": "HumanML3D-263 (recover_from_ric)", "fps": 20, "color": "#7dd3fc",
        "joint_names": list(SMPL_JOINT_NAMES), "edges": parents_to_edges(SMPL22_PARENTS),
        "positions": np.round(hml_joints, 4).tolist(),
        "info": f"{hml_joints.shape[0]} frames @20fps · 22-joint skeleton (input space)",
    })

    # 2. HML263 -> SMPL motion_135 (IK) -> SMPL mesh (30 fps)
    ik = retarget_hml263_clip(feats, device=device, refine_iters=refine_iters)
    m135 = ik["motion_135"]
    fit_mm = float(np.mean(ik["fit_mpjpe_mm"]))
    smpl_v = smpl_mesh.vertices(ik["global_orient"], ik["body_pose"], ik["transl"])
    smpl_v, ref_xz = _floor_center(smpl_v)
    f = _save_verts(out_dir, sid, "smpl", smpl_v)
    reps.append({
        "name": "smpl", "type": "mesh", "label": "SMPL mesh (motion_135 IK fit)",
        "fps": 30, "color": "#86efac", "num_frames": int(smpl_v.shape[0]),
        "num_verts": int(smpl_v.shape[1]), "faces_b64": smpl_faces_b64, "verts_file": f,
        "info": f"{smpl_v.shape[0]} frames @30fps · 6890-vtx SMPL · IK MPJPE = {fit_mm:.1f} mm",
    })

    # 3 & 4. SMPL <-> SOMA30 round trip
    rt = smpl_soma30_roundtrip(m135)
    soma30_global = rt["soma30_global_rots"]
    soma30_root = rt["soma30_joints"][:, 0, :]
    soma_v = soma_mesh.vertices(soma30_global, soma30_root)
    soma_v, _ = _floor_center(soma_v, ref_xz)
    fs = _save_verts(out_dir, sid, "soma", soma_v)
    reps.append({
        "name": "soma", "type": "mesh", "label": "SOMA mesh (SMPL -> SOMA30 -> SOMA77 LBS)",
        "fps": 30, "color": "#fca5a5", "num_frames": int(soma_v.shape[0]),
        "num_verts": int(soma_v.shape[1]), "faces_b64": soma_faces_b64, "verts_file": fs,
        "info": f"{soma_v.shape[0]} frames @30fps · {soma_v.shape[1]}-vtx KIMODO SOMA mesh",
    })

    smpl_from_soma_v = smpl_mesh.vertices(rt["global_orient"], rt["body_pose"], rt["transl"])
    smpl_from_soma_v, _ = _floor_center(smpl_from_soma_v, ref_xz)
    rt_mm = float(np.mean(np.linalg.norm(rt["fitted_joints"] - ik["fitted_joints"], axis=-1)) * 1000.0)
    fr = _save_verts(out_dir, sid, "smpl_from_soma", smpl_from_soma_v)
    reps.append({
        "name": "smpl_from_soma", "type": "mesh", "label": "SMPL mesh <- SOMA (round trip)",
        "fps": 30, "color": "#fcd34d", "num_frames": int(smpl_from_soma_v.shape[0]),
        "num_verts": int(smpl_from_soma_v.shape[1]), "faces_b64": smpl_faces_b64, "verts_file": fr,
        "info": f"{smpl_from_soma_v.shape[0]} frames @30fps · round-trip joint err = {rt_mm:.1f} mm",
    })

    # 5. SMPL motion_135 -> SMPL-X -> GMR -> G1 robot mesh (optional)
    if g1 is not None:
        g1_data = g1.robot_frames(ik["global_orient"], ik["body_pose"], ik["transl"], 30, sid)
        reps.append({
            "name": "g1", "type": "robot", "label": "Unitree G1 (GMR mink IK retarget)",
            "fps": 30, "color": "#c4b5fd", "num_frames": g1_data["num_frames"],
            "bodies": g1_data["bodies"], "frames": g1_data["frames"],
            "info": f"{g1_data['num_frames']} frames @30fps · {g1_data['num_bodies']} G1 links (GMR)",
        })

    return {"id": sid, "text": text, "reps": reps}


# --------------------------------------------------------------------------- #
def pick_ids(in_dir: Path, ids_arg, num: int) -> list[str]:
    if ids_arg:
        return [s.strip() for s in ids_arg.split(",") if s.strip()]
    files = sorted(in_dir.glob("[0-9]*.npy"))[: num * 6]
    out = []
    for f in files:
        m = np.load(f)
        if m.ndim == 2 and m.shape[-1] == 263 and 40 <= m.shape[0] <= 200:
            out.append(f.stem)
        if len(out) >= num:
            break
    return out


def read_text(sid: str, text_dir: Path) -> str:
    p = text_dir / f"{sid}.txt"
    if not p.exists():
        return sid
    body = p.read_text().strip()
    if not body:
        return sid
    return body.splitlines()[0].split("#")[0].strip() or sid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", default=str(DEFAULT_IN_DIR))
    ap.add_argument("--text-dir", default=str(DEFAULT_TEXT_DIR))
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--ids", default=None, help="comma-separated clip ids (default: auto-pick)")
    ap.add_argument("--num-cases", type=int, default=3)
    ap.add_argument("--max-frames", type=int, default=120)
    ap.add_argument("--device", default=None)
    ap.add_argument("--id-prefix", default="", help="prefix for case ids (read files by bare stem; keeps GT cases distinct)")
    ap.add_argument("--refine-iters", type=int, default=0, help="IK Adam refine iters (match the eval pipeline, e.g. 80)")
    args = ap.parse_args()

    import torch

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    in_dir, text_dir, out_dir = Path(args.in_dir), Path(args.text_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ids = pick_ids(in_dir, args.ids, args.num_cases)
    if not ids:
        raise SystemExit(f"no usable clips found under {in_dir}")
    print(f"[setup] device={device} cases={ids}", flush=True)

    _bootstrap_kimodo_skeleton()
    smpl_mesh = SMPLMesh(SMPL_MODEL_DIR, device)
    soma_mesh = SOMAMesh(SOMA_SKIN, device)
    try:
        g1 = G1GMR(G1_MJCF, out_dir / "g1_work")
        print("[setup] SMPL / SOMA / G1(GMR) backends ready", flush=True)
    except Exception as e:  # noqa: BLE001
        g1 = None
        print(f"[setup] SMPL / SOMA ready; G1(GMR) backend unavailable, skipping ({e})", flush=True)

    new_cases = []
    for stem in ids:
        feats = np.load(in_dir / f"{stem}.npy").astype(np.float32)
        if feats.shape[0] > args.max_frames:
            feats = feats[: args.max_frames]
        text = read_text(stem, text_dir)
        disp = f"{args.id_prefix}{stem}"
        print(f"[case] {disp}: T={feats.shape[0]} text={text!r}", flush=True)
        case = build_case(feats, text, disp, device, out_dir, smpl_mesh, soma_mesh, g1,
                          refine_iters=args.refine_iters)
        (out_dir / f"case_{disp}.json").write_text(json.dumps(case))
        new_cases.append({
            "id": disp, "text": text,
            "reps": [{"name": r["name"], "type": r["type"], "label": r["label"], "info": r["info"]}
                     for r in case["reps"]],
        })
        for r in case["reps"]:
            print(f"    - {r['name']:<16} [{r['type']:<8}] {r['info']}", flush=True)

    # merge into existing index.json (replace same-id, keep others) so multiple runs coexist
    index_path = out_dir / "index.json"
    existing = []
    if index_path.exists():
        try:
            existing = json.loads(index_path.read_text()).get("cases", [])
        except Exception:
            existing = []
    new_ids = {c["id"] for c in new_cases}
    merged = [c for c in existing if c.get("id") not in new_ids] + new_cases
    index_path.write_text(json.dumps({"cases": merged}, indent=2))
    print(f"[done] wrote {len(new_cases)} cases ({len(merged)} total) -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
