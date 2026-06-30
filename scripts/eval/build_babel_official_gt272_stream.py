#!/usr/bin/env python3
"""Convert official-BABEL GT AMASS clips to MotionStreamer native 272.

This consumes the manifest produced by
``build_babel_official_seq_protocol.py`` and writes ``<id>.npz`` files with a
``motion_272`` array, one file per retained sequence.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


ZUP_TO_YUP = np.array(
    [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]],
    dtype=np.float64,
)

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def _interp_trans(trans: np.ndarray, src_t: np.ndarray, dst_t: np.ndarray) -> np.ndarray:
    out = np.empty((len(dst_t), 3), dtype=np.float32)
    for k in range(3):
        out[:, k] = np.interp(dst_t, src_t, trans[:, k]).astype(np.float32)
    return out


def _interp_axis_angle(aa: np.ndarray, src_t: np.ndarray, dst_t: np.ndarray) -> np.ndarray:
    """Slerp axis-angle rotations with linear fallback if SciPy is unavailable."""
    aa = np.asarray(aa, dtype=np.float64)
    T, J, _ = aa.shape
    if len(dst_t) == T and np.allclose(dst_t, src_t):
        return aa.astype(np.float32)
    try:
        from scipy.spatial.transform import Rotation, Slerp

        out = np.empty((len(dst_t), J, 3), dtype=np.float32)
        clipped = np.clip(dst_t, src_t[0], src_t[-1])
        for j in range(J):
            rots = Rotation.from_rotvec(aa[:, j])
            out[:, j] = Slerp(src_t, rots)(clipped).as_rotvec().astype(np.float32)
        return out
    except Exception:
        flat = aa.reshape(T, J * 3)
        out = np.empty((len(dst_t), J * 3), dtype=np.float32)
        for k in range(flat.shape[1]):
            out[:, k] = np.interp(dst_t, src_t, flat[:, k]).astype(np.float32)
        return out.reshape(len(dst_t), J, 3)


def _load_clip_smpl85(npz_path: Path, start_t: float, end_t: float, target_fps: float) -> tuple[np.ndarray, str]:
    data = np.load(npz_path, allow_pickle=True)
    fps = float(data["mocap_frame_rate"]) if "mocap_frame_rate" in data else target_fps
    root = np.asarray(data["root_orient"] if "root_orient" in data else data["poses"][:, :3], dtype=np.float32)
    body = np.asarray(data["pose_body"] if "pose_body" in data else data["poses"][:, 3:66], dtype=np.float32)
    trans = np.asarray(data["trans"], dtype=np.float32)
    betas = np.asarray(data["betas"], dtype=np.float32).reshape(-1)[:10]
    gender = str(data["gender"].item() if getattr(data["gender"], "shape", ()) == () else data["gender"]).lower()
    if gender not in {"male", "female", "neutral"}:
        gender = "neutral"

    src_t = np.arange(len(trans), dtype=np.float64) / fps
    n = max(2, int(round((end_t - start_t) * target_fps)))
    dst_t = start_t + np.arange(n, dtype=np.float64) / target_fps
    dst_t = np.clip(dst_t, src_t[0], src_t[-1])

    aa = np.zeros((len(trans), 22, 3), dtype=np.float32)
    aa[:, 0] = root
    aa[:, 1:] = body[:, :63].reshape(len(trans), 21, 3)
    aa30 = _interp_axis_angle(aa, src_t, dst_t)
    tr30 = _interp_trans(trans, src_t, dst_t)

    from hftrainer.motion.representation.motion272 import pack_smpl85

    smpl85 = pack_smpl85(aa30[:, 0], aa30[:, 1:].reshape(len(dst_t), -1), tr30, betas)
    return smpl85, gender


def zup_to_yup_smpl85(smpl85: np.ndarray) -> np.ndarray:
    """Convert AMASS/BABEL global coordinates from Z-up to MotionStreamer Y-up.

    AMASS root orientation and translation are global, while body pose entries
    are local joint rotations. To change the world basis before MotionStreamer
    face-Z canonicalization, rotate only the global root orientation and
    translation by ``ZUP_TO_YUP`` and leave local body rotations unchanged.
    """
    arr = np.asarray(smpl85, dtype=np.float64).copy()
    root = Rotation.from_rotvec(arr[:, :3]).as_matrix()
    arr[:, :3] = Rotation.from_matrix(np.einsum("ij,tjk->tik", ZUP_TO_YUP, root)).as_rotvec()
    arr[:, 72:75] = np.einsum("ij,tj->ti", ZUP_TO_YUP, arr[:, 72:75])
    return arr.astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="outputs/evaluation/babel/official_val/msstyle_30fps_gt/manifest.jsonl")
    ap.add_argument("--babel-root", default="data/babel_official")
    ap.add_argument("--out-dir", default="outputs/evaluation/babel/official_val/msstyle_30fps_gt/gt_272_stream_yup")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--no-zup-to-yup", dest="zup_to_yup", action="store_false",
                    help="Disable AMASS/BABEL Z-up -> MotionStreamer Y-up conversion.")
    ap.set_defaults(zup_to_yup=True)
    args = ap.parse_args()

    import torch
    import smplx
    from hftrainer.motion.representation.motion272 import _default_smpl_model_dir, smpl85_to_272

    device = args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"
    manifest = Path(args.manifest)
    babel_root = Path(args.babel_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    records = [json.loads(l) for l in manifest.open() if l.strip()]
    if args.limit:
        records = records[: args.limit]
    print(f"[gt272] records={len(records)} out={out_dir} device={device}", flush=True)

    model_cache = {}

    def get_model(gender: str):
        if gender not in model_cache:
            model = smplx.create(
                _default_smpl_model_dir(),
                model_type="smplx",
                gender=gender,
                ext="npz",
                num_betas=10,
                batch_size=args.batch_size,
                use_pca=False,
                flat_hand_mean=True,
            ).to(torch.device(device))
            model.eval()
            model_cache[gender] = model
        return model_cache[gender]

    ok = skip = fail = 0
    for i, rec in enumerate(records, 1):
        sid = rec["id"]
        out_path = out_dir / f"{sid}.npz"
        if out_path.exists() and not args.overwrite:
            skip += 1
        else:
            try:
                amass_path = Path(rec["amass_path"])
                if not amass_path.is_absolute():
                    amass_path = babel_root / amass_path
                smpl85, gender = _load_clip_smpl85(
                    amass_path,
                    float(rec["source_start_t"]),
                    float(rec["source_end_t"]),
                    float(rec.get("target_fps", 30.0)),
                )
                if args.zup_to_yup:
                    smpl85 = zup_to_yup_smpl85(smpl85)
                m272 = smpl85_to_272(
                    smpl85,
                    model_type="smplx",
                    gender=gender,
                    device=device,
                    batch_size=args.batch_size,
                    apply_face_z=True,
                    model=get_model(gender),
                ).astype(np.float32)
                expected = int(rec["total_frames"])
                if m272.shape[0] != expected:
                    raise RuntimeError(f"length mismatch: got {m272.shape[0]}, expected {expected}")
                np.savez_compressed(
                    out_path,
                    motion_272=m272,
                    source_amass=str(amass_path),
                    protocol=rec.get("protocol", ""),
                    coordinate_transform="amass_zup_to_motionstreamer_yup" if args.zup_to_yup else "none",
                )
                ok += 1
            except Exception as exc:  # noqa: BLE001
                fail += 1
                print(f"[fail] {sid}: {type(exc).__name__}: {exc}", flush=True)
        if i % 50 == 0 or i == len(records):
            print(f"[gt272] {i}/{len(records)} ok={ok} skip={skip} fail={fail}", flush=True)

    print(f"[gt272] done ok={ok} skip={skip} fail={fail} -> {out_dir}", flush=True)
    if fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
