#!/usr/bin/env python3
"""Build a side-by-side SMPL mesh cache for BABEL sequential T2M methods."""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("HFTRAINER_SKIP_AUTOREGISTER", "1")

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(REPO / "scripts/eval") not in sys.path:
    sys.path.insert(0, str(REPO / "scripts/eval"))

from babel_caption import rewrite_caption  # noqa: E402
from hftrainer.motion.representation.rotation import (  # noqa: E402
    matrix_to_axis_angle,
    rotation_6d_to_matrix,
)


SMPL_MODEL_DIR = REPO / "ref_repo/MDM/body_models"
DEFAULT_ROOT = REPO / "outputs/evaluation/babel/official_val/msstyle_30fps_gt"
METHOD_COLORS = {
    "GT": "#d7e3ea",
    "PRISM": "#33d6a6",
    "MotionStreamer": "#85b7ff",
    "FlowMDM": "#f5b14c",
    "DoubleTake": "#c59cff",
}
SEGMENT_COLORS = ["#5bd2ff", "#ffd166", "#8be28b", "#ff8da1", "#b69cff", "#f59e0b"]


def _b64(arr: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(arr).tobytes()).decode("ascii")


def _save_verts(out_dir: Path, sid: str, method: str, verts: np.ndarray) -> str:
    safe_method = method.lower().replace(" ", "_")
    fname = f"{sid}__{safe_method}.f32"
    (out_dir / "verts").mkdir(parents=True, exist_ok=True)
    verts = np.round(np.asarray(verts, dtype=np.float32), 4)
    (out_dir / "verts" / fname).write_bytes(np.ascontiguousarray(verts).tobytes())
    return fname


def _selected_frame_indices(total: int, max_frames: int) -> np.ndarray:
    if total <= max_frames:
        return np.arange(total, dtype=np.int64)
    return np.unique(np.round(np.linspace(0, total - 1, max_frames)).astype(np.int64))


def _display_span(keep: np.ndarray, start: int, end: int) -> tuple[int, int]:
    mask = (keep >= start) & (keep < end)
    hits = np.flatnonzero(mask)
    if hits.size:
        return int(hits[0]), int(hits[-1] + 1)
    center = (float(start) + float(end)) * 0.5
    nearest = int(np.abs(keep.astype(np.float64) - center).argmin())
    return nearest, min(nearest + 1, len(keep))


def _segments_for_view(rec: dict[str, Any], keep: np.ndarray, total: int) -> list[dict[str, Any]]:
    out = []
    for idx, seg in enumerate(rec.get("segments", [])):
        start = max(0, min(int(seg.get("start", 0)), total))
        end = max(start + 1, min(int(seg.get("end", total)), total))
        display_start, display_end = _display_span(keep, start, end)
        raw = str(seg.get("caption", "")).strip()
        out.append(
            {
                "index": idx,
                "start": start,
                "end": end,
                "display_start": display_start,
                "display_end": display_end,
                "raw": raw,
                "rewrite": rewrite_caption(raw),
                "color": SEGMENT_COLORS[idx % len(SEGMENT_COLORS)],
            }
        )
    return out


def _motion135_to_params(motion135: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    motion135 = np.asarray(motion135, dtype=np.float32)
    if motion135.ndim != 2 or motion135.shape[1] != 135:
        raise ValueError(f"expected motion_135 shaped (T,135), got {motion135.shape}")
    transl = motion135[:, :3].astype(np.float32)
    rot6d = motion135[:, 3:135].reshape(-1, 22, 6)
    mats = rotation_6d_to_matrix(rot6d, convention="row")
    aa = matrix_to_axis_angle(mats.reshape(-1, 3, 3)).reshape(-1, 22, 3).astype(np.float32)
    return aa[:, 0], aa[:, 1:22].reshape(len(aa), 63), transl


def _floor_center(verts: np.ndarray) -> np.ndarray:
    verts = np.asarray(verts, dtype=np.float32).copy()
    verts[..., 1] -= float(verts[..., 1].min())
    center = verts[0].reshape(-1, 3).mean(axis=0)
    verts[..., 0] -= float(center[0])
    verts[..., 2] -= float(center[2])
    return verts


def _auto_upright(verts: np.ndarray, threshold: float = 1.25) -> tuple[np.ndarray, str]:
    """Rotate clearly non-Y-up source variants upright for visualization only.

    Do not rotate when the largest axis only barely exceeds Y.  Wide T-poses can
    have an arm span a few millimeters larger than height; treating that as
    "X-up" makes the person lie sideways even though the motion is already Y-up.
    """
    verts = np.asarray(verts, dtype=np.float32)
    first = verts[0].reshape(-1, 3)
    size = first.max(axis=0) - first.min(axis=0)
    axis = int(np.argmax(size))
    y_size = max(float(size[1]), 1e-6)
    if axis == 1 or float(size[axis]) < threshold * y_size:
        return verts.copy(), "y-up"
    out = np.empty_like(verts)
    if axis == 2:
        out[..., 0] = verts[..., 0]
        out[..., 1] = -verts[..., 2]
        out[..., 2] = verts[..., 1]
        return out, "-z-to-y"
    out[..., 0] = verts[..., 1]
    out[..., 1] = -verts[..., 0]
    out[..., 2] = verts[..., 2]
    return out, "-x-to-y"


class SMPLMesh:
    def __init__(self, model_dir: Path, device: str):
        from hftrainer.motion.retarget.smpl_soma import _import_smplx, _resolve_smpl_model_dir
        import torch

        self.torch = torch
        self.device = torch.device(device)
        smplx = _import_smplx()
        mdir = _resolve_smpl_model_dir(model_dir)
        self.model = smplx.create(
            str(mdir),
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
        batch: int = 128,
    ) -> np.ndarray:
        torch = self.torch
        chunks = []
        n = len(global_orient)
        for start in range(0, n, batch):
            end = min(start + batch, n)
            b = end - start
            body69 = np.zeros((b, 69), dtype=np.float32)
            body69[:, :63] = body_pose63[start:end]
            with torch.no_grad():
                out = self.model(
                    betas=torch.zeros(b, 10, device=self.device),
                    body_pose=torch.from_numpy(body69).to(self.device),
                    global_orient=torch.from_numpy(global_orient[start:end]).to(self.device),
                    transl=torch.from_numpy(transl[start:end]).to(self.device),
                )
            chunks.append(out.vertices.detach().cpu().numpy().astype(np.float32))
        return np.concatenate(chunks, axis=0)


def _parse_method(item: str) -> tuple[str, Path]:
    if "=" not in item:
        raise SystemExit(f"--method must be NAME=DIR, got: {item}")
    name, path = item.split("=", 1)
    name = name.strip()
    path = path.strip()
    if not name or not path:
        raise SystemExit(f"--method must be NAME=DIR, got: {item}")
    p = Path(path)
    if not p.is_absolute():
        p = REPO / p
    return name, p


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", default=str(DEFAULT_ROOT / "common_valid_manifest.jsonl"))
    ap.add_argument("--out-dir", default=str(REPO / "motion_annot_web/babel_seq_compare_viewer/data"))
    ap.add_argument("--ids", default="", help="comma-separated ids; default uses manifest order")
    ap.add_argument("--num-cases", type=int, default=72)
    ap.add_argument("--max-frames", type=int, default=140)
    ap.add_argument("--playback-fps", type=float, default=30.0)
    ap.add_argument("--auto-upright-threshold", type=float, default=1.25)
    ap.add_argument(
        "--no-auto-upright",
        action="store_true",
        help="Do not rotate meshes based on first-frame bbox. Use this for already-canonical evaluator outputs.",
    )
    ap.add_argument("--device", default="")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument(
        "--method",
        action="append",
        default=None,
        help="Method source in NAME=DIR form. Directory contains <id>.npz with motion_135.",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    import torch

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    manifest = [json.loads(line) for line in Path(args.manifest).read_text().splitlines() if line.strip()]
    by_id = {rec["id"]: rec for rec in manifest}
    ids = [s.strip() for s in args.ids.split(",") if s.strip()] if args.ids else [rec["id"] for rec in manifest]
    ids = ids[: args.num_cases]
    method_items = args.method or [
        f"GT={DEFAULT_ROOT / 'mesh135_evalcanon/GT'}",
        f"PRISM={DEFAULT_ROOT / 'mesh135_evalcanon/PRISM'}",
        f"MotionStreamer={DEFAULT_ROOT / 'mesh135_evalcanon/MotionStreamer'}",
        f"FlowMDM={DEFAULT_ROOT / 'mesh135_evalcanon/FlowMDM'}",
        f"DoubleTake={DEFAULT_ROOT / 'mesh135_evalcanon/DoubleTake'}",
    ]
    methods = [_parse_method(item) for item in method_items]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "verts").mkdir(parents=True, exist_ok=True)
    mesh = SMPLMesh(SMPL_MODEL_DIR, device)
    faces_b64 = _b64(mesh.faces.astype(np.int32))
    index_cases = []

    for idx, sid in enumerate(ids, 1):
        rec = by_id[sid]
        print(f"[case] {idx}/{len(ids)} {sid}", flush=True)
        reps = []
        max_display_frames = 0
        for method, method_dir in methods:
            path = method_dir / f"{sid}.npz"
            if not path.exists():
                print(f"[skip-rep] {method} missing {path}", flush=True)
                continue
            z = np.load(path, allow_pickle=True)
            motion135 = np.asarray(z["motion_135"], dtype=np.float32)
            keep = _selected_frame_indices(len(motion135), args.max_frames)
            global_orient, body_pose63, transl = _motion135_to_params(motion135[keep])
            verts = mesh.vertices(global_orient, body_pose63, transl, batch=args.batch_size)
            if args.no_auto_upright:
                upright_note = "y-up-no-auto"
            else:
                verts, upright_note = _auto_upright(verts, threshold=float(args.auto_upright_threshold))
            verts = _floor_center(verts)
            verts_file = _save_verts(out_dir, sid, method, verts)
            max_display_frames = max(max_display_frames, len(keep))
            reps.append(
                {
                    "name": method,
                    "color": METHOD_COLORS.get(method, "#9db4c0"),
                    "num_frames": int(verts.shape[0]),
                    "num_verts": int(verts.shape[1]),
                    "faces_b64": faces_b64,
                    "verts_file": verts_file,
                    "frame_indices": keep.astype(int).tolist(),
                    "fps": float(args.playback_fps),
                    "raw_frames": int(len(motion135)),
                    "upright": upright_note,
                }
            )
        if not reps:
            continue

        gt_keep = np.asarray(reps[0]["frame_indices"], dtype=np.int64)
        total_raw_frames = int(rec.get("total_frames", max(r["raw_frames"] for r in reps)))
        case = {
            "id": sid,
            "text": " -> ".join(str(s.get("caption", "")).strip() for s in rec.get("segments", [])),
            "meta": {
                "total_frames": total_raw_frames,
                "display_frames": int(max_display_frames),
                "source_fps": float(args.playback_fps),
                "playback_fps": float(args.playback_fps),
                "methods": [r["name"] for r in reps],
            },
            "segments": _segments_for_view(rec, gt_keep, int(rec.get("total_frames", max_display_frames))),
            "boundaries": [int(b) for b in rec.get("boundaries", [])],
            "reps": reps,
        }
        (out_dir / f"case_{sid}.json").write_text(json.dumps(case, ensure_ascii=False), encoding="utf-8")
        index_cases.append(
            {
                "id": sid,
                "text": case["text"],
                "total_frames": case["meta"]["total_frames"],
                "display_frames": case["meta"]["display_frames"],
                "methods": case["meta"]["methods"],
            }
        )

    (out_dir / "index.json").write_text(
        json.dumps({"cases": index_cases}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"[done] wrote {len(index_cases)} cases to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
