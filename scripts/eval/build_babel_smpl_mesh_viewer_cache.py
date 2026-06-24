#!/usr/bin/env python3
"""Build SMPL-mesh web-viewer cache for BABEL caption audits."""

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

from hftrainer.datasets.motion.representation.humanml_repr import (  # noqa: E402
    recover_local_rotations_and_root,
)
from hftrainer.motion.representation.rotation import matrix_to_axis_angle  # noqa: E402


SMPL_MODEL_DIR = REPO / "ref_repo/MDM/body_models"
SEGMENT_COLORS = [
    "#6ee7f9",
    "#fbbf24",
    "#a78bfa",
    "#34d399",
    "#fb7185",
    "#60a5fa",
    "#f472b6",
    "#c4b5fd",
]


def _b64(arr: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(arr).tobytes()).decode("ascii")


def _floor_center(pos: np.ndarray, ref_xz: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    pos = np.asarray(pos, dtype=np.float32).copy()
    pos[..., 1] -= float(pos[..., 1].min())
    if ref_xz is None:
        ref_xz = pos[0].reshape(-1, 3).mean(axis=0)[[0, 2]]
    pos[..., 0] -= float(ref_xz[0])
    pos[..., 2] -= float(ref_xz[1])
    return pos, np.asarray(ref_xz, dtype=np.float32)


def _save_verts(out_dir: Path, sid: str, verts: np.ndarray) -> str:
    fname = f"{sid}__smpl.f32"
    verts = np.round(np.asarray(verts, dtype=np.float32), 4)
    (out_dir / "verts").mkdir(parents=True, exist_ok=True)
    (out_dir / "verts" / fname).write_bytes(np.ascontiguousarray(verts).tobytes())
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

    def vertices(self, global_orient: np.ndarray, body_pose63: np.ndarray, transl: np.ndarray,
                 batch: int = 128) -> np.ndarray:
        torch = self.torch
        chunks = []
        n = len(global_orient)
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


def _selected_frame_indices(total: int, max_frames: int) -> np.ndarray:
    if total <= max_frames:
        return np.arange(total, dtype=np.int64)
    return np.unique(np.round(np.linspace(0, total - 1, max_frames)).astype(np.int64))


def _source_fps(gt_smpl_dir: Path, sid: str, fallback: float = 30.0) -> float:
    path = gt_smpl_dir / sid / "gt.npz"
    if not path.exists():
        return fallback
    try:
        z = np.load(path, allow_pickle=True)
        return float(z["mocap_framerate"])
    except Exception:  # noqa: BLE001
        return fallback


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
        raw = seg.get("raw", seg.get("caption", ""))
        rewrite = seg.get("cache", raw)
        out.append(
            {
                "index": idx,
                "start": start,
                "end": end,
                "display_start": display_start,
                "display_end": display_end,
                "raw": raw,
                "caption": seg.get("caption", raw),
                "rewrite": rewrite,
                "cache": rewrite,
                "rule": seg.get("rule", ""),
                "flags": seg.get("flags") or [],
                "score": int(seg.get("score", 0) or 0),
                "color": SEGMENT_COLORS[idx % len(SEGMENT_COLORS)],
            }
        )
    return out


def _segment_text(rec: dict[str, Any]) -> str:
    lines = []
    for seg in rec.get("segments", []):
        flags = seg.get("flags") or []
        flag_text = f" | flags: {', '.join(flags)}" if flags else ""
        lines.append(
            f"[{seg.get('start', 0)}-{seg.get('end', 0)}] "
            f"raw: {seg.get('raw', seg.get('caption', ''))} | "
            f"rewrite: {seg.get('cache', '')}{flag_text}"
        )
    return "\n".join(lines)


def _normalize_case_record(rec: dict[str, Any]) -> dict[str, Any]:
    # Records from audit_records.json already contain raw/cache/rule/flags. If
    # called on the plain BABEL manifest, keep a minimal fallback shape.
    out = dict(rec)
    segs = []
    for seg in rec.get("segments", []):
        if "raw" in seg:
            segs.append(seg)
        else:
            raw = seg.get("caption", "")
            segs.append({**seg, "raw": raw, "cache": raw, "rule": raw, "flags": []})
    out["segments"] = segs
    out.setdefault("rewrite_score", 0)
    return out


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--audit-json", default=str(REPO / "outputs/evaluation/babel_caption_audit_20260623/audit_records.json"))
    ap.add_argument("--motion-dir", default=str(REPO / "data/babel_272_stream/val_stream"))
    ap.add_argument("--gt-smpl-dir", default=str(REPO / "data/babel_272_stream/gt_smpl"))
    ap.add_argument("--out-dir", default=str(REPO / "motion_annot_web/babel_smpl_mesh_viewer/data"))
    ap.add_argument("--ids", default="", help="comma-separated BABEL seq ids; default uses audit-json order")
    ap.add_argument("--num-cases", type=int, default=36)
    ap.add_argument("--max-frames", type=int, default=240)
    ap.add_argument(
        "--playback-fps",
        type=float,
        default=60.0,
        help=(
            "Viewer playback fps. MotionStreamer BABEL stream files carry 30fps metadata, "
            "but visual inspection and segment durations match a 60fps time axis."
        ),
    )
    ap.add_argument("--device", default="")
    ap.add_argument("--batch-size", type=int, default=128)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    import torch

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "verts").mkdir(parents=True, exist_ok=True)

    audit_rows = [_normalize_case_record(r) for r in json.loads(Path(args.audit_json).read_text(encoding="utf-8"))]
    by_id = {r["id"]: r for r in audit_rows}
    ids = [s.strip() for s in args.ids.split(",") if s.strip()] if args.ids else [r["id"] for r in audit_rows]
    ids = ids[: args.num_cases]
    mesh = SMPLMesh(SMPL_MODEL_DIR, device)
    faces_b64 = _b64(mesh.faces.astype(np.int32))
    index_cases = []
    motion_dir = Path(args.motion_dir)
    gt_smpl_dir = Path(args.gt_smpl_dir)
    for idx, sid in enumerate(ids, 1):
        rec = by_id[sid]
        path = motion_dir / f"{sid}.npy"
        print(f"[case] {idx}/{len(ids)} {sid} {path}", flush=True)
        m272_raw = np.load(path).astype(np.float32)
        total_frames = int(len(m272_raw))
        keep = _selected_frame_indices(total_frames, args.max_frames)
        m272 = m272_raw[keep]
        source_fps = _source_fps(gt_smpl_dir, sid, fallback=30.0)
        display_fps = source_fps * (float(len(keep)) / float(total_frames)) if total_frames > 0 else source_fps
        playback_fps = float(args.playback_fps) * (float(len(keep)) / float(total_frames)) if total_frames > 0 else float(args.playback_fps)
        segments = _segments_for_view(rec, keep, total_frames)
        rot, root = recover_local_rotations_and_root(m272)
        aa = matrix_to_axis_angle(rot.reshape(-1, 3, 3)).reshape(rot.shape[0], rot.shape[1], 3).astype(np.float32)
        verts = mesh.vertices(aa[:, 0], aa[:, 1:22].reshape(len(aa), 63), root.astype(np.float32), batch=args.batch_size)
        verts, _ = _floor_center(verts)
        verts_file = _save_verts(out_dir, sid, verts)
        text = _segment_text(rec)
        case = {
            "id": sid,
            "text": text,
            "meta": {
                "total_frames": total_frames,
                "display_frames": int(len(keep)),
                "source_fps": source_fps,
                "display_fps": display_fps,
                "playback_fps": playback_fps,
                "duration_sec": float(total_frames / source_fps) if source_fps > 0 else None,
                "playback_duration_sec": float(len(keep) / playback_fps) if playback_fps > 0 else None,
                "rewrite_score": int(rec.get("rewrite_score", 0)),
                "frame_stride_note": "uniform temporal downsample" if len(keep) < total_frames else "native frames",
                "frame_indices": keep.astype(int).tolist(),
            },
            "segments": segments,
            "reps": [
                {
                    "name": "smpl",
                    "type": "mesh",
                    "label": "BABEL SMPL mesh from MotionStreamer-272",
                    "fps": float(display_fps),
                    "source_fps": float(source_fps),
                    "playback_fps": float(playback_fps),
                    "frame_indices": keep.astype(int).tolist(),
                    "color": "#8db7ff",
                    "num_frames": int(verts.shape[0]),
                    "num_verts": int(verts.shape[1]),
                    "faces_b64": faces_b64,
                    "verts_file": verts_file,
                    "info": (
                        f"{verts.shape[0]} displayed frames from {total_frames} raw frames "
                        f"@{playback_fps:.2f}fps playback ({source_fps:.2f}fps metadata) · 6890-vtx neutral SMPL · "
                        f"rewrite score {rec.get('rewrite_score', 0)}"
                    ),
                }
            ],
        }
        (out_dir / f"case_{sid}.json").write_text(json.dumps(case, ensure_ascii=False), encoding="utf-8")
        index_cases.append({
            "id": sid,
            "text": text.splitlines()[0] if text else sid,
            "total_frames": total_frames,
            "display_frames": int(len(keep)),
            "source_fps": source_fps,
            "display_fps": display_fps,
            "playback_fps": playback_fps,
            "rewrite_score": int(rec.get("rewrite_score", 0)),
        })
    (out_dir / "index.json").write_text(json.dumps({"cases": index_cases}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[done] wrote {len(index_cases)} mesh cases to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
