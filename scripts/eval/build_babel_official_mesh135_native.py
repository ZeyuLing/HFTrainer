#!/usr/bin/env python3
"""Materialize official-BABEL GT/PRISM/MotionStreamer as 135-dim SMPL motions.

The compare mesh viewer consumes ``motion_135`` files.  This script builds those
files from the corrected official-val protocol rooted at
``outputs/evaluation/babel/official_val/msstyle_30fps_gt``:

* GT: evaluator-facing corrected MS272 stream -> recover local rotations/root.
* PRISM: native SMPL axis-angle params -> row-major 6D rotations.
* MotionStreamer: generated ``motion_135`` saved by the inference script.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("HFTRAINER_SKIP_AUTOREGISTER", "1")

REPO = Path(__file__).resolve().parents[2]
for p in (REPO, REPO / "scripts" / "eval"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from h3d_272_to_135 import humanml272_to_motion135  # noqa: E402
from hftrainer.models.motion.components.utils.geometry.rotation_convert import (  # noqa: E402
    axis_angle_to_matrix,
    matrix_to_rotation_6d,
)


DEFAULT_ROOT = REPO / "outputs/evaluation/babel/official_val/msstyle_30fps_gt"


def native_smpl_to_135(
    z: np.lib.npyio.NpzFile,
    *,
    zup_to_yup: bool = False,
    transpose_rot: bool = False,
) -> np.ndarray:
    import torch

    go = np.asarray(z["global_orient"], np.float32).reshape(-1, 1, 3)
    bp = np.asarray(z["body_pose"], np.float32).reshape(go.shape[0], 21, 3)
    transl = np.asarray(z["transl"], np.float32).reshape(go.shape[0], 3)
    aa = np.concatenate([go, bp], axis=1)
    rot = axis_angle_to_matrix(torch.from_numpy(aa)).numpy().astype(np.float32)
    if transpose_rot:
        rot = np.swapaxes(rot, -1, -2).astype(np.float32)
    if zup_to_yup:
        rx = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], np.float32)
        transl = np.stack([transl[:, 0], transl[:, 1], -transl[:, 2]], axis=-1).astype(np.float32)
        rot[:, 0] = np.einsum("ij,tjk->tik", rx, rot[:, 0]).astype(np.float32)
    rot = torch.from_numpy(rot)
    d6 = matrix_to_rotation_6d(rot, convention="row").numpy().reshape(go.shape[0], 132)
    return np.concatenate([transl, d6], axis=-1).astype(np.float32)


def save_motion(out_path: Path, motion_135: np.ndarray, sid: str, source: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        motion_135=np.asarray(motion_135, dtype=np.float32),
        source_id=np.array(sid, dtype=object),
        source=np.array(source, dtype=object),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=str(DEFAULT_ROOT))
    ap.add_argument("--manifest", default="")
    ap.add_argument("--out-dir", default="")
    ap.add_argument("--gt-dir", default="")
    ap.add_argument("--prism-dir", default="")
    ap.add_argument("--motionstreamer-dir", default="")
    ap.add_argument(
        "--source-mode",
        default="native",
        choices=["native", "evalcanon"],
        help=(
            "native keeps generator SMPL global yaw where available; evalcanon "
            "renders the exact evaluator-facing MotionStreamer-272 streams."
        ),
    )
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()

    root = Path(args.root)
    manifest = Path(args.manifest) if args.manifest else root / "manifest.jsonl"
    out_dir = Path(args.out_dir) if args.out_dir else root / "mesh135_native"
    records = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    if args.limit:
        records = records[: args.limit]

    if args.source_mode == "evalcanon":
        sources: dict[str, tuple[Path, dict[str, bool]]] = {
            "GT": (Path(args.gt_dir) if args.gt_dir else root / "gt_272_stream_yup", {}),
            "PRISM": (root / "prism_prep272_segfix", {}),
            "MotionStreamer": (
                Path(args.motionstreamer_dir) if args.motionstreamer_dir else root / "motionstreamer_gen",
                {},
            ),
            "FlowMDM": (root / "flowmdm_272f_yup_tr", {}),
            "DoubleTake": (root / "doubletake_272f_yup_tr", {}),
        }
    else:
        sources = {
            "GT": (Path(args.gt_dir) if args.gt_dir else root / "gt_272_stream_yup", {}),
            "PRISM": (Path(args.prism_dir) if args.prism_dir else root / "prism_gen", {}),
            "MotionStreamer": (
                Path(args.motionstreamer_dir) if args.motionstreamer_dir else root / "motionstreamer_gen",
                {},
            ),
            "FlowMDM": (root / "flowmdm_gen", {"zup_to_yup": True, "transpose_rot": True}),
            "DoubleTake": (root / "doubletake_gen", {"zup_to_yup": True, "transpose_rot": True}),
        }
    ok = {name: 0 for name in sources}
    fail: list[str] = []

    for i, rec in enumerate(records, 1):
        sid = rec["id"]
        expected = int(rec["total_frames"])
        for method, (src_dir, smpl_flags) in sources.items():
            dst = out_dir / method / f"{sid}.npz"
            if args.skip_existing and dst.exists():
                ok[method] += 1
                continue
            src = src_dir / f"{sid}.npz"
            if not src.exists():
                fail.append(f"{method}:{sid}:missing")
                continue
            try:
                z = np.load(src, allow_pickle=True)
                if "motion_272" in z.files:
                    motion_135 = humanml272_to_motion135(np.asarray(z["motion_272"], dtype=np.float32))
                elif method == "GT":
                    motion_135 = humanml272_to_motion135(np.asarray(z["motion_272"], dtype=np.float32))
                elif "motion_135" in z.files:
                    motion_135 = np.asarray(z["motion_135"], dtype=np.float32)
                else:
                    motion_135 = native_smpl_to_135(z, **smpl_flags)
                if motion_135.shape != (expected, 135):
                    raise ValueError(f"shape {motion_135.shape}, expected {(expected, 135)}")
                save_motion(dst, motion_135, sid, str(src))
                ok[method] += 1
            except Exception as exc:  # noqa: BLE001
                fail.append(f"{method}:{sid}:{type(exc).__name__}:{exc}")
        if i % 100 == 0 or i == len(records):
            print(f"[mesh135-official] {i}/{len(records)} ok={ok} fail={len(fail)}", flush=True)

    print(f"[mesh135-official] DONE ok={ok} fail={len(fail)} -> {out_dir}", flush=True)
    if fail:
        for row in fail[:30]:
            print(f"[mesh135-official] FAIL {row}", flush=True)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
