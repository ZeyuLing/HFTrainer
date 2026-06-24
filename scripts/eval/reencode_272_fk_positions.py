#!/usr/bin/env python3
"""Rebuild MS272 position channels from decoded rotations and root translation.

This is a diagnostic for native-272 autoencoders such as MotionStreamer TAE:
keep the reconstructed rotation/root trajectory, discard the reconstructed
stored-position channels, and re-encode MS272 from FK joints.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
os.environ.setdefault("HFTRAINER_SKIP_AUTOREGISTER", "1")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts/eval"))

from hftrainer.datasets.motion.representation.humanml_repr import (  # noqa: E402
    fk_smplh_joints,
    recover_local_rotations_and_root,
)
from hftrainer.motion.representation.motion272 import (  # noqa: E402
    _canonical_272_offsets,
    encode_smpl_to_272,
)
from hftrainer.motion.skeleton.fk import forward_kinematics  # noqa: E402


POS_SLICE = slice(8, 74)
VEL_SLICE = slice(74, 140)
ROT_SLICE = slice(140, 272)


def _ids_from_split(split: Path | None) -> list[str] | None:
    if split is None:
        return None
    return [line.strip() for line in split.read_text(encoding="utf-8").splitlines() if line.strip()]


def _fk_reencode_full(m272: np.ndarray, fk_mode: str) -> np.ndarray:
    rot, root = recover_local_rotations_and_root(np.asarray(m272, dtype=np.float32))
    if fk_mode == "canon272":
        import torch

        with torch.no_grad():
            joints, _ = forward_kinematics(
                torch.from_numpy(rot).float(),
                torch.from_numpy(root).float(),
                torch.from_numpy(_canonical_272_offsets()).float(),
            )
        return np.asarray(encode_smpl_to_272(joints.numpy(), rot), dtype=np.float32)
    if fk_mode == "smplh":
        joints = fk_smplh_joints(rot, root)
        return np.asarray(encode_smpl_to_272(joints, rot), dtype=np.float32)
    raise ValueError(f"unknown fk_mode={fk_mode!r}")


def _reencode(m272: np.ndarray, fk_mode: str, rewrite_mode: str) -> np.ndarray:
    rebuilt = _fk_reencode_full(m272, fk_mode)
    if rewrite_mode == "full":
        return rebuilt
    if rewrite_mode == "position-blocks":
        out = np.asarray(m272, dtype=np.float32).copy()
        out[:, POS_SLICE] = rebuilt[:, POS_SLICE]
        out[:, VEL_SLICE] = rebuilt[:, VEL_SLICE]
        return out
    raise ValueError(f"unknown rewrite_mode={rewrite_mode!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--fk-mode", choices=["canon272", "smplh"], default="canon272")
    parser.add_argument(
        "--rewrite-mode",
        choices=["position-blocks", "full"],
        default="position-blocks",
        help="position-blocks keeps TAE root/heading/rotation channels unchanged.",
    )
    parser.add_argument("--split", default="")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ids = _ids_from_split(Path(args.split) if args.split else None)
    if ids is None:
        paths = sorted(in_dir.glob("*.npz"))
    else:
        paths = [in_dir / f"{sid}.npz" for sid in ids]
    if args.max_samples:
        paths = paths[: args.max_samples]

    stats: dict[str, list[float]] = {
        "pos_l2_mm": [],
        "vel_l2_mm": [],
        "rot_block_l2": [],
        "root_vel_l2": [],
    }
    skipped = {"missing": 0, "exists": 0, "error": 0}
    failures: list[dict[str, str]] = []
    written = 0

    for idx, src in enumerate(paths, 1):
        if not src.exists():
            skipped["missing"] += 1
            continue
        dst = out_dir / src.name
        if dst.exists() and not args.overwrite:
            skipped["exists"] += 1
            continue
        try:
            data = np.load(src, allow_pickle=True)
            if "motion_272" not in data.files:
                raise KeyError("missing motion_272")
            m272 = np.asarray(data["motion_272"], dtype=np.float32)
            rebuilt = _reencode(m272, args.fk_mode, args.rewrite_mode)
            payload = {key: data[key] for key in data.files}
            payload["motion_272"] = rebuilt
            np.savez(dst, **payload)
            written += 1

            t = min(len(m272), len(rebuilt))
            stats["pos_l2_mm"].append(float(np.linalg.norm(rebuilt[:t, POS_SLICE] - m272[:t, POS_SLICE], axis=1).mean() * 1000.0))
            stats["vel_l2_mm"].append(float(np.linalg.norm(rebuilt[:t, VEL_SLICE] - m272[:t, VEL_SLICE], axis=1).mean() * 1000.0))
            stats["rot_block_l2"].append(float(np.linalg.norm(rebuilt[:t, ROT_SLICE] - m272[:t, ROT_SLICE], axis=1).mean()))
            stats["root_vel_l2"].append(float(np.linalg.norm(rebuilt[:t, :2] - m272[:t, :2], axis=1).mean()))
        except Exception as exc:  # noqa: BLE001
            skipped["error"] += 1
            failures.append({"file": src.name, "error": f"{type(exc).__name__}: {exc}"})
            if len(failures) <= 10:
                print(f"[fail] {src.name}: {type(exc).__name__}: {exc}", flush=True)
        if idx % 500 == 0:
            print(f"[progress] {idx}/{len(paths)} written={written} skipped={skipped}", flush=True)

    summary = {
        key: {
            "mean": float(np.mean(vals)) if vals else None,
            "std": float(np.std(vals)) if vals else None,
            "n": len(vals),
        }
        for key, vals in stats.items()
    }
    manifest = {
        "input_dir": str(in_dir),
        "output_dir": str(out_dir),
        "fk_mode": args.fk_mode,
        "rewrite_mode": args.rewrite_mode,
        "selected": len(paths),
        "written": written,
        "skipped": skipped,
        "summary": summary,
        "failures": failures,
    }
    (out_dir / "reencode_fk_positions_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
