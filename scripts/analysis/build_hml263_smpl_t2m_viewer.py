#!/usr/bin/env python3
"""Pack HML263-to-SMPL retargeted T2M baselines for m2m_eval_viewer."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "scripts" / "eval") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "eval"))

from h3d_272_to_135 import humanml272_to_motion135  # noqa: E402


DEFAULT_METHOD_DIRS = {
    "flowmdm": "outputs/evaluation/humanml3d_smpl135/flowmdm",
    "motionlab": "outputs/evaluation/humanml3d_smpl135/motionlab",
}


def read_first_caption(text_file: Path) -> str:
    if not text_file.exists():
        return ""
    for line in text_file.read_text().splitlines():
        parts = line.strip().split("#")
        if len(parts) < 4:
            continue
        caption = parts[0].strip()
        if not caption:
            continue
        try:
            f_tag = 0.0 if parts[2] == "nan" else float(parts[2])
            t_tag = 0.0 if parts[3] == "nan" else float(parts[3])
        except ValueError:
            continue
        if f_tag == 0.0 and t_tag == 0.0:
            return caption
    return ""


def parse_method_dirs(values: list[str] | None) -> dict[str, Path]:
    if not values:
        return {k: PROJECT_ROOT / v for k, v in DEFAULT_METHOD_DIRS.items()}
    out: dict[str, Path] = {}
    for item in values:
        if "=" not in item:
            raise ValueError(f"expected METHOD=DIR, got {item!r}")
        name, path = item.split("=", 1)
        out[name.strip()] = Path(path).expanduser().resolve()
    return out


def load_motion_135(path: Path) -> np.ndarray:
    data = np.load(path, allow_pickle=True)
    if "motion_135" not in data:
        raise KeyError(f"{path} has no motion_135")
    motion = np.asarray(data["motion_135"], dtype=np.float32)
    if motion.ndim != 2 or motion.shape[1] != 135:
        raise ValueError(f"{path} expected (T,135), got {motion.shape}")
    return motion


def common_ids(method_dirs: dict[str, Path], gt_dir: Path) -> list[str]:
    ids: set[str] | None = None
    for method, d in method_dirs.items():
        cur = {p.stem for p in d.glob("*.npz") if not p.name.startswith("_")}
        print(f"[scan] {method}: {len(cur)} retargeted NPZ in {d}")
        ids = cur if ids is None else ids & cur
    ids = ids or set()
    ids &= {p.stem for p in gt_dir.glob("*.npy")}
    return sorted(ids)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method-dir", action="append",
                        help="METHOD=DIR. Default: flowmdm/motionlab under outputs/evaluation/humanml3d_smpl135.")
    parser.add_argument("--gt-dir", default=str(
        PROJECT_ROOT / "ref_repo/MotionStreamer/MotionStreamer/humanml3d_272/motion_data"))
    parser.add_argument("--text-dir", default=str(
        PROJECT_ROOT / "ref_repo/MotionStreamer/MotionStreamer/humanml3d_272/texts"))
    parser.add_argument("--out-dir", default=str(
        PROJECT_ROOT / "outputs/evaluation/t2m_hml263_smpl_viewer"))
    parser.add_argument("--task-key", default="E1_HumanML3D_T2M_HML263_SMPL")
    parser.add_argument("--ids", default=None,
                        help="Comma-separated ids or a text file with one id per line.")
    parser.add_argument("--max-cases", type=int, default=0)
    args = parser.parse_args()

    method_dirs = parse_method_dirs(args.method_dir)
    gt_dir = Path(args.gt_dir).expanduser().resolve()
    text_dir = Path(args.text_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()

    ids = common_ids(method_dirs, gt_dir)
    if args.ids:
        ids_arg = Path(args.ids)
        if ids_arg.exists():
            wanted = [x.strip() for x in ids_arg.read_text().splitlines() if x.strip()]
        else:
            wanted = [x.strip() for x in args.ids.split(",") if x.strip()]
        ids = [sid for sid in wanted if sid in set(ids)]
    if args.max_cases > 0:
        ids = ids[: args.max_cases]
    print(f"[pack] cases={len(ids)} out={out_dir}")

    gt_cache: dict[str, np.ndarray] = {}
    wrote = {method: 0 for method in method_dirs}
    failed = []
    for i, sid in enumerate(ids, 1):
        gt_path = gt_dir / f"{sid}.npy"
        try:
            gt_motion = gt_cache.get(sid)
            if gt_motion is None:
                gt_motion = humanml272_to_motion135(np.load(gt_path).astype(np.float32))
                gt_cache[sid] = gt_motion
            caption = read_first_caption(text_dir / f"{sid}.txt")
            for method, src_dir in method_dirs.items():
                pred_path = src_dir / f"{sid}.npz"
                pred_motion = load_motion_135(pred_path)
                dst = out_dir / method / args.task_key / "npz" / f"{sid}.npz"
                dst.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    dst,
                    motion_135=pred_motion.astype(np.float32),
                    gt_motion_135=gt_motion.astype(np.float32),
                    caption=np.array(caption),
                    task_key=np.array(args.task_key),
                    source_id=np.array(sid),
                    method=np.array(method),
                    pred_source_path=np.array(str(pred_path)),
                    gt_source_path=np.array(str(gt_path)),
                )
                wrote[method] += 1
        except Exception as exc:  # noqa: BLE001
            failed.append((sid, type(exc).__name__, str(exc)))
        if i % 250 == 0 or i == len(ids):
            print(f"[progress] {i}/{len(ids)} wrote={wrote} failed={len(failed)}", flush=True)

    manifest = {
        "task_key": args.task_key,
        "out_dir": str(out_dir),
        "num_cases": len(ids),
        "methods": {k: str(v) for k, v in method_dirs.items()},
        "wrote": wrote,
        "failed": failed[:50],
    }
    (out_dir / "_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
