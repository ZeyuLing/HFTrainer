#!/usr/bin/env python3
"""Build KIMODO pred272 -> motion_135 NPZs for the SMPL mesh viewer.

The KIMODO T2M debug outputs store SOMA-77/SMPL-22 joint positions, while the
existing SMPL mesh viewer expects ``motion_135``. This script uses the validated
MotionStreamer-272 -> motion_135 conversion and packs per-sample NPZs that
``motion_annot_web/m2m_eval_viewer/retarget_smpl_app.py`` can render.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "scripts/eval") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts/eval"))

from h3d_272_to_135 import humanml272_to_motion135  # noqa: E402


def _read_caption(debug_path: Path) -> str:
    if not debug_path.exists():
        return ""
    try:
        with np.load(debug_path, allow_pickle=True) as data:
            if "caption" in data.files:
                return str(np.asarray(data["caption"]).item())
    except Exception:
        return ""
    return ""


def _resample_positions(pos: np.ndarray, target_len: int) -> np.ndarray:
    pos = np.asarray(pos, dtype=np.float32)
    if len(pos) == target_len or len(pos) < 2:
        return pos[:target_len]
    src = np.linspace(0.0, 1.0, len(pos), dtype=np.float64)
    dst = np.linspace(0.0, 1.0, target_len, dtype=np.float64)
    flat = pos.reshape(len(pos), -1)
    out = np.empty((target_len, flat.shape[1]), dtype=np.float32)
    for c in range(flat.shape[1]):
        out[:, c] = np.interp(dst, src, flat[:, c])
    return out.reshape(target_len, *pos.shape[1:])


def _fk_positions(motion_135: np.ndarray) -> np.ndarray | None:
    try:
        import torch
        from hftrainer.pipelines.motion.differentiable_fk import motion135_to_fk
        from motionstreamer_272_encoder import _canonical_272_offsets

        bo = torch.from_numpy(_canonical_272_offsets()).float()
        with torch.no_grad():
            pos, _wr, _tr, _lr = motion135_to_fk(
                torch.from_numpy(motion_135[:, :135]).float(),
                bo,
                rotation_space="local",
            )
        return pos.detach().cpu().numpy().astype(np.float32)
    except Exception as exc:  # noqa: BLE001
        print(f"  [warn] FK diagnostic failed: {type(exc).__name__}: {exc}", flush=True)
        return None


def _debug_positions(debug_path: Path) -> np.ndarray | None:
    if not debug_path.exists():
        return None
    try:
        with np.load(debug_path, allow_pickle=True) as data:
            if "positions" in data.files:
                return np.asarray(data["positions"], dtype=np.float32)
    except Exception:
        return None
    return None


def _fit_error_mm(motion_135: np.ndarray, debug_path: Path) -> np.ndarray:
    fk = _fk_positions(motion_135)
    dbg = _debug_positions(debug_path)
    if fk is None or dbg is None or len(fk) == 0 or len(dbg) == 0:
        return np.zeros((len(motion_135),), dtype=np.float32)
    dbg = _resample_positions(dbg, len(fk))
    return (np.linalg.norm(fk[:, :22] - dbg[:, :22], axis=-1).mean(axis=1) * 1000.0).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pred272-dir",
        default=str(
            PROJECT_ROOT
            / "outputs/evaluation/humanml3d_t2m_kimodo_20260605_genfix"
            / "kimodo_official/pred272"
        ),
    )
    parser.add_argument(
        "--debug-dir",
        default=str(
            PROJECT_ROOT
            / "outputs/evaluation/humanml3d_t2m_kimodo_20260605_genfix"
            / "kimodo_official/debug_npz"
        ),
    )
    parser.add_argument(
        "--out-dir",
        default=str(
            PROJECT_ROOT
            / "outputs/evaluation/humanml3d_smpl135_kimodo_20260605_genfix"
            / "kimodo_official"
        ),
    )
    parser.add_argument("--limit", type=int, default=0, help="0 = all files")
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    pred_dir = Path(args.pred272_dir)
    debug_dir = Path(args.debug_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(pred_dir.glob("*.npy"))
    if args.limit > 0:
        files = files[: args.limit]
    print(f"[build] pred272={pred_dir} files={len(files)} -> {out_dir}", flush=True)

    ok = skip = fail = 0
    for i, src in enumerate(files, 1):
        dst = out_dir / f"{src.stem}.npz"
        if args.skip_existing and dst.exists():
            skip += 1
            continue
        try:
            m272 = np.load(src).astype(np.float32)
            motion_135 = humanml272_to_motion135(m272)
            debug_path = debug_dir / f"{src.stem}.npz"
            fit_mpjpe_mm = _fit_error_mm(motion_135, debug_path)
            caption = _read_caption(debug_path)
            np.savez_compressed(
                dst,
                motion_135=motion_135.astype(np.float32),
                fit_mpjpe_mm=fit_mpjpe_mm.astype(np.float32),
                caption=np.array(caption, dtype=object),
                source_id=np.array(src.stem, dtype=object),
                source_pred272_path=np.array(str(src), dtype=object),
                source_debug_path=np.array(str(debug_path), dtype=object),
            )
            ok += 1
        except Exception as exc:  # noqa: BLE001
            fail += 1
            if fail <= 10:
                print(f"  [fail] {src.name}: {type(exc).__name__}: {exc}", flush=True)
        if i % 250 == 0 or i == len(files):
            print(f"  {i}/{len(files)} ok={ok} skip={skip} fail={fail}", flush=True)
    print(f"[done] ok={ok} skip={skip} fail={fail} out={out_dir}", flush=True)


if __name__ == "__main__":
    main()
