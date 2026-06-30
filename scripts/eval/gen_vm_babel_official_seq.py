#!/usr/bin/env python3
"""FlowMDM / DoubleTake generation for the corrected official-BABEL protocol.

This runner intentionally consumes
``outputs/evaluation/babel/official_val/msstyle_30fps_gt/manifest.jsonl``
instead of the legacy two-segment ``data/babel/babel_seq_*`` files.  It reuses
the checked-out VersatileMotion baseline loaders for the BABEL checkpoints, but
writes flat exact-length SMPL NPZ files into this repository's canonical
official-val output tree.
"""

import argparse
import importlib.util
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R, Slerp

REPO = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(SCRIPT_DIR))

from babel_caption import rewrite_caption  # noqa: E402

VM_ROOT = Path(os.environ.get("VM_ROOT", "/apdcephfs_cq11/share_1467498/home/zeyuling/versatilemotion"))
if not VM_ROOT.is_dir():
    VM_ROOT = Path("/apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/versatilemotion")

DEFAULT_ROOT = REPO / "outputs/evaluation/babel/official_val/msstyle_30fps_gt"
DEFAULT_MANIFEST = DEFAULT_ROOT / "manifest.jsonl"


def _load_vm_module():
    path = VM_ROOT / "scripts/evaluation/eval_babel_baseline.py"
    if not path.exists():
        raise FileNotFoundError(f"VersatileMotion baseline script not found: {path}")
    old_cwd = os.getcwd()
    spec = importlib.util.spec_from_file_location("vm_eval_babel_baseline", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    os.chdir(old_cwd)
    return module


def _read_manifest(path: Path) -> List[Dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        raise RuntimeError(f"empty manifest: {path}")
    return rows


def _resolve_repo_path(value: Union[str, Path]) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (REPO / path)


def _fit_length_linear(x: np.ndarray, target_len: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    target_len = int(target_len)
    if len(x) == target_len:
        return x
    if len(x) < 2:
        return np.repeat(x[:1], target_len, axis=0).astype(np.float32)
    grid = np.linspace(0.0, len(x) - 1, target_len, dtype=np.float64)
    lo = np.floor(grid).astype(np.int64)
    hi = np.minimum(lo + 1, len(x) - 1)
    w = (grid - lo).astype(np.float32)
    shape = (target_len,) + (1,) * (x.ndim - 1)
    return (x[lo] * (1.0 - w.reshape(shape)) + x[hi] * w.reshape(shape)).astype(np.float32)


def _fit_length_axis_angle(aa: np.ndarray, target_len: int) -> np.ndarray:
    aa = np.asarray(aa, dtype=np.float32)
    target_len = int(target_len)
    if len(aa) == target_len:
        return aa
    if len(aa) < 2:
        return np.repeat(aa[:1], target_len, axis=0).astype(np.float32)
    flat = aa.reshape(len(aa), -1, 3)
    src_t = np.arange(len(aa), dtype=np.float64)
    dst_t = np.linspace(0.0, len(aa) - 1, target_len, dtype=np.float64)
    out = np.empty((target_len, flat.shape[1], 3), dtype=np.float32)
    for j in range(flat.shape[1]):
        out[:, j] = Slerp(src_t, R.from_rotvec(flat[:, j]))(dst_t).as_rotvec().astype(np.float32)
    return out.reshape((target_len,) + aa.shape[1:]).astype(np.float32)


def _enforce_smpl_length(smpl: Dict[str, Any], target_len: int) -> Tuple[Dict[str, Any], bool]:
    current = int(np.asarray(smpl["transl"]).shape[0])
    if current == int(target_len):
        return smpl, False
    out = dict(smpl)
    for key in ("global_orient", "body_pose", "poses"):
        if key in out:
            arr = np.asarray(out[key], dtype=np.float32)
            out[key] = _fit_length_axis_angle(arr, target_len)
    for key in ("trans", "transl"):
        if key in out:
            out[key] = _fit_length_linear(np.asarray(out[key], dtype=np.float32), target_len)
    for key in (
        "jaw_pose",
        "leye_pose",
        "reye_pose",
        "left_hand_pose",
        "right_hand_pose",
        "expression",
    ):
        if key in out:
            out[key] = _fit_length_linear(np.asarray(out[key], dtype=np.float32), target_len)
    return out, True


def _save_npz(path: Path, smpl: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **smpl)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--method", required=True, choices=["flowmdm", "doubletake"])
    ap.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    ap.add_argument("--output-dir", default="")
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--max-episodes", type=int, default=0)
    ap.add_argument("--min-total", type=int, default=0)
    ap.add_argument("--max-total", type=int, default=0, help="0 means no cap")
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--rewrite-captions", action="store_true", default=True)
    ap.add_argument("--no-rewrite-captions", dest="rewrite_captions", action="store_false")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--flowmdm-model", default="")
    ap.add_argument("--flow-guidance-param", type=float, default=1.5)
    ap.add_argument("--flow-bpe-denoising-step", type=int, default=125)
    ap.add_argument("--flow-use-chunked-att", action="store_true", default=True)
    ap.add_argument("--flow-no-chunked-att", dest="flow_use_chunked_att", action="store_false")
    ap.add_argument("--doubletake-model", default="")
    ap.add_argument("--doubletake-guidance-param", type=float, default=2.5)
    ap.add_argument("--handshake-size", type=int, default=20)
    ap.add_argument("--blend-len", type=int, default=20)
    ap.add_argument("--skip-steps-double-take", type=int, default=100)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_shards < 1 or not (0 <= args.shard_index < args.num_shards):
        raise ValueError("invalid shard configuration")

    random.seed(args.seed + args.shard_index)
    np.random.seed(args.seed + args.shard_index)
    torch.manual_seed(args.seed + args.shard_index)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed + args.shard_index)

    manifest = _resolve_repo_path(args.manifest).resolve()
    rows = _read_manifest(manifest)
    rows = [r for r in rows if int(r["total_frames"]) >= int(args.min_total)]
    if args.max_total:
        rows = [r for r in rows if int(r["total_frames"]) <= int(args.max_total)]
    if args.max_episodes:
        rows = rows[: int(args.max_episodes)]
    rows = rows[int(args.shard_index) :: int(args.num_shards)]
    if not rows:
        raise RuntimeError("no episodes selected")

    output_dir = (
        _resolve_repo_path(args.output_dir).resolve()
        if args.output_dir
        else (DEFAULT_ROOT / f"{args.method}_gen").resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    meta_dir = output_dir / "_meta"
    meta_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"[setup] method={args.method} episodes={len(rows)} "
        f"shard={args.shard_index}/{args.num_shards} out={output_dir}",
        flush=True,
    )
    vm = _load_vm_module()

    def build_pipeline():
        os.chdir(str(VM_ROOT))
        if args.method == "doubletake":
            return (
                vm._build_doubletake(
                    0,
                    doubletake_model=args.doubletake_model or vm._DOUBLETAKE_MODEL,
                    guidance_param=args.doubletake_guidance_param,
                    handshake_size=args.handshake_size,
                    blend_len=args.blend_len,
                    skip_steps_double_take=args.skip_steps_double_take,
                ),
                vm._generate_doubletake,
            )
        return (
            vm._build_flowmdm(
                0,
                flowmdm_model=args.flowmdm_model or vm._FLOWMDM_MODEL,
                guidance_param=args.flow_guidance_param,
                bpe_denoising_step=args.flow_bpe_denoising_step,
                use_chunked_att=args.flow_use_chunked_att,
            ),
            vm._generate_flowmdm,
        )

    pipe, generate = build_pipeline()

    run_meta = {
        "protocol": "official_babel_transition_midpoint_30fps",
        "method": args.method,
        "manifest": str(manifest),
        "output_dir": str(output_dir),
        "num_shards": int(args.num_shards),
        "shard_index": int(args.shard_index),
        "selected_episodes": len(rows),
        "rewrite_captions": bool(args.rewrite_captions),
        "seed": int(args.seed),
        "flow_guidance_param": float(args.flow_guidance_param),
        "flow_bpe_denoising_step": int(args.flow_bpe_denoising_step),
        "flow_use_chunked_att": bool(args.flow_use_chunked_att),
        "doubletake_guidance_param": float(args.doubletake_guidance_param),
        "handshake_size": int(args.handshake_size),
        "blend_len": int(args.blend_len),
        "skip_steps_double_take": int(args.skip_steps_double_take),
    }
    (meta_dir / f"run_meta_shard{args.shard_index}of{args.num_shards}.json").write_text(
        json.dumps(run_meta, indent=2, ensure_ascii=False) + "\n"
    )

    t0 = time.time()
    ok = skip = fail = resampled = 0
    for i, rec in enumerate(rows, 1):
        sid = str(rec["id"])
        out_path = output_dir / f"{sid}.npz"
        if args.skip_existing and out_path.exists():
            skip += 1
            continue
        prompts: List[str] = []
        lengths: List[int] = []
        for seg in rec.get("segments", []):
            cap = str(seg.get("caption") or "").strip()
            prompts.append(rewrite_caption(cap) if args.rewrite_captions else cap)
            lengths.append(max(1, int(seg["end"]) - int(seg["start"])))
        try:
            def generate_once():
                with torch.no_grad():
                    out = generate(pipe, prompts, lengths)
                    out = vm._complete_smplx_dict(out, fps=30.0)
                return _enforce_smpl_length(out, int(rec["total_frames"]))

            try:
                smpl, changed = generate_once()
            except Exception as exc:
                if args.method != "flowmdm":
                    raise
                print(
                    f"[retry-rebuild] {sid}: {type(exc).__name__}: {exc}",
                    flush=True,
                )
                del pipe
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                pipe, generate = build_pipeline()
                smpl, changed = generate_once()
            resampled += int(changed)
            T = int(np.asarray(smpl["transl"]).shape[0])
            if T != int(rec["total_frames"]):
                raise RuntimeError(f"length mismatch after enforce: got {T}, expected {rec['total_frames']}")
            _save_npz(out_path, smpl)
            meta = {
                "captions": prompts,
                "segment_lengths": lengths,
                "total_frames": int(rec["total_frames"]),
                "raw_generated_frames": int(np.asarray(smpl["segment_lengths"]).sum())
                if "segment_lengths" in smpl
                else T,
                "length_resampled": bool(changed),
            }
            (meta_dir / f"{sid}.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n")
            ok += 1
        except Exception as exc:  # noqa: BLE001
            fail += 1
            print(f"[fail] {sid}: {type(exc).__name__}: {exc}", flush=True)
        if i % 5 == 0 or i == len(rows):
            print(
                f"[{args.method}-official] shard={args.shard_index}/{args.num_shards} "
                f"{i}/{len(rows)} ok={ok} skip={skip} fail={fail} "
                f"resampled={resampled} elapsed={time.time() - t0:.0f}s",
                flush=True,
            )
    print(
        f"[done] method={args.method} ok={ok} skip={skip} fail={fail} "
        f"resampled={resampled} out={output_dir}",
        flush=True,
    )
    if fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
