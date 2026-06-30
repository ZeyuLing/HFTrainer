#!/usr/bin/env python3
"""PRISM generation for the corrected official-BABEL long protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(SCRIPT_DIR))

from babel_caption import rewrite_caption  # noqa: E402
from eval_prism_kafs_ablation import (  # noqa: E402
    load_prism_bundle,
    save_smplx_npz,
)
from hftrainer.datasets.motion.representation.humanml_repr import fk_smplh_joints  # noqa: E402
from hftrainer.motion.representation.rotation import axis_angle_to_matrix  # noqa: E402


DEFAULT_MANIFEST = (
    REPO
    / "outputs"
    / "evaluation"
    / "babel"
    / "official_val"
    / "msstyle_30fps_gt"
    / "manifest.jsonl"
)
DEFAULT_OUT = (
    REPO
    / "outputs"
    / "evaluation"
    / "babel"
    / "official_val"
    / "msstyle_30fps_gt"
    / "prism_gen"
)


def _read_manifest(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        raise RuntimeError(f"empty manifest: {path}")
    return rows


def _set_seed(seed: int) -> None:
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _sample_seed(base_seed: int, sample_id: str, ar_cond_frames: int, attempt: int) -> int:
    payload = f"{sample_id}|ar{int(ar_cond_frames)}|attempt{int(attempt)}".encode("utf-8")
    offset = int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little")
    return (int(base_seed) + offset) % (2**31 - 1)


def _first_frame_orientation(smplx_dict: dict[str, Any]) -> dict[str, float]:
    transl = np.asarray(smplx_dict.get("transl", smplx_dict.get("trans")), dtype=np.float32)
    go = np.asarray(smplx_dict["global_orient"], dtype=np.float32)
    bp = np.asarray(smplx_dict["body_pose"], dtype=np.float32)
    root_rot = axis_angle_to_matrix(go[:1])[0]
    poses = np.concatenate([go[:1], bp[:1]], axis=-1).reshape(1, 22, 3)
    rot = axis_angle_to_matrix(poses.reshape(-1, 3)).reshape(1, 22, 3, 3)
    joints = fk_smplh_joints(rot, transl[:1])[0]
    return {
        "root_up_y": float(root_rot[1, 1]),
        "root_y": float(joints[0, 1]),
        "head_y": float(joints[15, 1]),
        "head_root_y": float(joints[15, 1] - joints[0, 1]),
        "min_y": float(joints[:, 1].min()),
        "max_y": float(joints[:, 1].max()),
    }


_HORIZONTAL_OR_INVERTED_WORDS = (
    "lie",
    "lay",
    "floor",
    "ground",
    "crawl",
    "swim",
    "roll",
    "somersault",
    "cartwheel",
    "flip",
    "handstand",
)


def _allows_low_head(prompt: str) -> bool:
    text = str(prompt).lower()
    return any(word in text for word in _HORIZONTAL_OR_INVERTED_WORDS)


def _expand_generation_segments(
    prompts: list[str],
    seg_lens: list[int],
    max_segment_len: int,
) -> tuple[list[str], list[int], list[int]]:
    """Split overlong generation segments while preserving original boundaries."""
    gen_prompts: list[str] = []
    gen_lens: list[int] = []
    gen_source: list[int] = []
    max_segment_len = max(1, int(max_segment_len))
    for source_idx, (prompt, seg_len) in enumerate(zip(prompts, seg_lens)):
        remaining = max(1, int(seg_len))
        while remaining > max_segment_len:
            gen_prompts.append(prompt)
            gen_lens.append(max_segment_len)
            gen_source.append(source_idx)
            remaining -= max_segment_len
        gen_prompts.append(prompt)
        gen_lens.append(remaining)
        gen_source.append(source_idx)
    return gen_prompts, gen_lens, gen_source


def _np_int_list(value: Any) -> list[int]:
    if value is None:
        return []
    return [int(x) for x in np.asarray(value).reshape(-1).tolist()]


def _orientation_is_bad(
    diag: dict[str, float],
    first_prompt: str,
    min_root_up_y: float,
    min_head_root_y: float,
) -> bool:
    if float(diag["root_up_y"]) < float(min_root_up_y):
        return True
    if not _allows_low_head(first_prompt) and float(diag["head_root_y"]) < float(min_head_root_y):
        return True
    return False


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/prism/prism_1b_tp2m_multiframe_kt_spectral_unified.py")
    ap.add_argument("--checkpoint", default="work_dirs/prism_1b_tp2m_multiframe_kt_spectral_unified_t5cached/checkpoint-epoch_16")
    ap.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    ap.add_argument("--output-dir", default=str(DEFAULT_OUT))
    ap.add_argument("--num-inference-steps", type=int, default=50)
    ap.add_argument("--guidance-scale", type=float, default=5.0)
    ap.add_argument("--kafs-mode", default="none", choices=["none", "depth_driven", "uniform", "random"])
    ap.add_argument("--length-policy", choices=["direct_len", "pad360_crop", "legacy"], default="pad360_crop",
                    help="PRISM generation length policy. pad360_crop is the training-aligned default: "
                         "use a 360-frame canvas per segment and crop. direct_len is kept for ablations.")
    ap.add_argument("--pad-to-frames", type=int, default=360)
    ap.add_argument("--strict-length", action="store_true",
                    help="fail if a decoded segment is shorter than its target. "
                         "Default is off for BABEL: a 360-frame PRISM canvas "
                         "currently decodes to 357 raw frames, so the backend "
                         "pads the last frame to preserve manifest boundaries.")
    ap.add_argument("--rewrite-captions", action="store_true", default=True)
    ap.add_argument("--no-rewrite-captions", dest="rewrite_captions", action="store_false")
    ap.add_argument("--ar-cond-frames", type=int, default=5)
    ap.add_argument("--blend", action="store_true")
    ap.add_argument("--use-rollout-trans", action="store_true", default=False)
    ap.add_argument("--absolute-trans", dest="use_rollout_trans", action="store_false")
    ap.add_argument(
        "--translation-decode-mode",
        choices=["rollout", "absolute", "xz_rollout_y_absolute"],
        default="xz_rollout_y_absolute",
        help=(
            "How to decode PRISM abs_rel translation. Overrides "
            "--use-rollout-trans/--absolute-trans when provided. Current "
            "default is xz rollout + absolute y."
        ),
    )
    ap.add_argument("--min-total", type=int, default=0)
    ap.add_argument("--max-total", type=int, default=0, help="0 means no cap")
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--shard-idx", type=int, default=0)
    ap.add_argument("--max-episodes", type=int, default=0)
    ap.add_argument(
        "--id-file",
        default=None,
        help="Optional newline-separated BABEL episode ids to keep before sharding.",
    )
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--orientation-retries",
        type=int,
        default=4,
        help="Retry a sample with new deterministic seeds when the first frame is physically inverted.",
    )
    ap.add_argument(
        "--orientation-min-root-up-y",
        type=float,
        default=-0.25,
        help="Reject non-final attempts whose first-frame root local up axis points below this Y component.",
    )
    ap.add_argument(
        "--orientation-min-head-root-y",
        type=float,
        default=-0.20,
        help="Reject ordinary upright actions when first-frame head is this far below pelvis.",
    )
    ap.add_argument(
        "--legacy-shard-rng",
        action="store_true",
        help="Use the old single RNG stream per shard instead of deterministic per-sample seeds.",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    _set_seed(args.seed + int(args.shard_idx))

    manifest = Path(args.manifest)
    rows = _read_manifest(manifest)
    rows = [r for r in rows if int(r["total_frames"]) >= int(args.min_total)]
    if args.max_total:
        rows = [r for r in rows if int(r["total_frames"]) <= int(args.max_total)]
    if args.max_episodes:
        rows = rows[: int(args.max_episodes)]
    if args.id_file:
        keep_ids = {
            line.strip()
            for line in Path(args.id_file).read_text().splitlines()
            if line.strip()
        }
        rows = [r for r in rows if str(r["id"]) in keep_ids]
        print(f"[setup] id filter: {len(keep_ids)} ids from {args.id_file}", flush=True)
    rows = rows[int(args.shard_idx) :: int(args.num_shards)]
    if not rows:
        raise RuntimeError("no episodes selected")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"[setup] PRISM official-BABEL episodes={len(rows)} "
        f"shard={args.shard_idx}/{args.num_shards} device={device}",
        flush=True,
    )
    bundle = load_prism_bundle(args.config, args.checkpoint, device)
    from hftrainer.pipelines.motion.prism_pipeline import PrismPipeline

    pipeline = PrismPipeline(bundle=bundle)
    pipeline.backend.set_kafs_alpha(mode=args.kafs_mode)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_dir = out_dir / "_meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    run_meta = {
        "protocol": "official_babel_transition_midpoint_30fps",
        "manifest": str(manifest),
        "output_dir": str(out_dir),
        "config": args.config,
        "checkpoint": args.checkpoint,
        "num_inference_steps": int(args.num_inference_steps),
        "guidance_scale": float(args.guidance_scale),
        "kafs_mode": args.kafs_mode,
        "length_policy": args.length_policy,
        "pad_to_frames": int(args.pad_to_frames),
        "rewrite_captions": bool(args.rewrite_captions),
        "ar_cond_frames": int(args.ar_cond_frames),
        "blend": bool(args.blend),
        "use_rollout_trans": bool(args.use_rollout_trans),
        "translation_decode_mode": (
            args.translation_decode_mode
            if args.translation_decode_mode is not None
            else "rollout" if args.use_rollout_trans else "absolute"
        ),
        "num_shards": int(args.num_shards),
        "shard_idx": int(args.shard_idx),
        "id_file": str(args.id_file) if args.id_file else None,
        "selected_episodes": len(rows),
        "seed": int(args.seed),
    }
    (meta_dir / f"run_meta_shard{args.shard_idx}of{args.num_shards}.json").write_text(
        json.dumps(run_meta, indent=2, ensure_ascii=False) + "\n"
    )

    t0 = time.time()
    ok = skip = fail = 0
    for i, rec in enumerate(rows, start=1):
        sid = str(rec["id"])
        out_path = out_dir / f"{sid}.npz"
        if args.skip_existing and out_path.exists():
            skip += 1
            continue
        prompts = []
        seg_lens = []
        for seg in rec.get("segments", []):
            cap = str(seg.get("caption") or "").strip()
            prompts.append(rewrite_caption(cap) if args.rewrite_captions else cap)
            seg_lens.append(max(1, int(seg["end"]) - int(seg["start"])))
        gen_prompts = prompts
        gen_lens = seg_lens
        gen_source_segments = list(range(len(seg_lens)))
        if args.length_policy == "pad360_crop":
            gen_prompts, gen_lens, gen_source_segments = _expand_generation_segments(
                prompts,
                seg_lens,
                args.pad_to_frames,
            )
        try:
            max_attempts = max(1, int(args.orientation_retries) + 1)
            attempt_diags: list[dict[str, Any]] = []
            smplx_dict = None
            for attempt in range(max_attempts):
                if not args.legacy_shard_rng:
                    sample_seed = _sample_seed(args.seed, sid, args.ar_cond_frames, attempt)
                    _set_seed(sample_seed)
                else:
                    sample_seed = None
                smplx_try = pipeline(
                    prompts=gen_prompts,
                    num_frames_per_segment=gen_lens,
                    num_inference_steps=args.num_inference_steps,
                    guidance_scale=args.guidance_scale,
                    ar_condition_frames=args.ar_cond_frames,
                    use_blend=args.blend,
                    use_rollout_trans=(
                        True if run_meta["translation_decode_mode"] == "rollout"
                        else False if run_meta["translation_decode_mode"] == "absolute"
                        else run_meta["translation_decode_mode"]
                    ),
                    length_policy=args.length_policy,
                    pad_to_frames=args.pad_to_frames,
                    strict_length=args.strict_length,
                )
                actual_frames = int(np.asarray(smplx_try["transl"]).shape[0])
                expected_frames = int(rec["total_frames"])
                if actual_frames != expected_frames:
                    raise ValueError(
                        f"PRISM length mismatch for {sid}: "
                        f"actual={actual_frames}, expected={expected_frames}"
                    )
                orient = _first_frame_orientation(smplx_try)
                bad = _orientation_is_bad(
                    orient,
                    gen_prompts[0] if gen_prompts else "",
                    args.orientation_min_root_up_y,
                    args.orientation_min_head_root_y,
                )
                attempt_diags.append(
                    {
                        "attempt": int(attempt),
                        "seed": sample_seed,
                        "orientation": orient,
                        "bad": bool(bad),
                    }
                )
                smplx_dict = smplx_try
                if not bad or attempt == max_attempts - 1:
                    if bad:
                        print(
                            f"[warn-orient] {sid}: accepted final bad orientation "
                            f"after {attempt + 1} attempts: {orient}",
                            flush=True,
                        )
                    break
                print(
                    f"[retry-orient] {sid}: attempt={attempt} seed={sample_seed} "
                    f"orientation={orient}",
                    flush=True,
                )
            assert smplx_dict is not None
            save_smplx_npz(str(out_path), smplx_dict)
            meta = {
                "captions": prompts,
                "segment_lengths": seg_lens,
                "generation_captions": gen_prompts,
                "generation_segment_lengths": gen_lens,
                "generation_source_segments": gen_source_segments,
                "total_frames": int(rec["total_frames"]),
                "seed_mode": (
                    "legacy_shard_rng"
                    if args.legacy_shard_rng
                    else "per_sample_blake2b_id_arcond_attempt"
                ),
                "orientation_retries": int(args.orientation_retries),
                "orientation_attempts": attempt_diags,
                "length_policy": args.length_policy,
                "pad_to_frames": int(args.pad_to_frames),
                "strict_length": bool(args.strict_length),
                "translation_decode_mode": run_meta["translation_decode_mode"],
                "raw_decoded_segment_lengths": _np_int_list(
                    smplx_dict.get("_prism_raw_decoded_num_frames")
                ),
                "pretrim_segment_lengths": _np_int_list(
                    smplx_dict.get("_prism_pretrim_num_frames")
                ),
                "final_segment_lengths": _np_int_list(
                    smplx_dict.get("_prism_final_num_frames")
                ),
                "generation_num_frames": _np_int_list(
                    smplx_dict.get("_prism_generation_num_frames")
                ),
                "valid_num_frames": _np_int_list(
                    smplx_dict.get("_prism_valid_num_frames")
                ),
            }
            (meta_dir / f"{sid}.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n")
            ok += 1
        except Exception as exc:  # noqa: BLE001
            fail += 1
            print(f"[fail] {sid}: {type(exc).__name__}: {exc}", flush=True)
        if i % 5 == 0 or i == len(rows):
            elapsed = time.time() - t0
            print(
                f"[prism-official] shard={args.shard_idx}/{args.num_shards} "
                f"{i}/{len(rows)} ok={ok} skip={skip} fail={fail} "
                f"elapsed={elapsed:.0f}s",
                flush=True,
            )
    print(f"[done] ok={ok} skip={skip} fail={fail} out={out_dir}", flush=True)
    if fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
