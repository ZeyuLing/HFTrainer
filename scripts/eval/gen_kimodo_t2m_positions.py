#!/usr/bin/env python3
"""Batch KIMODO text-to-motion generation on HumanML3D test captions.

Outputs one ``<id>.npy`` file per HumanML3D test motion, containing SMPL-22
joint positions at 30 fps.  The downstream T2M table pipeline can then use
``scripts/eval/joints_to_272_npz.py --input-kind joints`` followed by the
MotionStreamer-272 evaluator.

The generator loads KIMODO through the hftrainer artifact wrapper by default,
using the in-repo ``hftrainer.models.motion.kimodo.network`` runtime rather
than an external upstream checkout.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch


REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

DEFAULT_KIMODO_ARTIFACT = REPO / "checkpoints/kimodo/hftrainer_soma_rp"
DEFAULT_KIMODO_MODEL_NAME = "Kimodo-SOMA-RP-v1"
KIMODO_SAFE_LEN = 240
DIFFUSION_STEPS = 100

# SOMA-77 indices corresponding to the SMPL-X/SMPL body 22-joint order.
SOMA77_TO_SMPL22 = np.asarray([
    0, 67, 72, 1, 68, 73, 2, 69, 74, 3, 70, 75,
    4, 11, 39, 6, 12, 40, 13, 41, 14, 42,
], dtype=np.int64)


def _read_first_caption(txt: Path) -> Optional[str]:
    if not txt.exists():
        return None
    for line in txt.read_text().splitlines():
        parts = line.strip().split("#")
        if len(parts) < 4:
            continue
        cap = parts[0].strip()
        try:
            f_tag = float(parts[2]) if parts[2] != "nan" else 0.0
            t_tag = float(parts[3]) if parts[3] != "nan" else 0.0
        except ValueError:
            f_tag = t_tag = 0.0
        if cap and f_tag == 0.0 and t_tag == 0.0:
            return cap
    return None


def _load_humanml3d_jobs(
    root: Path,
    min_len: int,
    max_len_exclusive: int,
    max_samples: int,
) -> List[Tuple[str, str, int]]:
    ids = [s.strip() for s in (root / "split" / "test.txt").read_text().splitlines() if s.strip()]
    jobs: List[Tuple[str, str, int]] = []
    for sid in ids:
        mfile = root / "motion_data" / f"{sid}.npy"
        if not mfile.exists():
            continue
        length = int(np.load(str(mfile), mmap_mode="r").shape[0])
        if length < min_len or length >= max_len_exclusive:
            continue
        cap = _read_first_caption(root / "texts" / f"{sid}.txt")
        if not cap:
            continue
        jobs.append((sid, cap, length))
        if max_samples and len(jobs) >= max_samples:
            break
    return jobs


def _load_corpus_jobs(
    corpus: Path,
    min_len: int,
    max_len_exclusive: int,
    max_samples: int,
) -> List[Tuple[str, str, int]]:
    jobs: List[Tuple[str, str, int]] = []
    for raw in corpus.read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if not raw:
            continue
        item = json.loads(raw)
        sid = str(item.get("id") or item.get("sample_id") or "").strip()
        caption = str(item.get("prompt") or item.get("caption") or "").strip()
        length = int(item.get("length") or item.get("target_length") or 0)
        if not sid or not caption:
            continue
        if length < min_len or length >= max_len_exclusive:
            continue
        jobs.append((sid, caption, length))
        if max_samples and len(jobs) >= max_samples:
            break
    return jobs


def _write_corpus(path: Path, jobs: List[Tuple[str, str, int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for sid, caption, length in jobs:
            f.write(json.dumps({
                "id": sid,
                "split": "test",
                "prompt": caption,
                "length": int(length),
            }, ensure_ascii=False) + "\n")


def _split_num_frames(num_frames: int, safe_len: int | None = None) -> list[int]:
    """Split long KIMODO requests into repeated-caption chunks."""
    safe = int(safe_len or KIMODO_SAFE_LEN)
    if num_frames <= safe:
        return [int(num_frames)]
    chunks: list[int] = []
    remaining = int(num_frames)
    while remaining > 0:
        n = min(safe, remaining)
        chunks.append(n)
        remaining -= n
    return chunks


def _positions22_from_posed(posed: np.ndarray) -> np.ndarray:
    posed = np.asarray(posed, dtype=np.float32)
    if posed.ndim != 3 or posed.shape[-1] != 3:
        raise ValueError(f"expected posed_joints as (T,J,3), got {posed.shape}")
    if posed.shape[1] == 22:
        return posed
    if posed.shape[1] == 77:
        return posed[:, SOMA77_TO_SMPL22]
    raise ValueError(
        "KIMODO output skeleton cannot be reduced to SMPL-22 positions: "
        f"posed_joints shape={posed.shape}"
    )


def _load_kimodo_model(args, *, use_feature_cache: bool):
    from hftrainer.models.motion.kimodo import KIMODOBundle

    kwargs = {"device": args.device, "diffusion_steps": args.diffusion_steps}
    if use_feature_cache:
        kwargs.update({"text_encoder": "dummy", "text_encoder_mode": "local"})

    model_path = Path(args.model_path) if args.model_path else None
    if model_path is not None and model_path.exists():
        bundle = KIMODOBundle.from_pretrained(str(model_path), **kwargs)
    else:
        bundle = KIMODOBundle(
            model_name=args.model_name or DEFAULT_KIMODO_MODEL_NAME,
            **kwargs,
        )
    model = bundle.model

    import hftrainer.models.motion.kimodo.network as kimodo_network

    kimodo_file = str(Path(kimodo_network.__file__).resolve())
    return model, bundle, kimodo_file


def _select_shard(items: List[Tuple[str, str, int]], num_shards: int, shard_index: int):
    if num_shards <= 1:
        return items
    return [item for i, item in enumerate(items) if i % num_shards == shard_index]


class CacheOnlyTextEncoder:
    """Read KIMODO LLM2Vec features from the native disk-cache format."""

    def __init__(
        self,
        *,
        namespace: str,
        cache_dir: str | Path,
        encoder_id: str = "LLM2VecEncoder",
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.namespace = namespace
        self.cache_dir = Path(cache_dir)
        self.encoder_id = encoder_id
        self.device = torch.device(device)
        self.dtype = dtype

        meta_path = self.cache_dir / self.namespace / "meta.json"
        self.llm_dim = 4096
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                self.llm_dim = int(meta.get("llm_dim") or self.llm_dim)
            except Exception:
                pass

    def to(self, device=None, dtype=None):
        if device is not None:
            self.device = torch.device(device)
        if dtype is not None:
            self.dtype = dtype
        return self

    def _key(self, sanitized_text: str) -> str:
        src = f"{self.namespace}|{self.encoder_id}|{sanitized_text}"
        return hashlib.sha256(src.encode("utf-8")).hexdigest()

    def __call__(self, texts):
        from hftrainer.models.motion.kimodo.network.sanitize import sanitize_texts

        if isinstance(texts, str):
            texts = [texts]
        clean = sanitize_texts(list(texts))
        arrays = []
        misses = []
        for text in clean:
            key = self._key(text)
            path = self.cache_dir / self.namespace / f"{key}.npy"
            if not path.exists():
                misses.append(text)
                continue
            arr = np.load(str(path))
            if arr.ndim != 2 or arr.shape[-1] != self.llm_dim:
                raise ValueError(f"bad cached text feature shape for {path}: {arr.shape}")
            arrays.append(np.asarray(arr, dtype=np.float32))
        if misses:
            preview = "; ".join(misses[:3])
            raise FileNotFoundError(
                f"{len(misses)} KIMODO text feature(s) missing in "
                f"{self.cache_dir / self.namespace}; first miss: {preview}"
            )
        if not arrays:
            empty = torch.empty((0, 0, self.llm_dim), device=self.device, dtype=self.dtype)
            return empty, []
        lengths = [int(arr.shape[0]) for arr in arrays]
        max_len = max(lengths)
        padded = np.zeros((len(arrays), max_len, self.llm_dim), dtype=np.float32)
        for idx, arr in enumerate(arrays):
            padded[idx, : arr.shape[0]] = arr
        feats = torch.from_numpy(padded).to(device=self.device, dtype=self.dtype)
        return feats, lengths


def _resample_positions(pos: np.ndarray, n_out: int) -> np.ndarray:
    pos = np.asarray(pos, dtype=np.float32)
    if len(pos) == n_out or len(pos) < 2:
        return pos[:n_out]
    src = np.linspace(0.0, 1.0, len(pos))
    dst = np.linspace(0.0, 1.0, n_out)
    flat = pos.reshape(len(pos), -1)
    out = np.empty((n_out, flat.shape[1]), dtype=np.float32)
    for c in range(flat.shape[1]):
        out[:, c] = np.interp(dst, src, flat[:, c])
    return out.reshape(n_out, pos.shape[1], pos.shape[2])


def _to_numpy_sequence(value) -> np.ndarray | None:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    arr = np.asarray(value)
    if arr.ndim >= 3 and arr.shape[0] == 1:
        arr = arr[0]
    return arr.astype(np.float32)


def _resample_nearest(arr: np.ndarray, n_out: int) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    if len(arr) == n_out or len(arr) < 2:
        return arr[:n_out]
    idx = np.rint(np.linspace(0, len(arr) - 1, n_out)).astype(np.int64)
    return arr[idx]


def _fit_length(arr: np.ndarray, n_out: int) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    if len(arr) > n_out:
        return arr[:n_out]
    if len(arr) < n_out and len(arr) > 0:
        pad = np.repeat(arr[-1:], n_out - len(arr), axis=0)
        return np.concatenate([arr, pad], axis=0)
    return arr


def _run_one(
    model,
    caption: str,
    num_frames_30: int,
    target_fps: float,
    postprocess: bool,
    *,
    force_single_segment: bool = False,
    max_segment_frames: int | None = None,
):
    model_fps = float(model.fps)
    model_frames = max(10, int(round(num_frames_30 * model_fps / target_fps)))
    seg_lens = (
        [model_frames]
        if force_single_segment
        else _split_num_frames(model_frames, safe_len=max_segment_frames)
    )
    is_multi = len(seg_lens) > 1
    prompts = [caption] * len(seg_lens)
    output = model(
        prompts,
        seg_lens,
        num_denoising_steps=DIFFUSION_STEPS,
        cfg_weight=[2.0, 2.0],
        num_samples=1,
        return_numpy=True,
        multi_prompt=is_multi,
        post_processing=postprocess,
    )
    posed = _to_numpy_sequence(output["posed_joints"])
    global_rot_mats = _to_numpy_sequence(output.get("global_rot_mats"))
    local_rot_mats = _to_numpy_sequence(output.get("local_rot_mats"))
    root_positions = _to_numpy_sequence(output.get("root_positions"))
    if posed is None:
        raise KeyError("KIMODO output has no posed_joints")
    pos22 = _positions22_from_posed(posed)
    if abs(model_fps - target_fps) > 1e-6:
        pos22 = _resample_positions(pos22, num_frames_30)
        posed = _resample_positions(posed, num_frames_30)
        if global_rot_mats is not None:
            global_rot_mats = _resample_nearest(global_rot_mats, num_frames_30)
        if local_rot_mats is not None:
            local_rot_mats = _resample_nearest(local_rot_mats, num_frames_30)
        if root_positions is not None:
            root_positions = _resample_positions(root_positions[:, None, :], num_frames_30)[:, 0]
    if len(pos22) > num_frames_30:
        pos22 = pos22[:num_frames_30]
    elif len(pos22) < num_frames_30 and len(pos22) > 0:
        pad = np.repeat(pos22[-1:], num_frames_30 - len(pos22), axis=0)
        pos22 = np.concatenate([pos22, pad], axis=0)
    payload = {
        "positions": pos22.astype(np.float32),
        "posed_joints": _fit_length(posed, num_frames_30).astype(np.float32),
    }
    if global_rot_mats is not None:
        payload["global_rot_mats"] = _fit_length(global_rot_mats, num_frames_30).astype(np.float32)
    if local_rot_mats is not None:
        payload["local_rot_mats"] = _fit_length(local_rot_mats, num_frames_30).astype(np.float32)
    if root_positions is not None:
        payload["root_positions"] = _fit_length(root_positions, num_frames_30).astype(np.float32)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--humanml3d-272", default="ref_repo/MotionStreamer/MotionStreamer/humanml3d_272")
    parser.add_argument("--model-path", default=str(DEFAULT_KIMODO_ARTIFACT))
    parser.add_argument("--model-name", default=DEFAULT_KIMODO_MODEL_NAME)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--debug-npz-dir", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--min-len", type=int, default=60)
    parser.add_argument("--max-len", type=int, default=300)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--diffusion-steps", type=int, default=100)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--postprocess", action="store_true")
    parser.add_argument("--corpus", default=None, help="Optional JSONL prompt bank with id/prompt/length.")
    parser.add_argument("--write-corpus", default=None, help="Write the resolved full job list for provenance.")
    parser.add_argument(
        "--force-single-segment",
        action="store_true",
        help="Disable the long-motion repeated-caption split and use KIMODO's single-prompt path.",
    )
    parser.add_argument(
        "--max-segment-frames",
        type=int,
        default=None,
        help="Override KIMODO_SAFE_LEN for the repeated-caption split; ignored with --force-single-segment.",
    )
    parser.add_argument("--text-feature-cache-dir", default=None)
    parser.add_argument("--text-feature-namespace", default=None)
    parser.add_argument("--text-feature-encoder-id", default="LLM2VecEncoder")
    args = parser.parse_args()

    if not (0 <= args.shard_index < args.num_shards):
        raise ValueError("--shard-index must be in [0, num_shards)")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    debug = Path(args.debug_npz_dir) if args.debug_npz_dir else None
    if debug:
        debug.mkdir(parents=True, exist_ok=True)

    if args.corpus:
        all_jobs = _load_corpus_jobs(
            Path(args.corpus),
            min_len=args.min_len,
            max_len_exclusive=args.max_len,
            max_samples=args.max_samples,
        )
    else:
        all_jobs = _load_humanml3d_jobs(
            Path(args.humanml3d_272),
            min_len=args.min_len,
            max_len_exclusive=args.max_len,
            max_samples=args.max_samples,
        )
    if args.write_corpus:
        _write_corpus(Path(args.write_corpus), all_jobs)
    jobs = _select_shard(all_jobs, args.num_shards, args.shard_index)
    print(f"[setup] total={len(all_jobs)} shard={args.shard_index}/{args.num_shards} jobs={len(jobs)}", flush=True)

    use_feature_cache = bool(args.text_feature_cache_dir and args.text_feature_namespace)
    if use_feature_cache:
        os.environ["TEXT_ENCODER"] = "dummy"
        os.environ["TEXT_ENCODER_MODE"] = "local"

    global DIFFUSION_STEPS
    DIFFUSION_STEPS = int(args.diffusion_steps)
    model, bundle, kimodo_file = _load_kimodo_model(args, use_feature_cache=use_feature_cache)
    if use_feature_cache:
        model.text_encoder = CacheOnlyTextEncoder(
            namespace=args.text_feature_namespace,
            cache_dir=args.text_feature_cache_dir,
            encoder_id=args.text_feature_encoder_id,
            device=args.device,
        )
        print(
            "[setup] using cached text features "
            f"{Path(args.text_feature_cache_dir) / args.text_feature_namespace}",
            flush=True,
        )
    skeleton_type = type(model.skeleton).__name__
    print(
        f"[setup] KIMODO loaded fps={model.fps} skeleton={skeleton_type} "
        f"source={args.model_path or args.model_name}",
        flush=True,
    )
    print(f"[setup] kimodo_import={kimodo_file}", flush=True)

    manifest = out / f"manifest_shard{args.shard_index}of{args.num_shards}.jsonl"
    ok = skipped = failed = 0
    with manifest.open("w") as mf:
        for i, (sid, caption, length) in enumerate(jobs):
            out_file = out / f"{sid}.npy"
            if args.skip_existing and out_file.exists():
                skipped += 1
                continue
            try:
                payload = _run_one(
                    model,
                    caption,
                    length,
                    args.fps,
                    args.postprocess,
                    force_single_segment=args.force_single_segment,
                    max_segment_frames=args.max_segment_frames,
                )
                pos22 = payload["positions"]
                if not np.isfinite(pos22).all() or pos22.shape != (length, 22, 3):
                    raise ValueError(f"bad position shape/range: {pos22.shape}")
                np.save(str(out_file), pos22)
                if debug is not None:
                    np.savez_compressed(
                        str(debug / f"{sid}.npz"),
                        **payload,
                        caption=np.array(caption, dtype=object),
                        sample_id=np.array(sid, dtype=object),
                        target_length=np.array(length, dtype=np.int32),
                    )
                mf.write(json.dumps({
                    "sample_id": sid,
                    "caption": caption,
                    "target_length": length,
                    "path": str(out_file),
                    "text_feature_namespace": args.text_feature_namespace,
                    "text_feature_encoder_id": args.text_feature_encoder_id,
                    "force_single_segment": bool(args.force_single_segment),
                    "model_path": args.model_path,
                    "model_name": getattr(bundle, "resolved_model_name", args.model_name),
                    "skeleton_type": skeleton_type,
                    "kimodo_import": kimodo_file,
                }, ensure_ascii=False) + "\n")
                mf.flush()
                ok += 1
            except Exception as exc:  # noqa: BLE001
                failed += 1
                print(f"[fail] {sid}: {type(exc).__name__}: {exc}", flush=True)
            if (i + 1) % 25 == 0 or (i + 1) == len(jobs):
                print(f"[progress] {i+1}/{len(jobs)} ok={ok} skipped={skipped} failed={failed}", flush=True)

    summary = {
        "all_jobs": len(all_jobs),
        "jobs": len(jobs),
        "ok": ok,
        "skipped": skipped,
        "failed": failed,
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "model_path": args.model_path,
        "model_name": getattr(bundle, "resolved_model_name", args.model_name),
        "skeleton_type": skeleton_type,
        "kimodo_import": kimodo_file,
    }
    (out / f"summary_shard{args.shard_index}of{args.num_shards}.json").write_text(json.dumps(summary, indent=2))
    print("[done] " + json.dumps(summary), flush=True)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
