#!/usr/bin/env python3
"""Batch KIMODO text-to-motion generation on HumanML3D test captions.

Outputs one ``<id>.npy`` file per HumanML3D test motion, containing SMPL-22
joint positions at 30 fps.  The downstream T2M table pipeline can then use
``scripts/eval/joints_to_272_npz.py --input-kind joints`` followed by the
MotionStreamer-272 evaluator.
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
KIMODO_ROOT = REPO / "ref_repo" / "KIMODO" / "kimodo"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(KIMODO_ROOT))


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
        from kimodo.sanitize import sanitize_texts

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


def _run_one(model, caption: str, num_frames_30: int, target_fps: float, postprocess: bool):
    from scripts.kimodo.run_kimodo_all_tasks import (
        DIFFUSION_STEPS,
        _split_num_frames,
        soma77_to_smpl22,
    )

    model_fps = float(model.fps)
    model_frames = max(10, int(round(num_frames_30 * model_fps / target_fps)))
    seg_lens = _split_num_frames(model_frames)
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
    posed = output["posed_joints"]
    if isinstance(posed, torch.Tensor):
        posed = posed.detach().cpu().numpy()
    if posed.ndim == 4:
        posed = posed[0]
    pos22 = soma77_to_smpl22(posed)
    if isinstance(pos22, torch.Tensor):
        pos22 = pos22.detach().cpu().numpy()
    pos22 = np.asarray(pos22, dtype=np.float32)
    if abs(model_fps - target_fps) > 1e-6:
        pos22 = _resample_positions(pos22, num_frames_30)
    if len(pos22) > num_frames_30:
        pos22 = pos22[:num_frames_30]
    elif len(pos22) < num_frames_30 and len(pos22) > 0:
        pad = np.repeat(pos22[-1:], num_frames_30 - len(pos22), axis=0)
        pos22 = np.concatenate([pos22, pad], axis=0)
    return pos22.astype(np.float32), posed.astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--humanml3d-272", default="ref_repo/MotionStreamer/MotionStreamer/humanml3d_272")
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
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--postprocess", action="store_true")
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

    all_jobs = _load_humanml3d_jobs(
        Path(args.humanml3d_272),
        min_len=args.min_len,
        max_len_exclusive=args.max_len,
        max_samples=args.max_samples,
    )
    jobs = _select_shard(all_jobs, args.num_shards, args.shard_index)
    print(f"[setup] total={len(all_jobs)} shard={args.shard_index}/{args.num_shards} jobs={len(jobs)}", flush=True)

    use_feature_cache = bool(args.text_feature_cache_dir and args.text_feature_namespace)
    if use_feature_cache:
        os.environ["TEXT_ENCODER"] = "dummy"
        os.environ["TEXT_ENCODER_MODE"] = "local"

    from kimodo import load_model

    model = load_model("kimodo-soma-rp", device=args.device)
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
    print(f"[setup] KIMODO loaded fps={model.fps}", flush=True)

    manifest = out / f"manifest_shard{args.shard_index}of{args.num_shards}.jsonl"
    ok = skipped = failed = 0
    with manifest.open("w") as mf:
        for i, (sid, caption, length) in enumerate(jobs):
            out_file = out / f"{sid}.npy"
            if args.skip_existing and out_file.exists():
                skipped += 1
                continue
            try:
                pos22, posed77 = _run_one(model, caption, length, args.fps, args.postprocess)
                if not np.isfinite(pos22).all() or pos22.shape != (length, 22, 3):
                    raise ValueError(f"bad position shape/range: {pos22.shape}")
                np.save(str(out_file), pos22)
                if debug is not None:
                    np.savez_compressed(
                        str(debug / f"{sid}.npz"),
                        positions=pos22,
                        posed_joints=posed77,
                        caption=np.array(caption, dtype=object),
                        sample_id=np.array(sid, dtype=object),
                        target_length=np.array(length, dtype=np.int32),
                    )
                mf.write(json.dumps({
                    "sample_id": sid,
                    "caption": caption,
                    "target_length": length,
                    "path": str(out_file),
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
    }
    (out / f"summary_shard{args.shard_index}of{args.num_shards}.json").write_text(json.dumps(summary, indent=2))
    print("[done] " + json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
