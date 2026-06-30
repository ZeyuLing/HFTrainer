#!/usr/bin/env python3
"""MotionStreamer latent-prefix generation for Table 2 TP2M evaluation."""
from __future__ import annotations

import argparse
import json
import random
import traceback
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch


def _install_from_numpy_fallback() -> None:
    """Work around Taiji hosts where torch's NumPy C-API binding is broken.

    A small subset of long-lived A100PRO containers reports
    ``TypeError: expected np.ndarray (got numpy.ndarray)`` for every
    ``torch.from_numpy`` call. MotionStreamer inference only converts small
    arrays in this wrapper/model path, so falling back to a copied Python-list
    tensor is much cheaper than leaving an otherwise idle 8-GPU host unusable.
    """
    original = torch.from_numpy

    def safe_from_numpy(array):  # noqa: ANN001
        try:
            return original(array)
        except (TypeError, RuntimeError):
            if not isinstance(array, np.ndarray):
                raise
            dtype_map = {
                np.dtype("float16"): torch.float16,
                np.dtype("float32"): torch.float32,
                np.dtype("float64"): torch.float64,
                np.dtype("int16"): torch.int16,
                np.dtype("int32"): torch.int32,
                np.dtype("int64"): torch.int64,
                np.dtype("uint8"): torch.uint8,
                np.dtype("bool"): torch.bool,
            }
            dtype = dtype_map.get(array.dtype)
            if dtype is None:
                return torch.tensor(array.tolist())
            return torch.tensor(array.tolist(), dtype=dtype)

    torch.from_numpy = safe_from_numpy


_install_from_numpy_fallback()

from gen_motionstreamer_smpl_npz import (
    _build_gt_path_map,
    _load_h3d_pairs,
    _load_model,
    _load_motionhub_pairs,
    _motion272_to_npz_fields,
    _safe_name,
    _select_shard,
)


def _load_json(path: Path):
    return json.loads(path.read_text())


def _iter_entries(raw):
    data = raw.get("data_list", raw) if isinstance(raw, dict) else raw
    if isinstance(data, dict):
        yield from data.items()
        return
    for i, entry in enumerate(data):
        yield str(entry.get("motion_id") or entry.get("id") or i), entry


def _build_gt272_map(anno_file: Optional[Path], gt_272_dir: Path) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    if anno_file is not None and anno_file.exists():
        for name, entry in _iter_entries(_load_json(anno_file)):
            stem = Path(str(entry.get("smplx_path") or "")).stem
            candidates = [gt_272_dir / f"{name}.npy"]
            if stem:
                candidates.append(gt_272_dir / f"{stem}.npy")
            found = next((p for p in candidates if p.exists()), None)
            if found is not None:
                out[str(name)] = found
                if stem:
                    out[stem] = found
    for path in gt_272_dir.glob("*.npy"):
        out.setdefault(path.stem, path)
    return out


def _load_only_ids(value: Optional[str]) -> Optional[set[str]]:
    if not value:
        return None
    path = Path(value)
    if path.exists():
        ids = [line.strip() for line in path.read_text().splitlines()]
    else:
        ids = [part.strip() for part in value.split(",")]
    return {sid for sid in ids if sid}


def _encode_prefix_latents(
    net,
    motion_272: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    condition_num_frames: int,
    device: torch.device,
    latent_source: str = "sample",
) -> tuple[torch.Tensor, int]:
    n = min(condition_num_frames, int(motion_272.shape[0]))
    if n <= 0:
        raise ValueError("empty prefix")
    prefix = np.asarray(motion_272[:n], dtype=np.float32)
    encoded_frames = n
    if encoded_frames < 4:
        pad = np.repeat(prefix[-1:], 4 - encoded_frames, axis=0)
        prefix = np.concatenate([prefix, pad], axis=0)
        encoded_frames = 4
    prefix_norm = (prefix - mean) / std
    x = torch.from_numpy(prefix_norm).float().unsqueeze(0).to(device)
    latents, mu, _ = net.encode(x)
    if latent_source == "mu":
        latents = mu.squeeze(0)
    elif latent_source != "sample":
        raise ValueError(f"unsupported latent_source={latent_source!r}")
    if latents.ndim == 2 and latents.shape[0] == 16 and latents.shape[1] != 16:
        latents = latents.transpose(0, 1).contiguous()
    if latents.ndim == 2:
        return latents, encoded_frames
    if latents.ndim == 3:
        latents = latents.squeeze(0)
        if latents.ndim == 2 and latents.shape[0] == 16 and latents.shape[1] != 16:
            latents = latents.transpose(0, 1).contiguous()
        return latents, encoded_frames
    raise ValueError(f"unexpected prefix latent shape: {tuple(latents.shape)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["humanml3d", "motionhub"], required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--gt-272-dir", required=True)
    parser.add_argument("--condition-num-frames", type=int, required=True)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--caption-protocol", choices=["rewritten", "original", "fallback"], default="original")
    parser.add_argument("--anno-file", default=None)
    parser.add_argument("--rewritten-file", default=None)
    parser.add_argument("--data-dir", default="data/motionhub")
    parser.add_argument("--humanml3d-272", default="ref_repo/MotionStreamer/MotionStreamer/humanml3d_272")
    parser.add_argument("--humanml3d-min-motion-length", type=int, default=0,
                        help="Optional minimum length for official HumanML3D-272 protocol checks.")
    parser.add_argument("--resume-pth", default="ref_repo/MotionStreamer/MotionStreamer/MotionStreamer_HF/Causal_TAE/net_last.pth")
    parser.add_argument("--resume-trans", default="ref_repo/MotionStreamer/MotionStreamer/MotionStreamer_HF/Experiments/t2m_model/latest.pth")
    parser.add_argument("--t5-model", default=None)
    parser.add_argument("--mean", default="ref_repo/MotionStreamer/MotionStreamer/humanml3d_272/mean_std/Mean.npy")
    parser.add_argument("--std", default="ref_repo/MotionStreamer/MotionStreamer/humanml3d_272/mean_std/Std.npy")
    parser.add_argument("--max-motion-length", type=int, default=300)
    parser.add_argument(
        "--only-ids",
        default=None,
        help="Optional comma-separated id list or newline text file for targeted TP2M reruns.",
    )
    parser.add_argument(
        "--flat-out-dir",
        action="store_true",
        help="Write files directly into --out-dir instead of --out-dir/condX_latent_prefix.",
    )
    parser.add_argument("--align-to-gt-root", action="store_true")
    parser.add_argument("--align-root-mode", choices=["yaw", "full"], default="yaw")
    parser.add_argument("--prefix-latent-source", choices=["sample", "mu"], default="sample",
                        help="Use the stochastic VAE sample or posterior mean for the observed prefix.")
    parser.add_argument("--sampling-method", choices=["new_demo", "new"], default="new_demo",
                        help="MotionStreamer BABEL inference sampler to use for latent-prefix continuation.")
    parser.add_argument("--cfg", type=float, default=4.5,
                        help="Classifier-free guidance scale for MotionStreamer sampling.")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Diffusion sampling temperature for MotionStreamer latent tokens.")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--hidden_size", default=1024, type=int)
    parser.add_argument("--down-t", type=int, default=2)
    parser.add_argument("--stride-t", type=int, default=2)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--dilation-growth-rate", type=int, default=3)
    parser.add_argument("--num_diffusion_head_layers", type=int, default=9)
    parser.add_argument("--latent_dim", type=int, default=16)
    parser.add_argument("--disable-out-proj", dest="use_out_proj", action="store_false")
    parser.set_defaults(use_out_proj=True)
    args = parser.parse_args()

    if args.condition_num_frames < 1:
        raise ValueError("--condition-num-frames must be >= 1")
    if not (0 <= args.shard_index < args.num_shards):
        raise ValueError("--shard-index must be in [0, num_shards)")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    anno_file = Path(args.anno_file) if args.anno_file else None
    rewritten_file = Path(args.rewritten_file) if args.rewritten_file else None
    if args.dataset == "humanml3d":
        min_length = args.condition_num_frames + 1
        if anno_file is None and args.humanml3d_min_motion_length > 0:
            min_length = max(min_length, int(args.humanml3d_min_motion_length))
        pairs = _load_h3d_pairs(
            humanml3d_272=Path(args.humanml3d_272),
            anno_file=anno_file,
            rewritten_file=rewritten_file,
            data_dir=Path(args.data_dir),
            caption_protocol=args.caption_protocol,
            min_length=min_length,
            # _load_h3d_pairs uses an exclusive upper bound. The official
            # HumanML3D selected split contains many 300-frame clips, so keep
            # MotionStreamer's 300-frame cap inclusive here.
            max_length_exclusive=args.max_motion_length + 1,
            limit=0,
        )
    else:
        if anno_file is None:
            raise ValueError("motionhub requires --anno-file")
        pairs = _load_motionhub_pairs(
            anno_file=anno_file,
            data_dir=Path(args.data_dir),
            rewritten_file=rewritten_file,
            caption_protocol=args.caption_protocol,
        )
        pairs = [p for p in pairs if p[2] > args.condition_num_frames]
    only_ids = _load_only_ids(args.only_ids)
    if only_ids is not None:
        pairs = [p for p in pairs if str(p[0]) in only_ids]
    pairs = _select_shard(pairs, args.num_shards, args.shard_index)
    if args.max_samples:
        pairs = pairs[: args.max_samples]
    print(f"[setup] device={device} dataset={args.dataset} cond={args.condition_num_frames} pairs={len(pairs)}", flush=True)

    out_dir = Path(args.out_dir) if args.flat_out_dir else Path(args.out_dir) / f"cond{args.condition_num_frames}_latent_prefix"
    out_dir.mkdir(parents=True, exist_ok=True)
    gt272_map = _build_gt272_map(anno_file, Path(args.gt_272_dir))
    gt_path_map = _build_gt_path_map(anno_file, Path(args.data_dir)) if args.align_to_gt_root else {}
    mean = np.load(args.mean).astype(np.float32)
    std = np.load(args.std).astype(np.float32)
    t5_model, net, trans_encoder = _load_model(args, device)

    ok = skipped = failed = 0
    manifest = []
    with torch.no_grad():
        for idx, (name, caption, target_len) in enumerate(pairs):
            out_path = out_dir / f"{_safe_name(name)}.npz"
            if args.skip_existing and out_path.exists():
                skipped += 1
                continue
            try:
                gt272_path = gt272_map.get(name)
                if gt272_path is None:
                    raise FileNotFoundError(f"missing GT 272 for {name}")
                gt272 = np.load(gt272_path).astype(np.float32)
                prefix_latents, encoded_frames = _encode_prefix_latents(
                    net,
                    gt272,
                    mean,
                    std,
                    args.condition_num_frames,
                    device,
                    args.prefix_latent_source,
                )
                prefix_tokens = int(prefix_latents.shape[0])
                eval_len = min(int(target_len), int(args.max_motion_length))
                sample_total_frames = max(((eval_len + 3) // 4) * 4, (prefix_tokens + 1) * 4)
                if args.sampling_method == "new_demo":
                    _xs, b_latents = trans_encoder.sample_for_eval_CFG_babel_inference_new_demo(
                        B_text=caption,
                        A_motion=prefix_latents,
                        length=sample_total_frames,
                        clip_model=t5_model,
                        device=device,
                        tokenizer="t5-xxl",
                        unit_length=4,
                        cfg=args.cfg,
                        temperature=args.temperature,
                    )
                else:
                    continuation_frames = max(eval_len - prefix_tokens * 4, 4)
                    _xs, b_latents = trans_encoder.sample_for_eval_CFG_babel_inference_new(
                        B_text=[caption],
                        A_motion=prefix_latents,
                        length=continuation_frames,
                        clip_model=t5_model,
                        device=device,
                        tokenizer="t5-xxl",
                        unit_length=4,
                        cfg=args.cfg,
                    )
                full_latents = torch.cat([prefix_latents.unsqueeze(0), b_latents], dim=1)
                motion_norm = net.forward_decoder(full_latents).squeeze(0).detach().cpu().numpy()
                motion_norm = motion_norm[:eval_len]
                motion_272 = (motion_norm * std + mean).astype(np.float32)
                gt_path = gt_path_map.get(name) if args.align_to_gt_root else None
                fields = _motion272_to_npz_fields(
                    motion_272,
                    gt_path=gt_path,
                    align_mode=args.align_root_mode,
                )
                np.savez_compressed(
                    out_path,
                    **fields,
                    text=caption,
                    sample_id=name,
                    target_length=int(target_len),
                    generated_length=int(motion_272.shape[0]),
                    condition_num_frames=int(args.condition_num_frames),
                    prefix_encoded_frames=int(encoded_frames),
                    prefix_tokens=int(prefix_tokens),
                    prefix_latent_source=args.prefix_latent_source,
                    motionstreamer_sampling_method=args.sampling_method,
                    motionstreamer_cfg=float(args.cfg),
                    motionstreamer_temperature=float(args.temperature),
                    gt272_path=str(gt272_path),
                    aligned_to_gt_root=bool(gt_path is not None),
                    align_root_mode=args.align_root_mode if gt_path is not None else "",
                    motionstreamer_use_out_proj=bool(args.use_out_proj),
                )
                ok += 1
                manifest.append({
                    "sample_id": name,
                    "path": str(out_path),
                    "status": "ok",
                    "motionstreamer_use_out_proj": bool(args.use_out_proj),
                    "prefix_latent_source": args.prefix_latent_source,
                    "motionstreamer_sampling_method": args.sampling_method,
                    "motionstreamer_cfg": float(args.cfg),
                    "motionstreamer_temperature": float(args.temperature),
                })
            except Exception as exc:  # noqa: BLE001
                failed += 1
                print(f"[fail] {name}: {type(exc).__name__}: {exc}", flush=True)
                if failed <= 3:
                    traceback.print_exc()
                manifest.append({"sample_id": name, "status": f"{type(exc).__name__}: {exc}"})
            if (idx + 1) % 25 == 0 or idx + 1 == len(pairs):
                print(f"[progress] {idx + 1}/{len(pairs)} ok={ok} skipped={skipped} failed={failed}", flush=True)
    (out_dir / f"manifest_shard{args.shard_index}of{args.num_shards}.json").write_text(json.dumps(manifest, indent=2))
    print(f"[done] ok={ok} skipped={skipped} failed={failed} out={out_dir}", flush=True)


if __name__ == "__main__":
    main()
