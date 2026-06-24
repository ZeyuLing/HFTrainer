#!/usr/bin/env python3
"""Batch MotionStreamer text-to-motion inference and dump SMPL-style NPZ files.

The upstream MotionStreamer model generates 272-dim motions. This script
converts each generated clip to:
  - motion_272: raw denormalized MotionStreamer representation
  - motion_135: transl + 22x row-major 6D local rotations
  - transl / global_orient / body_pose: SMPL-22 axis-angle fields readable by
    the local MotionCLIP evaluator loader.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch


REPO = Path(__file__).resolve().parents[2]
MS_ROOT = REPO / "ref_repo" / "MotionStreamer" / "MotionStreamer"

sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "eval"))
sys.path.insert(0, str(MS_ROOT))


def _load_json(path: Path):
    return json.loads(path.read_text())


def _load_rewritten(path: Optional[Path]) -> Dict[str, str]:
    if path is None:
        return {}
    raw = _load_json(path)
    if isinstance(raw, dict) and "data_list" in raw:
        raw = raw["data_list"]
    if not isinstance(raw, dict):
        raise ValueError(f"rewritten caption file must be a dict: {path}")
    out: Dict[str, str] = {}
    for key, value in raw.items():
        if isinstance(value, str):
            cap = value
        elif isinstance(value, dict):
            cap = value.get("caption") or value.get("text") or value.get("short_caption")
        else:
            cap = None
        if isinstance(cap, str) and cap.strip():
            out[str(key)] = cap.strip()
    return out


def _iter_motionhub_entries(raw) -> Iterable[Tuple[str, Dict]]:
    if isinstance(raw, dict) and "data_list" in raw:
        data_list = raw["data_list"]
        if isinstance(data_list, dict):
            for name, entry in data_list.items():
                yield str(name), entry
        else:
            for i, entry in enumerate(data_list):
                yield str(entry.get("motion_id") or entry.get("id") or i), entry
    elif isinstance(raw, list):
        for i, entry in enumerate(raw):
            yield str(entry.get("motion_id") or entry.get("id") or i), entry
    else:
        raise ValueError("Unrecognized annotation format")


def _load_caption_from_json(path: Path) -> Optional[str]:
    try:
        data = _load_json(path)
    except Exception:
        return None
    pool: List[str] = []
    if isinstance(data, dict) and all(isinstance(data.get(k), list) for k in ("macro", "meso", "micro")):
        for group in ("macro", "meso", "micro"):
            for item in data[group]:
                if isinstance(item, str) and item.strip():
                    pool.append(item.strip())
    elif isinstance(data, dict) and isinstance(data.get("result"), list):
        for item in data["result"]:
            if not isinstance(item, dict):
                continue
            for key in ("short_caption_rewritten", "short caption_rewritten"):
                vals = item.get(key)
                if isinstance(vals, list):
                    pool.extend(v.strip() for v in vals if isinstance(v, str) and v.strip())
                    break
            else:
                for key in ("short_caption", "short caption"):
                    val = item.get(key)
                    if isinstance(val, str) and val.strip():
                        pool.append(val.strip())
                        break
    return pool[0] if pool else None


def _load_h3d_pairs(
    humanml3d_272: Path,
    anno_file: Optional[Path],
    rewritten_file: Optional[Path],
    data_dir: Path,
    caption_protocol: str,
    min_length: Optional[int] = None,
    max_length_exclusive: Optional[int] = None,
    limit: int = 0,
) -> List[Tuple[str, str, int]]:
    def accept_length(length: int) -> bool:
        if min_length is not None and length < min_length:
            return False
        if max_length_exclusive is not None and length >= max_length_exclusive:
            return False
        return True

    rewritten = _load_rewritten(rewritten_file)
    if anno_file is not None and anno_file.exists():
        raw = _load_json(anno_file)
        pairs: List[Tuple[str, str, int]] = []
        for name, entry in _iter_motionhub_entries(raw):
            caption = None
            if caption_protocol == "rewritten":
                caption = rewritten.get(name)
            if not caption and caption_protocol in {"original", "fallback"} and entry.get("hierarchical_caption_path"):
                caption = _load_caption_from_json(data_dir / entry["hierarchical_caption_path"])
            if not caption and caption_protocol in {"rewritten", "fallback"}:
                caption = rewritten.get(name)
            if not caption:
                continue
            length = int(entry.get("num_frames") or round(float(entry.get("duration", 0.0)) * float(entry.get("fps", 30.0))))
            if length <= 0 or not accept_length(length):
                continue
            pairs.append((name, caption.strip(), length))
            if limit and len(pairs) >= limit:
                break
        return pairs

    motion_dir = humanml3d_272 / "motion_data"
    ids = [ln.strip() for ln in (humanml3d_272 / "split" / "test.txt").read_text().splitlines() if ln.strip()]

    id_to_name: Dict[str, str] = {}
    name_to_entry: Dict[str, Dict] = {}
    if anno_file is not None and anno_file.exists():
        raw = _load_json(anno_file)
        for name, entry in _iter_motionhub_entries(raw):
            name_to_entry[name] = entry
            smplx_path = str(entry.get("smplx_path") or "")
            stem = Path(smplx_path).stem
            if stem:
                id_to_name[stem] = name

    pairs: List[Tuple[str, str, int]] = []
    for cid in ids:
        mfile = motion_dir / f"{cid}.npy"
        if not mfile.exists():
            continue
        canonical_name = id_to_name.get(cid, cid)
        caption = None
        if caption_protocol == "rewritten":
            caption = rewritten.get(canonical_name) or rewritten.get(cid)
        if not caption and caption_protocol in {"original", "fallback"}:
            text_file = humanml3d_272 / "texts" / f"{cid}.txt"
            if text_file.exists():
                for line in text_file.read_text().splitlines():
                    parts = line.strip().split("#")
                    if len(parts) >= 4:
                        try:
                            f_tag = float(parts[2]) if parts[2] != "nan" else 0.0
                            t_tag = float(parts[3]) if parts[3] != "nan" else 0.0
                        except ValueError:
                            f_tag = t_tag = 0.0
                        if f_tag == 0.0 and t_tag == 0.0 and parts[0].strip():
                            caption = parts[0].strip()
                            break
        if not caption and caption_protocol in {"rewritten", "fallback"}:
            entry = name_to_entry.get(canonical_name)
            if entry is not None and entry.get("hierarchical_caption_path"):
                caption = _load_caption_from_json(data_dir / entry["hierarchical_caption_path"])
        if not caption:
            continue
        entry = name_to_entry.get(canonical_name)
        length = int(entry.get("num_frames", 0)) if entry is not None else 0
        if length <= 0:
            length = int(np.load(mfile, mmap_mode="r").shape[0])
        if not accept_length(length):
            continue
        pairs.append((canonical_name, caption, length))
        if limit and len(pairs) >= limit:
            break
    return pairs


def _load_motionhub_pairs(
    anno_file: Path,
    data_dir: Path,
    rewritten_file: Optional[Path],
    caption_protocol: str,
) -> List[Tuple[str, str, int]]:
    rewritten = _load_rewritten(rewritten_file)
    raw = _load_json(anno_file)
    pairs: List[Tuple[str, str, int]] = []
    for name, entry in _iter_motionhub_entries(raw):
        caption = None
        if caption_protocol == "rewritten":
            caption = rewritten.get(name)
        if not caption and caption_protocol in {"original", "fallback"} and entry.get("hierarchical_caption_path"):
            caption = _load_caption_from_json(data_dir / entry["hierarchical_caption_path"])
        if not caption and caption_protocol in {"rewritten", "fallback"}:
            caption = rewritten.get(name)
        if not caption:
            continue
        length = int(entry.get("num_frames") or round(float(entry.get("duration", 0.0)) * float(entry.get("fps", 30.0))))
        if length <= 0:
            continue
        pairs.append((name, caption.strip(), length))
    return pairs


def _select_shard(items: List[Tuple[str, str, int]], num_shards: int, shard_index: int):
    if num_shards <= 1:
        return items
    return [item for i, item in enumerate(items) if i % num_shards == shard_index]


def _resolve_t5_model(ms_root: Path, user_path: Optional[str]) -> str:
    candidates = []
    if user_path:
        candidates.append(Path(user_path))
    candidates.extend([
        ms_root / "sentencet5-xxl",
        REPO / "sentencet5-xxl",
        Path.home() / ".cache" / "huggingface" / "hub" / "models--sentence-transformers--sentence-t5-xxl" / "snapshots",
    ])
    for cand in candidates:
        if cand.name == "snapshots" and cand.is_dir():
            snaps = sorted([p for p in cand.iterdir() if p.is_dir()])
            if snaps:
                return str(snaps[-1])
        elif cand.exists():
            return str(cand)
    return "sentence-transformers/sentence-t5-xxl"


def _load_model(args, device):
    from sentence_transformers import SentenceTransformer
    from models.llama_model import LLaMAHF, LLaMAHFConfig
    import models.tae as tae

    t5_path = _resolve_t5_model(MS_ROOT, args.t5_model)
    print(f"[load] SentenceTransformer: {t5_path}", flush=True)
    if os.environ.get("T5_FP16_GPU") == "1":
        # Load fp32 on CPU, cast to fp16, then move to GPU so only ~9.4GB lands on
        # the device (fits a 15GB T4; fp32 GPU load would OOM). Embeddings are
        # L2-normalised so the fp16 numerical drift is negligible for conditioning.
        t5_model = SentenceTransformer(t5_path, device="cpu").half()
        t5_model = t5_model.to("cuda")
        t5_model._target_device = torch.device("cuda")
        print("[load] T5 fp16 on cuda", flush=True)
    else:
        t5_model = SentenceTransformer(t5_path)
    t5_model.eval()
    for p in t5_model.parameters():
        p.requires_grad = False

    net = tae.Causal_HumanTAE(
        hidden_size=args.hidden_size,
        down_t=args.down_t,
        stride_t=args.stride_t,
        depth=args.depth,
        dilation_growth_rate=args.dilation_growth_rate,
        activation="relu",
        latent_dim=args.latent_dim,
        clip_range=[-30, 20],
    )
    config = LLaMAHFConfig.from_name("Normal_size")
    config.block_size = 78
    trans_encoder = LLaMAHF(config, args.num_diffusion_head_layers, args.latent_dim, device)

    print(f"[load] TAE: {args.resume_pth}", flush=True)
    ckpt = torch.load(args.resume_pth, map_location="cpu")
    net.load_state_dict(ckpt["net"], strict=True)
    net.eval().to(device)

    print(f"[load] Transformer: {args.resume_trans}", flush=True)
    ckpt = torch.load(args.resume_trans, map_location="cpu")
    trans_sd = {}
    for key, value in ckpt["trans"].items():
        new_key = ".".join(key.split(".")[1:]) if key.split(".")[0] == "module" else key
        trans_sd[new_key] = value
    trans_encoder.load_state_dict(trans_sd, strict=True)
    trans_encoder.use_out_proj = bool(getattr(args, "use_out_proj", True))
    trans_encoder.eval().to(device)
    return t5_model, net, trans_encoder


def _motion272_to_npz_fields(
    m272: np.ndarray,
    gt_path: Optional[Path] = None,
    align_mode: str = "yaw",
):
    from hftrainer.datasets.motion.representation.humanml_repr import recover_local_rotations_and_root
    from hftrainer.models.motion.components.utils.geometry.rotation_convert import (
        axis_angle_to_matrix,
        matrix_to_axis_angle,
        matrix_to_rotation_6d,
    )

    rot, root = recover_local_rotations_and_root(np.asarray(m272, dtype=np.float32))
    rot = np.asarray(rot, dtype=np.float32)
    root = np.asarray(root, dtype=np.float32)
    if gt_path is not None and gt_path.exists():
        gt = np.load(str(gt_path), allow_pickle=True)
        if "global_orient" in gt.files and "transl" in gt.files:
            gt_go0 = torch.from_numpy(np.asarray(gt["global_orient"], dtype=np.float32)[:1]).reshape(1, 3)
            gt_mat0 = axis_angle_to_matrix(gt_go0)[0]
            pred_mat0 = torch.from_numpy(np.asarray(rot[0, 0], dtype=np.float32))
            if align_mode == "full":
                delta = gt_mat0 @ pred_mat0.transpose(0, 1)
            elif align_mode == "yaw":
                def yaw_from_mat(mat: torch.Tensor) -> torch.Tensor:
                    fwd = mat[:, 2]
                    return torch.atan2(fwd[0], fwd[2])

                yaw = yaw_from_mat(gt_mat0) - yaw_from_mat(pred_mat0)
                c, s = torch.cos(yaw), torch.sin(yaw)
                delta = torch.stack([
                    torch.stack([c, torch.zeros_like(c), s]),
                    torch.stack([torch.zeros_like(c), torch.ones_like(c), torch.zeros_like(c)]),
                    torch.stack([-s, torch.zeros_like(c), c]),
                ])
            else:
                raise ValueError(f"unsupported align_mode={align_mode!r}")
            rot = rot.copy()
            rot[:, 0] = np.matmul(delta.numpy()[None], rot[:, 0]).astype(np.float32)
            root_t = torch.from_numpy(root)
            gt_tr0 = torch.from_numpy(np.asarray(gt["transl"], dtype=np.float32)[0])
            root = ((delta @ (root_t - root_t[0]).T).T + gt_tr0).numpy().astype(np.float32)
    rot_t = torch.from_numpy(rot)
    d6 = matrix_to_rotation_6d(rot_t, convention="row").numpy().reshape(rot.shape[0], -1)
    aa = matrix_to_axis_angle(rot_t).numpy().astype(np.float32)
    motion_135 = np.concatenate([root, d6], axis=-1).astype(np.float32)
    return {
        "motion_272": np.asarray(m272, dtype=np.float32),
        "motion_135": motion_135,
        "transl": root.astype(np.float32),
        "global_orient": aa[:, 0].astype(np.float32),
        "body_pose": aa[:, 1:].reshape(rot.shape[0], -1).astype(np.float32),
    }


def _fit_motion_length(motion: np.ndarray, target_len: int) -> np.ndarray:
    """Trim or last-frame-pad a motion array to the exact requested length."""
    motion = np.asarray(motion, dtype=np.float32)
    target_len = int(target_len)
    if motion.shape[0] == target_len:
        return motion
    if motion.shape[0] > target_len:
        return motion[:target_len]
    if motion.shape[0] <= 0:
        return motion
    pad = np.repeat(motion[-1:], target_len - motion.shape[0], axis=0)
    return np.concatenate([motion, pad], axis=0).astype(np.float32)


def _safe_name(name: str) -> str:
    return name.replace("/", "_")


def _build_gt_path_map(anno_file: Optional[Path], data_dir: Path) -> Dict[str, Path]:
    if anno_file is None or not anno_file.exists():
        return {}
    raw = _load_json(anno_file)
    out: Dict[str, Path] = {}
    for name, entry in _iter_motionhub_entries(raw):
        smplx_path = entry.get("smplx_path")
        if not smplx_path:
            continue
        path = data_dir / smplx_path
        out[str(name)] = path
        stem = Path(str(smplx_path)).stem
        if stem:
            out[stem] = path
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["humanml3d", "motionhub"], required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--caption-protocol", choices=["rewritten", "original", "fallback"], default=None,
                        help="Caption source. Defaults to original for HumanML3D official_eval, "
                             "rewritten otherwise.")
    parser.add_argument("--anno-file", default=None)
    parser.add_argument("--rewritten-file", default=None)
    parser.add_argument("--data-dir", default="data/motionhub")
    parser.add_argument("--humanml3d-272", default="ref_repo/MotionStreamer/MotionStreamer/humanml3d_272")
    parser.add_argument("--humanml3d-protocol", choices=["official_eval", "all"], default="official_eval",
                        help="official_eval follows MotionStreamer's HumanML3D evaluator length range; "
                             "all keeps every available test pair.")
    parser.add_argument("--humanml3d-min-motion-length", type=int, default=60,
                        help="Minimum HumanML3D length for official_eval. MotionStreamer uses 60 at 30 fps.")
    parser.add_argument("--resume-pth", default="ref_repo/MotionStreamer/MotionStreamer/MotionStreamer_HF/Causal_TAE/net_last.pth")
    parser.add_argument("--resume-trans", default="ref_repo/MotionStreamer/MotionStreamer/MotionStreamer_HF/Experiments/t2m_model/latest.pth")
    parser.add_argument("--t5-model", default=None)
    parser.add_argument("--generation-mode", choices=["paper_eval", "demo_inference"], default="paper_eval",
                        help="paper_eval matches MotionStreamer's eval_t2m.py length-conditioned sampler; "
                             "demo_inference matches demo_t2m.py open-ended end-latent stopping.")
    parser.add_argument("--reference-end-latent", default="ref_repo/MotionStreamer/MotionStreamer/reference_end_latent_t2m_272.npy")
    parser.add_argument("--mean", default="ref_repo/MotionStreamer/MotionStreamer/humanml3d_272/mean_std/Mean.npy")
    parser.add_argument("--std", default="ref_repo/MotionStreamer/MotionStreamer/humanml3d_272/mean_std/Std.npy")
    parser.add_argument("--threshold", type=float, default=0.1)
    parser.add_argument("--max-motion-length", type=int, default=300,
                        help="Maximum generated frame length before unit-length rounding; "
                             "MotionStreamer HumanML3D eval uses max_motion_length=300.")
    parser.add_argument("--align-to-gt-root", action="store_true",
                        help="Align the decoded 272 motion to each paired GT first-frame root before saving SMPL fields.")
    parser.add_argument("--align-root-mode", choices=["yaw", "full"], default="yaw")
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

    if not (0 <= args.shard_index < args.num_shards):
        raise ValueError("--shard-index must be in [0, num_shards)")
    if args.caption_protocol is None:
        args.caption_protocol = (
            "original"
            if args.dataset == "humanml3d" and args.humanml3d_protocol == "official_eval"
            else "rewritten"
        )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[setup] device={device} dataset={args.dataset} shard={args.shard_index}/{args.num_shards}", flush=True)

    anno_file = Path(args.anno_file) if args.anno_file else None
    rewritten_file = Path(args.rewritten_file) if args.rewritten_file else None
    if args.dataset == "humanml3d":
        min_length = None
        max_length_exclusive = None
        limit_hint = 0
        if args.humanml3d_protocol == "official_eval":
            min_length = int(args.humanml3d_min_motion_length)
            max_length_exclusive = int(args.max_motion_length)
            if args.max_samples > 0:
                limit_hint = int(args.max_samples) * max(int(args.num_shards), 1)
        pairs = _load_h3d_pairs(
            humanml3d_272=Path(args.humanml3d_272),
            anno_file=anno_file,
            rewritten_file=rewritten_file,
            data_dir=Path(args.data_dir),
            caption_protocol=args.caption_protocol,
            min_length=min_length,
            max_length_exclusive=max_length_exclusive,
            limit=limit_hint,
        )
        if args.humanml3d_protocol == "official_eval":
            print(
                f"[setup] HumanML3D official_eval length filter applied "
                f"(range=[{args.humanml3d_min_motion_length},{args.max_motion_length}))",
                flush=True,
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
    pairs = _select_shard(pairs, args.num_shards, args.shard_index)
    if args.max_samples > 0:
        pairs = pairs[: args.max_samples]
    print(f"[setup] pairs={len(pairs)}", flush=True)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = out_dir / f"manifest_shard{args.shard_index}of{args.num_shards}.jsonl"
    gt_path_map = _build_gt_path_map(anno_file, Path(args.data_dir)) if args.align_to_gt_root else {}

    t5_model, net, trans_encoder = _load_model(args, device)
    reference_end_latent = None
    if args.generation_mode == "demo_inference":
        reference_end_latent = torch.from_numpy(np.load(args.reference_end_latent)).to(device)
    mean = np.load(args.mean)
    std = np.load(args.std)

    ok = skipped = failed = 0
    with meta_path.open("w") as meta_f:
        with torch.no_grad():
            for local_i, (name, caption, target_len) in enumerate(pairs):
                out_path = out_dir / f"{_safe_name(name)}.npz"
                if args.skip_existing and out_path.exists():
                    skipped += 1
                    continue
                try:
                    if args.dataset == "humanml3d":
                        capped_target_len = int(target_len)
                    else:
                        capped_target_len = min(int(target_len), int(args.max_motion_length))
                    eval_len = (capped_target_len // 4) * 4
                    if eval_len <= 0:
                        raise ValueError(f"invalid target length: {target_len}")
                    if args.generation_mode == "paper_eval":
                        latents = trans_encoder.sample_for_eval_CFG(
                            text=[caption],
                            length=eval_len,
                            tokenize_model=t5_model,
                            device=device,
                            unit_length=4,
                            cfg=4.0,
                        )
                    else:
                        latents = trans_encoder.sample_for_eval_CFG_inference(
                            text=caption,
                            length=max(eval_len, 312),
                            tokenizer=t5_model,
                            device=device,
                            unit_length=4,
                            reference_end_latent=reference_end_latent,
                            threshold=args.threshold,
                            cfg=4.0,
                        )
                    motion_norm = net.forward_decoder(latents).squeeze(0).detach().cpu().numpy()
                    motion_norm = motion_norm[:eval_len]
                    motion_272 = (motion_norm * std + mean).astype(np.float32)
                    motion_272 = _fit_motion_length(motion_272, capped_target_len)
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
                        capped_target_length=int(capped_target_len),
                        generated_length=int(motion_272.shape[0]),
                        caption_protocol=args.caption_protocol,
                        dataset=args.dataset,
                        aligned_to_gt_root=bool(gt_path is not None),
                        align_root_mode=args.align_root_mode if gt_path is not None else "",
                        motionstreamer_use_out_proj=bool(args.use_out_proj),
                    )
                    meta_f.write(json.dumps({
                        "sample_id": name,
                        "path": str(out_path),
                        "text": caption,
                        "target_length": int(target_len),
                        "generated_length": int(motion_272.shape[0]),
                        "capped_target_length": int(capped_target_len),
                        "gt_path": str(gt_path) if gt_path is not None else "",
                        "motionstreamer_use_out_proj": bool(args.use_out_proj),
                    }, ensure_ascii=False) + "\n")
                    meta_f.flush()
                    ok += 1
                except Exception as exc:  # noqa: BLE001
                    failed += 1
                    print(f"[fail] {name}: {type(exc).__name__}: {exc}", flush=True)
                if (local_i + 1) % 25 == 0:
                    print(f"[progress] {local_i + 1}/{len(pairs)} ok={ok} skipped={skipped} failed={failed}", flush=True)

    summary = {
        "dataset": args.dataset,
        "out_dir": str(out_dir),
        "caption_protocol": args.caption_protocol,
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "pairs": len(pairs),
        "ok": ok,
        "skipped": skipped,
        "failed": failed,
    }
    (out_dir / f"summary_shard{args.shard_index}of{args.num_shards}.json").write_text(json.dumps(summary, indent=2))
    print("[done] " + json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
