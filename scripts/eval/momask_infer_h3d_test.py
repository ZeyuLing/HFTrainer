#!/usr/bin/env python3
"""Run MoMask T2M inference on the HumanML3D test split.

Drives MoMask's released checkpoints (rvq + t2m_transformer + tres + length_estimator)
over every (caption, length) pair in MotionStreamer's ``humanml3d_272/`` test split,
and saves the **un-standardized 263-dim HumanML3D feature** for each sample as
``<out_dir>/<id>.npy``. Each id corresponds to the matching ``humanml3d_272/<split>/<id>``,
so a downstream 263→272 converter + MotionStreamer evaluator can compare directly.

Usage::

    python3 tools/momask_infer_h3d_test.py \
        --momask_root  ref_repo/Momask/momask-codes \
        --humanml3d_272 ref_repo/MotionStreamer/MotionStreamer/humanml3d_272 \
        --out_dir      work_dirs/momask_eval/momask_pred_263 \
        --vq_name      rvq_nq6_dc512_nc512_noshare_qdp0.2 \
        --t2m_name     t2m_nlayer8_nhead6_ld384_ff1024_cdp0.1_rvq6ns \
        --res_name     tres_nlayer8_ld384_ff1024_rvq6ns_cdp0.2_sw \
        [--cond_scale 4 --time_steps 10 --topkr 0.9 --temperature 1.0]
        [--max_samples 100]

Caveats:
* Generation length is forced to the ground-truth ``T_gt = humanml3d_272/<id>.npy.shape[0]``
  rounded down to a multiple of 4 (MoMask's ``unit_length``). This matches the
  protocol used by MoMask's own ``eval_t2m_trans_res.py`` (which also takes m_lens
  from the test loader).
* We pick the **first non-tag-restricted caption** from each
  ``humanml3d_272/texts/<id>.txt`` to drive the generation. (Tagged sub-clips
  ``f_tag != 0 or t_tag != 0`` are skipped — they would require time-slicing the
  reference motion to match.)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from tqdm import tqdm


def _import_momask(momask_root: Path):
    """Add MoMask root to sys.path and import its model classes."""
    sys.path.insert(0, str(momask_root.resolve()))
    sys.path.insert(0, str((momask_root / "models").resolve()))

    from models.mask_transformer.transformer import MaskTransformer, ResidualTransformer  # noqa: E402
    from models.vq.model import RVQVAE, LengthEstimator  # noqa: E402
    from utils.get_opt import get_opt  # noqa: E402

    return MaskTransformer, ResidualTransformer, RVQVAE, LengthEstimator, get_opt


CLIP_VERSION = "ViT-B/32"


def _load_test_pairs(humanml3d_272: Path):
    """Mirror ``ref_repo/MotionStreamer/eval_with_motionstreamer_evaluator.py::load_test_pairs``.

    Returns a list of ``(name, caption, m_length)`` tuples — each *(name, caption)*
    is one prompt to drive MoMask, with *m_length* the ground-truth motion length
    (after unit_length=4 rounding).
    """
    motion_dir = humanml3d_272 / "motion_data"
    text_dir = humanml3d_272 / "texts"
    split = (humanml3d_272 / "split" / "test.txt").read_text().splitlines()

    pairs = []
    for name in split:
        name = name.strip()
        if not name:
            continue
        m_file = motion_dir / f"{name}.npy"
        t_file = text_dir / f"{name}.txt"
        if not (m_file.exists() and t_file.exists()):
            continue
        gt = np.load(m_file)
        T_gt = len(gt)
        if T_gt < 60 or T_gt >= 300:
            continue
        for line in t_file.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split("#")
            if len(parts) < 4:
                continue
            caption = parts[0]
            f_tag = float(parts[2]) if parts[2] != "nan" else 0.0
            t_tag = float(parts[3]) if parts[3] != "nan" else 0.0
            if f_tag != 0.0 or t_tag != 0.0:
                # tagged sub-clip — skip (requires alignment we don't track)
                continue
            ml = (T_gt // 4) * 4
            if ml < 60:
                continue
            pairs.append((name, caption, ml, 30.0))
    return pairs


def _load_json(path: Path):
    return json.loads(path.read_text())


def _iter_motionhub_entries(raw):
    if isinstance(raw, dict) and "data_list" in raw:
        data = raw["data_list"]
        if isinstance(data, dict):
            for name, entry in data.items():
                yield str(name), entry
        else:
            for i, entry in enumerate(data):
                yield str(entry.get("motion_id") or entry.get("id") or i), entry
    elif isinstance(raw, list):
        for i, entry in enumerate(raw):
            yield str(entry.get("motion_id") or entry.get("id") or i), entry
    else:
        raise ValueError("Unrecognized annotation format")


def _load_rewritten(path: Optional[Path]):
    if path is None:
        return {}
    raw = _load_json(path)
    if isinstance(raw, dict) and "data_list" in raw:
        raw = raw["data_list"]
    if not isinstance(raw, dict):
        raise ValueError(f"rewritten caption file must be a dict: {path}")
    out = {}
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


def _load_caption_from_json(path: Path):
    try:
        data = _load_json(path)
    except Exception:
        return None
    pool = []
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


def _load_annotation_pairs(anno_file: Path, data_dir: Path, rewritten_file: Optional[Path],
                           caption_protocol: str, default_fps: float):
    rewritten = _load_rewritten(rewritten_file)
    pairs = []
    for name, entry in _iter_motionhub_entries(_load_json(anno_file)):
        caption = None
        if caption_protocol == "rewritten":
            caption = rewritten.get(name)
        if not caption and caption_protocol in {"original", "fallback"} and entry.get("hierarchical_caption_path"):
            caption = _load_caption_from_json(data_dir / entry["hierarchical_caption_path"])
        if not caption and caption_protocol in {"rewritten", "fallback"}:
            caption = rewritten.get(name)
        if not isinstance(caption, str) or not caption.strip():
            continue
        src_fps = float(entry.get("fps") or default_fps)
        length = int(entry.get("num_frames") or round(float(entry.get("duration", 0.0)) * src_fps))
        if length <= 0:
            continue
        pairs.append((name, caption.strip(), length, src_fps))
    return pairs


def _select_shard(items, num_shards: int, shard_index: int):
    if num_shards <= 1:
        return items
    return [item for i, item in enumerate(items) if i % num_shards == shard_index]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--momask_root", required=True)
    p.add_argument("--humanml3d_272", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--vq_name", default="rvq_nq6_dc512_nc512_noshare_qdp0.2")
    p.add_argument("--t2m_name", default="t2m_nlayer8_nhead6_ld384_ff1024_cdp0.1_rvq6ns")
    p.add_argument("--res_name", default="tres_nlayer8_ld384_ff1024_rvq6ns_cdp0.2_sw")
    p.add_argument("--checkpoints_dir", default="checkpoints",
                   help="Relative to momask_root. Default 'checkpoints'.")
    p.add_argument("--dataset_name", default="t2m")
    p.add_argument("--cond_scale", type=float, default=4.0)
    p.add_argument("--time_steps", type=int, default=10)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--topkr", type=float, default=0.9)
    p.add_argument("--gumbel_sample", action="store_true")
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--gt_fps", type=float, default=30.0,
                   help="FPS of the reference motions used to source m_length. "
                        "MoMask runs at 20 fps internally, so we scale lengths "
                        "by 20/gt_fps before passing to the model.")
    p.add_argument("--momask_fps", type=float, default=20.0)
    p.add_argument("--anno_file", default=None,
                   help="Optional motionhub-format annotation. When set, sample ids/captions/lengths "
                        "come from this file instead of the official HumanML3D split.")
    p.add_argument("--data_dir", default="data/motionhub")
    p.add_argument("--rewritten_file", default=None)
    p.add_argument("--caption_protocol", choices=["rewritten", "original", "fallback"], default="rewritten")
    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--shard_index", type=int, default=0)
    p.add_argument("--skip_existing", action="store_true")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    momask_root = Path(args.momask_root).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    humanml3d_272 = Path(args.humanml3d_272).resolve()

    MaskTransformer, ResidualTransformer, RVQVAE, LengthEstimator, get_opt = _import_momask(momask_root)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[+] device       = {device}")
    print(f"[+] momask_root  = {momask_root}")
    print(f"[+] out_dir      = {out_dir}")

    # ----- 1. Load opts (paths are interpreted relative to momask_root/cwd) -----
    cwd_save = os.getcwd()
    os.chdir(str(momask_root))
    try:
        ckpt_dir = Path(args.checkpoints_dir)
        vq_opt = get_opt(str(ckpt_dir / args.dataset_name / args.vq_name / "opt.txt"), device=device)
        vq_opt.dim_pose = 263
        model_opt = get_opt(str(ckpt_dir / args.dataset_name / args.t2m_name / "opt.txt"), device=device)
        res_opt = get_opt(str(ckpt_dir / args.dataset_name / args.res_name / "opt.txt"), device=device)

        # ----- 2. Build models -----
        print(f"[+] Loading RVQ {args.vq_name} ...")
        vq_model = RVQVAE(
            vq_opt, vq_opt.dim_pose, vq_opt.nb_code, vq_opt.code_dim,
            vq_opt.output_emb_width, vq_opt.down_t, vq_opt.stride_t,
            vq_opt.width, vq_opt.depth, vq_opt.dilation_growth_rate,
            vq_opt.vq_act, vq_opt.vq_norm,
        )
        vq_ckpt = torch.load(
            str(ckpt_dir / args.dataset_name / args.vq_name / "model" / "net_best_fid.tar"),
            map_location="cpu",
        )
        vq_model.load_state_dict(vq_ckpt.get("vq_model", vq_ckpt.get("net")))

        model_opt.num_tokens = vq_opt.nb_code
        model_opt.num_quantizers = vq_opt.num_quantizers
        model_opt.code_dim = vq_opt.code_dim
        res_opt.num_tokens = vq_opt.nb_code
        res_opt.num_quantizers = vq_opt.num_quantizers

        print(f"[+] Loading t2m_transformer {args.t2m_name} ...")
        t2m_transformer = MaskTransformer(
            code_dim=model_opt.code_dim,
            cond_mode="text",
            latent_dim=model_opt.latent_dim,
            ff_size=model_opt.ff_size,
            num_layers=model_opt.n_layers,
            num_heads=model_opt.n_heads,
            dropout=model_opt.dropout,
            clip_dim=512,
            cond_drop_prob=model_opt.cond_drop_prob,
            clip_version=CLIP_VERSION,
            opt=model_opt,
        )
        t2m_ckpt = torch.load(
            str(ckpt_dir / args.dataset_name / args.t2m_name / "model" / "latest.tar"),
            map_location="cpu",
        )
        miss, unex = t2m_transformer.load_state_dict(t2m_ckpt["t2m_transformer"], strict=False)
        assert len(unex) == 0
        assert all(k.startswith("clip_model.") for k in miss)

        print(f"[+] Loading res_transformer {args.res_name} ...")
        res_transformer = ResidualTransformer(
            code_dim=vq_opt.code_dim,
            cond_mode="text",
            latent_dim=res_opt.latent_dim,
            ff_size=res_opt.ff_size,
            num_layers=res_opt.n_layers,
            num_heads=res_opt.n_heads,
            dropout=res_opt.dropout,
            clip_dim=512,
            shared_codebook=vq_opt.shared_codebook,
            cond_drop_prob=res_opt.cond_drop_prob,
            share_weight=res_opt.share_weight,
            clip_version=CLIP_VERSION,
            opt=res_opt,
        )
        res_ckpt = torch.load(
            str(ckpt_dir / args.dataset_name / args.res_name / "model" / "net_best_fid.tar"),
            map_location="cpu",
        )
        miss, unex = res_transformer.load_state_dict(res_ckpt["res_transformer"], strict=False)
        assert len(unex) == 0
        assert all(k.startswith("clip_model.") for k in miss)

        # ----- 3. Mean / std for inv_transform -----
        mean = np.load(str(ckpt_dir / args.dataset_name / args.vq_name / "meta" / "mean.npy"))
        std = np.load(str(ckpt_dir / args.dataset_name / args.vq_name / "meta" / "std.npy"))
    finally:
        os.chdir(cwd_save)

    vq_model.eval().to(device)
    t2m_transformer.eval().to(device)
    res_transformer.eval().to(device)

    # ----- 4. Build test prompt list -----
    print("[+] Loading test pairs ...")
    if args.anno_file:
        pairs = _load_annotation_pairs(
            Path(args.anno_file),
            Path(args.data_dir),
            Path(args.rewritten_file) if args.rewritten_file else None,
            args.caption_protocol,
            args.gt_fps,
        )
    else:
        pairs = _load_test_pairs(humanml3d_272)
    pairs = _select_shard(pairs, args.num_shards, args.shard_index)
    print(f"    {len(pairs)} (id, caption, T) prompts (one per caption line)")

    # The downstream MotionStreamer evaluator expects ONE pred motion per `name`
    # (= one ``<name>.npy`` file in --pred_dir), then pairs that motion with each
    # caption listed in ``texts/<name>.txt`` for retrieval.  Mirror that protocol:
    # generate one motion per name using the *first* valid caption as the prompt.
    #
    # Also rescale the GT motion length (which is in --gt_fps frames) to MoMask's
    # internal 20 fps so the *physical duration* matches the GT once we later
    # upsample MoMask outputs back to 30 fps.
    indexed_pairs = []
    seen = set()
    src_fps_values = []
    for name, caption, ml, src_fps in pairs:
        if name in seen:
            continue
        seen.add(name)
        ml_momask = int(round(ml * args.momask_fps / src_fps))
        ml_momask = (ml_momask // 4) * 4
        if ml_momask < 40:  # MoMask expects at least 10 latent tokens
            ml_momask = 40
        indexed_pairs.append((name, caption, ml_momask))
        src_fps_values.append(src_fps)

    if args.max_samples:
        indexed_pairs = indexed_pairs[: args.max_samples]
    print(f"    will generate {len(indexed_pairs)} motions (one per name, first caption)")
    if src_fps_values:
        print(f"    source_fps min/median/max="
              f"{min(src_fps_values):.3g}/{float(np.median(src_fps_values)):.3g}/{max(src_fps_values):.3g}; "
              f"momask_fps={args.momask_fps:.3g}")

    # ----- 5. Generate in batches -----
    bs = args.batch_size
    written = 0
    t0 = time.time()
    with torch.no_grad():
        for i in tqdm(range(0, len(indexed_pairs), bs), ncols=80):
            j = min(i + bs, len(indexed_pairs))
            chunk = indexed_pairs[i:j]
            ids_b = [c[0] for c in chunk]
            caps_b = [c[1] for c in chunk]
            lens_b = [c[2] for c in chunk]

            token_lens = torch.tensor(lens_b, dtype=torch.long, device=device) // 4

            try:
                mids = t2m_transformer.generate(
                    caps_b, token_lens,
                    timesteps=args.time_steps,
                    cond_scale=args.cond_scale,
                    temperature=args.temperature,
                    topk_filter_thres=args.topkr,
                    gsample=args.gumbel_sample,
                )
                mids = res_transformer.generate(mids, caps_b, token_lens, temperature=1.0, cond_scale=5.0)
                pred_motions = vq_model.forward_decoder(mids).detach().cpu().numpy()
            except Exception as e:
                print(f"  [!] batch {i}-{j} failed: {e}")
                continue

            data = pred_motions * std + mean

            for k, (sid, caption, ml) in enumerate(chunk):
                out_path = out_dir / f"{sid}.npy"
                if args.skip_existing and out_path.exists():
                    continue
                m = data[k, : int(ml)]
                np.save(out_path, m.astype(np.float32))
                written += 1

    elapsed = time.time() - t0
    print(f"[+] done: {written} / {len(indexed_pairs)} motions  elapsed={elapsed:.1f}s")
    print(f"[+] outputs in {out_dir}")


if __name__ == "__main__":
    main()
