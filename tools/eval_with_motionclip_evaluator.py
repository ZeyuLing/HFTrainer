#!/usr/bin/env python3
"""Standalone MotionCLIP-evaluator evaluation script.

Mirrors :file:`ref_repo/MotionStreamer/eval_with_motionstreamer_evaluator.py`
in protocol (32-chunk R-Precision / MM-Dist, 20-repeat averaging, Frechet
distance on encoded embeddings) but uses the *MotionCLIP* evaluator
(SMPL-22, 135-dim) trained in our framework — i.e. "our evaluator" in the
TMM paper.

Three input modes:
  --gt_only            : real motions, real motions; sanity-check (FID -> 0).
  --pred_dir DIR       : (motion_<id>.npy, 135-dim) predictions vs GT motions.
  --pred_npz NPZ       : single .npz with arrays {names, motions, lengths},
                         each motion 135-dim aligned with the test split.

Test-split source:
  --anno_file <test json> + --data_dir <motionhub root>
  Caption is loaded via the same `LoadCompatibleCaption` transform used at
  training time, motion via `LoadSmplx55(rot=rot6d, smpl_22, transl=abs)`.

Metrics — same as MotionStreamer protocol (within-chunk):
  R-Precision @ 1/2/3, FID, MM-Dist (matching score), Diversity.

Output:
  --out_json <path> JSON dict with mean / std of all metrics over n_repeats.

Example:
  python tools/eval_with_motionclip_evaluator.py \
      --evaluator_ckpt checkpoints/motion_clip/motionclip_base_1p_aug_hq \
      --anno_file data/annotation/test_hml3d.json \
      --data_dir data/motionhub \
      --gt_only \
      --out_json work_dirs/mc_eval/gt_h3d.json \
      --n_repeats 20
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor

THIS_DIR = Path(__file__).resolve().parent
HF_ROOT = THIS_DIR.parent
sys.path.insert(0, str(HF_ROOT))


# ---------------------------------------------------------------------------
# MotionCLIP loader
# ---------------------------------------------------------------------------

def load_motionclip(ckpt_dir: Path,
                    device: torch.device,
                    clip_pretrained: str = 'checkpoints/clip-vit-base-patch32',
                    stats_file: str = 'data/statistic/smplx55_stats_hymotion_aug.json'):
    """Load the bundle from a converted checkpoint dir.

    Expects::
        <ckpt_dir>/motionclip_model.safetensors
        <ckpt_dir>/bundle_config.json
    """
    from safetensors.torch import load_file
    from hftrainer.models.motion.motion_clip import MotionCLIPBundle

    cfg_path = ckpt_dir / 'bundle_config.json'
    weight_path = ckpt_dir / 'motionclip_model.safetensors'
    if not cfg_path.exists() or not weight_path.exists():
        raise FileNotFoundError(
            f'Expected motionclip_model.safetensors and bundle_config.json '
            f'in {ckpt_dir}'
        )
    with open(cfg_path) as f:
        bcfg = json.load(f)

    bundle = MotionCLIPBundle(
        text_config=bcfg['text_config'],
        motion_config=bcfg['motion_config'],
        projection_dim=bcfg['projection_dim'],
        logit_scale_init_value=bcfg['logit_scale_init_value'],
        tokenizer={
            'type': 'CLIPTokenizer',
            'from_pretrained': {
                'pretrained_model_name_or_path': str(clip_pretrained),
            },
        },
        smpl_pose_processor={
            'type': 'SMPLPoseProcessor',
            'do_normalize': True,
            'stats_file': str(stats_file),
            'rot_type': 'rotation_6d',
            'transl_type': 'abs',
            'smpl_type': 'smpl_22',
            'smpl_model': None,
            'smooth_model': None,
        },
        clip_pretrained=None,
        freeze_text_encoder=False,
    )
    sd = load_file(str(weight_path))
    missing, unexpected = bundle.motionclip_model.load_state_dict(sd, strict=True)
    assert not missing and not unexpected, (missing, unexpected)
    bundle = bundle.eval().to(device)
    return bundle


# ---------------------------------------------------------------------------
# Test-split loading (motion + caption pairs)
# ---------------------------------------------------------------------------

def _load_caption(caption_path: Path) -> Optional[str]:
    """Mirror LoadCompatibleCaption: supports hierarchical (macro/meso/micro)
    and hymotion (result -> short_caption[_rewritten]) formats.
    Returns ONE randomly chosen caption."""
    if not caption_path.exists():
        return None
    try:
        data = json.loads(caption_path.read_text())
    except Exception:
        return None
    if not isinstance(data, dict):
        return None

    # Hierarchical
    if all(k in data and isinstance(data[k], list) for k in ('macro', 'meso', 'micro')):
        pool = []
        for g in ('macro', 'meso', 'micro'):
            for c in data[g]:
                if isinstance(c, str) and c.strip():
                    pool.append(c.strip())
        return random.choice(pool) if pool else None

    # HYMotion
    if 'result' in data and isinstance(data['result'], list):
        pool = []
        for item in data['result']:
            if not isinstance(item, dict):
                continue
            for rk in ('short_caption_rewritten', 'short caption_rewritten'):
                if isinstance(item.get(rk), list):
                    for v in item[rk]:
                        if isinstance(v, str) and v.strip():
                            pool.append(v.strip())
                    break
            else:
                for ck in ('short_caption', 'short caption'):
                    if isinstance(item.get(ck), str) and item[ck].strip():
                        pool.append(item[ck].strip())
                        break
        return random.choice(pool) if pool else None

    return None


def _load_smpl22_motion(motion_path: Path) -> Optional[np.ndarray]:
    """Minimal SMPL-22 loader: returns (T, 135) array (transl + 22 joints * 6D rot).

    Reads NPZ produced by motionhub data pipeline. Expects fields:
      transl: (T, 3)
      global_orient: (T, 3) axis-angle  [will be converted to rot6d]
      body_pose: (T, 21*3) axis-angle    [will be converted to rot6d]
    Either axis-angle or rotation_6d direct fields are acceptable.
    """
    if not motion_path.exists():
        return None
    try:
        npz = np.load(str(motion_path), allow_pickle=True)
    except Exception:
        return None

    keys = list(npz.keys()) if hasattr(npz, 'keys') else []
    if 'transl' not in keys:
        return None
    transl = np.asarray(npz['transl'], dtype=np.float32)  # (T, 3)
    T = transl.shape[0]

    # Try rotation_6d direct first
    if 'global_orient_rot6d' in keys and 'body_pose_rot6d' in keys:
        go = np.asarray(npz['global_orient_rot6d'], dtype=np.float32).reshape(T, -1)  # (T, 6)
        bp = np.asarray(npz['body_pose_rot6d'], dtype=np.float32).reshape(T, -1)      # (T, 21*6)
    else:
        # axis-angle path
        from hftrainer.models.motion.components.utils.geometry.rotation_convert import (
            axis_angle_to_matrix,
            matrix_to_rotation_6d,
        )
        go_aa = torch.from_numpy(np.asarray(npz['global_orient'], dtype=np.float32)).reshape(T, 3)
        bp_aa = torch.from_numpy(np.asarray(npz['body_pose'], dtype=np.float32)).reshape(T, 21, 3)

        go_rotmat = axis_angle_to_matrix(go_aa)
        bp_rotmat = axis_angle_to_matrix(bp_aa)
        go = matrix_to_rotation_6d(go_rotmat).numpy().reshape(T, -1)
        bp = matrix_to_rotation_6d(bp_rotmat).numpy().reshape(T, -1)

    motion135 = np.concatenate([transl, go, bp], axis=-1)
    if motion135.shape[-1] != 135:
        return None
    return motion135.astype(np.float32)


def load_test_pairs(anno_file: Path,
                    data_dir: Path,
                    motion_key: str = 'smplx',
                    caption_key: str = 'hierarchical_caption',
                    min_frames: int = 24,
                    max_frames: int = 360,
                    max_pairs: Optional[int] = None) -> List[Tuple[str, str, np.ndarray, int]]:
    """Load (name, caption, motion[135], num_frames) pairs from a motionhub anno file.

    Supports two layouts:
      - flat list[dict]
      - {meta_info, data_list: dict[name -> entry]} (motionhub canonical)
    """
    raw = json.loads(Path(anno_file).read_text())
    if isinstance(raw, dict) and 'data_list' in raw:
        dl = raw['data_list']
        if isinstance(dl, dict):
            entries = [(name, e) for name, e in dl.items()]
        else:
            entries = [(e.get('motion_id') or e.get('id') or str(i), e)
                       for i, e in enumerate(dl)]
    elif isinstance(raw, list):
        entries = [(e.get('motion_id') or e.get('id') or str(i), e) for i, e in enumerate(raw)]
    else:
        raise ValueError(f'Unrecognized annotation format in {anno_file}')

    pairs = []
    for name, entry in entries:
        m_rel = entry.get(f'{motion_key}_path')
        c_rel = entry.get(f'{caption_key}_path')
        if not (m_rel and c_rel):
            continue
        m_path = Path(data_dir) / m_rel
        c_path = Path(data_dir) / c_rel
        caption = _load_caption(c_path)
        if caption is None:
            continue
        motion = _load_smpl22_motion(m_path)
        if motion is None:
            continue
        T = motion.shape[0]
        if T < min_frames:
            continue
        nf = min(T, max_frames)
        pairs.append((name, caption, motion[:nf], nf))
        if max_pairs and len(pairs) >= max_pairs:
            break
    return pairs


# ---------------------------------------------------------------------------
# Metric helpers (same protocol as MotionStreamer)
# ---------------------------------------------------------------------------

def _euclidean_dist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    d1 = -2 * a @ b.T
    d2 = (a ** 2).sum(axis=1, keepdims=True)
    d3 = (b ** 2).sum(axis=1)
    return np.sqrt(np.maximum(d1 + d2 + d3, 0))


def _calc_top_k(argmax: np.ndarray, k: int) -> np.ndarray:
    n = argmax.shape[0]
    gt = np.arange(n)[:, None].repeat(n, 1)
    correct = np.zeros(n, dtype=bool)
    out = np.zeros((n, k), dtype=bool)
    for i in range(k):
        correct = correct | (argmax[:, i] == gt[:, i])
        out[:, i] = correct
    return out


def _r_precision(text_emb: np.ndarray, motion_emb: np.ndarray, top_k: int = 3):
    d = _euclidean_dist(text_emb, motion_emb)
    matching = d.trace()
    arg = np.argsort(d, axis=1)
    top = _calc_top_k(arg, top_k)
    return top.sum(0), matching


def _diversity(emb: np.ndarray, n: int = 300) -> float:
    n = min(n, len(emb))
    a = emb[np.random.choice(len(emb), n, replace=False)]
    b = emb[np.random.choice(len(emb), n, replace=False)]
    return float(np.linalg.norm(a - b, axis=1).mean())


def _frechet(mu1, c1, mu2, c2, eps=1e-6) -> float:
    from scipy import linalg
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(c1.dot(c2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(c1.shape[0]) * eps
        covmean = linalg.sqrtm((c1 + offset).dot(c2 + offset))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff.dot(diff) + np.trace(c1) + np.trace(c2) - 2 * np.trace(covmean))


def _activation_stats(x: np.ndarray):
    return x.mean(axis=0), np.cov(x, rowvar=False)


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------

def encode_dataset(bundle, captions: List[str], motions: List[np.ndarray],
                   lengths: List[int], device: torch.device,
                   batch_size: int = 32, max_frames: int = 360):
    """Run MotionCLIP over the dataset, return (text_emb, motion_emb, [unnorm_text/motion]).

    Note: we return the *projected, non-normalized* embeddings — matching
    the convention used by MotionStreamer's TMR latent for FID computation.
    """
    n = len(captions)
    text_embs, motion_embs = [], []
    real_text_norm, real_motion_norm = [], []

    with torch.no_grad():
        for i in range(0, n, batch_size):
            j = min(i + batch_size, n)
            cap_b = captions[i:j]
            mot_b = motions[i:j]
            len_b = lengths[i:j]

            T = max(int(l) for l in len_b)
            T = min(T, max_frames)
            B = j - i

            # pad to common T via replicate
            mot_padded = np.zeros((B, T, 135), dtype=np.float32)
            for k, (m, ml) in enumerate(zip(mot_b, len_b)):
                ml = min(int(ml), T)
                mot_padded[k, :ml] = m[:ml]
                if ml < T:
                    mot_padded[k, ml:] = m[ml - 1]  # replicate last
            mot_t = torch.from_numpy(mot_padded).to(device)

            # normalize via SMPL processor
            mot_n = bundle.smpl_pose_processor.normalize(mot_t)

            # build mask
            attn = torch.zeros(B, T, device=device, dtype=mot_n.dtype)
            for k, ml in enumerate(len_b):
                attn[k, : min(int(ml), T)] = 1.0

            # ---- motion embedding (raw projection, used for FID) ----
            mot_feat = bundle.encode_motion(mot_n, attn)
            motion_embs.append(mot_feat.cpu().numpy())

            # ---- text embedding ----
            enc = bundle.tokenize(list(cap_b))
            text_feat = bundle.encode_text(
                enc['input_ids'].to(device),
                enc['attention_mask'].to(device),
            )
            text_embs.append(text_feat.cpu().numpy())

    text_emb = np.concatenate(text_embs, axis=0)
    motion_emb = np.concatenate(motion_embs, axis=0)
    return text_emb, motion_emb


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--evaluator_ckpt', required=True,
                   help='Directory with motionclip_model.safetensors + bundle_config.json')
    p.add_argument('--anno_file', required=True,
                   help='motionhub-format JSON with motion_path/caption_path entries.')
    p.add_argument('--data_dir', required=True, help='motionhub data root')
    p.add_argument('--motion_key', default='smplx')
    p.add_argument('--caption_key', default='hierarchical_caption')
    p.add_argument('--clip_pretrained', default='checkpoints/clip-vit-base-patch32')
    p.add_argument('--stats_file', default='data/statistic/smplx55_stats_hymotion_aug.json')

    p.add_argument('--gt_only', action='store_true',
                   help='Use real motions for both pred and real (sanity check, FID -> 0).')
    p.add_argument('--pred_dir', default=None,
                   help='Directory of pred_<name>.npy 135-dim motions, keyed by sample name.')
    p.add_argument('--pred_npz', default=None,
                   help='Single NPZ with arrays {names: list[str], motions: object[ndarray], lengths: int[]}.')

    p.add_argument('--out_json', required=True)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--max_frames', type=int, default=360)
    p.add_argument('--min_frames', type=int, default=24)
    p.add_argument('--max_pairs', type=int, default=None)
    p.add_argument('--batch_size', type=int, default=32,
                   help='R-Precision/MM-Dist computed per chunk of this size.')
    p.add_argument('--n_repeats', type=int, default=20,
                   help='Average metrics over this many random shuffles.')
    args = p.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[+] device = {device}')

    print('[+] Loading test pairs ...')
    pairs = load_test_pairs(
        Path(args.anno_file), Path(args.data_dir),
        motion_key=args.motion_key, caption_key=args.caption_key,
        min_frames=args.min_frames, max_frames=args.max_frames,
        max_pairs=args.max_pairs,
    )
    print(f'    loaded: {len(pairs)} (motion_135-dim, caption) pairs')
    if not pairs:
        raise RuntimeError('No pairs loaded; check --anno_file / --data_dir.')

    print('[+] Loading MotionCLIP evaluator ...')
    bundle = load_motionclip(
        Path(args.evaluator_ckpt), device,
        clip_pretrained=args.clip_pretrained,
        stats_file=args.stats_file,
    )

    # Aligned (real, pred) selection
    captions, real_motions, pred_motions, lengths = [], [], [], []
    pred_dir = Path(args.pred_dir) if args.pred_dir else None
    pred_lookup = None
    if args.pred_npz:
        z = np.load(args.pred_npz, allow_pickle=True)
        # supports two shapes: object array with named entries, or arrays
        names = list(z['names']) if 'names' in z.files else list(z.files)
        mots = z['motions'] if 'motions' in z.files else None
        lens = z['lengths'] if 'lengths' in z.files else None
        pred_lookup = {}
        for i, nm in enumerate(names):
            arr = mots[i] if mots is not None else z[nm]
            ln = int(lens[i]) if lens is not None else int(arr.shape[0])
            pred_lookup[str(nm)] = (np.asarray(arr, dtype=np.float32), ln)

    for name, caption, gt, ml in pairs:
        if args.gt_only:
            pred = gt
            pred_ml = ml
        elif pred_dir is not None:
            pred_file = pred_dir / f'{name}.npy'
            if not pred_file.exists():
                continue
            pred = np.load(str(pred_file)).astype(np.float32)
            pred_ml = int(pred.shape[0])
        elif pred_lookup is not None:
            if name not in pred_lookup:
                continue
            pred, pred_ml = pred_lookup[name]
        else:
            raise ValueError('Specify --gt_only, --pred_dir or --pred_npz')

        captions.append(caption)
        real_motions.append(gt)
        pred_motions.append(pred)
        lengths.append(min(int(ml), int(pred_ml), args.max_frames))

    n = len(captions)
    print(f'[+] aligned samples: {n}')

    print('[+] Encoding ...')
    t0 = time.time()
    text_emb_real, motion_emb_real = encode_dataset(
        bundle, captions, real_motions, lengths, device,
        batch_size=args.batch_size, max_frames=args.max_frames,
    )
    if args.gt_only:
        text_emb_pred, motion_emb_pred = text_emb_real, motion_emb_real
    else:
        text_emb_pred, motion_emb_pred = encode_dataset(
            bundle, captions, pred_motions, lengths, device,
            batch_size=args.batch_size, max_frames=args.max_frames,
        )
    print(f'    encoding done in {time.time() - t0:.1f}s')

    chunk = args.batch_size
    rp_real_runs, rp_pred_runs = [], []
    ms_real_runs, ms_pred_runs = [], []
    fid_runs, div_real_runs, div_pred_runs = [], [], []

    rng = np.random.default_rng(args.seed)
    for rep in range(args.n_repeats):
        idx = rng.permutation(n)
        rp_real = np.zeros(3)
        rp_pred = np.zeros(3)
        ms_real = 0.0
        ms_pred = 0.0
        nb = 0
        for i in range(0, n // chunk * chunk, chunk):
            j = i + chunk
            sub = idx[i:j]
            tr = text_emb_real[sub]
            mr = motion_emb_real[sub]
            mp_ = motion_emb_pred[sub]
            top_r, ms_r = _r_precision(tr, mr, top_k=3)
            top_p, ms_p = _r_precision(tr, mp_, top_k=3)
            rp_real += top_r
            rp_pred += top_p
            ms_real += ms_r
            ms_pred += ms_p
            nb += 1
        rp_real /= nb * chunk
        rp_pred /= nb * chunk
        ms_real /= nb * chunk
        ms_pred /= nb * chunk

        # FID over the FULL set (not chunked)
        mu_r, c_r = _activation_stats(motion_emb_real)
        mu_p, c_p = _activation_stats(motion_emb_pred)
        fid = _frechet(mu_p, c_p, mu_r, c_r)

        div_real = _diversity(motion_emb_real)
        div_pred = _diversity(motion_emb_pred)

        rp_real_runs.append(rp_real)
        rp_pred_runs.append(rp_pred)
        ms_real_runs.append(ms_real)
        ms_pred_runs.append(ms_pred)
        fid_runs.append(fid)
        div_real_runs.append(div_real)
        div_pred_runs.append(div_pred)
        print(f'    [rep {rep+1}/{args.n_repeats}] '
              f'R-P real={rp_real} pred={rp_pred}  '
              f'MM-Dist real={ms_real:.4f} pred={ms_pred:.4f}  '
              f'FID={fid:.4f}  Div real={div_real:.3f} pred={div_pred:.3f}')

    def _ms(x):
        a = np.asarray(x)
        return float(a.mean()), float(a.std())

    rp_real_arr = np.stack(rp_real_runs)
    rp_pred_arr = np.stack(rp_pred_runs)
    res = {
        'samples': n,
        'n_repeats': args.n_repeats,
        'gt_only': bool(args.gt_only),
        'evaluator': 'motionclip',
        'evaluator_ckpt': str(args.evaluator_ckpt),
        'anno_file': str(args.anno_file),
        'r_precision_real_top1_mean': float(rp_real_arr[:, 0].mean()),
        'r_precision_real_top1_std': float(rp_real_arr[:, 0].std()),
        'r_precision_real_top2_mean': float(rp_real_arr[:, 1].mean()),
        'r_precision_real_top3_mean': float(rp_real_arr[:, 2].mean()),
        'r_precision_pred_top1_mean': float(rp_pred_arr[:, 0].mean()),
        'r_precision_pred_top1_std': float(rp_pred_arr[:, 0].std()),
        'r_precision_pred_top2_mean': float(rp_pred_arr[:, 1].mean()),
        'r_precision_pred_top3_mean': float(rp_pred_arr[:, 2].mean()),
        'mm_dist_real_mean': _ms(ms_real_runs)[0],
        'mm_dist_real_std': _ms(ms_real_runs)[1],
        'mm_dist_pred_mean': _ms(ms_pred_runs)[0],
        'mm_dist_pred_std': _ms(ms_pred_runs)[1],
        'fid_mean': _ms(fid_runs)[0],
        'fid_std': _ms(fid_runs)[1],
        'diversity_real_mean': _ms(div_real_runs)[0],
        'diversity_real_std': _ms(div_real_runs)[1],
        'diversity_pred_mean': _ms(div_pred_runs)[0],
        'diversity_pred_std': _ms(div_pred_runs)[1],
    }
    print()
    print('===== Final =====')
    for k, v in res.items():
        if isinstance(v, float):
            print(f'  {k}: {v:.4f}')
        else:
            print(f'  {k}: {v}')

    out_json.write_text(json.dumps(res, indent=2))
    print(f'[+] wrote {out_json}')


if __name__ == '__main__':
    main()
