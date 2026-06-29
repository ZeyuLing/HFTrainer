#!/usr/bin/env python3
"""Batch evaluation script for PRISM KAFS ablation experiments.

Generates motions from a PRISM model under different KAFS (Kinematic-Adaptive
Flow Scheduling) modes and saves them as per-sample NPZ files for downstream
metric computation.

Usage:
    python scripts/eval/eval_prism_kafs_ablation.py \
        --config configs/prism/prism_1b_tp2m_multiframe.py \
        --checkpoint work_dirs/prism_1b_tp2m_multiframe/checkpoint-iter_15000 \
        --kafs-mode depth_driven \
        --anno-file data/annotation/test_hml3d.json \
        --data-dir data/motionhub \
        --output-dir work_dirs/prism_kafs_ablation \
        --num-inference-steps 50 \
        --max-samples 100

KAFS modes:
    none          — Standard baseline (no per-joint timestep scaling)
    depth_driven  — Per-joint scaling based on kinematic tree depth
    uniform       — All joints get alpha=1.0 (equivalent to baseline, sanity check)
    random        — Random alphas in [0.85, 1.15] for ablation control
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

# ---------------------------------------------------------------------------
# Ensure hftrainer is importable
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
HF_ROOT = SCRIPT_DIR.parent.parent  # scripts/eval/.. -> scripts/.. -> hf_trainer/
sys.path.insert(0, str(HF_ROOT))


# ---------------------------------------------------------------------------
# Caption loader (mirrors eval_with_motionclip_evaluator.py)
# ---------------------------------------------------------------------------

def _load_caption(caption_path: Path) -> Optional[str]:
    """Load a single randomly chosen caption from hierarchical or hymotion format."""
    if not caption_path.exists():
        return None
    try:
        data = json.loads(caption_path.read_text())
    except Exception:
        return None
    if not isinstance(data, dict):
        return None

    # Hierarchical format: {macro: [...], meso: [...], micro: [...]}
    if all(k in data and isinstance(data[k], list) for k in ('macro', 'meso', 'micro')):
        pool = []
        for g in ('macro', 'meso', 'micro'):
            for c in data[g]:
                if isinstance(c, str) and c.strip():
                    pool.append(c.strip())
        return random.choice(pool) if pool else None

    # HYMotion format: {result: [{short_caption_rewritten: [...], ...}]}
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


# ---------------------------------------------------------------------------
# Annotation / test sample loading
# ---------------------------------------------------------------------------

def load_test_samples(
    anno_file: Path,
    data_dir: Path,
    motion_key: str = 'smplx',
    caption_key: str = 'hierarchical_caption',
    min_frames: int = 24,
    max_frames: int = 360,
    max_samples: Optional[int] = None,
    require_motion_file: bool = True,
    keep_ids: Optional[set[str]] = None,
    shard_idx: Optional[int] = None,
    num_shards: int = 1,
) -> List[Dict]:
    """Load test samples from a motionhub-format annotation JSON.

    Returns list of dicts with keys:
        name, caption, num_frames, motion_path, caption_path
    """
    raw = json.loads(anno_file.read_text())

    # Parse annotation: dict with data_list or flat list
    if isinstance(raw, dict) and 'data_list' in raw:
        dl = raw['data_list']
        if isinstance(dl, dict):
            entries = list(dl.items())
        else:
            entries = [
                (e.get('motion_id') or e.get('id') or str(i), e)
                for i, e in enumerate(dl)
            ]
    elif isinstance(raw, list):
        entries = [
            (e.get('motion_id') or e.get('id') or str(i), e)
            for i, e in enumerate(raw)
        ]
    else:
        raise ValueError(f'Unrecognized annotation format in {anno_file}')

    def resolve_data_path(rel_path: str) -> Path:
        raw_path = Path(rel_path)
        if raw_path.is_absolute():
            return raw_path
        data_path = Path(data_dir) / raw_path
        if data_path.exists():
            return data_path
        repo_path = HF_ROOT / raw_path
        if repo_path.exists():
            return repo_path
        return data_path

    samples = []
    candidate_idx = 0
    for name, entry in entries:
        if keep_ids is not None and name not in keep_ids:
            continue

        m_rel = entry.get(f'{motion_key}_path')
        c_rel = entry.get(f'{caption_key}_path')
        num_frames = entry.get('num_frames')
        if not (m_rel and c_rel and num_frames):
            continue

        num_frames = int(num_frames)
        if num_frames < min_frames:
            continue
        num_frames = min(num_frames, max_frames)

        if shard_idx is not None and num_shards > 1:
            if candidate_idx % num_shards != shard_idx:
                candidate_idx += 1
                continue
            candidate_idx += 1

        m_path = resolve_data_path(m_rel)
        c_path = resolve_data_path(c_rel)

        caption = _load_caption(c_path)
        if caption is None:
            continue

        if require_motion_file and not m_path.exists():
            continue

        samples.append({
            'name': name,
            'caption': caption,
            'num_frames': num_frames,
            'motion_path': str(m_path),
        })

        if max_samples and len(samples) >= max_samples:
            break

    return samples


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_prism_bundle(config_path: str, checkpoint_dir: str, device: torch.device):
    """Load PrismBundle from config + checkpoint directory.

    The checkpoint directory should contain model.pt (the standard
    AccelerateRunner checkpoint format with per-module state dicts).
    """
    # PRISM generation needs model-bundle registration.  Some evaluation entry
    # points set this flag to speed up lightweight metric imports, so clear it
    # locally before importing the registry.
    os.environ.pop("HFTRAINER_SKIP_AUTOREGISTER", None)
    from mmengine.config import Config
    import hftrainer
    from hftrainer.registry import MODEL_BUNDLES

    hftrainer.register_all_modules()
    # Some Taiji V100 images are intentionally thin.  The package-level
    # auto-registration swallows optional import failures, so explicitly import
    # the modules required by the PRISM config before building the bundle.
    import hftrainer.motion.processing.smpl_processor  # noqa: F401
    import hftrainer.models.motion.prism.autoencoder_kl_2d  # noqa: F401
    import hftrainer.models.motion.prism.network.transformer_prism  # noqa: F401
    import hftrainer.models.motion.prism.bundle  # noqa: F401

    cfg = Config.fromfile(config_path)
    bundle = MODEL_BUNDLES.build(cfg.model)

    # Load checkpoint (model.pt is a dict keyed by module name)
    ckpt_path = os.path.join(checkpoint_dir, 'model.pt')
    if os.path.isfile(ckpt_path):
        print(f'[+] Loading checkpoint from {ckpt_path}')
        state_dict = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        bundle.load_state_dict_selective(state_dict, strict=False)
    else:
        # Fallback: try model.safetensors (only has transformer weights)
        st_path = os.path.join(checkpoint_dir, 'model.safetensors')
        if os.path.isfile(st_path):
            print(f'[+] Loading safetensors from {st_path}')
            from safetensors.torch import load_file
            st_dict = load_file(st_path)
            bundle.transformer.load_state_dict(st_dict, strict=False)
        else:
            raise FileNotFoundError(
                f'No model.pt or model.safetensors found in {checkpoint_dir}'
            )

    bundle = bundle.eval().to(device)
    return bundle


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_motion(
    pipeline,
    caption: str,
    num_frames: int,
    num_inference_steps: int = 50,
    guidance_scale: float = 5.0,
    use_rollout_trans: bool = True,
    length_policy: str = 'pad360_crop',
    pad_to_frames: int = 360,
) -> Dict:
    """Generate a single motion sample via the PRISM pipeline.

    Returns the smplx_dict from pipeline output.
    """
    if length_policy == 'direct_len':
        generation_num_frames = num_frames
        valid_num_frames = num_frames
    elif length_policy == 'pad360_crop':
        generation_num_frames = pad_to_frames
        valid_num_frames = num_frames
    else:
        raise ValueError(f'Unsupported length_policy: {length_policy}')

    output = pipeline(
        prompts=caption,
        num_frames_per_segment=num_frames,
        generation_num_frames_per_segment=generation_num_frames,
        valid_num_frames_per_segment=valid_num_frames,
        preserve_segment_lengths=True,
        allow_segment_padding=False,
        # Direct-length still needs the minimal VAE-compatible canvas.  The
        # valid mask stays at the official length and the decoded result is
        # cropped back to GT length, with no last-frame repair.
        align_generation_frames=(length_policy == 'direct_len'),
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        use_rollout_trans=use_rollout_trans,
    )
    return output


# ---------------------------------------------------------------------------
# Save utility
# ---------------------------------------------------------------------------

def save_smplx_npz(out_path: str, smplx_dict: Dict):
    """Save smplx_dict to compressed NPZ."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    pack = {}
    for k, v in smplx_dict.items():
        if isinstance(v, np.ndarray):
            if k.startswith('_prism_'):
                pack[k] = v
            else:
                pack[k] = v.astype(np.float32, copy=False)
        elif isinstance(v, torch.Tensor):
            arr = v.detach().cpu().numpy()
            pack[k] = arr if k.startswith('_prism_') else arr.astype(np.float32)
        else:
            pack[k] = v
    np.savez_compressed(out_path, **pack)


def _fit_first_dim(arr, target_len: int):
    """Trim or last-frame-pad arrays/tensors whose first dim is time."""
    if not hasattr(arr, "shape") or len(arr.shape) == 0:
        return arr
    cur = int(arr.shape[0])
    if cur == target_len:
        return arr
    if cur > target_len:
        return arr[:target_len]
    if cur <= 0:
        return arr
    pad_n = target_len - cur
    if isinstance(arr, torch.Tensor):
        pad = arr[-1:].repeat((pad_n,) + (1,) * (arr.ndim - 1))
        return torch.cat([arr, pad], dim=0)
    if isinstance(arr, np.ndarray):
        pad = np.repeat(arr[-1:], pad_n, axis=0)
        return np.concatenate([arr, pad], axis=0)
    return arr


def enforce_smplx_num_frames(smplx_dict: Dict, target_len: int) -> Dict:
    """Make generated SMPL-X fields exactly match the requested frame count.

    PRISM's latent decoder can return an off-by-one/two sequence when the target
    length is not aligned with the VAE temporal stride.  For protocol-consistent
    HumanML3D evaluation and visualization, every time-indexed output field must
    match the official GT length exactly.
    """
    out = dict(smplx_dict)
    for key, value in list(out.items()):
        out[key] = _fit_first_dim(value, target_len)
    return out


def _time_len(value) -> int:
    if isinstance(value, np.ndarray):
        return int(value.shape[0])
    if isinstance(value, torch.Tensor):
        return int(value.shape[0])
    return -1


def get_smplx_num_frames(smplx_dict: Dict) -> int:
    for key in ('transl', 'trans', 'global_orient', 'body_pose', 'poses'):
        if key in smplx_dict:
            n = _time_len(smplx_dict[key])
            if n >= 0:
                return n
    return -1


def _meta_int(smplx_dict: Dict, key: str, default: int = -1) -> int:
    value = smplx_dict.get(key)
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray) and value.size:
        return int(value.reshape(-1)[0])
    return default


def validate_smplx_length(smplx_dict: Dict, target_len: int, sample_name: str) -> int:
    out_len = get_smplx_num_frames(smplx_dict)
    if out_len != int(target_len):
        raise ValueError(
            f'Length mismatch for {sample_name}: final_len={out_len}, '
            f'official_gt_len={target_len}. Refusing to save benchmark output.'
        )
    return out_len


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='PRISM KAFS ablation evaluation: generate motions under different KAFS modes.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--config', type=str,
        default='configs/prism/prism_1b_tp2m_multiframe.py',
        help='Path to PRISM config file.',
    )
    parser.add_argument(
        '--checkpoint', type=str,
        default='work_dirs/prism_1b_tp2m_multiframe/checkpoint-iter_15000',
        help='Path to checkpoint directory (containing model.pt or model.safetensors).',
    )
    parser.add_argument(
        '--kafs-mode', type=str, default='none',
        choices=['none', 'depth_driven', 'uniform', 'random'],
        help='KAFS mode for per-joint adaptive timestep scaling.',
    )
    parser.add_argument(
        '--kafs-alpha', type=str, default=None,
        help='Comma-separated 23 alpha values -> custom KAFS mode (overrides '
             '--kafs-mode). Order: [trans,pelvis,L_hip,R_hip,spine1,L_knee,'
             'R_knee,spine2,L_ankle,R_ankle,spine3,L_foot,R_foot,neck,L_collar,'
             'R_collar,head,L_shoulder,R_shoulder,L_elbow,R_elbow,L_wrist,R_wrist].',
    )
    parser.add_argument(
        '--out-subdir', type=str, default=None,
        help='Override output subdir name (default = kafs-mode). Use to keep '
             'multiple custom-alpha sweeps separate under --output-dir.',
    )
    parser.add_argument(
        '--anno-file', type=str,
        default='data/annotation/test_hml3d.json',
        help='Annotation JSON file (motionhub format).',
    )
    parser.add_argument(
        '--data-dir', type=str,
        default='data/motionhub',
        help='Root data directory for resolving relative paths in annotation.',
    )
    parser.add_argument(
        '--skip-motion-existence-check', action='store_true',
        help='Do not require each annotation motion path to exist when loading '
             'T2M samples. This is useful for official HumanML3D-272 annotations '
             'where captions and target lengths are sufficient for generation.',
    )
    parser.add_argument(
        '--output-dir', type=str,
        default='work_dirs/prism_kafs_ablation',
        help='Root output directory. Results saved under <output-dir>/<kafs-mode>/.',
    )
    parser.add_argument(
        '--num-samples', type=int, default=None,
        help='Total number of samples to generate. None = all available.',
    )
    parser.add_argument(
        '--max-samples', type=int, default=None,
        help='Limit number of samples loaded from annotation (for debugging).',
    )
    parser.add_argument(
        '--id-file', type=str, default=None,
        help='Optional newline-separated sample ids to keep after loading annotations.',
    )
    parser.add_argument(
        '--num-inference-steps', type=int, default=50,
        help='Number of denoising steps per segment.',
    )
    parser.add_argument(
        '--guidance-scale', type=float, default=5.0,
        help='Classifier-free guidance scale.',
    )
    parser.add_argument(
        '--batch-size', type=int, default=1,
        help='Batch size for generation (currently only batch_size=1 is supported).',
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed for reproducibility.',
    )
    parser.add_argument(
        '--motion-key', type=str, default='smplx',
        help='Key prefix for motion path in annotation (e.g. smplx -> smplx_path).',
    )
    parser.add_argument(
        '--caption-key', type=str, default='hierarchical_caption',
        help='Key prefix for caption path in annotation.',
    )
    parser.add_argument(
        '--rewritten-caption-file', type=str, default=None,
        help='Optional {motion_id: caption} JSON. When set, overrides each sample\'s '
             'generation caption with the rewritten one (matched by name). Frame counts '
             'and motion paths still come from --anno-file. This matches the main-table '
             'protocol: GENERATE on rewritten captions, but EVALUATE metrics against the '
             'original captions in --anno-file.',
    )
    parser.add_argument(
        '--min-frames', type=int, default=24,
        help='Minimum number of frames to include a sample.',
    )
    parser.add_argument(
        '--max-frames', type=int, default=360,
        help='Maximum number of frames per sample.',
    )
    parser.add_argument(
        '--num-shards', type=int, default=1,
        help='Total number of parallel shards (for multi-GPU sharding).',
    )
    parser.add_argument(
        '--shard-idx', type=int, default=0,
        help='Index of this shard in [0, num_shards). Samples are split by stride.',
    )
    parser.add_argument(
        '--skip-existing', action='store_true',
        help='Skip samples whose output NPZ already exists (resume-friendly).',
    )
    parser.add_argument(
        '--smooth-output', action='store_true',
        help='Apply HY-Motion-style temporal smoothing to generated SMPLX output '
             'before saving (quaternion Gaussian rotations + Savitzky-Golay root).',
    )
    parser.add_argument(
        '--translation-decode-mode',
        choices=['rollout', 'absolute'],
        default='rollout',
        help='How to decode PRISM abs_rel translation channels. rollout uses '
             'initial absolute translation plus cumulative relative deltas '
             '(previous default); absolute uses decoded absolute channels directly.',
    )
    parser.add_argument(
        '--length-policy',
        choices=['direct_len', 'pad360_crop'],
        default='pad360_crop',
        help='Official benchmark length policy. pad360_crop generates a fixed '
             '360-frame training canvas and masks/crops back to the official '
             'length. direct_len is retained only for exact-length ablations.',
    )
    parser.add_argument(
        '--pad-to-frames',
        type=int,
        default=360,
        help='Generation canvas length for --length-policy pad360_crop.',
    )
    args = parser.parse_args()

    # ---- Seed ----
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # ---- Device ----
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[+] Device: {device}')
    print(f'[+] KAFS mode: {args.kafs_mode}')
    print(f'[+] Length policy: {args.length_policy} (pad_to_frames={args.pad_to_frames})')
    if args.smooth_output:
        print('[+] Output smoothing: HY-Motion-style temporal smoothing enabled')

    # ---- Load test samples ----
    anno_path = Path(args.anno_file)
    data_path = Path(args.data_dir)
    print(f'[+] Loading test samples from {anno_path} ...')

    keep_ids = None
    if args.id_file:
        keep_ids = {
            line.strip()
            for line in Path(args.id_file).read_text().splitlines()
            if line.strip()
        }
        print(f'    ID prefilter enabled: {len(keep_ids)} ids from {args.id_file}')

    loader_shard_idx = None
    loader_num_shards = 1
    if keep_ids is not None and args.num_shards > 1:
        loader_shard_idx = args.shard_idx
        loader_num_shards = args.num_shards
        print(f'    Loader shard prefilter enabled: '
              f'{loader_shard_idx}/{loader_num_shards}')

    samples = load_test_samples(
        anno_file=anno_path,
        data_dir=data_path,
        motion_key=args.motion_key,
        caption_key=args.caption_key,
        min_frames=args.min_frames,
        max_frames=args.max_frames,
        max_samples=args.max_samples,
        require_motion_file=not args.skip_motion_existence_check,
        keep_ids=keep_ids,
        shard_idx=loader_shard_idx,
        num_shards=loader_num_shards,
    )
    print(f'    Loaded {len(samples)} test samples')
    if not samples:
        raise RuntimeError(
            f'No valid test samples found. Check --anno-file ({args.anno_file}) '
            f'and --data-dir ({args.data_dir}).'
        )

    if args.id_file:
        print(f'    ID filter: kept {len(samples)} samples from {args.id_file}')
        if not samples:
            raise RuntimeError(f'No samples left after --id-file filter: {args.id_file}')

    # ---- Optional: override GENERATION captions with rewritten ones ----
    # (metrics are still computed against --anno-file's original captions downstream)
    if args.rewritten_caption_file:
        rw_raw = json.loads(Path(args.rewritten_caption_file).read_text())
        rw_map = rw_raw['data_list'] if isinstance(rw_raw, dict) and 'data_list' in rw_raw else rw_raw
        n_over = 0
        kept = []
        for s in samples:
            cap = rw_map.get(s['name'])
            if isinstance(cap, dict):  # tolerate nested {caption: ...}
                cap = cap.get('caption') or cap.get('text')
            if isinstance(cap, str) and cap.strip():
                s['caption'] = cap.strip()
                n_over += 1
                kept.append(s)
            # samples without a rewritten caption are dropped to stay consistent
            # with the main-table rewritten test set
        samples = kept
        print(f'    Rewritten-caption override: {n_over} samples kept '
              f'(from {args.rewritten_caption_file})')
        if not samples:
            raise RuntimeError('No samples left after rewritten-caption override; '
                               'check motion_id keys match between files.')

    # Multi-GPU sharding: keep a deterministic stride subset for this shard.
    if args.num_shards > 1 and loader_shard_idx is None:
        samples = samples[args.shard_idx::args.num_shards]
        print(f'    Shard {args.shard_idx}/{args.num_shards}: {len(samples)} samples')
    elif args.num_shards > 1:
        print(f'    Shard {args.shard_idx}/{args.num_shards}: {len(samples)} samples')

    # Optionally limit number of samples to generate
    if args.num_samples is not None and args.num_samples < len(samples):
        samples = samples[:args.num_samples]
        print(f'    Limited to {len(samples)} samples (--num-samples)')

    # ---- Load model ----
    print(f'[+] Loading PRISM bundle from config: {args.config}')
    print(f'    Checkpoint: {args.checkpoint}')
    t_load = time.time()
    bundle = load_prism_bundle(args.config, args.checkpoint, device)
    print(f'    Bundle loaded in {time.time() - t_load:.1f}s')

    # ---- Build pipeline ----
    from hftrainer.pipelines.motion.prism_pipeline import PrismPipeline
    pipeline = PrismPipeline(bundle=bundle)

    # ---- Set KAFS mode ----
    if args.kafs_alpha:
        import torch as _torch
        _vals = [float(x) for x in args.kafs_alpha.split(',') if x.strip() != '']
        assert len(_vals) == 23, f'--kafs-alpha needs 23 values, got {len(_vals)}'
        pipeline.backend.set_kafs_alpha(
            mode='custom', alpha_vals=_torch.tensor(_vals))
        print(f'[+] KAFS custom alpha: {_vals}')
    else:
        pipeline.backend.set_kafs_alpha(mode=args.kafs_mode)

    # ---- Output directory ----
    _subdir = args.out_subdir if args.out_subdir else args.kafs_mode
    out_dir = Path(args.output_dir) / _subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f'[+] Output directory: {out_dir}')

    # Save run metadata
    meta = {
        'config': args.config,
        'checkpoint': args.checkpoint,
        'kafs_mode': args.kafs_mode,
        'anno_file': args.anno_file,
        'data_dir': args.data_dir,
        'num_inference_steps': args.num_inference_steps,
        'guidance_scale': args.guidance_scale,
        'smooth_output': bool(args.smooth_output),
        'translation_decode_mode': args.translation_decode_mode,
        'length_policy': args.length_policy,
        'pad_to_frames': args.pad_to_frames,
        'strict_length': True,
        'seed': args.seed,
        'num_samples': len(samples),
    }
    if args.num_shards > 1:
        meta_path = out_dir / f'run_meta_shard{args.shard_idx}of{args.num_shards}.json'
    else:
        meta_path = out_dir / 'run_meta.json'
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f'    Saved run metadata to {meta_path}')

    # ---- Generate motions ----
    print(f'[+] Generating {len(samples)} motions (steps={args.num_inference_steps}, '
          f'guidance={args.guidance_scale}) ...')
    t_start = time.time()
    n_success = 0
    n_fail = 0
    results_manifest = []

    for i, sample in enumerate(samples):
        sample_name = sample['name']
        caption = sample['caption']
        num_frames = sample['num_frames']
        out_path = out_dir / f'{sample_name}.npz'

        if args.skip_existing and out_path.exists():
            try:
                existing = np.load(out_path, allow_pickle=True)
                existing_dict = {k: existing[k] for k in existing.files}
                out_frames = validate_smplx_length(existing_dict, num_frames, sample_name)
                results_manifest.append({
                    'name': sample_name,
                    'caption': caption,
                    'official_gt_num_frames': num_frames,
                    'requested_num_frames': _meta_int(existing_dict, '_prism_requested_num_frames', num_frames),
                    'generation_num_frames': _meta_int(existing_dict, '_prism_generation_num_frames', num_frames),
                    'valid_num_frames': _meta_int(existing_dict, '_prism_valid_num_frames', num_frames),
                    'raw_decoded_num_frames': _meta_int(existing_dict, '_prism_raw_decoded_num_frames', out_frames),
                    'pretrim_num_frames': _meta_int(existing_dict, '_prism_pretrim_num_frames', out_frames),
                    'final_num_frames': out_frames,
                    'length_policy': args.length_policy,
                    'npz_path': str(out_path),
                    'status': 'skipped_existing',
                })
                n_success += 1
                continue
            except Exception as e:
                print(f'  [!] Existing output invalid for {sample_name}: {e}; regenerating')

        if args.length_policy == 'pad360_crop' and num_frames > args.pad_to_frames:
            results_manifest.append({
                'name': sample_name,
                'caption': caption,
                'official_gt_num_frames': num_frames,
                'requested_num_frames': num_frames,
                'generation_num_frames': args.pad_to_frames,
                'valid_num_frames': num_frames,
                'raw_decoded_num_frames': -1,
                'pretrim_num_frames': -1,
                'final_num_frames': -1,
                'length_policy': args.length_policy,
                'npz_path': '',
                'status': (
                    f'error: official length {num_frames} exceeds '
                    f'pad_to_frames={args.pad_to_frames}'
                ),
            })
            n_fail += 1
            continue

        try:
            smplx_dict = generate_motion(
                pipeline=pipeline,
                caption=caption,
                num_frames=num_frames,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                use_rollout_trans=(args.translation_decode_mode == 'rollout'),
                length_policy=args.length_policy,
                pad_to_frames=args.pad_to_frames,
            )
            if args.smooth_output:
                from hftrainer.motion.processing.temporal_smoothing import (
                    smooth_smplx_dict_hymotion,
                )
                smplx_dict = smooth_smplx_dict_hymotion(smplx_dict)
            out_frames = validate_smplx_length(smplx_dict, num_frames, sample_name)

            # Save output NPZ
            save_smplx_npz(str(out_path), smplx_dict)

            results_manifest.append({
                'name': sample_name,
                'caption': caption,
                'official_gt_num_frames': num_frames,
                'requested_num_frames': _meta_int(smplx_dict, '_prism_requested_num_frames', num_frames),
                'generation_num_frames': _meta_int(
                    smplx_dict,
                    '_prism_generation_num_frames',
                    args.pad_to_frames if args.length_policy == 'pad360_crop' else num_frames,
                ),
                'valid_num_frames': _meta_int(smplx_dict, '_prism_valid_num_frames', num_frames),
                'raw_decoded_num_frames': _meta_int(smplx_dict, '_prism_raw_decoded_num_frames', out_frames),
                'pretrim_num_frames': _meta_int(smplx_dict, '_prism_pretrim_num_frames', out_frames),
                'final_num_frames': out_frames,
                'length_policy': args.length_policy,
                'npz_path': str(out_path),
                'status': 'success',
            })
            n_success += 1

        except Exception as e:
            print(f'  [!] Failed on sample {sample_name}: {e}')
            results_manifest.append({
                'name': sample_name,
                'caption': caption,
                'official_gt_num_frames': num_frames,
                'requested_num_frames': num_frames,
                'generation_num_frames': args.pad_to_frames if args.length_policy == 'pad360_crop' else num_frames,
                'valid_num_frames': num_frames,
                'raw_decoded_num_frames': -1,
                'pretrim_num_frames': -1,
                'final_num_frames': -1,
                'length_policy': args.length_policy,
                'npz_path': '',
                'status': f'error: {e}',
            })
            n_fail += 1

        # Progress report every 10 samples
        if (i + 1) % 10 == 0 or (i + 1) == len(samples):
            elapsed = time.time() - t_start
            avg_time = elapsed / (i + 1)
            eta = avg_time * (len(samples) - i - 1)
            print(f'  [{i + 1}/{len(samples)}] '
                  f'success={n_success} fail={n_fail} '
                  f'elapsed={elapsed:.1f}s avg={avg_time:.2f}s/sample '
                  f'ETA={eta:.0f}s')

    # ---- Summary ----
    total_time = time.time() - t_start
    print()
    print('=' * 60)
    print(f'PRISM KAFS Ablation — {args.kafs_mode}')
    print(f'  Total samples:  {len(samples)}')
    print(f'  Successful:     {n_success}')
    print(f'  Failed:         {n_fail}')
    print(f'  Total time:     {total_time:.1f}s')
    if n_success > 0:
        print(f'  Avg time/sample: {total_time / n_success:.2f}s')
    print(f'  Output dir:     {out_dir}')
    print('=' * 60)

    # Save manifest (shard-aware to avoid collisions across parallel shards)
    if args.num_shards > 1:
        manifest_path = out_dir / f'manifest_shard{args.shard_idx}of{args.num_shards}.json'
    else:
        manifest_path = out_dir / 'manifest.json'
    manifest_path.write_text(json.dumps(results_manifest, indent=2))
    print(f'[+] Saved manifest ({len(results_manifest)} entries) to {manifest_path}')


if __name__ == '__main__':
    main()
