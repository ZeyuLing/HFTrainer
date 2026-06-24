"""Run OUR M2M model as the automatic motion-repair method on BrokenAMASS*,
producing a StableMotion-format ``results.npy`` so the SAME official eval
(``eval.eval_scripts``) scores ours, StableMotion and corrupted input
identically.

Pipeline (per clip), reusing the validated conversion helpers in
``scripts/run_stablemotion_e9.py``:

    SM corrupted smpldata  (z-up canonical, 20fps)         ← results.npy['motion'][i]
      → z-up→y-up  +  smpldata→m2m135  (subtract bone_offsets[0])
      → resample 20→30 fps (slerp on rot6d, linear on trans)
      → evaluate_sample(E9, automatic QC defect mask, inpaint)   ← OUR model
      → repaired motion_135 (30fps)
      → resample 30→20 fps
      → m2m135→smpldata(24)  +  y-up→z-up   (back into SM canonical frame)
      → store as motion_fix[i]

Then ``{motion: SM corrupted, motion_fix: ours, lengths}`` is written and the
official ``eval.eval_scripts`` is launched (TMR optional).

The repaired clip is produced in the SAME canonical frame as the SM corrupted
input (all transforms are reversible), so it is directly comparable against the
clean GT collected by StableMotion's ``--collect_dataset``.

Usage:
    CUDA_VISIBLE_DEVICES=1 python3 scripts/eval/run_ours_repair_brokenamass.py \
        --sm-results ref_repo/StableMotion/output/brokenamass_star_sm_enhanced/results.npy \
        --gt ref_repo/StableMotion/output/brokenamass_star_clean_v2/results_collected.npy \
        --output-dir ref_repo/StableMotion/output/brokenamass_star_ours \
        --max-samples 9999
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

# Validated 135 <-> smpldata <-> fps conversion helpers (StableMotion E9 wrapper)
from scripts.run_stablemotion_e9 import (  # noqa: E402
    smpldata_to_m2m135,
    m2m135_to_smpldata_24,
    _resample_motion135_slerp,
    _upright_fix_135,
)

# Validated per-sample M2M repair (mask build + normalize + pad + pipeline +
# denorm + 198->135 extraction), and the model registry entry.
from scripts.eval.eval_m2m_v2_all_tasks import (  # noqa: E402
    evaluate_sample,
    V2_MODELS,
    MOTION_DIM_V2,
)


def _to_torch(x):
    if isinstance(x, torch.Tensor):
        return x.float()
    return torch.from_numpy(np.asarray(x)).float()


def build_model(model_name: str, device: str):
    """Mirror eval_m2m_v2_all_tasks.load_model but resolve the (moved) config
    path under configs/hymotion_m2m_v2/."""
    from mmengine.config import Config
    from hftrainer.registry import MODEL_BUNDLES
    from hftrainer.utils.checkpoint_utils import (
        load_checkpoint, find_latest_checkpoint,
    )
    from hftrainer.pipelines.motion.hymotion_m2m_pipeline import (
        HyMotionM2MPipeline,
    )
    import hftrainer.models.motion  # noqa: F401 — registers HyMotionM2MBundle

    info = dict(V2_MODELS[model_name])
    cfg_path = info['config']
    if not os.path.exists(cfg_path):
        # registry path is stale (configs/hymotion_m2m/...); real file lives
        # under configs/hymotion_m2m_v2/hymotion_m2m_v2_<name>.py
        base = os.path.basename(cfg_path)
        alt = os.path.join('configs/hymotion_m2m_v2',
                           base.replace('hymotion_m2m_', 'hymotion_m2m_v2_', 1))
        if os.path.exists(alt):
            cfg_path = alt
        else:
            raise FileNotFoundError(f'config not found: {cfg_path} / {alt}')
    print(f'[model] config={cfg_path}')
    cfg = Config.fromfile(cfg_path)
    bundle = MODEL_BUNDLES.build(cfg.model.to_dict())
    ckpt = find_latest_checkpoint(info['work_dir'])
    assert ckpt is not None, f'no checkpoint under {info["work_dir"]}'
    print(f'[model] checkpoint={ckpt}')
    sd = load_checkpoint(ckpt, map_location='cpu')
    bundle.load_state_dict_selective(sd)
    del sd
    bundle.eval().to(device)
    pipeline = HyMotionM2MPipeline(
        bundle=bundle, num_steps=50, replacement_guidance='none',
    )
    model_info = {
        'motion_dim': MOTION_DIM_V2,
        'rotation_space': info.get('rotation_space', 'local'),
        'has_caption': info.get('has_caption', False),
    }
    return bundle, pipeline, model_info, ckpt


def build_e9_qc_task(sdedit_tau: float = 0.0):
    """E9 repair task with automatic-mask settings (inpaint mode).

    Two settings:
      AUTO_qc       - QC-rule per-joint defect mask (under-detects dynamic
                      BrokenAMASS corruption; kept for reference/ablation).
      AUTO_adaptive - MoGenDIT change-based adaptive mask (the intended
                      method). Loaded from the adaptive_masks_mogendit cache
                      keyed by sample['path']; generate it first with
                      scripts/eval/gen_mogendit_masks_brokenamass.py.

    ``sdedit_tau`` > 0 turns the masked-region repair from full regeneration
    (τ=0, noise to t=T) into SDEdit-style partial noising (start denoising from
    σ(τ)). This keeps masked cells close to the corrupted input while still
    cleaning jitter — appropriate when corruption is mild and full regeneration
    invents GT-deviating motion. Requires replacement_guidance='skip_last' so
    the pipeline takes its SDEdit branch (use_replacement=True).
    """
    from hftrainer.evaluation.motion.m2m_eval_tasks import get_task, TaskSetting
    task = get_task('E9')
    task.settings['AUTO_qc'] = TaskSetting(
        'AUTO_qc',
        'Automatic repair: QC-detected per-joint defect mask, inpaint mode.',
        {
            '_qc_defect_mask': True,
            '_qc_dilate_temp': 2,
            '_qc_dilate_spatial': True,
            '_qc_include_borderline': True,
            '_editing_mode': False,
        },
    )
    adaptive_kwargs = {
        '_use_adaptive_mask': True,
        '_editing_mode': False,
    }
    if sdedit_tau and sdedit_tau > 0:
        adaptive_kwargs['_sdedit_tau'] = float(sdedit_tau)
        adaptive_kwargs['_replacement_guidance'] = 'skip_last'
    task.settings['AUTO_adaptive'] = TaskSetting(
        'AUTO_adaptive',
        'Automatic repair: MoGenDIT change-based adaptive mask, inpaint mode.',
        adaptive_kwargs,
    )
    # Canonical dashboard repair method (m2m_eval_tasks.py E9
    # D_strict_mask_d2_b3_bsmooth_combo). A_adaptive_inpaint above is
    # DEPRECATED/disabled there: the raw MoGenDIT mask is user-confirmed
    # inaccurate. The strict path instead (1) TIGHTENS the cached mask
    # (kinematic spatial + temporal dilate=2, suppress blobs < 3 frames×joints)
    # so only "definitely defective" cells are touched, (2) runs a standard
    # skip_last inpaint with clean_motion=LQ at τ=0 (unmasked locked to LQ),
    # and (3) applies boundary smoothing + Savitzky-Golay + accel-spike median
    # to kill the velocity spikes at mask 0↔1 transitions.
    task.settings['AUTO_strict'] = TaskSetting(
        'AUTO_strict',
        'Canonical D_strict_mask_d2_b3_bsmooth_combo repair.',
        {
            '_strict_adaptive_mask': True,
            '_strict_dilate': 2,
            '_strict_min_blob': 3,
            '_editing_mode': False,
            '_presmooth_clean_sigma': 1.0,
            '_boundary_smooth_radius': 3,
            '_boundary_smooth_sigma': 2.0,
            '_accel_spike_k': 3.0,
            '_savgol_window': 7,
            '_savgol_poly': 3,
        },
    )
    # Best-of-both for a GT-based benchmark: strict (tight, reliable) mask
    # like D_strict, but SDEdit τ=0.5 partial-noise instead of τ=0 full
    # regeneration so masked global-drift frames stay near the input (avoids
    # the ~50cm root-trajectory drift D_strict shows on BrokenAMASS*). Local
    # boundary smoothing kept; the GLOBAL savgol7/accel-spike from
    # bsmooth_combo is dropped because it smooths the whole root trajectory.
    task.settings['AUTO_strict_sdedit'] = TaskSetting(
        'AUTO_strict_sdedit',
        'Strict mask + lock root + SDEdit τ=0.5 + local boundary smooth.',
        {
            '_strict_adaptive_mask': True,
            '_strict_dilate': 2,
            '_strict_min_blob': 3,
            '_strict_lock_trans': True,
            '_editing_mode': False,
            '_sdedit_tau': 0.5,
            '_replacement_guidance': 'skip_last',
            '_presmooth_clean_sigma': 1.0,
            '_boundary_smooth_radius': 3,
            '_boundary_smooth_sigma': 2.0,
        },
    )
    return task


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sm-results', type=str, required=True,
                    help='StableMotion results.npy (provides corrupted motion).')
    ap.add_argument('--gt', type=str, required=True,
                    help='clean GT results_collected.npy (for the launched eval).')
    ap.add_argument('--output-dir', type=str, required=True)
    ap.add_argument('--model-name', type=str, default='uncond_local')
    ap.add_argument('--max-samples', type=int, default=9999)
    ap.add_argument('--device', type=str, default='cuda')
    ap.add_argument('--num-steps', type=int, default=50)
    ap.add_argument('--replacement-guidance', type=str, default='skip_last')
    ap.add_argument('--src-fps', type=float, default=20.0,
                    help='fps of the SM smpldata clips (StableMotion = 20).')
    ap.add_argument('--work-fps', type=float, default=30.0,
                    help='fps our M2M model operates at.')
    ap.add_argument('--identity-check', action='store_true',
                    help='Skip M2M; just round-trip corrupted→135→smpldata to '
                         'validate the conversion is near-identity vs corrupted '
                         'baseline.')
    ap.add_argument('--seed-base', type=int, default=1234)
    # ── HyMotionM2MPipeline.infer_repair axes (single canonical entry) ──
    ap.add_argument('--mask-source', choices=['provided', 'self_denoise'],
                    default='provided',
                    help="axis 2: 'provided' = MoGenDIT cached mask (proven "
                         "best); 'self_denoise' = our model's own SDEdit-from-LQ "
                         'change detection.')
    ap.add_argument('--translation-mode', choices=['lock', 'detected', 'all'],
                    default='lock',
                    help="axis 1: 'lock' never regenerates root translation "
                         "(proven best, no drift); 'detected' only on flagged "
                         "frames; 'all' every frame.")
    ap.add_argument('--mask-granularity', choices=['joint', 'frame', 'channel'],
                    default='joint',
                    help="axis 4: 'joint' regenerates only flagged joints; "
                         "'frame' regenerates whole flagged frames; 'channel' "
                         'regenerates only the flagged per-channel cells '
                         '(MoGenDIT-faithful, no joint OR / strict tighten).')
    ap.add_argument('--sdedit-tau', type=float, default=0.5,
                    help='axis 3: 0 = regenerate masked region from scratch; '
                         '>0 = add τ noise then regenerate (gentle, default 0.5).')
    ap.add_argument('--detect-tau', type=float, default=0.3,
                    help='self_denoise: SDEdit τ for the stage-1 projection.')
    ap.add_argument('--detect-metric', choices=['angle', 'abs'], default='angle',
                    help="self_denoise change metric: 'angle' (MoGenDIT-style "
                         'per-joint geodesic angle, physical) or legacy z-scored '
                         "'abs'.")
    ap.add_argument('--detect-joint-thr-rad', type=float, default=0.15,
                    help='angle metric: per-joint geodesic-angle threshold (rad).')
    ap.add_argument('--detect-trans-thr-m', type=float, default=0.05,
                    help='angle metric: translation change threshold (meters).')
    ap.add_argument('--detect-threshold', type=float, default=0.1,
                    help='abs metric: |LQ-projection| z-scored change threshold.')
    ap.add_argument('--no-strict-tighten', action='store_true',
                    help='disable kinematic/temporal/blob tightening of the mask.')
    ap.add_argument('--no-upright-fix', action='store_true',
                    help='disable the per-clip upside-down (gravity-flipped) '
                         'detection + 180°-X upright correction applied before '
                         'the model sees the clip.')
    ap.add_argument('--presmooth-sigma', type=float, default=0.0,
                    help='Gaussian temporal pre-smooth (frames) of the kept LQ '
                         'region before imputation; reduces residual jitter and '
                         'mask-boundary seams. 0 = off.')
    args = ap.parse_args()
    print(f'[run] infer_repair: mask_source={args.mask_source} '
          f'translation_mode={args.translation_mode} '
          f'granularity={args.mask_granularity} sdedit_tau={args.sdedit_tau}')
    MOG_MASK_DIR = (
        PROJECT_ROOT / 'data/eval/hymotion_m2m/adaptive_masks_mogendit/brokenamass_star'
    )

    def _load_mog_joint_mask(idx: int, t30: int):
        f = MOG_MASK_DIR / f'{idx:05d}.npz'
        if not f.is_file():
            return np.zeros((t30, 22), dtype=np.float32)
        jm = np.load(f)['joint_mask'].astype(np.float32)  # (Tc, 22)
        if jm.shape[0] != t30:  # resample to model frame count
            src = np.clip(np.round(np.linspace(0, jm.shape[0] - 1, t30)).astype(int),
                          0, jm.shape[0] - 1)
            jm = jm[src]
        return jm

    bone_offsets = torch.load(
        str(PROJECT_ROOT / 'data/hymotion_m2m_data/bone_offsets_22.pt'),
        map_location='cpu', weights_only=False,
    ).float()
    bone_offsets_np = bone_offsets.numpy()

    sm = np.load(args.sm_results, allow_pickle=True).item()
    corrupted = sm['motion']            # list of smpldata dicts (y-up, SM frame)
    lengths = np.asarray(sm['lengths']).reshape(-1)
    N = min(len(corrupted), args.max_samples)
    print(f'[run] {N} clips from {args.sm_results}')

    bundle = pipeline = model_info = None
    if not args.identity_check:
        bundle, pipeline, model_info, ckpt = build_model(args.model_name, args.device)

    out_fix = []
    out_masks = []
    t0 = time.time()
    n_detected_total = 0
    n_flipped = 0
    _cov_frame, _cov_joint = [], []
    for i in range(N):
        sd = {k: _to_torch(v) for k, v in corrupted[i].items()}
        T20 = sd['poses'].shape[0]
        L = int(lengths[i]) if i < len(lengths) else T20
        L = min(L, T20)
        # crop to valid length (drop right padding)
        sd = {k: v[:L] for k, v in sd.items()}

        # SM results.npy smpldata is already y-up (verified). Convert straight
        # to m2m135 (20fps) — NO axis swap. The old z_up_to_y_up double-rotated
        # the clip, feeding a lying-down/mirrored motion to the y-up model.
        m135_20 = smpldata_to_m2m135(sd, bone_offsets)

        # Some BrokenAMASS clips come out of StableMotion's RIFKE decode with
        # gravity flipped (head below feet) -- a per-clip source-frame issue.
        # Stand them upright BEFORE the (y-up trained) model sees them, else the
        # repair of an upside-down motion is meaningless.
        if not args.no_upright_fix:
            m135_20, flipped = _upright_fix_135(m135_20, bone_offsets)
            if flipped:
                n_flipped += 1

        # 20 → 30 fps
        T30 = max(2, int(round(L * args.work_fps / args.src_fps)))
        m135_30 = _resample_motion135_slerp(m135_20, T30)

        jflag30 = np.zeros((T30, 22), dtype=bool)   # real regenerated mask (30fps)
        if args.identity_check:
            repaired_30 = m135_30
            n_det = 0
        else:
            seed = args.seed_base + i
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            np.random.seed(seed & 0xFFFFFFFF)
            try:
                motion_t = torch.from_numpy(
                    np.asarray(m135_30, dtype=np.float32)).to(args.device)
                kw = dict(
                    lengths=[T30],
                    mask_source=args.mask_source,
                    translation_mode=args.translation_mode,
                    mask_granularity=args.mask_granularity,
                    sdedit_tau=args.sdedit_tau,
                    strict_tighten=not args.no_strict_tighten,
                    replacement_guidance=args.replacement_guidance,
                    presmooth_sigma=args.presmooth_sigma,
                )
                if args.mask_source == 'provided':
                    jm = _load_mog_joint_mask(i, T30)            # (T30,22)
                    kw['adaptive_mask'] = torch.from_numpy(jm).unsqueeze(0)
                else:
                    kw['detect_tau'] = args.detect_tau
                    kw['detect_metric'] = args.detect_metric
                    kw['detect_joint_thr_rad'] = args.detect_joint_thr_rad
                    kw['detect_trans_thr_m'] = args.detect_trans_thr_m
                    kw['detect_threshold'] = args.detect_threshold
                out = pipeline.infer_repair(motion_t, **kw)
                repaired_30 = out['motion'][0].detach().cpu().numpy()
                jm_out = out['joint_mask'][0][:T30].float()
                jflag30 = jm_out.cpu().numpy().astype(bool)
                n_det = int(jm_out.any(-1).sum().item())
                _cov_frame.append(float(jm_out.any(-1).float().mean().item()))
                _cov_joint.append(float(jm_out.mean().item()))
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f'[{i+1}/{N}] FAIL: {e}; falling back to corrupted')
                repaired_30 = m135_30
                n_det = 0

        # 30 → 20 fps
        repaired_20 = _resample_motion135_slerp(
            np.asarray(repaired_30, dtype=np.float32), L)

        # m2m135 → smpldata(24), still y-up (matches SM results frame). NO swap.
        # Keep torch tensors (matches SM smpldata payloads; eval calls .to()).
        sd_fixed_y = m2m135_to_smpldata_24(repaired_20, bone_offsets)
        out_fix.append({
            'poses': sd_fixed_y['poses'].reshape(L, -1).cpu().float(),
            'trans': sd_fixed_y['trans'].cpu().float(),
            'joints': sd_fixed_y['joints'].cpu().float(),
        })
        # Resample the REAL regenerated mask 30fps->20fps (nearest) so the
        # viewer can show what hymotion-m2m actually regenerated (self_denoise),
        # not the MoGenDIT mask.
        midx = np.clip(np.round(np.linspace(0, T30 - 1, L)).astype(int), 0, T30 - 1)
        out_masks.append(jflag30[midx].astype(bool))     # (L, 22)
        n_detected_total += n_det
        if (i + 1) % 10 == 0 or i == N - 1:
            dt = time.time() - t0
            print(f'[{i+1}/{N}] L={L} T30={T30} '
                  f'({dt/(i+1):.2f}s/clip, eta {dt/(i+1)*(N-i-1):.0f}s)')

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    npy_path = out_dir / 'results.npy'
    np.save(npy_path, {
        'motion': corrupted[:N],
        'motion_fix': out_fix,
        'joint_masks': out_masks,   # list of (L,22) bool, real self_denoise mask
        'lengths': lengths[:N],
    })
    with open(str(npy_path).replace('.npy', '_len.txt'), 'w') as fw:
        fw.write('\n'.join(str(int(l)) for l in lengths[:N]))
    print(f'[save] {npy_path}  ({len(out_fix)} clips)')
    print(f'[upright-fix] corrected {n_flipped}/{N} upside-down clips')
    if _cov_frame:
        import numpy as _np
        print(f'[coverage] mask_source={args.mask_source} '
              f'granularity={args.mask_granularity} | '
              f'frame-cov={_np.mean(_cov_frame)*100:.1f}% '
              f'(p50={_np.median(_cov_frame)*100:.0f}% '
              f'max={_np.max(_cov_frame)*100:.0f}%) '
              f'joint-cov={_np.mean(_cov_joint)*100:.1f}%')

    # Launch official eval (motion_fix vs GT).
    sm_root = PROJECT_ROOT / 'ref_repo' / 'StableMotion'
    cmd = (
        f'cd {sm_root} && python3 -m eval.eval_scripts '
        f'--data_path {os.path.relpath(npy_path, sm_root)} '
        f'--gt_data_path {os.path.relpath(os.path.abspath(args.gt), sm_root)} '
        f'--motiontypes motion_fix motion --force_redo'
    )
    print(f'[eval] {cmd}')
    os.system(cmd)
    return 0


if __name__ == '__main__':
    sys.exit(main())
