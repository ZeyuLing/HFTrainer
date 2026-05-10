"""HyMotion M2M HQ post-processing via MoGenDIT trans_regen.

Goal (2026-04-27): The HyMotion M2M v4 outputs achieve qc_pass = 43.6% on
E9 v2 (374 cases). Per-checker breakdown shows foot_sliding alone causes
50% of single-failure cases. MoGenDIT's `trans_regen` mode is purpose-
built for foot-skating / trajectory drift fixes — keep pose rotations,
re-generate the root translation only.

Pipeline:
  HyM motion_135 NPZ -> SMPL-H NPZ (poses 156 + trans 3)
    -> MoGenDIT trans_regen -> SMPL-H NPZ
    -> HyM motion_135 NPZ (motion_135, positions, translation)
    -> QC checker -> per-sample metrics
    -> flat import JSON for dashboard

Usage:
    python3 scripts/postprocess_hymotion_with_mogendit.py \\
        --src work_dirs/e9_v2_rerun_20260427/m2m_combo_v4/uncond_local/E9_D_strict_mask_d2_b3_bsmooth_combo \\
        --out-dir work_dirs/e9_v2_rerun_20260427/m2m_combo_v5_mogendit \\
        --eval-datalist data/eval/m2m_v2/eval_e9_repair_v2.json \\
        --setting D_strict_mask_d2_b3_bsmooth_combo_mogendit \\
        --mode trans_regen
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'tools'))

from hftrainer.evaluation.motion.m2m_eval_metrics import (  # noqa: E402
    compute_all_metrics, aggregate_metrics, motion135_to_positions_np,
)
from hftrainer.models.motion.components.utils.geometry.rotation_convert import (  # noqa: E402
    rotation_6d_to_matrix, matrix_to_axis_angle, axis_angle_to_matrix,
    matrix_to_rotation_6d,
)
from hftrainer.pipelines.motion.mogendit_pipeline import (  # noqa: E402
    MoGenDITRepairPipeline,
)
from eval_m2m_v2_all_tasks import (  # noqa: E402
    _run_quality_checker, _compute_qc_defect_mask,
)


def _rot6d_row_to_col(rot6d_row: np.ndarray) -> np.ndarray:
    """M2M row-major -> project col-major (last dim is 6)."""
    return rot6d_row[..., [0, 2, 4, 1, 3, 5]]


def _rot6d_col_to_row(rot6d_col: np.ndarray) -> np.ndarray:
    return rot6d_col[..., [0, 3, 1, 4, 2, 5]]


def motion135_to_smplh_npz(motion_135: np.ndarray, dst_path: str) -> None:
    """Convert HyM motion_135 -> minimal SMPL-H NPZ (poses 156 + trans 3)."""
    T = motion_135.shape[0]
    trans = motion_135[:, :3].astype(np.float32)
    rot6d_row = motion_135[:, 3:135].reshape(T, 22, 6)
    rot6d_col = _rot6d_row_to_col(rot6d_row)
    R = rotation_6d_to_matrix(torch.from_numpy(rot6d_col).float())  # (T, 22, 3, 3)
    aa_22 = matrix_to_axis_angle(R)  # (T, 22, 3)
    poses_52 = np.zeros((T, 52, 3), dtype=np.float32)
    poses_52[:, :22] = aa_22.numpy().astype(np.float32)
    poses_flat = poses_52.reshape(T, 156)
    np.savez(
        dst_path,
        poses=poses_flat,
        trans=trans,
        betas=np.zeros(16, dtype=np.float32),
        gender='neutral',
        mocap_framerate=30.0,
    )


def _build_obs_mask(
    lq_motion135: np.ndarray,
    bone_offsets: np.ndarray,
    source: str = 'qc',
    min_keep_ratio: float = 0.30,
    device: str = 'cuda',
) -> np.ndarray:
    """Build per-frame obs_mask (1=keep LQ, 0=re-generate).

    Strategy:
      * source='qc': run LQ quality checker, OR all per-checker invalid_masks
        across the 22 joints into a per-frame defect indicator. A frame is
        marked defect (obs=0) if ANY joint flagged.
      * source='none': all-ones (degenerate to plain denoise).

    A safety floor of ``min_keep_ratio`` ensures we never let MoGenDIT
    re-generate more than ``1 - min_keep_ratio`` of frames; if QC flags
    too many, we keep the frames with the fewest joint-defects as
    anchors. This protects motion semantics: the model needs anchored
    LQ context to reconstruct the bridge, otherwise it free-generates.
    """
    T = lq_motion135.shape[0]
    if source == 'none':
        return np.ones(T, dtype=np.float32)

    # _compute_qc_defect_mask returns (T, motion_dim) but uses defect_joints
    # internally. Recompute joint-level here to be explicit.
    try:
        from eval_m2m_v2_all_tasks import _run_quality_checker as _qc
        r_dict = _qc(lq_motion135, bone_offsets, device=device)
    except Exception as e:
        print(f'    [obs_mask] QC failed ({e!r}); fallback to all-ones')
        return np.ones(T, dtype=np.float32)
    r = (r_dict or {}).get('_raw')
    if r is None:
        return np.ones(T, dtype=np.float32)

    defect_joints = np.zeros((T, 22), dtype=bool)
    failed = list(r.failed_checks) + list(r.borderline_checks)
    for name in failed:
        res = r.all_results.get(name, {})
        im = res.get('invalid_mask', None)
        if im is None:
            defect_joints[:, :] = True
            continue
        im_arr = np.asarray(im)
        if im_arr.ndim != 2 or im_arr.shape[0] == 0:
            defect_joints[:, :] = True
            continue
        tcap = min(T, im_arr.shape[0])
        defect_joints[:tcap] |= im_arr[:tcap, :22].astype(bool)

    # Frame defect = ANY joint flagged
    frame_defect = defect_joints.any(axis=1)  # (T,)
    obs_mask = (~frame_defect).astype(np.float32)

    # Safety floor: ensure at least min_keep_ratio frames are kept
    n_keep = int(obs_mask.sum())
    n_min = int(np.ceil(min_keep_ratio * T))
    if n_keep < n_min:
        # Sort defect frames by joint-defect count (fewest = best anchor)
        defect_count = defect_joints.sum(axis=1)  # (T,)
        # Sort defect frames by ascending defect count, take the cleanest
        defect_idx = np.where(frame_defect)[0]
        if defect_idx.size > 0:
            order = defect_idx[np.argsort(defect_count[defect_idx])]
            n_promote = n_min - n_keep
            promote = order[:n_promote]
            obs_mask[promote] = 1.0

    return obs_mask


def smplh_npz_to_motion135(npz_path: str) -> np.ndarray:
    """Convert MoGenDIT-output SMPL-H NPZ back to motion_135."""
    d = np.load(npz_path, allow_pickle=True)
    poses = np.asarray(d['poses'], dtype=np.float32)
    trans = np.asarray(d.get('trans', d.get('transl')), dtype=np.float32)
    T = trans.shape[0]
    aa_22 = poses[:, :66].reshape(T, 22, 3)
    R = axis_angle_to_matrix(torch.from_numpy(aa_22).float())  # (T, 22, 3, 3)
    rot6d_col = matrix_to_rotation_6d(R).numpy()  # (T, 22, 6)
    rot6d_row = _rot6d_col_to_row(rot6d_col)
    motion_135 = np.concatenate(
        [trans, rot6d_row.reshape(T, 132)], axis=-1).astype(np.float32)
    return motion_135


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', type=str, required=False, default=None,
                        help='HyM v4 src dir, parent of npz/. Required '
                             'unless --input-source=lq (then ignored).')
    parser.add_argument(
        '--input-source', type=str, default='src',
        choices=['src', 'lq'],
        help=("Where the MoGenDIT input motion comes from. 'src' = use "
              "--src/npz/ (e.g. v4 HyM HQ outputs). 'lq' = load original "
              "LQ NPZ from eval-datalist's motion_path (preserves LQ "
              "translation in obs frames bit-exactly under impute)."),
    )
    parser.add_argument('--out-dir', type=str, required=True)
    parser.add_argument('--eval-datalist', type=str,
                        default='data/eval/m2m_v2/eval_e9_repair_v2.json')
    parser.add_argument('--model-name', type=str, default='uncond_local')
    parser.add_argument('--setting', type=str,
                        default='D_strict_mask_d2_b3_bsmooth_combo_mogendit')
    parser.add_argument('--mode', type=str, default='trans_regen',
                        choices=['denoise', 'ada_denoise',
                                 'trans_regen', 'impute'])
    parser.add_argument('--mogendit-model', type=str, default='MoreDiff-0.1B')
    parser.add_argument('--step', type=int, default=10)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--max-samples', type=int, default=99999)
    parser.add_argument('--skip-existing', action='store_true',
                        help='Skip per-sample if output NPZ already exists.')
    # 2026-04-27: When mode='impute', compute a per-frame obs_mask from the
    # LQ QC defect mask: only frames with QC defects are re-generated; clean
    # frames keep their LQ trans+pose verbatim. This addresses user feedback
    # that v5/v6 (trans_regen + ada_denoise) regenerated *every* frame's
    # translation, breaking original-motion semantics.
    parser.add_argument(
        '--obs-mask-source', type=str, default='qc',
        choices=['qc', 'none'],
        help=("Source for obs_mask when mode='impute'. 'qc' uses LQ quality-"
              "checker defect mask (clean frames frozen). Ignored otherwise."),
    )
    parser.add_argument(
        '--obs-min-keep-ratio', type=float, default=0.30,
        help=('When mode=impute, ensure at least this fraction of frames '
              'are kept as obs (anchors). If QC marks more frames defective '
              'than 1-this, top frames by defect count are still re-generated '
              'but a minimum number of clean anchors is preserved.'),
    )
    args = parser.parse_args()

    with open(args.eval_datalist) as f:
        dl = json.load(f)
    items = dl.get('data_list', dl)

    if args.input_source == 'src':
        assert args.src, '--src is required when input-source=src'
        src_npz_dir = Path(args.src) / 'npz'
        assert src_npz_dir.is_dir(), \
            f'Source npz dir missing: {src_npz_dir}'
        files = sorted(src_npz_dir.glob('*.npz'))
    else:
        # Synthesize per-index placeholder Path; actual NPZ comes from
        # items[idx]['motion_path'].
        src_npz_dir = None  # unused
        files = [Path(f'{i:05d}.npz') for i in range(len(items))]
    files = files[:args.max_samples]
    print(f'[plan] {len(files)} src(={args.input_source}) NPZs -> '
          f'MoGenDIT {args.mode}')

    out_root = Path(args.out_dir)
    task_key = f'E9_{args.setting}'
    out_npz_dir = out_root / args.model_name / task_key / 'npz'
    out_npz_dir.mkdir(parents=True, exist_ok=True)
    import_json_dir = out_root / 'import_jsons'
    import_json_dir.mkdir(exist_ok=True)
    log_dir = out_root / 'logs'
    log_dir.mkdir(exist_ok=True)

    bone_offsets_t = torch.load(
        str(PROJECT_ROOT / 'data/hymotion_m2m_data/bone_offsets_22.pt'),
        map_location='cpu', weights_only=False).float()
    bone_offsets = bone_offsets_t.numpy()

    # Init MoGenDIT pipeline
    print('[init] loading MoGenDIT pipeline...')
    t0 = time.time()
    pipe = MoGenDITRepairPipeline(
        model_name=args.mogendit_model, device=args.device, use_ema=True,
    )
    print(f'[init] MoGenDIT loaded in {time.time()-t0:.1f}s')

    per_sample_list = []
    n_done = n_skip = n_fail = 0
    for i, src_npz in enumerate(files):
        idx_str = src_npz.stem
        try:
            idx = int(idx_str)
        except ValueError:
            idx = i
        out_npz = out_npz_dir / src_npz.name

        if args.skip_existing and out_npz.exists():
            try:
                d = np.load(out_npz, allow_pickle=True)
                hq = d['motion_135']
                metrics = json.loads((log_dir / f'{idx_str}.metrics.json').read_text())
                per_sample_list.append(metrics)
                n_skip += 1
                continue
            except Exception:
                pass

        try:
            if args.input_source == 'lq':
                # Resolve LQ NPZ from eval_datalist[idx]['motion_path'].
                if idx >= len(items):
                    raise ValueError(
                        f'idx {idx} out of range for eval_datalist '
                        f'(len={len(items)})')
                mp = items[idx]['motion_path']
                lq_path = (
                    mp if os.path.isabs(mp)
                    else str(PROJECT_ROOT / mp)
                )
                src = np.load(lq_path, allow_pickle=True)
                use_raw_lq_npz = (
                    'motion_135' not in src.files
                    and ('poses' in src.files or 'body_pose' in src.files)
                    and ('trans' in src.files or 'transl' in src.files)
                )
                if 'motion_135' in src.files:
                    src_135 = np.asarray(
                        src['motion_135'], dtype=np.float32)
                else:
                    # Raw SMPL-H NPZ: convert poses (T, 156) + trans
                    # (T, 3) -> motion_135 via the SMPL-H decoder we
                    # already use for MoGenDIT outputs.
                        src_135 = smplh_npz_to_motion135(lq_path)
            else:
                src = np.load(src_npz, allow_pickle=True)
                use_raw_lq_npz = False
                src_135 = np.asarray(src['motion_135'], dtype=np.float32)
            T = src_135.shape[0]

            with tempfile.TemporaryDirectory() as tmpdir:
                in_path = os.path.join(tmpdir, 'in.npz')
                out_path = os.path.join(tmpdir, 'out.npz')
                if args.input_source == 'lq' and use_raw_lq_npz:
                    # Match m2m_database runtime: feed the original source NPZ
                    # directly into MoGenDIT instead of round-tripping through
                    # motion_135 -> minimal SMPL-H. This preserves the exact
                    # original SMPL-H fields/semantics used by online repair.
                    np.savez(
                        in_path,
                        **{k: src[k] for k in src.files}
                    )
                else:
                    motion135_to_smplh_npz(src_135, in_path)

                if args.mode == 'impute':
                    # Build per-frame obs_mask from LQ QC defect mask.
                    obs_mask = _build_obs_mask(
                        src_135, bone_offsets,
                        source=args.obs_mask_source,
                        min_keep_ratio=args.obs_min_keep_ratio,
                        device=args.device,
                    )  # (T,) float, 1=keep LQ, 0=generate
                    in_npz = np.load(in_path, allow_pickle=True)
                    motion_dict_in = {
                        'poses': in_npz['poses'],
                        'trans': in_npz['trans'],
                        'betas': in_npz['betas'],
                        'gender': str(in_npz['gender']),
                        'mocap_framerate': float(in_npz['mocap_framerate']),
                    }
                    out_dict = pipe.impute_with_obs_mask(
                        motion_dict_in, obs_mask, step=args.step,
                    )
                    np.savez(
                        out_path,
                        poses=np.array(
                            out_dict['poses'], dtype=np.float32),
                        trans=np.array(
                            out_dict.get('trans', out_dict.get('transl')),
                            dtype=np.float32),
                        betas=np.zeros(16, dtype=np.float32),
                        gender='neutral',
                        mocap_framerate=30.0,
                    )
                    if i < 3:  # Verbose for first few
                        n_keep = int((obs_mask > 0.5).sum())
                        print(f'    [obs_mask] {n_keep}/{T} frames kept '
                              f'({100.0*n_keep/max(T,1):.0f}%)')
                else:
                    pipe.repair_npz(
                        in_path, out_path,
                        mode=args.mode, step=args.step,
                        use_windowed=True, window_size=224,
                        prev_padding=20,
                    )
                hq = smplh_npz_to_motion135(out_path)

                # When mode='impute', enforce hard preservation of LQ
                # trans+pose on obs frames (impute_with_obs_mask sometimes
                # leaks small drift through normalization round-trip).
                if args.mode == 'impute':
                    keep = obs_mask > 0.5  # (T,)
                    T_min = min(hq.shape[0], src_135.shape[0])
                    keep = keep[:T_min]
                    hq[:T_min][keep] = src_135[:T_min][keep]

            T_out = hq.shape[0]
            if T_out != T:
                if T_out > T:
                    hq = hq[:T]
                else:
                    pad = np.repeat(hq[-1:], T - T_out, axis=0)
                    hq = np.concatenate([hq, pad], axis=0)

            pos = motion135_to_positions_np(hq, bone_offsets).astype(np.float32)
            np.savez_compressed(
                out_npz, motion_135=hq.astype(np.float32),
                positions=pos, translation=hq[:, :3].astype(np.float32),
            )

            metrics = compute_all_metrics(
                pred_motion=hq, gt_motion=None, mask=None,
                bone_offsets=bone_offsets, rotation_space='local',
                fps=30.0, compute_fk=True,
            )

            qc = _run_quality_checker(hq, bone_offsets, device=args.device)
            if qc is not None:
                metrics['qc_pass'] = float(qc.get('is_valid', False))
                metrics['qc_num_failed'] = float(len(qc.get('failed_checks') or []))
                metrics['qc_num_borderline'] = float(
                    len(qc.get('borderline_checks') or []))
                # Use HyM convention: pass=1.0, fail=0.0
                for ch_name, ch_info in (qc.get('per_checker') or {}).items():
                    is_valid = (
                        ch_info.get('is_valid', True)
                        if isinstance(ch_info, dict) else True
                    )
                    metrics[f'qc_{ch_name}'] = 1.0 if is_valid else 0.0

            item = items[idx] if idx < len(items) else {}
            metrics['_npz_path'] = str(out_npz.resolve())
            metrics['_sample_idx'] = idx
            metrics['_caption'] = item.get('prompt_id', '') or item.get('caption_en', '')
            metrics['_num_frames'] = int(T)
            metrics['inference_time'] = 0.0  # postprocess only

            per_sample_list.append(metrics)
            (log_dir / f'{idx_str}.metrics.json').write_text(json.dumps(metrics, default=float))
            n_done += 1
            if (i + 1) % 10 == 0 or (i + 1) == len(files):
                rate = sum(1 for m in per_sample_list if m.get('qc_pass') == 1.0) / max(len(per_sample_list), 1)
                print(f'  [{i+1}/{len(files)}] qc_pass running rate={rate*100:.1f}% '
                      f'(done={n_done} skip={n_skip} fail={n_fail})')
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f'[fail] {src_npz.name}: {e!r}')
            n_fail += 1
            continue

    aggregated = aggregate_metrics(per_sample_list)
    from datetime import datetime
    flat = {
        'model': args.model_name,
        'rotation_space': 'local',
        'has_caption': False,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'task_id': 'E9',
        'setting': args.setting,
        'num_prompts': len(per_sample_list),
        'aggregated': aggregated,
        'per_sample': per_sample_list,
        '_postprocess_mode': args.mode,
        '_mogendit_model': args.mogendit_model,
        '_mogendit_step': args.step,
    }
    json_path = import_json_dir / f'{args.model_name}__E9_{args.setting}.json'
    with open(json_path, 'w') as f:
        json.dump(flat, f, indent=2, default=float)
    qc_pass_rate = aggregated.get('qc_pass', {}).get('mean', None)
    print(f'\n=== summary ===')
    print(f'  done={n_done}, skip={n_skip}, fail={n_fail}, total={len(per_sample_list)}')
    if qc_pass_rate is not None:
        print(f'  qc_pass mean: {qc_pass_rate:.1%}')
    print(f'  flat JSON: {json_path}')
    if n_fail > 0 or n_done == 0:
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
