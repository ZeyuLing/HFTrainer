"""Surgical LQ overlay on MoGenDIT-chained HQ output.

Motivation (2026-04-27): The HyM v6 = v4 → trans_regen → ada_denoise
chain achieves qc_pass = 60.2% but regenerates EVERY frame's
translation. User feedback: "all translations are now being regenerated
— I think this is inappropriate." The fix: for frames that the LQ
input already passes (per QC), force HQ[frame] = LQ[frame] (both rot
and trans). Frames that LQ fails are kept as v6's repair output.

A linear cross-fade window of ``--blend-frames`` frames is applied at
each obs/non-obs boundary to avoid pose discontinuities at the seam.

Usage:
    python3 scripts/lq_overlay_clean_frames.py \\
        --hq-dir work_dirs/.../m2m_combo_v6_mogendit_chained/uncond_local/E9_..._chained \\
        --eval-datalist data/eval/m2m_v2/eval_e9_repair_v2.json \\
        --out-dir work_dirs/.../m2m_combo_v8_lq_overlay
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
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
from eval_m2m_v2_all_tasks import _run_quality_checker  # noqa: E402


def _rot6d_col_to_row(rot6d_col: np.ndarray) -> np.ndarray:
    return rot6d_col[..., [0, 3, 1, 4, 2, 5]]


def smplh_to_motion135(npz_path: str) -> np.ndarray:
    """Convert raw SMPL-H NPZ -> motion_135 (T, 3+132 row-major)."""
    d = np.load(npz_path, allow_pickle=True)
    poses = np.asarray(d['poses'], dtype=np.float32)
    trans = np.asarray(d.get('trans', d.get('transl')), dtype=np.float32)
    T = trans.shape[0]
    aa_22 = poses[:, :66].reshape(T, 22, 3)
    R = axis_angle_to_matrix(torch.from_numpy(aa_22).float())
    rot6d_col = matrix_to_rotation_6d(R).numpy()
    rot6d_row = _rot6d_col_to_row(rot6d_col)
    return np.concatenate(
        [trans, rot6d_row.reshape(T, 132)], axis=-1
    ).astype(np.float32)


def _qc_clean_mask(
    motion_135: np.ndarray, bone_offsets: np.ndarray, device: str = 'cuda',
) -> np.ndarray:
    """Return per-frame bool mask: True = QC-clean (no joint flagged)."""
    T = motion_135.shape[0]
    try:
        r_dict = _run_quality_checker(motion_135, bone_offsets, device=device)
    except Exception as e:
        print(f'    [warn] QC failed: {e!r}; treating all frames clean')
        return np.ones(T, dtype=bool)
    r = (r_dict or {}).get('_raw')
    if r is None:
        return np.ones(T, dtype=bool)

    defect_joints = np.zeros((T, 22), dtype=bool)
    failed = list(r.failed_checks) + list(r.borderline_checks)
    for name in failed:
        res = r.all_results.get(name, {})
        im = res.get('invalid_mask', None)
        if im is None:
            return np.zeros(T, dtype=bool)  # everything defective
        im_arr = np.asarray(im)
        if im_arr.ndim != 2 or im_arr.shape[0] == 0:
            return np.zeros(T, dtype=bool)
        tcap = min(T, im_arr.shape[0])
        defect_joints[:tcap] |= im_arr[:tcap, :22].astype(bool)

    return ~defect_joints.any(axis=1)


def _qc_checker_clean_mask(
    motion_135: np.ndarray,
    bone_offsets: np.ndarray,
    checker_names: tuple[str, ...],
    device: str = 'cuda',
) -> np.ndarray:
    """Return per-frame clean mask for a subset of QC checkers.

    If a named checker is valid globally, all frames remain clean. If it fails
    and exposes an invalid_mask, frames with any flagged joint are marked dirty.
    Missing checker masks are treated conservatively as all-dirty.
    """
    T = motion_135.shape[0]
    try:
        r_dict = _run_quality_checker(motion_135, bone_offsets, device=device)
    except Exception as e:
        print(f'    [warn] QC failed: {e!r}; treating all frames clean')
        return np.ones(T, dtype=bool)
    r = (r_dict or {}).get('_raw')
    if r is None:
        return np.ones(T, dtype=bool)

    active = set(list(r.failed_checks) + list(r.borderline_checks))
    frame_defect = np.zeros(T, dtype=bool)
    for name in checker_names:
        if name not in active:
            continue
        res = r.all_results.get(name, {})
        im = res.get('invalid_mask', None)
        if im is None:
            return np.zeros(T, dtype=bool)
        im_arr = np.asarray(im)
        if im_arr.ndim == 1:
            tcap = min(T, im_arr.shape[0])
            frame_defect[:tcap] |= im_arr[:tcap].astype(bool)
        elif im_arr.ndim == 2 and im_arr.shape[0] > 0:
            tcap = min(T, im_arr.shape[0])
            frame_defect[:tcap] |= im_arr[:tcap, :22].astype(bool).any(axis=1)
        else:
            return np.zeros(T, dtype=bool)

    return ~frame_defect


def _slerp_blend_window(
    hq_seg: np.ndarray, lq_seg: np.ndarray, w: np.ndarray,
) -> np.ndarray:
    """Linear blend (lerp) on rot6d + trans for a window of frames.

    For 6D rotation we lerp the 6D vector then renormalize via
    matrix_to_rotation_6d(rotation_6d_to_matrix(blend)) to keep on SO(3).
    w shape (n,), each value in [0, 1]. w=1 means full HQ, w=0 means full LQ.
    """
    out = w[:, None] * hq_seg + (1.0 - w)[:, None] * lq_seg  # (n, 135)
    # Renormalize rot6d portion via SO(3) projection.
    T = out.shape[0]
    rot6d_row = out[:, 3:135].reshape(T, 22, 6)
    rot6d_col = rot6d_row[..., [0, 2, 4, 1, 3, 5]]
    R = rotation_6d_to_matrix(torch.from_numpy(rot6d_col).float())
    rot6d_col_re = matrix_to_rotation_6d(R).numpy()
    rot6d_row_re = rot6d_col_re[..., [0, 3, 1, 4, 2, 5]]
    out[:, 3:135] = rot6d_row_re.reshape(T, 132)
    return out.astype(np.float32)


def overlay_lq_on_hq(
    hq_135: np.ndarray, lq_135: np.ndarray, lq_clean: np.ndarray,
    blend_frames: int = 4,
) -> np.ndarray:
    """Where LQ is QC-clean, replace HQ with LQ; cross-fade the seams."""
    T = min(hq_135.shape[0], lq_135.shape[0], lq_clean.shape[0])
    out = hq_135[:T].copy()
    keep = lq_clean[:T]

    if not keep.any():
        return out  # no clean frames -> keep HQ as-is

    # Step 1: hard set kept frames to LQ.
    out[keep] = lq_135[:T][keep]

    if blend_frames <= 0:
        return out

    # Step 2: cross-fade at boundaries.
    # Find runs of kept=True; at each boundary, blend `blend_frames` frames
    # on the non-kept side from LQ towards HQ.
    keep_int = keep.astype(np.int8)
    diff = np.diff(np.concatenate([[0], keep_int, [0]]))
    starts = np.where(diff == 1)[0]   # transitions 0->1
    ends = np.where(diff == -1)[0]    # transitions 1->0 (exclusive)

    # Blend window at each boundary: weights ramp from 1 (LQ-side) to 0 (HQ-side)
    for s in starts:
        # Frames [s-blend_frames, s) are HQ side; ramp from HQ to LQ
        i0 = max(0, s - blend_frames)
        i1 = s
        if i0 >= i1:
            continue
        n = i1 - i0
        w = np.linspace(1.0, 0.0, n + 2)[1:-1]  # weights for HQ contribution
        # final = w * HQ + (1-w) * LQ
        out[i0:i1] = _slerp_blend_window(
            hq_135[i0:i1], lq_135[i0:i1], w,
        )
    for e in ends:
        # Frames [e, e+blend_frames) are HQ side; ramp from LQ back to HQ
        i0 = e
        i1 = min(T, e + blend_frames)
        if i0 >= i1:
            continue
        n = i1 - i0
        w = np.linspace(0.0, 1.0, n + 2)[1:-1]
        out[i0:i1] = _slerp_blend_window(
            hq_135[i0:i1], lq_135[i0:i1], w,
        )

    return out


def overlay_lq_translation_on_hq(
    hq_135: np.ndarray, lq_135: np.ndarray, trans_keep: np.ndarray,
    blend_frames: int = 4,
) -> np.ndarray:
    """Use LQ root translation, interpolating spans that fail translation QC.

    We do not paste HQ translation into dirty spans: the MoGenDIT-chain output
    may live on a different regenerated trajectory, and mixing a few HQ frames
    into an otherwise-LQ path creates new velocity spikes. Interpolating dirty
    spans from neighboring LQ-clean anchors preserves action semantics while
    removing local translation jumps.
    """
    T = min(hq_135.shape[0], lq_135.shape[0], trans_keep.shape[0])
    out = hq_135[:T].copy()
    keep = trans_keep[:T]

    if not keep.any():
        return out

    repaired_trans = lq_135[:T, :3].astype(np.float32).copy()
    if not keep.all():
        x = np.arange(T, dtype=np.float32)
        clean_idx = np.where(keep)[0].astype(np.float32)
        for c in range(3):
            repaired_trans[:, c] = np.interp(
                x, clean_idx, lq_135[:T, c][keep],
            ).astype(np.float32)

    out[:, :3] = repaired_trans
    return out.astype(np.float32)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--hq-dir', type=str, required=True,
                        help='Dir containing HQ npz/ subdir from v6 chain.')
    parser.add_argument('--eval-datalist', type=str,
                        default='data/eval/m2m_v2/eval_e9_repair_v2.json')
    parser.add_argument('--out-dir', type=str, required=True)
    parser.add_argument('--model-name', type=str, default='uncond_local')
    parser.add_argument(
        '--setting', type=str,
        default='D_strict_mask_d2_b3_bsmooth_combo_chained_lqOverlay')
    parser.add_argument('--blend-frames', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--max-samples', type=int, default=99999)
    # 2026-04-27: After running the default frame-level overlay we found
    # that on E9 LQ the QC-clean ratio averages only ~2% (LQ is heavily
    # defective on every frame). To still satisfy "translation should not
    # be regenerated", an alternative is the trans-only overlay: keep the
    # 6D pose from v6 HQ (so spine/arm/joint defects ARE fixed) but
    # restore the root translation to LQ on every frame. This preserves
    # the original walking/running trajectory verbatim.
    parser.add_argument(
        '--mode', type=str, default='frame',
        choices=[
            'frame', 'trans_only', 'trans_qc_clean',
            'trans_only_clean_pose',
        ],
        help=("'frame' = replace clean frames in HQ with LQ (rot+trans). "
              "'trans_only' = always use LQ root translation, keep v6 "
              "rotations. 'trans_qc_clean' = use LQ translation but "
              "interpolate spans failing translation QC. "
              "'trans_only_clean_pose' = trans always LQ, rotations only "
              "restored to LQ on QC-clean frames."),
    )
    args = parser.parse_args()

    hq_npz_dir = Path(args.hq_dir) / 'npz'
    assert hq_npz_dir.is_dir(), f'HQ npz dir missing: {hq_npz_dir}'
    files = sorted(hq_npz_dir.glob('*.npz'))[:args.max_samples]
    print(f'[plan] {len(files)} HQ NPZs -> LQ overlay')

    with open(args.eval_datalist) as f:
        dl = json.load(f)
    items = dl.get('data_list', dl)

    out_root = Path(args.out_dir)
    setting_name = args.setting
    if args.mode != 'frame':
        setting_name = f'{setting_name}_{args.mode}'
    task_key = f'E9_{setting_name}'
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

    per_sample_list = []
    n_done = n_fail = 0
    keep_ratios = []
    for i, hq_npz in enumerate(files):
        idx_str = hq_npz.stem
        try:
            idx = int(idx_str)
        except ValueError:
            idx = i

        try:
            d = np.load(hq_npz, allow_pickle=True)
            hq_135 = np.asarray(d['motion_135'], dtype=np.float32)
            T = hq_135.shape[0]

            mp = items[idx]['motion_path'] if idx < len(items) else None
            if mp is None:
                print(f'    [skip] no motion_path for idx={idx}')
                continue
            lq_path = mp if os.path.isabs(mp) else str(PROJECT_ROOT / mp)
            lq_135 = smplh_to_motion135(lq_path)
            T_eff = min(T, lq_135.shape[0])

            if args.mode == 'frame':
                lq_clean = _qc_clean_mask(
                    lq_135[:T_eff], bone_offsets, device=args.device,
                )
                keep_ratio = float(lq_clean.sum()) / max(T_eff, 1)
                keep_ratios.append(keep_ratio)

                final_135 = overlay_lq_on_hq(
                    hq_135[:T_eff], lq_135[:T_eff], lq_clean,
                    blend_frames=args.blend_frames,
                )
            elif args.mode == 'trans_only':
                final_135 = hq_135[:T_eff].copy()
                final_135[:, :3] = lq_135[:T_eff, :3]  # always LQ trans
                keep_ratio = 1.0
                keep_ratios.append(keep_ratio)
            elif args.mode == 'trans_qc_clean':
                trans_clean = _qc_checker_clean_mask(
                    lq_135[:T_eff], bone_offsets,
                    checker_names=('translation_velocity',),
                    device=args.device,
                )
                keep_ratio = float(trans_clean.sum()) / max(T_eff, 1)
                keep_ratios.append(keep_ratio)
                final_135 = overlay_lq_translation_on_hq(
                    hq_135[:T_eff], lq_135[:T_eff], trans_clean,
                    blend_frames=args.blend_frames,
                )
            elif args.mode == 'trans_only_clean_pose':
                lq_clean = _qc_clean_mask(
                    lq_135[:T_eff], bone_offsets, device=args.device,
                )
                keep_ratio = float(lq_clean.sum()) / max(T_eff, 1)
                keep_ratios.append(keep_ratio)
                final_135 = hq_135[:T_eff].copy()
                final_135[:, :3] = lq_135[:T_eff, :3]
                # Optional: restore rotations on clean frames to LQ
                final_135[lq_clean, 3:] = lq_135[:T_eff, 3:][lq_clean]
            else:
                raise ValueError(f'Unknown mode: {args.mode}')

            pos = motion135_to_positions_np(
                final_135, bone_offsets).astype(np.float32)
            out_npz = out_npz_dir / hq_npz.name
            np.savez_compressed(
                out_npz, motion_135=final_135.astype(np.float32),
                positions=pos,
                translation=final_135[:, :3].astype(np.float32),
            )

            metrics = compute_all_metrics(
                pred_motion=final_135, gt_motion=None, mask=None,
                bone_offsets=bone_offsets, rotation_space='local',
                fps=30.0, compute_fk=True,
            )
            qc = _run_quality_checker(
                final_135, bone_offsets, device=args.device)
            if qc is not None:
                metrics['qc_pass'] = float(qc.get('is_valid', False))
                metrics['qc_num_failed'] = float(
                    len(qc.get('failed_checks') or []))
                metrics['qc_num_borderline'] = float(
                    len(qc.get('borderline_checks') or []))
                for ch_name, ch_info in (qc.get('per_checker') or {}).items():
                    is_valid = (
                        ch_info.get('is_valid', True)
                        if isinstance(ch_info, dict) else True
                    )
                    metrics[f'qc_{ch_name}'] = (
                        1.0 if is_valid else 0.0
                    )

            item = items[idx] if idx < len(items) else {}
            metrics['_npz_path'] = str(out_npz.resolve())
            metrics['_sample_idx'] = idx
            metrics['_caption'] = (
                item.get('prompt_id', '') or item.get('caption_en', '')
            )
            metrics['_num_frames'] = int(T_eff)
            metrics['_lq_keep_ratio'] = keep_ratio
            metrics['inference_time'] = 0.0
            per_sample_list.append(metrics)
            (log_dir / f'{idx_str}.metrics.json').write_text(
                json.dumps(metrics, default=float))
            n_done += 1
            if (i + 1) % 20 == 0 or (i + 1) == len(files):
                rate = sum(
                    1 for m in per_sample_list if m.get('qc_pass') == 1.0
                ) / max(len(per_sample_list), 1)
                kr = float(np.mean(keep_ratios)) if keep_ratios else 0.0
                print(f'  [{i+1}/{len(files)}] qc_pass={rate*100:.1f}% '
                      f'mean_lq_keep={kr*100:.0f}% (done={n_done} '
                      f'fail={n_fail})')
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f'[fail] {hq_npz.name}: {e!r}')
            n_fail += 1
            continue

    aggregated = aggregate_metrics(per_sample_list)
    flat = {
        'model': args.model_name,
        'rotation_space': 'local',
        'has_caption': False,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'task_id': 'E9',
        'setting': setting_name,
        'num_prompts': len(per_sample_list),
        'aggregated': aggregated,
        'per_sample': per_sample_list,
        '_overlay_blend_frames': args.blend_frames,
        '_mean_lq_keep_ratio': (
            float(np.mean(keep_ratios)) if keep_ratios else 0.0
        ),
    }
    json_path = (
        import_json_dir / f'{args.model_name}__E9_{setting_name}.json'
    )
    with open(json_path, 'w') as f:
        json.dump(flat, f, indent=2, default=float)
    qc_pass_rate = aggregated.get('qc_pass', {}).get('mean', None)
    print('\n=== summary ===')
    print(f'  done={n_done}, fail={n_fail}, total={len(per_sample_list)}')
    if qc_pass_rate is not None:
        print(f'  qc_pass mean: {qc_pass_rate:.1%}')
    if keep_ratios:
        kr = float(np.mean(keep_ratios))
        print(f'  mean LQ-clean ratio: {kr:.1%} (={kr*100:.0f}% of '
              f'frames in HQ replaced by LQ)')
    print(f'  flat JSON: {json_path}')
    if n_fail > 0 or n_done == 0:
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
