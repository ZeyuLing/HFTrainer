#!/usr/bin/env python3.8
"""
CJGame MB NPZ 数据切片、坐标系标准化、质检 - Phase 2+3 脚本
(Phase 1 已完成，此脚本直接从 Phase 2 开始)
"""

import sys, types, importlib.util, os

_HFT_ROOT = '/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer'
sys.path.insert(0, _HFT_ROOT)

def _make_pkg(name, path=None):
    parts = name.split('.')
    parent = None
    cur = ''
    for part in parts:
        cur = cur + '.' + part if cur else part
        if cur not in sys.modules:
            m = types.ModuleType(cur)
            sys.modules[cur] = m
            if parent:
                setattr(parent, part, m)
        parent = sys.modules[cur]
    if path:
        parent.__path__ = [path]
    parent.__package__ = name
    return parent

_BASE = os.path.join(_HFT_ROOT, 'hftrainer')
for _pkg in [
    'hftrainer',
    'hftrainer.evaluation', 'hftrainer.evaluation.quality_check_rules',
    'hftrainer.models', 'hftrainer.models.motion',
    'hftrainer.models.motion.components',
    'hftrainer.models.motion.components.utils',
    'hftrainer.models.motion.components.utils.geometry',
    'hftrainer.motion.body_models',
]:
    _make_pkg(_pkg, _BASE + _pkg[len('hftrainer'):])

_reg = _make_pkg('hftrainer.registry')
class _MockReg:
    def __init__(self, *a, **kw): pass
    def register_module(self, *a, **kw): return lambda x: x
_reg.MODELS = _MockReg()
_reg.HF_MODELS = _MockReg()
_reg.build_hf_model_from_cfg = lambda *a, **kw: None


def _load(module_name, filepath):
    spec = importlib.util.spec_from_file_location(module_name, filepath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


_geo = os.path.join(_BASE, 'models/motion/components/utils/geometry/')
_load('hftrainer.models.motion.components.utils.geometry.rotation_convert', _geo + 'rotation_convert.py')
_load('hftrainer.models.motion.components.utils.geometry.matrix', _geo + 'matrix.py')
_body = os.path.join(_BASE, 'motion/body_models/')
_load('hftrainer.motion.body_models.smplx_lite', _body + 'smplx_lite.py')

_qc = os.path.join(_BASE, 'evaluation/quality_check_rules/')
for _m in [
    'mask_utils', 'tbs_utils', '_geometry_compat', 'rotation_classifier',
    '_model_compat', 'base_checker', 'root_motion_utils',
    'jitter_checker', 'joint_twist_checker', 'candy_wrapper_checker',
    'joint_jump_checker', 'arm_penetration_checker', 'small_wobble_checker',
    'foot_sliding_checker', 'rotation_velocity_checker',
    'translation_velocity_checker', 'rotation_validity_checker',
    'ported_hymotion_data_checkers', 'motion_quality_checker',
]:
    _load(f'hftrainer.evaluation.quality_check_rules.{_m}', _qc + f'{_m}.py')

import json
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from hftrainer.evaluation.quality_check_rules.motion_quality_checker import MotionQualityChecker

NPZ_DIR  = Path(_HFT_ROOT) / 'data/lightai_data/CJGame_MB/npz'
OUT_DIR  = Path(_HFT_ROOT) / 'data/lightai_data/CJGame_MB/npz_split'
REPORT_DIR = Path(_HFT_ROOT) / 'data/lightai_data/CJGame_MB'
TARGET_FPS = 30
MAX_FRAMES = 360


def load_npz(path: Path) -> Dict:
    return dict(np.load(path, allow_pickle=True))


def fps_of(data: Dict) -> float:
    return float(data.get('mocap_framerate', 30.0))


def resample_to_30fps(data: Dict) -> Dict:
    src_fps = fps_of(data)
    if abs(src_fps - TARGET_FPS) < 0.5:
        return data
    ratio = src_fps / TARGET_FPS
    T = data['poses'].shape[0]
    new_T = max(1, int(round(T / ratio)))
    indices = np.round(np.linspace(0, T - 1, new_T)).astype(int)
    new_data = {}
    for k, v in data.items():
        if isinstance(v, np.ndarray) and v.ndim >= 1 and v.shape[0] == T:
            new_data[k] = v[indices]
        else:
            new_data[k] = v
    new_data['mocap_framerate'] = np.float64(TARGET_FPS)
    return new_data


def canonicalize_smplh(data: Dict) -> Dict:
    """
    坐标系标准化：第0帧朝向 +Z，xz原点，y=0地面。
    """
    from hftrainer.models.motion.components.utils.geometry.rotation_convert import (
        axis_angle_to_matrix, matrix_to_axis_angle
    )
    data = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in data.items()}
    poses = data['poses'].copy().astype(np.float64)
    trans = data['trans'].copy().astype(np.float64) if 'trans' in data else data.get('transl', np.zeros((poses.shape[0], 3))).copy().astype(np.float64)

    global_orient = poses[:, :3]
    R0 = axis_angle_to_matrix(torch.tensor(global_orient[0], dtype=torch.float32)).numpy()
    forward = R0 @ np.array([0.0, 0.0, 1.0])
    yaw = np.arctan2(forward[0], forward[2])
    c, s = np.cos(-yaw), np.sin(-yaw)
    R_yaw = np.array([[c, 0., s], [0., 1., 0.], [-s, 0., c]])

    T_len = global_orient.shape[0]
    go_t = torch.tensor(global_orient, dtype=torch.float32)
    R_t_all = axis_angle_to_matrix(go_t).numpy()
    R_yaw_t = np.tile(R_yaw[None], (T_len, 1, 1))
    R_corrected = np.einsum('tij,tjk->tik', R_yaw_t, R_t_all)
    corrected_go = matrix_to_axis_angle(torch.tensor(R_corrected, dtype=torch.float32)).numpy()

    trans = (R_yaw @ trans.T).T
    trans[:, 0] -= trans[0, 0]
    trans[:, 2] -= trans[0, 2]
    min_y = trans[:, 1].min()
    trans[:, 1] -= min_y

    poses[:, :3] = corrected_go
    data['poses'] = poses.astype(np.float32)
    data['trans'] = trans.astype(np.float32)
    if 'transl' in data:
        data['transl'] = data['trans']
    return data


def save_npz_safe(path: Path, data: Dict):
    """Save NPZ, removing existing file first to avoid permission issues."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        try:
            path.unlink()
        except PermissionError:
            # If we can't delete, try to use a temp name then rename
            import tempfile, shutil
            tmp = path.parent / (path.stem + '.tmp.npz')
            save_dict = {k: v for k, v in data.items() if not k.startswith('_cached')}
            np.savez(str(tmp), **save_dict)
            try:
                os.rename(str(tmp), str(path))
            except Exception:
                pass
            return
    save_dict = {k: v for k, v in data.items() if not k.startswith('_cached')}
    np.savez(str(path), **save_dict)


def find_changed_frames(poses_raw: np.ndarray, poses_clean: np.ndarray, threshold: float = 0.05) -> np.ndarray:
    """
    判断哪些帧被"显著修复"了。
    由于清洗过程会对整体运动做全局平滑处理，几乎每一帧都会有微小差异。
    使用 0.05 弧度阈值来区分"真正被修复的帧"和"仅有数值噪声的帧"。
    """
    frame_diffs = np.abs(poses_raw - poses_clean).max(axis=tuple(range(1, poses_raw.ndim)))
    return frame_diffs > threshold


def compute_slice_segments(T: int, changed_mask: np.ndarray, max_frames: int = MAX_FRAMES) -> List[Tuple[int, int]]:
    if T == 0:
        return []

    segments = []
    covered = np.zeros(T, dtype=bool)
    change_indices = np.where(changed_mask)[0]

    if len(change_indices) > 0:
        runs = []
        cur_start = change_indices[0]
        cur_end = change_indices[0]
        for idx in change_indices[1:]:
            if idx == cur_end + 1:
                cur_end = idx
            else:
                runs.append((cur_start, cur_end + 1))
                cur_start = idx
                cur_end = idx
        runs.append((cur_start, cur_end + 1))

        i = 0
        while i < len(runs):
            win_start = runs[i][0]
            win_end = runs[i][1]
            while i + 1 < len(runs) and (runs[i + 1][1] - win_start) <= max_frames:
                i += 1
                win_end = runs[i][1]
            run_len = win_end - win_start
            pad = max_frames - run_len
            pad_left = min(win_start, pad // 2)
            pad_right = min(T - win_end, pad - pad_left)
            pad_left = min(win_start, pad - pad_right)
            seg_start = win_start - pad_left
            seg_end = min(T, seg_start + max_frames)
            seg_start = max(0, seg_start)
            seg_end = min(T, seg_end)
            if seg_end - seg_start < min(max_frames, T) and seg_end < T:
                seg_end = min(T, seg_start + max_frames)
            elif seg_end - seg_start < min(max_frames, T) and seg_start > 0:
                seg_start = max(0, seg_end - max_frames)
            segments.append((seg_start, seg_end))
            covered[seg_start:seg_end] = True
            i += 1

    uncov_start = None
    for i in range(T + 1):
        in_uncov = (i < T) and (not covered[i])
        if in_uncov and uncov_start is None:
            uncov_start = i
        elif (not in_uncov or i == T) and uncov_start is not None:
            uncov_end = i
            cur = uncov_start
            while cur < uncov_end:
                end = min(cur + max_frames, uncov_end)
                segments.append((cur, end))
                cur = end
            uncov_start = None

    segments.sort(key=lambda x: x[0])
    merged = []
    for seg in segments:
        if merged and seg[0] < merged[-1][1]:
            prev = merged[-1]
            merged[-1] = (prev[0], max(prev[1], seg[1]))
        else:
            merged.append(seg)

    final = []
    for (s, e) in merged:
        while e - s > max_frames:
            final.append((s, s + max_frames))
            s += max_frames
        if s < e:
            final.append((s, e))
    return final


def slice_and_save(
    raw_path: Path, clean_path: Optional[Path],
    checker: MotionQualityChecker, out_dir: Path
) -> List[Dict]:
    raw_data = load_npz(raw_path)
    raw_data = resample_to_30fps(raw_data)

    if clean_path is not None:
        clean_data = load_npz(clean_path)
        clean_data = resample_to_30fps(clean_data)
    else:
        clean_data = None

    T = raw_data['poses'].shape[0]
    stem = raw_path.stem
    if '__' in stem:
        base_part, date_suffix = stem.split('__', 1)
        out_stem = f'{base_part}__{date_suffix}'
    else:
        out_stem = stem

    if clean_data is not None and clean_data['poses'].shape[0] == T:
        changed_mask = find_changed_frames(raw_data['poses'], clean_data['poses'])
        segments = compute_slice_segments(T, changed_mask, MAX_FRAMES)
    else:
        changed_mask = np.zeros(T, dtype=bool)
        segments = [(i, min(i + MAX_FRAMES, T)) for i in range(0, T, MAX_FRAMES)]

    results = []
    for seg_start, seg_end in segments:
        if seg_end <= seg_start:
            continue

        raw_seg = {
            k: (v[seg_start:seg_end] if isinstance(v, np.ndarray) and v.ndim >= 1 and v.shape[0] == T else v)
            for k, v in raw_data.items()
        }
        raw_seg = canonicalize_smplh(raw_seg)

        if clean_data is not None and clean_data['poses'].shape[0] == T:
            clean_seg = {
                k: (v[seg_start:seg_end] if isinstance(v, np.ndarray) and v.ndim >= 1 and v.shape[0] == T else v)
                for k, v in clean_data.items()
            }
            clean_seg = canonicalize_smplh(clean_seg)
            seg_changed = int(changed_mask[seg_start:seg_end].sum())
            is_identical = (seg_changed == 0)
        else:
            clean_seg = None
            seg_changed = 0
            is_identical = True

        raw_out_name = f'{out_stem}_{seg_start}_{seg_end}.npz'
        clean_out_name = f'{out_stem}_{seg_start}_{seg_end}_cleaned.npz'
        raw_out_path = out_dir / raw_out_name
        clean_out_path = out_dir / clean_out_name

        save_npz_safe(raw_out_path, raw_seg)
        if clean_seg is not None and not is_identical:
            save_npz_safe(clean_out_path, clean_seg)

        try:
            qc_result = checker.check(dict(raw_seg))
            category = qc_result.category
            failed = qc_result.failed_checks
            borderline = qc_result.borderline_checks
        except Exception as e:
            category = 'error'
            failed = [f'checker_error: {e}']
            borderline = []

        results.append({
            'source': str(raw_path.name),
            'seg_start': seg_start,
            'seg_end': seg_end,
            'seg_frames': seg_end - seg_start,
            'changed_frames': seg_changed,
            'changed_ratio': seg_changed / (seg_end - seg_start),
            'is_identical': is_identical,
            'has_clean_pair': (clean_seg is not None and not is_identical),
            'raw_out': str(raw_out_path.name),
            'clean_out': str(clean_out_path.name) if (clean_seg is not None and not is_identical) else None,
            'qc_category': category,
            'qc_failed': failed,
            'qc_borderline': borderline,
        })
    return results


def build_pairs(npz_dir: Path) -> List[Tuple[Path, Optional[Path]]]:
    all_files = sorted(npz_dir.glob('*.npz'))
    all_names = {f.name: f for f in all_files}
    pairs = []
    for f in all_files:
        stem = f.stem
        if '_cleaned' in stem:
            continue
        if '__' in stem:
            base_part, date_suffix = stem.split('__', 1)
            clean_name = f'{base_part}_cleaned__{date_suffix}.npz'
        else:
            clean_name = f'{stem}_cleaned.npz'
        clean_path = all_names.get(clean_name)
        pairs.append((f, clean_path))
    return pairs


def main():
    print('Initializing MotionQualityChecker (CPU)...')
    checker = MotionQualityChecker(device='cpu')
    print('Done.')

    pairs = build_pairs(NPZ_DIR)
    print(f'Found {len(pairs)} pairs.')

    print('\n=== Phase 2: Slicing + canonicalizing ===')
    all_seg_results = []
    for i, (raw_path, clean_path) in enumerate(pairs):
        print(f'  [{i+1}/{len(pairs)}] {raw_path.name}', end='', flush=True)
        try:
            segs = slice_and_save(raw_path, clean_path, checker, OUT_DIR)
            all_seg_results.extend(segs)
            changed_segs = sum(1 for s in segs if not s['is_identical'])
            print(f' -> {len(segs)} segs, {changed_segs} with changes')
        except Exception as e:
            print(f' -> ERROR: {e}')
            import traceback; traceback.print_exc()

    print('\n=== Phase 3: Build post-slice report ===')
    total_segs = len(all_seg_results)
    identical_segs = sum(1 for r in all_seg_results if r['is_identical'])
    changed_segs = total_segs - identical_segs
    cat_counter = Counter(r['qc_category'] for r in all_seg_results)
    fail_counter: Counter = Counter()
    borderline_counter: Counter = Counter()
    for r in all_seg_results:
        for f in r['qc_failed']:
            fail_counter[f] += 1
        for b in r['qc_borderline']:
            borderline_counter[b] += 1

    hq_clean = sum(1 for r in all_seg_results if r['has_clean_pair'])
    low_cnt = cat_counter.get('low', 0)
    high_cnt = cat_counter.get('high', 0)

    lines = [
        '# 质检报告2：切片后数据质检',
        '',
        f'生成时间: {time.strftime("%Y-%m-%d %H:%M:%S")}',
        f'切片输出目录: {OUT_DIR}',
        f'最大帧数限制: {MAX_FRAMES} 帧 @ {TARGET_FPS} fps',
        '',
        '## 切片统计',
        '',
        f'| 指标 | 数量 |',
        f'|------|------|',
        f'| 总切片数 | {total_segs} |',
        f'| 含有修复帧的切片（有对应cleaned版本） | {hq_clean} |',
        f'| 完全相同切片（只保存1份） | {identical_segs} |',
        '',
        '## 切片后修复前数据质量分布',
        '',
        f'| 质量等级 | 数量 | 比例 |',
        f'|----------|------|------|',
    ]
    for cat in ['high', 'borderline', 'low', 'error']:
        cnt = cat_counter.get(cat, 0)
        lines.append(f'| {cat} | {cnt} | {cnt/max(total_segs,1)*100:.1f}% |')
    lines.append('')

    if fail_counter:
        lines += [
            '## 切片后修复前数据：失败检查项统计',
            '',
            '| 检查项 | 出现次数 | 占低质量比例 | 占总切片比例 |',
            '|--------|----------|-------------|-------------|',
        ]
        for check_name, cnt in fail_counter.most_common():
            lines.append(
                f'| {check_name} | {cnt} | {cnt/max(low_cnt,1)*100:.1f}% | {cnt/max(total_segs,1)*100:.1f}% |'
            )
        lines.append('')

    if borderline_counter:
        lines += [
            '## 切片后修复前数据：边界检查项统计',
            '',
            '| 检查项 | 出现次数 | 占总切片比例 |',
            '|--------|----------|-------------|',
        ]
        for check_name, cnt in borderline_counter.most_common():
            lines.append(f'| {check_name} | {cnt} | {cnt/max(total_segs,1)*100:.1f}% |')
        lines.append('')

    lines += [
        '## 修复配对数据汇总',
        '',
        f'修复前后配对切片总数（有changed帧的）: **{hq_clean}**',
        f'其中修复前为高质量(high)的: **{sum(1 for r in all_seg_results if r["has_clean_pair"] and r["qc_category"]=="high")}**',
        f'其中修复前为边界(borderline)的: **{sum(1 for r in all_seg_results if r["has_clean_pair"] and r["qc_category"]=="borderline")}**',
        f'其中修复前为低质量(low)的: **{sum(1 for r in all_seg_results if r["has_clean_pair"] and r["qc_category"]=="low")}**',
        '',
    ]

    # Low quality detail
    low_segs = [r for r in all_seg_results if r['qc_category'] == 'low']
    if low_segs:
        lines += [
            '## 低质量切片详情（前100条）',
            '',
            '| 切片文件 | 帧数 | 修复帧数 | 失败检查项 |',
            '|----------|------|----------|-----------|',
        ]
        for r in low_segs[:100]:
            lines.append(
                f'| {r["raw_out"]} | {r["seg_frames"]} | {r["changed_frames"]} | {", ".join(r["qc_failed"])} |'
            )
        if len(low_segs) > 100:
            lines.append(f'| ... ({len(low_segs)-100} more) | | | |')
        lines.append('')

    # Changed pairs detail
    changed_pairs = [r for r in all_seg_results if r['has_clean_pair']]
    if changed_pairs:
        lines += [
            '## 修复配对切片（前80条）',
            '',
            '| 原始切片 | 修复切片 | 帧数 | 修复帧数 | 修复比例 | 原始质量 |',
            '|----------|----------|------|----------|----------|----------|',
        ]
        for r in changed_pairs[:80]:
            lines.append(
                f'| {r["raw_out"]} | {r["clean_out"]} | {r["seg_frames"]} | '
                f'{r["changed_frames"]} | {r["changed_ratio"]*100:.1f}% | {r["qc_category"]} |'
            )
        if len(changed_pairs) > 80:
            lines.append(f'| ... ({len(changed_pairs)-80} more) | | | | | |')
        lines.append('')

    report2 = '\n'.join(lines)
    report2_path = REPORT_DIR / 'quality_report_2_post_slice.md'
    report2_path.write_text(report2, encoding='utf-8')
    print(f'Report 2 saved: {report2_path}')

    with open(REPORT_DIR / 'quality_report_2_post_slice.json', 'w') as f:
        json.dump(all_seg_results, f, indent=2, ensure_ascii=False, default=str)

    print('\n=== Complete! ===')
    print(f'Total segments: {total_segs}')
    print(f'  With changes (paired): {changed_segs}')
    print(f'  Identical: {identical_segs}')
    print(f'Quality: high={high_cnt}, borderline={cat_counter.get("borderline",0)}, low={low_cnt}')


if __name__ == '__main__':
    main()
