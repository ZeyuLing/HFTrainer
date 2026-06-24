#!/usr/bin/env python3.8
"""
CJGame MB NPZ 数据质检、切片、坐标系标准化一体化处理脚本

步骤：
1. 对修复前数据进行质检，生成质检报告1（pre-slice）
2. 对数据进行切片（按规则：≤360帧@30fps，修复前后对应，集中修复帧）
3. 切片后进行坐标系标准化（第一帧面朝+Z，xz原点，y=0地面）
4. 对切片后修复前数据进行质检，生成质检报告2（post-slice）

用法：
    python3.8 scripts/process_cjgame_npz.py
"""

import sys, types, importlib.util, os

# ─── 0. Bootstrap: inject stub packages so quality checkers can be imported ─────
_HFT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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
    _subpath = _BASE + _pkg[len('hftrainer'):]
    _make_pkg(_pkg, _subpath)

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

# ─── 1. Real imports ────────────────────────────────────────────────────────────
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from hftrainer.evaluation.quality_check_rules.motion_quality_checker import MotionQualityChecker

# ─── Config ────────────────────────────────────────────────────────────────────
NPZ_DIR = Path(_HFT_ROOT) / 'data/lightai_data/CJGame_MB/npz'
OUT_DIR  = Path(_HFT_ROOT) / 'data/lightai_data/CJGame_MB/npz_split'
REPORT_DIR = Path(_HFT_ROOT) / 'data/lightai_data/CJGame_MB'

TARGET_FPS   = 30
MAX_FRAMES   = 360   # at 30 fps
# If a sliced segment has fewer changed frames than this threshold, skip saving both files
# and only save one copy (since they're identical)
IDENTITY_SEGMENT_SAVE_ONCE = True

OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# ─── Helpers ───────────────────────────────────────────────────────────────────

def load_npz(path: Path) -> Dict:
    return dict(np.load(path, allow_pickle=True))


def fps_of(data: Dict) -> float:
    return float(data.get('mocap_framerate', 30.0))


def resample_to_30fps(data: Dict) -> Dict:
    """Resample motion data to TARGET_FPS=30 if needed."""
    src_fps = fps_of(data)
    if abs(src_fps - TARGET_FPS) < 0.5:
        return data
    ratio = src_fps / TARGET_FPS
    T = data['poses'].shape[0]
    # Uniform re-sample by nearest-neighbor
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
    坐标系标准化：
    - 第0帧朝向 +Z 轴（绕Y轴旋转对齐）
    - xz 平移归零（第0帧在原点）
    - y轴地面归零（全序列最低点对齐 y=0）

    参考 smpl_processor.py::normalize_smplx_dict()
    SMPLH poses: (T, 156), global_orient = poses[:, :3]
    """
    from hftrainer.models.motion.components.utils.geometry.rotation_convert import (
        axis_angle_to_matrix, matrix_to_axis_angle
    )

    data = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in data.items()}
    poses = data['poses'].copy().astype(np.float64)   # (T, 156)
    trans = data['trans'].copy().astype(np.float64)   # (T, 3)

    global_orient = poses[:, :3]  # (T, 3)

    # Step 1: compute yaw from first-frame facing direction
    R0 = axis_angle_to_matrix(
        torch.tensor(global_orient[0], dtype=torch.float32)
    ).numpy()  # (3, 3)
    forward = R0 @ np.array([0.0, 0.0, 1.0])
    yaw = np.arctan2(forward[0], forward[2])

    # Build R_yaw (rotate -yaw around Y)
    c, s = np.cos(-yaw), np.sin(-yaw)
    R_yaw = np.array([[c, 0., s], [0., 1., 0.], [-s, 0., c]])

    # Step 2: apply yaw correction to all frames
    T_len = global_orient.shape[0]
    corrected_go = np.zeros_like(global_orient)
    go_t = torch.tensor(global_orient, dtype=torch.float32)
    R_t_all = axis_angle_to_matrix(go_t).numpy()  # (T, 3, 3)
    R_yaw_t = R_yaw[None].repeat(T_len, 0)  # broadcast
    R_corrected = np.einsum('tij,tjk->tik', R_yaw_t, R_t_all)
    # convert back to axis-angle
    R_corrected_t = torch.tensor(R_corrected, dtype=torch.float32)
    corrected_go_t = matrix_to_axis_angle(R_corrected_t)
    corrected_go = corrected_go_t.numpy()

    # Apply yaw to translation
    trans = (R_yaw @ trans.T).T

    # Step 3: XZ centering
    trans[:, 0] -= trans[0, 0]
    trans[:, 2] -= trans[0, 2]

    # Step 4: ground normalization (use pelvis height since no body model here)
    min_y = trans[:, 1].min()
    trans[:, 1] -= min_y

    # Step 5: write back
    poses[:, :3] = corrected_go
    data['poses'] = poses.astype(np.float32)
    data['trans'] = trans.astype(np.float32)
    if 'transl' in data:
        data['transl'] = data['trans']
    return data


def matrix_to_axis_angle(rot_mats: torch.Tensor) -> torch.Tensor:
    """Convert rotation matrices (T, 3, 3) to axis-angle (T, 3)."""
    # Use the rotation_convert module we loaded
    from hftrainer.models.motion.components.utils.geometry.rotation_convert import (
        matrix_to_axis_angle as _m2aa
    )
    return _m2aa(rot_mats)


def save_npz(path: Path, data: Dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    # Only save arrays + scalar values; skip cached computation keys
    save_dict = {
        k: v for k, v in data.items()
        if not k.startswith('_cached')
    }
    np.savez(str(path), **save_dict)


# ─── Pairing logic ─────────────────────────────────────────────────────────────

def build_pairs(npz_dir: Path) -> List[Tuple[Path, Optional[Path]]]:
    """
    Build (raw_path, clean_path_or_None) pairs.

    Naming rules:
    - Standard: Xxx.npz + Xxx_cleaned.npz
    - Date-variant: Xxx__YYYYMMDD.npz + Xxx_cleaned__YYYYMMDD.npz
    """
    all_files = sorted(npz_dir.glob('*.npz'))
    all_names = {f.name: f for f in all_files}

    pairs = []
    seen_raw = set()

    for f in all_files:
        name = f.name
        stem = f.stem  # without .npz

        # Skip if it's a cleaned file
        if '_cleaned' in stem:
            continue

        # Determine expected cleaned name
        # Handle date suffix: Xxx__YYYYMMDD.npz -> Xxx_cleaned__YYYYMMDD.npz
        if '__' in stem:
            base_part, date_suffix = stem.split('__', 1)
            clean_name = f'{base_part}_cleaned__{date_suffix}.npz'
        else:
            clean_name = f'{stem}_cleaned.npz'

        clean_path = all_names.get(clean_name)
        pairs.append((f, clean_path))
        seen_raw.add(name)

    return pairs


# ─── Slicing logic ──────────────────────────────────────────────────────────────

def find_changed_frames(poses_raw: np.ndarray, poses_clean: np.ndarray) -> np.ndarray:
    """Return boolean mask of frames where raw != clean."""
    diff = np.any(poses_raw != poses_clean, axis=tuple(range(1, poses_raw.ndim)))
    return diff  # (T,)


def compute_slice_segments(
    T: int,
    changed_mask: np.ndarray,
    max_frames: int = MAX_FRAMES,
) -> List[Tuple[int, int]]:
    """
    Compute non-overlapping slice segments [start, end) such that:
    - Each segment length ≤ max_frames
    - Segments cover the full sequence
    - Changed frames are concentrated within segments (greedy window centering)

    Strategy:
    1. Find contiguous blocks of changed frames.
    2. Greedily extend each block to max_frames, centered around the block.
    3. Fill uncovered regions with uniform cuts.
    """
    if T == 0:
        return []

    segments = []
    covered = np.zeros(T, dtype=bool)

    # Find changed-frame runs
    change_indices = np.where(changed_mask)[0]

    if len(change_indices) > 0:
        # Group consecutive changed frames into runs
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

        # Greedily merge runs into windows of max_frames
        merged = []
        i = 0
        while i < len(runs):
            win_start = runs[i][0]
            win_end = runs[i][1]
            # Expand window by including more runs
            while i + 1 < len(runs) and (runs[i + 1][1] - win_start) <= max_frames:
                i += 1
                win_end = runs[i][1]
            # Center the window
            run_len = win_end - win_start
            pad = max_frames - run_len
            pad_left = min(win_start, pad // 2)
            pad_right = min(T - win_end, pad - pad_left)
            pad_left = min(win_start, pad - pad_right)
            seg_start = win_start - pad_left
            seg_end = min(T, seg_start + max_frames)
            # Clip
            seg_start = max(0, seg_start)
            seg_end = min(T, seg_end)
            # Ensure we always get a window of max_frames if possible
            if seg_end - seg_start < min(max_frames, T) and seg_end < T:
                seg_end = min(T, seg_start + max_frames)
            elif seg_end - seg_start < min(max_frames, T) and seg_start > 0:
                seg_start = max(0, seg_end - max_frames)
            merged.append((seg_start, seg_end))
            covered[seg_start:seg_end] = True
            i += 1
        segments.extend(merged)

    # Fill uncovered regions with uniform cuts
    uncov_start = None
    for i in range(T + 1):
        in_uncov = (i < T) and (not covered[i])
        if in_uncov and uncov_start is None:
            uncov_start = i
        elif (not in_uncov or i == T) and uncov_start is not None:
            uncov_end = i
            # Split [uncov_start, uncov_end) into max_frames chunks
            cur = uncov_start
            while cur < uncov_end:
                end = min(cur + max_frames, uncov_end)
                segments.append((cur, end))
                cur = end
            uncov_start = None

    # Sort and merge overlapping segments (shouldn't happen, but be safe)
    segments.sort(key=lambda x: x[0])
    # Remove duplicates / exact overlaps
    merged_segs = []
    for seg in segments:
        if merged_segs and seg[0] < merged_segs[-1][1]:
            # Overlapping — keep the one with more changed frames or merge
            prev = merged_segs[-1]
            merged_segs[-1] = (prev[0], max(prev[1], seg[1]))
        else:
            merged_segs.append(seg)

    # Final check: split any segment > max_frames
    final_segs = []
    for (s, e) in merged_segs:
        while e - s > max_frames:
            final_segs.append((s, s + max_frames))
            s += max_frames
        if s < e:
            final_segs.append((s, e))

    return final_segs


def slice_and_save(
    raw_path: Path,
    clean_path: Optional[Path],
    checker: MotionQualityChecker,
    out_dir: Path,
) -> List[Dict]:
    """
    Slice a pair of (raw, clean) files and save to out_dir.
    Returns list of segment info dicts for reporting.
    """
    raw_data = load_npz(raw_path)
    raw_data = resample_to_30fps(raw_data)

    if clean_path is not None:
        clean_data = load_npz(clean_path)
        clean_data = resample_to_30fps(clean_data)
    else:
        clean_data = None

    T = raw_data['poses'].shape[0]
    stem = raw_path.stem  # e.g. "Base_Stand_Lobby_Performance01_012"
    # Determine output filename base (without _cleaned suffix distinction)
    if '__' in stem:
        base_part, date_suffix = stem.split('__', 1)
        out_stem = f'{base_part}__{date_suffix}'
    else:
        out_stem = stem

    # Determine segments
    if clean_data is not None and clean_data['poses'].shape[0] == T:
        # Same length — use changed-frame-aware slicing
        changed_mask = find_changed_frames(raw_data['poses'], clean_data['poses'])
        segments = compute_slice_segments(T, changed_mask, MAX_FRAMES)
    else:
        # Different length or no clean — simple uniform cuts
        changed_mask = np.zeros(T, dtype=bool)
        segments = [(i, min(i + MAX_FRAMES, T)) for i in range(0, T, MAX_FRAMES)]

    results = []
    for seg_start, seg_end in segments:
        if seg_end <= seg_start:
            continue

        # Slice raw
        raw_seg = {
            k: (v[seg_start:seg_end] if isinstance(v, np.ndarray) and v.ndim >= 1 and v.shape[0] == T else v)
            for k, v in raw_data.items()
        }
        raw_seg = canonicalize_smplh(raw_seg)

        # Slice clean (if available and same length)
        if clean_data is not None and clean_data['poses'].shape[0] == T:
            clean_seg = {
                k: (v[seg_start:seg_end] if isinstance(v, np.ndarray) and v.ndim >= 1 and v.shape[0] == T else v)
                for k, v in clean_data.items()
            }
            clean_seg = canonicalize_smplh(clean_seg)
            seg_changed = int(changed_mask[seg_start:seg_end].sum())
            seg_total = seg_end - seg_start
            is_identical = (seg_changed == 0)
        else:
            clean_seg = None
            seg_changed = 0
            seg_total = seg_end - seg_start
            is_identical = True

        # Build output paths
        # e.g.: out_stem_{start}_{end}_cleaned.npz
        raw_out_name = f'{out_stem}_{seg_start}_{seg_end}.npz'
        clean_out_name = f'{out_stem}_{seg_start}_{seg_end}_cleaned.npz'
        raw_out_path = out_dir / raw_out_name
        clean_out_path = out_dir / clean_out_name

        # Save
        save_npz(raw_out_path, raw_seg)
        if clean_seg is not None and not is_identical:
            save_npz(clean_out_path, clean_seg)
        # If identical, clean version is not saved (same as raw)

        # Quality check on raw segment
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


# ─── Report generation ─────────────────────────────────────────────────────────

def build_report(
    check_results: List[Dict],
    title: str,
) -> str:
    total = len(check_results)
    cat_counter = Counter(r['qc_category'] for r in check_results)
    high = cat_counter.get('high', 0)
    borderline = cat_counter.get('borderline', 0)
    low = cat_counter.get('low', 0)
    error = cat_counter.get('error', 0)

    # Failed check frequency
    fail_counter: Counter = Counter()
    borderline_counter: Counter = Counter()
    for r in check_results:
        for f in r['qc_failed']:
            fail_counter[f] += 1
        for b in r['qc_borderline']:
            borderline_counter[b] += 1

    lines = [
        f'# {title}',
        '',
        f'生成时间: {time.strftime("%Y-%m-%d %H:%M:%S")}',
        '',
        '## 总体统计',
        '',
        f'| 指标 | 数量 | 比例 |',
        f'|------|------|------|',
        f'| 总样本 | {total} | 100% |',
        f'| 高质量 (high) | {high} | {high/max(total,1)*100:.1f}% |',
        f'| 边界 (borderline) | {borderline} | {borderline/max(total,1)*100:.1f}% |',
        f'| 低质量 (low) | {low} | {low/max(total,1)*100:.1f}% |',
        f'| 检查错误 (error) | {error} | {error/max(total,1)*100:.1f}% |',
        '',
    ]

    if fail_counter:
        lines += [
            '## 失败检查项统计（低质量原因）',
            '',
            '| 检查项 | 出现次数 | 占低质量比例 |',
            '|--------|----------|-------------|',
        ]
        for check_name, cnt in fail_counter.most_common():
            lines.append(f'| {check_name} | {cnt} | {cnt/max(low,1)*100:.1f}% |')
        lines.append('')

    if borderline_counter:
        lines += [
            '## 边界检查项统计',
            '',
            '| 检查项 | 出现次数 |',
            '|--------|----------|',
        ]
        for check_name, cnt in borderline_counter.most_common():
            lines.append(f'| {check_name} | {cnt} |')
        lines.append('')

    # Per-file detail (low quality)
    low_results = [r for r in check_results if r['qc_category'] == 'low']
    if low_results:
        lines += [
            '## 低质量样本详情（前50条）',
            '',
            '| 文件 | 帧数 | 失败检查项 |',
            '|------|------|-----------|',
        ]
        for r in low_results[:50]:
            fname = r.get('raw_out', r.get('source', '?'))
            frames = r.get('seg_frames', r.get('frames', '?'))
            fails = ', '.join(r['qc_failed'])
            lines.append(f'| {fname} | {frames} | {fails} |')
        if len(low_results) > 50:
            lines.append(f'| ... ({len(low_results)-50} more) | | |')
        lines.append('')

    return '\n'.join(lines)


def build_pre_slice_report(pair_results: List[Dict]) -> str:
    """Report on original (pre-slice) files."""
    total = len(pair_results)
    cat_counter = Counter(r['qc_category'] for r in pair_results)
    high = cat_counter.get('high', 0)
    borderline_cnt = cat_counter.get('borderline', 0)
    low = cat_counter.get('low', 0)
    error = cat_counter.get('error', 0)

    fail_counter: Counter = Counter()
    borderline_counter_c: Counter = Counter()
    for r in pair_results:
        for f in r.get('qc_failed', []):
            fail_counter[f] += 1
        for b in r.get('qc_borderline', []):
            borderline_counter_c[b] += 1

    lines = [
        '# 质检报告1：切片前修复前数据质检',
        '',
        f'生成时间: {time.strftime("%Y-%m-%d %H:%M:%S")}',
        f'数据目录: {NPZ_DIR}',
        '',
        '## 总体统计',
        '',
        f'| 指标 | 数量 | 比例 |',
        f'|------|------|------|',
        f'| 总样本（修复前文件） | {total} | 100% |',
        f'| 高质量 (high) | {high} | {high/max(total,1)*100:.1f}% |',
        f'| 边界 (borderline) | {borderline_cnt} | {borderline_cnt/max(total,1)*100:.1f}% |',
        f'| 低质量 (low) | {low} | {low/max(total,1)*100:.1f}% |',
        f'| 检查错误 (error) | {error} | {error/max(total,1)*100:.1f}% |',
        '',
    ]

    if fail_counter:
        lines += [
            '## 失败检查项统计（低质量原因）',
            '',
            '| 检查项 | 出现次数 | 占低质量比例 | 占总样本比例 |',
            '|--------|----------|-------------|-------------|',
        ]
        for check_name, cnt in fail_counter.most_common():
            lines.append(
                f'| {check_name} | {cnt} | {cnt/max(low,1)*100:.1f}% | {cnt/max(total,1)*100:.1f}% |'
            )
        lines.append('')

    if borderline_counter_c:
        lines += [
            '## 边界检查项统计',
            '',
            '| 检查项 | 出现次数 | 占总样本比例 |',
            '|--------|----------|-------------|',
        ]
        for check_name, cnt in borderline_counter_c.most_common():
            lines.append(f'| {check_name} | {cnt} | {cnt/max(total,1)*100:.1f}% |')
        lines.append('')

    # Show some problematic files
    low_results = [r for r in pair_results if r['qc_category'] == 'low']
    if low_results:
        lines += [
            '## 低质量样本详情（前60条）',
            '',
            '| 文件 | FPS | 帧数 | 失败检查项 |',
            '|------|-----|------|-----------|',
        ]
        for r in low_results[:60]:
            fname = r.get('file', '?')
            fps = r.get('fps', '?')
            frames = r.get('frames', '?')
            fails = ', '.join(r.get('qc_failed', []))
            lines.append(f'| {fname} | {fps} | {frames} | {fails} |')
        if len(low_results) > 60:
            lines.append(f'| ... ({len(low_results)-60} more) | | | |')
        lines.append('')

    return '\n'.join(lines)


# ─── Main ──────────────────────────────────────────────────────────────────────

def main():
    print('Initializing MotionQualityChecker (CPU)...')
    # Force CPU: torch 1.13.1 on this host has CUDA 11.7 libs but system has CUDA 11.0,
    # causing libnvrtc.so errors in FK computations that produce false positive failures.
    device = 'cpu'
    print(f'  Using device: {device}')
    checker = MotionQualityChecker(device=device)
    print('  Done.')

    pairs = build_pairs(NPZ_DIR)
    print(f'Found {len(pairs)} raw/clean file pairs.')

    # ── Phase 1: Pre-slice quality check on raw files ──────────────────────────
    print('\n=== Phase 1: Pre-slice quality check ===')
    pre_slice_results = []
    for i, (raw_path, clean_path) in enumerate(pairs):
        print(f'  [{i+1}/{len(pairs)}] {raw_path.name}', end='', flush=True)
        try:
            data = load_npz(raw_path)
            fps = fps_of(data)
            T = data['poses'].shape[0]
            qc = checker.check(dict(data))
            pre_slice_results.append({
                'file': raw_path.name,
                'fps': fps,
                'frames': T,
                'has_clean': clean_path is not None,
                'qc_category': qc.category,
                'qc_failed': qc.failed_checks,
                'qc_borderline': qc.borderline_checks,
            })
            print(f' -> {qc.category} | failed: {qc.failed_checks}')
        except Exception as e:
            pre_slice_results.append({
                'file': raw_path.name,
                'fps': 0,
                'frames': 0,
                'has_clean': clean_path is not None,
                'qc_category': 'error',
                'qc_failed': [str(e)],
                'qc_borderline': [],
            })
            print(f' -> ERROR: {e}')

    # Save pre-slice report
    report1 = build_pre_slice_report(pre_slice_results)
    report1_path = REPORT_DIR / 'quality_report_1_pre_slice.md'
    report1_path.write_text(report1, encoding='utf-8')
    print(f'\nReport 1 saved: {report1_path}')

    # Save JSON for reference
    with open(REPORT_DIR / 'quality_report_1_pre_slice.json', 'w') as f:
        json.dump(pre_slice_results, f, indent=2, ensure_ascii=False, default=str)

    # ── Phase 2: Slice + canonicalize + save ─────────────────────────────────
    print('\n=== Phase 2: Slicing + canonicalizing ===')
    all_seg_results = []
    for i, (raw_path, clean_path) in enumerate(pairs):
        print(f'  [{i+1}/{len(pairs)}] {raw_path.name}', end='', flush=True)
        try:
            segs = slice_and_save(raw_path, clean_path, checker, OUT_DIR)
            all_seg_results.extend(segs)
            changed_segs = sum(1 for s in segs if not s['is_identical'])
            print(f' -> {len(segs)} segments, {changed_segs} with changes')
        except Exception as e:
            print(f' -> ERROR: {e}')
            import traceback; traceback.print_exc()

    # ── Phase 3: Post-slice quality check ──────────────────────────────────────
    print('\n=== Phase 3: Post-slice quality report ===')
    total_segs = len(all_seg_results)
    identical_segs = sum(1 for r in all_seg_results if r['is_identical'])
    changed_segs = total_segs - identical_segs

    cat_counter = Counter(r['qc_category'] for r in all_seg_results)
    fail_counter: Counter = Counter()
    for r in all_seg_results:
        for f in r['qc_failed']:
            fail_counter[f] += 1

    # Build post-slice report
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
        f'| 含有修复帧的切片（修复前后不同） | {changed_segs} |',
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

    # High-quality clean data stats
    hq_clean = sum(1 for r in all_seg_results if r['has_clean_pair'])
    lines += [
        '## 高质量修复数据（修复前后不同且已保存配对）',
        '',
        f'修复前后配对切片总数: **{hq_clean}**',
        '',
        '其中，切片后修复后数据的质量（修复前=low但有clean版本的）：',
    ]

    # Check clean segment quality for low-quality raw segments
    clean_to_check = [r for r in all_seg_results if r['has_clean_pair'] and r['clean_out'] is not None]
    lines += [
        '',
        f'需要人工验证的修复对: {len(clean_to_check)} 对',
        '',
    ]

    if fail_counter:
        lines += [
            '## 切片后修复前数据：失败检查项统计',
            '',
            '| 检查项 | 出现次数 | 占低质量比例 |',
            '|--------|----------|-------------|',
        ]
        low_cnt = cat_counter.get('low', 0)
        for check_name, cnt in fail_counter.most_common():
            lines.append(f'| {check_name} | {cnt} | {cnt/max(low_cnt,1)*100:.1f}% |')
        lines.append('')

    # Detailed per-segment table for low quality
    low_segs = [r for r in all_seg_results if r['qc_category'] == 'low']
    if low_segs:
        lines += [
            '## 低质量切片详情（前80条）',
            '',
            '| 切片文件 | 帧数 | 修复帧数 | 失败检查项 |',
            '|----------|------|----------|-----------|',
        ]
        for r in low_segs[:80]:
            lines.append(
                f'| {r["raw_out"]} | {r["seg_frames"]} | {r["changed_frames"]} | {", ".join(r["qc_failed"])} |'
            )
        if len(low_segs) > 80:
            lines.append(f'| ... ({len(low_segs)-80} more) | | | |')
        lines.append('')

    # High quality pairs detail
    if clean_to_check:
        lines += [
            '## 修复配对切片（有修复帧的切片，前50条）',
            '',
            '| 原始切片 | 修复切片 | 帧数 | 修复帧数 | 修复比例 | 原始质量 |',
            '|----------|----------|------|----------|----------|----------|',
        ]
        for r in clean_to_check[:50]:
            lines.append(
                f'| {r["raw_out"]} | {r["clean_out"]} | {r["seg_frames"]} | '
                f'{r["changed_frames"]} | {r["changed_ratio"]*100:.1f}% | {r["qc_category"]} |'
            )
        if len(clean_to_check) > 50:
            lines.append(f'| ... ({len(clean_to_check)-50} more) | | | | | |')
        lines.append('')

    report2 = '\n'.join(lines)
    report2_path = REPORT_DIR / 'quality_report_2_post_slice.md'
    report2_path.write_text(report2, encoding='utf-8')
    print(f'\nReport 2 saved: {report2_path}')

    # Save JSON
    with open(REPORT_DIR / 'quality_report_2_post_slice.json', 'w') as f:
        json.dump(all_seg_results, f, indent=2, ensure_ascii=False, default=str)

    print('\n=== Complete! ===')
    print(f'Pre-slice report: {report1_path}')
    print(f'Post-slice report: {report2_path}')
    print(f'Sliced data: {OUT_DIR}')
    print(f'Total segments produced: {total_segs}')
    print(f'  Changed pairs: {changed_segs}')
    print(f'  Identical (single file): {identical_segs}')


if __name__ == '__main__':
    main()
