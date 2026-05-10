"""Plot train loss curves around the v2->v3 sampler switch (2026-04-26 13:32).

Usage:
    python3 tools/m2m_v3_loss_curve.py \
        --out-dir tools/_v3_loss_curves \
        [--smooth 100] [--latest-only]

Reads `work_dirs/<task>/<run_ts>/train.log` for the 4 production m2m_v2 tasks
and emits per-task PNGs with epoch on x-axis and selected loss components
on y-axis. The vertical dashed line marks the v3 sampler restart.
"""
from __future__ import annotations

import argparse
import os
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

WORK_ROOT = Path('work_dirs')

TASKS = {
    'uncond_local': 'hymotion_m2m_v2_uncond_local_046b',
    'uncond_global': 'hymotion_m2m_v2_uncond_global_046b',
    'caption_local_p2': 'hymotion_m2m_v2_caption_local_phase2',
    'caption_global_p2': 'hymotion_m2m_v2_caption_global_phase2',
}

V3_RESTART = datetime(2026, 4, 26, 13, 32)

LINE_RE = re.compile(
    r'^\[(?P<ts>[\d/ :]+)\] hftrainer INFO: '
    r'epoch \[(?P<epoch>\d+)/\d+\]\s+step \[(?P<step>\d+)/(?P<steps_per_epoch>\d+)\]\s+'
    r'loss=(?P<loss>[\d.eE+-]+)\s+'
    r'loss_velocity=(?P<vel>[\d.eE+-]+)\s+'
    r'loss_x1=(?P<x1>[\d.eE+-]+)\s+'
    r'loss_smoothness=(?P<smooth>[\d.eE+-]+)\s+'
    r'loss_fk_consistency=(?P<fk>[\d.eE+-]+)'
)


def parse_log(log_path: Path) -> List[Dict]:
    rows = []
    if not log_path.exists():
        return rows
    with open(log_path, 'r', errors='ignore') as f:
        for line in f:
            m = LINE_RE.match(line)
            if not m:
                continue
            d = m.groupdict()
            try:
                ts = datetime.strptime(d['ts'], '%Y/%m/%d %H:%M:%S')
            except ValueError:
                continue
            rows.append({
                'ts': ts,
                'epoch': int(d['epoch']),
                'step': int(d['step']),
                'steps_per_epoch': int(d['steps_per_epoch']),
                'loss': float(d['loss']),
                'loss_velocity': float(d['vel']),
                'loss_x1': float(d['x1']),
                'loss_smoothness': float(d['smooth']),
                'loss_fk': float(d['fk']),
            })
    return rows


def collect_task_rows(task_dir: Path, latest_only: bool = False) -> List[Dict]:
    runs = sorted([p for p in task_dir.iterdir() if p.is_dir() and re.match(r'^\d{8}_\d{6}$', p.name)])
    if latest_only and runs:
        runs = runs[-1:]
    all_rows: List[Dict] = []
    for r in runs:
        rows = parse_log(r / 'train.log')
        all_rows.extend(rows)
    all_rows.sort(key=lambda x: x['ts'])
    return all_rows


def smooth(arr: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(arr) < window:
        return arr
    cumsum = np.cumsum(np.insert(arr, 0, 0.0))
    sm = (cumsum[window:] - cumsum[:-window]) / window
    pad = np.full(window - 1, sm[0])
    return np.concatenate([pad, sm])


def plot_task(name: str, rows: List[Dict], out_path: Path, smooth_w: int):
    if not rows:
        print(f'  [skip] {name}: no rows')
        return
    epochs = np.array([r['epoch'] + (r['step'] / max(r['steps_per_epoch'], 1)) for r in rows])
    ts = np.array([r['ts'].timestamp() for r in rows])

    components = {
        'loss': 'total',
        'loss_velocity': 'velocity',
        'loss_smoothness': 'smoothness',
        'loss_fk': 'fk_consistency',
    }
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True)
    axes = axes.flatten()

    v3_restart_ts = V3_RESTART.timestamp()
    pre_mask = ts < v3_restart_ts
    post_mask = ~pre_mask

    epoch_at_switch = float('nan')
    if pre_mask.any() and post_mask.any():
        epoch_at_switch = float(epochs[pre_mask].max())

    for ax, (key, label) in zip(axes, components.items()):
        vals = np.array([r[key] for r in rows])
        if key == 'loss_fk':
            vals = np.maximum(vals, 1e-12)
            sm = smooth(vals, smooth_w)
            ax.semilogy(epochs[pre_mask], sm[pre_mask], color='#1f77b4', lw=1.4, label='v2 sampler')
            ax.semilogy(epochs[post_mask], sm[post_mask], color='#d62728', lw=1.4, label='v3 sampler')
        else:
            sm = smooth(vals, smooth_w)
            ax.plot(epochs[pre_mask], sm[pre_mask], color='#1f77b4', lw=1.4, label='v2 sampler')
            ax.plot(epochs[post_mask], sm[post_mask], color='#d62728', lw=1.4, label='v3 sampler')
        if not np.isnan(epoch_at_switch):
            ax.axvline(epoch_at_switch, color='k', ls='--', lw=0.8, alpha=0.6)
            ax.text(epoch_at_switch, ax.get_ylim()[1], ' v3 restart',
                    va='top', ha='left', fontsize=8, color='k')
        ax.set_title(f'{label}  (smoothed window={smooth_w})')
        ax.set_xlabel('epoch')
        ax.set_ylabel(key)
        ax.grid(alpha=0.3)
        ax.legend(loc='best', fontsize=8)

    n_pre = int(pre_mask.sum())
    n_post = int(post_mask.sum())
    fig.suptitle(
        f'{name}: train loss across v2->v3 sampler switch  '
        f'(pre={n_pre} steps, post={n_post} steps)',
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f'  [ok]   {name}: {len(rows)} rows -> {out_path}')


def summarize(name: str, rows: List[Dict]):
    if not rows:
        return
    arr_pre = [r for r in rows if r['ts'] < V3_RESTART]
    arr_post = [r for r in rows if r['ts'] >= V3_RESTART]
    print(f'  {name}:')
    for arr, era in [(arr_pre, 'v2_last'), (arr_post, 'v3')]:
        if not arr:
            print(f'    {era}: (empty)')
            continue
        last_n = arr[-min(500, len(arr)):]
        L = lambda k: np.mean([r[k] for r in last_n])
        print(
            f'    {era}: epoch [{arr[0]["epoch"]}-{arr[-1]["epoch"]}], '
            f'last500_avg loss={L("loss"):.5f} vel={L("loss_velocity"):.5f} '
            f'smooth={L("loss_smoothness"):.5f} fk={L("loss_fk"):.2e}'
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-dir', default='tools/_v3_loss_curves')
    ap.add_argument('--smooth', type=int, default=100, help='moving-avg window over steps')
    ap.add_argument('--latest-only', action='store_true', help='use only latest run dir per task')
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f'[m2m_v3_loss_curve] V3 restart marker: {V3_RESTART.isoformat()}')
    print(f'[m2m_v3_loss_curve] Output dir: {out_dir}')

    task_rows: Dict[str, List[Dict]] = {}
    for short, dirname in TASKS.items():
        task_dir = WORK_ROOT / dirname
        rows = collect_task_rows(task_dir, latest_only=args.latest_only)
        task_rows[short] = rows
        plot_task(short, rows, out_dir / f'{short}.png', args.smooth)

    print()
    print('[summary] last-500-step averages per era:')
    for short, rows in task_rows.items():
        summarize(short, rows)


if __name__ == '__main__':
    main()
