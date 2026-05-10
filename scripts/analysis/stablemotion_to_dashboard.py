"""StableMotion E9 Dashboard Ingestion (2026-04-23).

Post-processes StableMotion inference outputs from
`output/eval_v2_e9_stablemotion_20260423/npz/XXXXX.npz` (containing
lq_motion_135 / hq_motion_135 / stablemotion_label) into:

  1) Dashboard-compatible NPZ at
     `<out-dir>/stablemotion/E9_StableMotion/npz/XXXXX.npz`
     containing (motion_135, positions, translation).
  2) Flat import JSON at
     `<out-dir>/import_jsons/stablemotion__E9_StableMotion.json`
     with `aggregated` + `per_sample` metrics (jitter, bone_length,
     foot, fk_consistency, qc_*) so data_importer.py can ingest it.

Usage:
    python3 scripts/stablemotion_to_dashboard.py \
        --src output/eval_v2_e9_stablemotion_20260423 \
        --out-dir output/eval_v2_e9_stablemotion_20260423 \
        --eval-datalist data/eval/m2m_v2/eval_e9_repair.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from hftrainer.evaluation.motion.m2m_eval_metrics import (  # noqa: E402
    compute_all_metrics, aggregate_metrics, motion135_to_positions_np,
)
# Reuse the eval script's QC helper so we produce keys in exactly the
# schema the dashboard already understands (qc_pass, qc_num_failed,
# qc_<checker_name>, etc.).
sys.path.insert(0, str(PROJECT_ROOT / 'tools'))
from eval_m2m_v2_all_tasks import _run_quality_checker  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', type=str,
                        default='output/eval_v2_e9_stablemotion_20260423',
                        help='Directory with npz/XXXXX.npz from run_stablemotion_e9.py')
    parser.add_argument('--out-dir', type=str,
                        default='output/eval_v2_e9_stablemotion_20260423',
                        help='Output directory for dashboard-ready NPZ + JSON')
    parser.add_argument('--eval-datalist', type=str,
                        default='data/eval/m2m_v2/eval_e9_repair.json')
    parser.add_argument('--model-name', type=str, default='stablemotion',
                        help='Name to register in dashboard.')
    parser.add_argument('--setting', type=str, default='StableMotion',
                        help='Setting name within E9 for dashboard.')
    parser.add_argument('--checkpoint-tag', type=str,
                        default='ref_repo/StableMotion/save/stablemotion_brokenamass.pt')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    src_dir = Path(args.src)
    src_npz_dir = src_dir / 'npz'
    out_root = Path(args.out_dir)
    task_key = f'E9_{args.setting}'
    out_npz_dir = out_root / args.model_name / task_key / 'npz'
    out_npz_dir.mkdir(parents=True, exist_ok=True)
    import_json_dir = out_root / 'import_jsons'
    import_json_dir.mkdir(exist_ok=True)

    # Load datalist to align prompt_id / captions with sample idx
    with open(args.eval_datalist) as f:
        dl = json.load(f)
    items = dl.get('data_list', dl)

    bone_offsets = torch.load(
        PROJECT_ROOT / 'data/hymotion_m2m_data/bone_offsets_22.pt',
        map_location='cpu', weights_only=False,
    ).numpy()

    per_sample_list = []
    npz_files = sorted(src_npz_dir.glob('*.npz'))
    print(f'[convert] {len(npz_files)} StableMotion NPZs → dashboard format')
    for i, p in enumerate(npz_files):
        d = np.load(p, allow_pickle=True)
        hq = d['hq_motion_135'].astype(np.float32)          # (T, 135)
        T = hq.shape[0]

        # ---------- Rewrite NPZ for dashboard ----------
        pos = motion135_to_positions_np(hq, bone_offsets)    # (T, 22, 3)
        out_path = out_npz_dir / p.name
        np.savez_compressed(
            out_path, motion_135=hq,
            positions=pos.astype(np.float32),
            translation=hq[:, :3],
        )

        # ---------- Compute metrics ----------
        metrics = compute_all_metrics(
            pred_motion=hq, gt_motion=None, mask=None,
            bone_offsets=bone_offsets, rotation_space='local',
            fps=30.0, compute_fk=True,
        )
        metrics['inference_time'] = 1.5   # approx (real run was 1.5s/sample)

        qc = _run_quality_checker(hq, bone_offsets, device=args.device)
        if qc is not None:
            metrics['qc_pass'] = float(qc.get('is_valid', False))
            metrics['qc_num_failed'] = float(len(qc.get('failed_checks') or []))
            metrics['qc_num_borderline'] = float(
                len(qc.get('borderline_checks') or []))
            for ch_name, ch_info in qc.get('per_checker', {}).items():
                if isinstance(ch_info, dict):
                    is_valid = ch_info.get('is_valid', True)
                else:
                    is_valid = True
                # Keep the dashboard-wide convention: qc_<checker> is a
                # PASS flag (1 = checker passed, 0 = failed). Aggregating
                # these fields gives per-checker pass rates, matching
                # tools/eval_m2m_v2_all_tasks.py and the E9 UI.
                metrics[f'qc_{ch_name}'] = 1.0 if is_valid else 0.0

        # Embed dashboard-required fields
        idx_str = p.stem
        try:
            idx = int(idx_str)
        except ValueError:
            idx = i
        item = items[idx] if idx < len(items) else {}
        metrics['_npz_path'] = str(out_path.resolve())
        metrics['_sample_idx'] = idx
        metrics['_caption'] = item.get('prompt_id', '') or os.path.basename(
            str(d.get('source_path', ''))).removesuffix('.npz')
        metrics['_num_frames'] = T
        # StableMotion detected-corrupt frame count (extra info)
        if 'stablemotion_label' in d.files:
            metrics['stablemotion_detected_frac'] = float(
                np.asarray(d['stablemotion_label']).sum() / max(T, 1))

        per_sample_list.append(metrics)
        if (i + 1) % 20 == 0:
            print(f'  {i+1}/{len(npz_files)} done')

    # ---------- Aggregate ----------
    aggregated = aggregate_metrics(per_sample_list)

    # ---------- Write import JSON ----------
    import json as _json
    from datetime import datetime
    flat = {
        'model': args.model_name,
        'checkpoint': args.checkpoint_tag,
        'rotation_space': 'local',       # HQ is in M2M 135-dim local rot6d
        'has_caption': False,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'task_id': 'E9',
        'setting': args.setting,
        'num_prompts': len(per_sample_list),
        'aggregated': aggregated,
        'per_sample': per_sample_list,
    }
    json_path = import_json_dir / f'{args.model_name}__E9_{args.setting}.json'
    with open(json_path, 'w') as f:
        _json.dump(flat, f, indent=2, default=float)
    print(f'\n✓ Wrote {json_path}')
    print(f'✓ NPZ dir: {out_npz_dir} ({len(per_sample_list)} files)')
    qc_pass_rate = aggregated.get('qc_pass', {}).get('mean', None)
    if qc_pass_rate is not None:
        print(f'  qc_pass mean: {qc_pass_rate:.1%}')
    print(f'  jitter_pos mean: {aggregated.get("jitter_pos", {}).get("mean", 0):.1f}')


if __name__ == '__main__':
    main()
