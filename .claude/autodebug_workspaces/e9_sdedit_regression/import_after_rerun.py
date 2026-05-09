"""After eval finishes:
1. Delete old uncond E2/E5/E9/E14/E15/E16 runs from dashboard DB
2. Split final eval_v2_*.json into per-(model,task_setting) JSON files
3. Import each into dashboard
4. Update model epoch column
"""
import json
import sqlite3
import subprocess
from pathlib import Path
import sys

HF = Path('/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer')
DB = HF / 'motion_annot_web/eval_dashboard/eval_dashboard.db'
OUT_DIR = HF / 'work_dirs/all_tasks_after_fix_20260421'
TASKS_TO_REPLACE = ['E2', 'E5', 'E9', 'E14', 'E15', 'E16']
MODEL_IDS = (5, 12)  # uncond_local, uncond_global


def find_result_json():
    candidates = sorted(OUT_DIR.glob('eval_v2_*.json'))
    if not candidates:
        raise FileNotFoundError(f'No eval_v2_*.json found in {OUT_DIR}')
    return candidates[-1]


def delete_old_runs():
    conn = sqlite3.connect(DB)
    cur = conn.cursor()
    placeholders = ','.join('?' * len(TASKS_TO_REPLACE))
    cur.execute(
        f"""SELECT id FROM eval_runs
            WHERE model_id IN {MODEL_IDS} AND task_id IN ({placeholders})""",
        TASKS_TO_REPLACE,
    )
    ids = [r[0] for r in cur.fetchall()]
    if not ids:
        print('  No old runs to delete')
        conn.close()
        return []
    print(f'  Deleting {len(ids)} old runs: {ids}')
    id_ph = ','.join('?' * len(ids))
    cur.execute(f'DELETE FROM agg_metrics WHERE eval_run_id IN ({id_ph})', ids)
    cur.execute(f'DELETE FROM sample_results WHERE eval_run_id IN ({id_ph})', ids)
    cur.execute(f'DELETE FROM eval_runs WHERE id IN ({id_ph})', ids)
    conn.commit()
    conn.close()
    return ids


def split_and_import(result_json):
    data = json.load(open(result_json))
    import_dir = OUT_DIR / 'import_jsons'
    import_dir.mkdir(exist_ok=True)
    total = 0
    for model_name, model_data in data.items():
        ckpt = model_data.get('checkpoint', '')
        ts = '2026-04-21 01:09:00'
        subdir = import_dir / model_name
        subdir.mkdir(exist_ok=True)
        for task_key, task_payload in model_data.get('tasks', {}).items():
            rec = {
                'model': model_name,
                'checkpoint': ckpt,
                'timestamp': ts,
                'task_id': task_payload['task_id'],
                'setting': task_payload['setting'],
                'num_prompts': task_payload.get('num_samples', 0),
                'aggregated': task_payload.get('aggregated', {}),
                'per_sample': task_payload.get('per_sample', []),
            }
            out_path = subdir / f'{task_key}.json'
            with open(out_path, 'w') as f:
                json.dump(rec, f, indent=2)
            total += 1

    # run importer
    importer = HF / 'motion_annot_web/eval_dashboard/data_importer.py'
    for f in sorted(import_dir.rglob('*.json')):
        r = subprocess.run(
            ['python3', str(importer), 'import', str(f),
             '--notes', 'Rerun after uncond pipeline fix (2026-04-21)'],
            cwd=str(importer.parent), capture_output=True, text=True,
        )
        if r.returncode != 0 or '"status": "ok"' not in r.stdout:
            print(f'  ❌ {f.name}: {r.stderr[:200]}')
        else:
            run_id = [ln for ln in r.stdout.splitlines() if 'run_id' in ln]
            print(f'  ✅ {f.parent.name}/{f.name}: {run_id[0].strip() if run_id else ""}')
    return total


def update_model_epochs():
    """Read ckpt epoch from result.json and update model table."""
    result = find_result_json()
    data = json.load(open(result))
    conn = sqlite3.connect(DB)
    cur = conn.cursor()
    for model_name, model_data in data.items():
        ckpt = model_data.get('checkpoint', '')
        epoch = None
        for part in ckpt.split('/'):
            if 'epoch' in part.lower():
                try:
                    epoch = int(''.join(filter(str.isdigit, part)))
                except ValueError:
                    pass
        if epoch:
            cur.execute(
                'UPDATE models SET epoch=?, checkpoint_path=? WHERE name=?',
                (epoch, ckpt, model_name),
            )
            print(f'  Set {model_name}.epoch = {epoch}, ckpt = {ckpt}')
    conn.commit()
    conn.close()


if __name__ == '__main__':
    print('=== 1. Delete old uncond runs ===')
    delete_old_runs()
    print('=== 2. Find result.json ===')
    result = find_result_json()
    print(f'  Using: {result}')
    print('=== 3. Split + import ===')
    total = split_and_import(result)
    print(f'  Imported {total} (model, task_setting) pairs')
    print('=== 4. Update model epochs ===')
    update_model_epochs()
    print('Done.')
