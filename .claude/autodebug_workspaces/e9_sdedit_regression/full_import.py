"""Import all full-rerun results into dashboard.

For each (model, task, setting) triple:
1. Delete any existing row in eval_runs (for uncond_* and caption_* models)
2. Import new result from the newest relevant eval_v2_*.json

Strategy:
- Prefer *_1_v2 over *_1 (the v2 runs used the fixed pipeline)
- Merge across _0 and _1_v2 partitions for each model
"""
import json
import sqlite3
import subprocess
from pathlib import Path

HF = Path('/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer')
DB = HF / 'motion_annot_web/eval_dashboard/eval_dashboard.db'
OUT_ROOT = HF / 'work_dirs/full_rerun_20260421'
IMPORTER = HF / 'motion_annot_web/eval_dashboard/data_importer.py'

# Model -> list of (result_json_path, partition_label)
# Order matters: later entries in the list overwrite earlier ones (via delete-then-import).
MODEL_SOURCES = {
    'uncond_local': [
        sorted((OUT_ROOT / 'uncond_local_0').glob('eval_v2_*.json'))[-1],
        sorted((OUT_ROOT / 'uncond_local_1').glob('eval_v2_*.json'))[-1],
    ],
    'uncond_global': [
        sorted((OUT_ROOT / 'uncond_global_0').glob('eval_v2_*.json'))[-1],
        sorted((OUT_ROOT / 'uncond_global_1').glob('eval_v2_*.json'))[-1],
    ],
    'caption_local': [
        sorted((OUT_ROOT / 'caption_local_0').glob('eval_v2_*.json'))[-1],
        sorted((OUT_ROOT / 'caption_local_1_v2').glob('eval_v2_*.json'))[-1],
    ],
    'caption_global': [
        sorted((OUT_ROOT / 'caption_global_0').glob('eval_v2_*.json'))[-1],
        sorted((OUT_ROOT / 'caption_global_1_v2').glob('eval_v2_*.json'))[-1],
    ],
}

# Backup DB
import shutil
backup = DB.with_suffix(f'.bak_before_full_import_{__import__("datetime").datetime.now():%Y%m%d_%H%M%S}.db')
shutil.copy(DB, backup)
print(f'DB backup: {backup}')

# Collect the model->ckpt mapping for updating model epochs
model_ckpt = {}
for model, json_files in MODEL_SOURCES.items():
    for fp in json_files:
        d = json.load(open(fp))
        if model in d and d[model].get('checkpoint'):
            model_ckpt[model] = d[model]['checkpoint']
            break

# Delete all old runs for these 4 models
conn = sqlite3.connect(DB)
cur = conn.cursor()
placeholders = ','.join('?' * len(MODEL_SOURCES))
cur.execute(
    f'SELECT id FROM models WHERE name IN ({placeholders})',
    list(MODEL_SOURCES.keys()),
)
mids = [r[0] for r in cur.fetchall()]
print(f'Model IDs to clear: {mids}')
if mids:
    mid_ph = ','.join('?' * len(mids))
    cur.execute(
        f'SELECT id FROM eval_runs WHERE model_id IN ({mid_ph})', mids
    )
    rids = [r[0] for r in cur.fetchall()]
    print(f'Deleting {len(rids)} existing eval_runs')
    if rids:
        rid_ph = ','.join('?' * len(rids))
        cur.execute(f'DELETE FROM agg_metrics WHERE eval_run_id IN ({rid_ph})', rids)
        cur.execute(f'DELETE FROM sample_results WHERE eval_run_id IN ({rid_ph})', rids)
        cur.execute(f'DELETE FROM eval_runs WHERE id IN ({rid_ph})', rids)
conn.commit()

# Import each result.json per-setting
imported = 0
import_tmp = OUT_ROOT / 'import_jsons'
import_tmp.mkdir(exist_ok=True)
for model, json_files in MODEL_SOURCES.items():
    for fp in json_files:
        data = json.load(open(fp))
        # data[model] = {checkpoint, tasks: {task_key: {task_id, setting, aggregated, per_sample}}}
        for mname, mdata in data.items():
            ckpt = mdata.get('checkpoint', '')
            for task_key, task_payload in mdata.get('tasks', {}).items():
                agg = task_payload.get('aggregated', {})
                samples = task_payload.get('per_sample', [])
                if not agg and not samples:
                    continue  # skipped task
                rec = {
                    'model': mname,
                    'checkpoint': ckpt,
                    'timestamp': '2026-04-21 03:00:00',
                    'task_id': task_payload['task_id'],
                    'setting': task_payload['setting'],
                    'num_prompts': task_payload.get('num_samples', 0),
                    'aggregated': agg,
                    'per_sample': samples,
                }
                tmpf = import_tmp / f'{mname}__{task_key}.json'
                with open(tmpf, 'w') as f:
                    json.dump(rec, f, indent=2)
                r = subprocess.run(
                    ['python3', str(IMPORTER), 'import', str(tmpf),
                     '--notes', 'Full rerun with fixed pipeline (uncond ctxt + CFG null_ctxt dyn)'],
                    cwd=str(IMPORTER.parent),
                    capture_output=True, text=True,
                )
                if '"status": "ok"' in r.stdout:
                    imported += 1
                    print(f'  ✅ {mname} {task_key}')
                else:
                    print(f'  ❌ {mname} {task_key}: {r.stderr[:200]}')

print(f'\nImported {imported} runs')

# Update model epochs
for model, ckpt in model_ckpt.items():
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
            (epoch, ckpt, model),
        )
        print(f'  Set {model}.epoch={epoch}')
conn.commit()
conn.close()
print('DB updated.')
