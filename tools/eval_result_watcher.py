#!/usr/bin/env python3
"""Watch eval output directories for new result.json files and auto-import to Dashboard DB.

Usage:
    # Watch multiple directories (default: all common eval output dirs)
    python tools/eval_result_watcher.py

    # Watch specific directories
    python tools/eval_result_watcher.py --watch-dirs work_dirs/m2m_v2_t2m_eval_compare work_dirs/m2m_v2_eval_kimodo

    # Custom poll interval
    python tools/eval_result_watcher.py --poll-interval 10
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DEFAULT_WATCH_DIRS = [
    'work_dirs/m2m_v2_t2m_eval',
    'work_dirs/m2m_v2_t2m_eval_compare',
    'work_dirs/m2m_v2_eval_kimodo',
    'work_dirs/m2m_v2_eval_report',
]

DEFAULT_DB_PATH = 'motion_annot_web/eval_dashboard/eval_dashboard.db'


def find_result_jsons(watch_dirs: list[str]) -> dict[str, float]:
    """Recursively find all result.json files and their mtime."""
    found = {}
    for d in watch_dirs:
        if not os.path.isdir(d):
            continue
        for root, dirs, files in os.walk(d):
            for f in files:
                if f == 'result.json':
                    path = os.path.join(root, f)
                    found[path] = os.path.getmtime(path)
    return found


def main():
    parser = argparse.ArgumentParser(description='Watch for new eval results and auto-import')
    parser.add_argument('--watch-dirs', nargs='+', default=DEFAULT_WATCH_DIRS,
                        help='Directories to watch for result.json files')
    parser.add_argument('--db', type=str, default=DEFAULT_DB_PATH,
                        help='Path to eval_dashboard.db')
    parser.add_argument('--poll-interval', type=int, default=15,
                        help='Seconds between checks')
    parser.add_argument('--import-existing', action='store_true',
                        help='Import all existing result.json on startup')
    args = parser.parse_args()

    from motion_annot_web.eval_dashboard.db_manager import EvalDashboardDB
    from motion_annot_web.eval_dashboard.data_importer import import_result_json

    db = EvalDashboardDB(args.db)

    # Track already-imported files: {path: mtime}
    imported = {}

    if not args.import_existing:
        # Mark existing files as already imported (don't re-import on startup)
        existing = find_result_jsons(args.watch_dirs)
        imported.update(existing)
        print(f'[watcher] Found {len(existing)} existing result.json files (skipping)')
        for p in sorted(existing.keys()):
            print(f'  [skip] {p}')
    else:
        print(f'[watcher] --import-existing: will import all found files')

    print(f'[watcher] Watching {len(args.watch_dirs)} directories, poll every {args.poll_interval}s')
    print(f'[watcher] DB: {args.db}')
    for d in args.watch_dirs:
        print(f'  📁 {d}' + (' (exists)' if os.path.isdir(d) else ' (not yet)'))
    print(f'[watcher] Press Ctrl+C to stop\n')

    try:
        while True:
            current = find_result_jsons(args.watch_dirs)

            for path, mtime in current.items():
                if path not in imported or imported[path] < mtime:
                    # New or updated result.json
                    print(f'[watcher] 🆕 Detected: {path}')
                    try:
                        result = import_result_json(db, path)
                        status = result.get('status', 'unknown')
                        if status == 'ok':
                            model = result.get('model_name', '?')
                            task = result.get('task_id', '?')
                            run_id = result.get('run_id', '?')
                            n_samples = result.get('num_samples', '?')
                            print(f'  ✅ Imported: model={model}, task={task}, '
                                  f'run_id={run_id}, samples={n_samples}')
                        elif status == 'skipped':
                            print(f'  ⏭️  Skipped (already exists): {result.get("reason", "")}')
                        else:
                            print(f'  ⚠️  Status: {status} — {result.get("reason", "")}')
                        imported[path] = mtime
                    except Exception as e:
                        print(f'  ❌ Import failed: {e}')
                        # Don't mark as imported so it retries next cycle
                        import traceback
                        traceback.print_exc()

            time.sleep(args.poll_interval)

    except KeyboardInterrupt:
        print(f'\n[watcher] Stopped. Imported {len(imported)} files total.')


if __name__ == '__main__':
    main()
