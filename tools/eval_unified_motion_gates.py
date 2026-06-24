#!/usr/bin/env python3
"""Summarize unified-motion gate metrics.

This is a lightweight gate definition tool for early Unified Motion Foundation
experiments. It intentionally accepts generic JSON/JSONL records so it can read
small eval exports before the full dashboard integration is ready.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional


@dataclass(frozen=True)
class GateSpec:
    name: str
    task: str
    metrics: Dict[str, float]
    comparison: str = 'max'


GATES = (
    GateSpec(
        name='caption_generation',
        task='E1',
        metrics={
            'jitter_pos': 800.0,
            'foot_skating_ratio': 0.35,
        },
    ),
    GateSpec(
        name='spatial_control',
        task='E4',
        metrics={
            'ee_error_mean': 0.08,
            'jitter_pos': 1000.0,
        },
    ),
    GateSpec(
        name='caption_condition_balance',
        task='E10',
        metrics={
            'jitter_pos': 900.0,
            'foot_skating_ratio': 0.35,
        },
    ),
    GateSpec(
        name='semantic_edit',
        task='MotionFix',
        metrics={
            'target_mpjpe': 0.12,
            'preserve_mpjpe': 0.05,
            'edit_success_error': 0.15,
        },
    ),
)


def _load_records(path: Path) -> List[Dict[str, Any]]:
    text = path.read_text(encoding='utf-8')
    if path.suffix == '.jsonl':
        return [json.loads(line) for line in text.splitlines() if line.strip()]

    data = json.loads(text)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ('records', 'results', 'data', 'runs'):
            if isinstance(data.get(key), list):
                return data[key]
        return [data]
    raise TypeError(f'Unsupported JSON root in {path}: {type(data)}')


def _record_task(record: Dict[str, Any]) -> Optional[str]:
    return record.get('task') or record.get('task_id') or record.get('eval_task')


def _record_metric(record: Dict[str, Any], metric: str) -> Optional[float]:
    value = record.get(metric)
    if value is None and isinstance(record.get('metrics'), dict):
        value = record['metrics'].get(metric)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _summarize_metric(records: Iterable[Dict[str, Any]], metric: str) -> Optional[float]:
    values = [_record_metric(record, metric) for record in records]
    values = [value for value in values if value is not None]
    if not values:
        return None
    return mean(values)


def summarize_gates(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    for gate in GATES:
        gate_records = [record for record in records if _record_task(record) == gate.task]
        row: Dict[str, Any] = {
            'gate': gate.name,
            'task': gate.task,
            'num_records': len(gate_records),
            'pass': bool(gate_records),
            'metrics': {},
        }
        for metric, threshold in gate.metrics.items():
            value = _summarize_metric(gate_records, metric)
            ok = value is not None and value <= threshold
            row['metrics'][metric] = {
                'mean': value,
                'threshold': threshold,
                'pass': ok,
            }
            row['pass'] = row['pass'] and ok
        rows.append(row)
    return rows


def _format_value(value: Optional[float]) -> str:
    if value is None:
        return 'missing'
    return f'{value:.4g}'


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('inputs', nargs='+', help='JSON/JSONL metric files')
    parser.add_argument('--fail-on-gate', action='store_true',
                        help='Return non-zero if any gate fails')
    parser.add_argument('--json', action='store_true',
                        help='Print machine-readable JSON')
    args = parser.parse_args()

    records: List[Dict[str, Any]] = []
    for filename in args.inputs:
        records.extend(_load_records(Path(filename)))

    rows = summarize_gates(records)
    if args.json:
        print(json.dumps(rows, ensure_ascii=False, indent=2))
    else:
        for row in rows:
            status = 'PASS' if row['pass'] else 'FAIL'
            print(f"{status} {row['gate']} ({row['task']}), n={row['num_records']}")
            for metric, item in row['metrics'].items():
                print(
                    f"  {metric}: mean={_format_value(item['mean'])} "
                    f"threshold<={item['threshold']}"
                )

    if args.fail_on_gate and any(not row['pass'] for row in rows):
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
