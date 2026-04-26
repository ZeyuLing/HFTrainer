#!/usr/bin/env python3
"""Serve keyframe pose guidance evaluation results as a web page.

Reads eval_summary.json and per-variant NPZ files, provides:
  - Comparison table with sorting
  - Per-case 3D motion visualization (via shared SMPL viewer)
  - Side-by-side GT vs Output comparison

Usage:
    python3 scripts/serve_kf_eval_results.py \
        --result-dir output/eval_keyframe_pose \
        --port 8095
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, request, send_from_directory, render_template_string

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

STATIC_DIR = os.path.join(str(PROJECT_ROOT), 'scripts', 'vis_npz_database', 'static')

app = Flask(
    __name__,
    static_folder=STATIC_DIR,
    static_url_path='/static',
)

RESULT_DIR = ''

INDEX_HTML = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Keyframe Pose Guidance — Evaluation Results</title>
<style>
  body { font-family: 'Segoe UI', sans-serif; margin: 20px; background: #f5f5f5; }
  h1 { color: #333; }
  .meta { color: #666; margin-bottom: 20px; }
  table { border-collapse: collapse; width: 100%; margin: 20px 0; background: white; box-shadow: 0 1px 4px rgba(0,0,0,0.1); }
  th, td { border: 1px solid #ddd; padding: 8px 12px; text-align: center; font-size: 13px; }
  th { background: #4a90d9; color: white; cursor: pointer; user-select: none; }
  th:hover { background: #357abd; }
  tr:nth-child(even) { background: #f9f9f9; }
  tr:hover { background: #e8f0fe; }
  .best { font-weight: bold; color: #2e7d32; }
  .worst { color: #c62828; }
  .tabs { display: flex; gap: 8px; margin: 15px 0; }
  .tab { padding: 8px 16px; border: 1px solid #ccc; border-radius: 4px; cursor: pointer; background: white; }
  .tab.active { background: #4a90d9; color: white; border-color: #4a90d9; }
  .case-list { max-height: 400px; overflow-y: auto; }
  .case-item { padding: 6px 12px; cursor: pointer; border-bottom: 1px solid #eee; }
  .case-item:hover { background: #e8f0fe; }
  #viewer-panel { display: flex; gap: 20px; margin-top: 20px; }
  .viewer-box { flex: 1; border: 1px solid #ddd; border-radius: 8px; padding: 10px; background: white; }
  .filter-bar { margin: 10px 0; display: flex; gap: 10px; align-items: center; }
  select, input { padding: 6px 10px; border: 1px solid #ccc; border-radius: 4px; }
</style>
</head>
<body>
<h1>🎯 Keyframe Pose Guidance — Evaluation Results</h1>
<div class="meta" id="meta-info">Loading...</div>

<div class="filter-bar">
  <label>Group by:</label>
  <select id="group-by" onchange="renderTable()">
    <option value="none">No grouping</option>
    <option value="model">Model</option>
    <option value="imp_mode">Imputation Mode</option>
    <option value="rep_mode">Replacement Guidance</option>
    <option value="rotation_space">Rotation Space</option>
  </select>
  <label>Sort by:</label>
  <select id="sort-by" onchange="renderTable()">
    <option value="kf_l2">KF L2 ↓</option>
    <option value="mpjpe">MPJPE ↓</option>
    <option value="bnd_smooth">Boundary Smoothness ↓</option>
    <option value="foot_skate">Foot Skating ↓</option>
    <option value="overall_smooth">Overall Smoothness ↓</option>
  </select>
</div>

<table id="results-table">
  <thead><tr id="table-header"></tr></thead>
  <tbody id="table-body"></tbody>
</table>

<h2>Per-Case Details</h2>
<div class="filter-bar">
  <label>Variant:</label>
  <select id="variant-select" onchange="loadVariantCases()"></select>
</div>
<div class="case-list" id="case-list"></div>

<div id="viewer-panel">
  <div class="viewer-box">
    <h3>Ground Truth</h3>
    <div id="gt-viewer" style="width:100%;height:400px;background:#1a1a2e;"></div>
  </div>
  <div class="viewer-box">
    <h3>Model Output</h3>
    <div id="output-viewer" style="width:100%;height:400px;background:#1a1a2e;"></div>
  </div>
</div>

<script>
let summaryData = null;

async function init() {
  const resp = await fetch('/api/summary');
  summaryData = await resp.json();

  document.getElementById('meta-info').innerHTML =
    `Date: ${summaryData.timestamp} | Samples: ${summaryData.num_test_samples} | Variants: ${summaryData.num_variants}`;

  // Populate variant select
  const sel = document.getElementById('variant-select');
  (summaryData.comparison || []).forEach(r => {
    const opt = document.createElement('option');
    opt.value = r.variant;
    opt.textContent = `${r.model} | ${r.imp_mode} | ${r.rep_mode}`;
    sel.appendChild(opt);
  });

  renderTable();
}

function renderTable() {
  const sortKey = document.getElementById('sort-by').value;
  const rows = [...(summaryData.comparison || [])];
  rows.sort((a, b) => (a[sortKey] || 999) - (b[sortKey] || 999));

  // Find best values
  const cols = ['kf_l2', 'kf_trans', 'mpjpe', 'bnd_smooth', 'foot_skate'];
  const best = {};
  cols.forEach(c => { best[c] = Math.min(...rows.map(r => r[c] || 999)); });

  const header = document.getElementById('table-header');
  header.innerHTML = `
    <th>Model</th><th>Imp Mode</th><th>Rep Mode</th><th>Rot</th>
    <th>KF L2↓</th><th>KF Trans↓</th><th>MPJPE↓</th>
    <th>Bnd Smooth↓</th><th>Foot Skate↓</th><th>Time(s)</th>
  `;

  const tbody = document.getElementById('table-body');
  tbody.innerHTML = '';
  rows.forEach(r => {
    const tr = document.createElement('tr');
    const cls = (key) => (r[key] || 999) <= best[key] * 1.05 ? 'best' : '';
    tr.innerHTML = `
      <td>${r.model}</td><td>${r.imp_mode}</td><td>${r.rep_mode}</td>
      <td>${r.rotation_space}</td>
      <td class="${cls('kf_l2')}">${(r.kf_l2||0).toFixed(4)}</td>
      <td class="${cls('kf_trans')}">${(r.kf_trans||0).toFixed(4)}</td>
      <td class="${cls('mpjpe')}">${(r.mpjpe||0).toFixed(4)}</td>
      <td class="${cls('bnd_smooth')}">${(r.bnd_smooth||0).toFixed(4)}</td>
      <td class="${cls('foot_skate')}">${(r.foot_skate||0).toFixed(4)}</td>
      <td>${(r.time_sec||0).toFixed(1)}</td>
    `;
    tbody.appendChild(tr);
  });
}

async function loadVariantCases() {
  const variant = document.getElementById('variant-select').value;
  if (!variant) return;
  const resp = await fetch(`/api/variant_cases?variant=${encodeURIComponent(variant)}`);
  const data = await resp.json();
  const list = document.getElementById('case-list');
  list.innerHTML = '';
  (data.cases || []).forEach(c => {
    const div = document.createElement('div');
    div.className = 'case-item';
    div.textContent = `${c.case_key} | kf_l2=${(c.kf_l2||0).toFixed(4)} | mpjpe=${(c.global_mpjpe||0).toFixed(4)}`;
    div.onclick = () => loadCase(variant, c.case_key);
    list.appendChild(div);
  });
}

async function loadCase(variant, caseKey) {
  // Load NPZ frames for 3D viewer
  const url = `/api/case_frames?variant=${encodeURIComponent(variant)}&case_key=${encodeURIComponent(caseKey)}`;
  const resp = await fetch(url);
  const data = await resp.json();
  // TODO: integrate with Three.js SMPL viewer
  console.log('Case data loaded:', data);
}

init();
</script>
</body>
</html>
"""


@app.route('/')
def index():
    return render_template_string(INDEX_HTML)


@app.route('/api/summary')
def api_summary():
    summary_path = os.path.join(RESULT_DIR, 'eval_summary.json')
    if not os.path.exists(summary_path):
        return jsonify({'error': 'No eval_summary.json found', 'comparison': []})
    with open(summary_path) as f:
        data = json.load(f)
    # Strip full_results to reduce payload
    data.pop('full_results', None)
    return jsonify(data)


@app.route('/api/variant_cases')
def api_variant_cases():
    variant = request.args.get('variant', '')
    results_path = os.path.join(RESULT_DIR, variant, 'results.json')
    if not os.path.exists(results_path):
        return jsonify({'error': f'No results for variant {variant}', 'cases': []})
    with open(results_path) as f:
        data = json.load(f)
    return jsonify({'cases': data.get('cases', [])})


@app.route('/api/case_frames')
def api_case_frames():
    variant = request.args.get('variant', '')
    case_key = request.args.get('case_key', '')
    npz_path = os.path.join(RESULT_DIR, variant, f'{case_key}.npz')
    if not os.path.exists(npz_path):
        return jsonify({'error': 'NPZ not found'})

    data = np.load(npz_path, allow_pickle=True)
    result = {
        'output_motion': data['output_motion'].tolist(),
        'gt_motion': data['gt_motion'].tolist(),
        'keyframe_idx': int(data['keyframe_idx']),
    }
    if 'src_mask' in data:
        # Just send per-frame mask summary (not full T*D)
        mask = data['src_mask']
        result['frame_mask'] = (mask.max(axis=-1) > 0.5).tolist()
    return jsonify(result)


@app.route('/api/report')
def api_report():
    report_path = os.path.join(RESULT_DIR, 'REPORT.md')
    if not os.path.exists(report_path):
        return jsonify({'error': 'No report found'})
    with open(report_path) as f:
        content = f.read()
    return jsonify({'content': content})


def main():
    global RESULT_DIR
    parser = argparse.ArgumentParser()
    parser.add_argument('--result-dir', default='output/eval_keyframe_pose')
    parser.add_argument('--port', type=int, default=8095)
    parser.add_argument('--host', default='0.0.0.0')
    args = parser.parse_args()

    RESULT_DIR = str(Path(args.result_dir).resolve())
    logger.info(f'Serving results from: {RESULT_DIR}')
    logger.info(f'URL: http://0.0.0.0:{args.port}/')
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == '__main__':
    main()
