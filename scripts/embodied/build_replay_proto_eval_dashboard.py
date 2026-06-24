#!/usr/bin/env python3
"""Build a static dashboard for the replay_proto tracker eval result."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


ROOT = Path("/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer")
LAFAN_SUMMARY = (
    ROOT
    / "output/lafan1_g1_proto_baseline_eval/"
    "gt_replay_genproto_formal2k_cached_eval_fix2_0618/summary.json"
)
AMASS_SUMMARY = (
    ROOT
    / "output/amass_g1_proto_baseline_eval/"
    "gt_replay_genproto_formal2k_cached_eval_fix2_0618/summary.json"
)
RUN_LOG = ROOT / "work_dirs/physflow_replay_proto_cached_eval_fix2_0618.log"
OUT_DIR = (
    ROOT
    / "output/physflow_visualizations/"
    "tracker_reward_proto_2k_fixed_noise_fourway/replay_proto_eval"
)
OUT_HTML = OUT_DIR / "index.html"

BASELINE = "protomotions_g1_bones"
CANDIDATE = "gt_replay_after"

METRICS = [
    ("eval/success_rate", "Success", "higher", "pct"),
    ("eval/relative_body_pos/failure_rate", "Rel-body fail", "lower", "pct"),
    ("eval/anchor_height_error/failure_rate", "Anchor fail", "lower", "pct"),
    ("eval/gt_error/mean_mm", "GT err", "lower", "mm"),
    ("eval/max_joint_error/mean", "Max joint err", "lower", "rad"),
    ("eval/gr_error/mean", "GR err", "lower", "rad"),
    ("eval/normalized_jerk_mean", "Jerk", "lower", "scalar"),
    ("eval/action_delta_mean_rad", "Action delta", "lower", "rad"),
]


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def clean_float(value: Any) -> float:
    value = float(value)
    if math.isnan(value) or math.isinf(value):
        return 0.0
    return value


def improvement(base: float, candidate: float, direction: str) -> float:
    if abs(base) < 1e-12:
        return 0.0
    raw = (candidate - base) / abs(base)
    return raw if direction == "higher" else -raw


def fmt_value(value: float, kind: str) -> str:
    if kind == "pct":
        return f"{value * 100:.2f}%"
    if kind == "mm":
        return f"{value:.1f} mm"
    if kind == "rad":
        return f"{value:.4f}"
    if abs(value) >= 100:
        return f"{value:.1f}"
    return f"{value:.4f}"


def fmt_delta(delta: float, kind: str) -> str:
    if kind == "pct":
        return f"{delta * 100:+.2f} pp"
    if kind == "mm":
        return f"{delta:+.1f} mm"
    if kind == "rad":
        return f"{delta:+.4f}"
    if abs(delta) >= 100:
        return f"{delta:+.1f}"
    return f"{delta:+.4f}"


def dataset_record(name: str, path: Path) -> dict[str, Any]:
    data = read_json(path)
    base = data["results"][BASELINE]
    cand = data["results"][CANDIDATE]
    rows = []
    wins = 0
    for key, label, direction, kind in METRICS:
        base_value = clean_float(base[key])
        cand_value = clean_float(cand[key])
        delta = cand_value - base_value
        imp = improvement(base_value, cand_value, direction)
        is_better = imp > 0
        is_worse = imp < 0
        wins += int(is_better)
        rows.append(
            {
                "key": key,
                "label": label,
                "direction": direction,
                "kind": kind,
                "base": base_value,
                "candidate": cand_value,
                "delta": delta,
                "improvement": imp,
                "status": "better" if is_better else "worse" if is_worse else "flat",
                "baseText": fmt_value(base_value, kind),
                "candidateText": fmt_value(cand_value, kind),
                "deltaText": fmt_delta(delta, kind),
            }
        )

    success_delta = cand["eval/success_rate"] - base["eval/success_rate"]
    error_delta = cand["eval/gt_error/mean_mm"] - base["eval/gt_error/mean_mm"]
    if success_delta > 0.01 and error_delta < 0 and wins >= 6:
        verdict = "small_improvement"
    elif success_delta < -0.001 and wins >= 5:
        verdict = "mixed"
    elif wins >= 6:
        verdict = "metric_improvement"
    else:
        verdict = "not_effective"

    return {
        "name": name,
        "path": str(path),
        "numMotions": int(cand["num_motions"]),
        "numShards": int(cand["num_shards"]),
        "wins": wins,
        "losses": len(rows) - wins,
        "verdict": verdict,
        "baseline": BASELINE,
        "candidate": CANDIDATE,
        "metrics": rows,
    }


def build_data() -> dict[str, Any]:
    datasets = [
        dataset_record("LAFAN1-G1", LAFAN_SUMMARY),
        dataset_record("AMASS-G1", AMASS_SUMMARY),
    ]
    total_motions = sum(d["numMotions"] for d in datasets)
    success = {d["name"]: next(r for r in d["metrics"] if r["key"] == "eval/success_rate") for d in datasets}
    gt_err = {d["name"]: next(r for r in d["metrics"] if r["key"] == "eval/gt_error/mean_mm") for d in datasets}
    return {
        "run": {
            "title": "replay_proto eval",
            "task": "physflow-replayprotoeval2-0618-V100-1x8-2032",
            "instance": "8b1d81419ed03968019edab823e4133b",
            "taijiStatus": "END true",
            "log": str(RUN_LOG),
            "artifactNote": "This eval ran with SAVE_PREDICTED_MOTION_LIB=0, so it has aggregate metrics only and no per-motion G1 mesh replay JSON.",
        },
        "datasets": datasets,
        "headline": {
            "verdict": "mixed_not_decisive",
            "totalMotions": total_motions,
            "lafanSuccessDelta": success["LAFAN1-G1"]["delta"],
            "amassSuccessDelta": success["AMASS-G1"]["delta"],
            "lafanGtErrDelta": gt_err["LAFAN1-G1"]["delta"],
            "amassGtErrDelta": gt_err["AMASS-G1"]["delta"],
        },
    }


def html_doc(data: dict[str, Any]) -> str:
    payload = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>replay_proto eval</title>
  <style>
    :root {{
      --paper: #f4f6f1;
      --ink: #151815;
      --muted: #657069;
      --line: #c8d1ca;
      --panel: #ffffff;
      --teal: #087c75;
      --rust: #bd4f30;
      --gold: #a97914;
      --blue: #3f6f92;
      --shadow: 0 14px 36px rgba(25, 38, 31, .13);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      min-height: 100vh;
      color: var(--ink);
      background:
        linear-gradient(90deg, rgba(20,23,19,.035) 1px, transparent 1px) 0 0 / 24px 24px,
        linear-gradient(0deg, rgba(20,23,19,.026) 1px, transparent 1px) 0 0 / 24px 24px,
        var(--paper);
      font-family: Avenir Next, Segoe UI, sans-serif;
      letter-spacing: 0;
    }}
    header {{
      display: grid;
      grid-template-columns: minmax(0, 1fr) minmax(330px, .72fr);
      gap: 24px;
      padding: 24px clamp(16px, 3vw, 42px) 18px;
      border-bottom: 1px solid var(--line);
    }}
    h1 {{
      margin: 0;
      font: 700 clamp(32px, 5vw, 62px)/.96 Georgia, Cambria, serif;
    }}
    .sub {{
      margin: 10px 0 0;
      color: var(--muted);
      line-height: 1.45;
      max-width: 900px;
      font-size: 14px;
    }}
    .nav {{
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      margin-bottom: 14px;
    }}
    .nav a, button {{
      border: 1px solid var(--line);
      background: #fbfcfa;
      color: var(--ink);
      min-height: 34px;
      padding: 8px 10px;
      font: 700 12px ui-monospace, SFMono-Regular, Menlo, monospace;
      text-decoration: none;
      cursor: pointer;
    }}
    button.active, .nav a.active {{
      background: var(--ink);
      color: white;
      border-color: var(--ink);
    }}
    .stats {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px;
      align-content: end;
    }}
    .stat, .panel, .notice {{
      background: rgba(255,255,255,.88);
      border: 1px solid var(--line);
      box-shadow: var(--shadow);
    }}
    .stat {{
      min-height: 88px;
      padding: 13px;
    }}
    .stat span, .metric-label {{
      display: block;
      color: var(--muted);
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: .08em;
    }}
    .stat strong {{
      display: block;
      margin-top: 7px;
      font: 700 24px Georgia, Cambria, serif;
    }}
    main {{
      padding: 18px clamp(16px, 3vw, 42px) 42px;
    }}
    .grid {{
      display: grid;
      grid-template-columns: minmax(360px, .58fr) minmax(0, 1fr);
      gap: 14px;
      align-items: start;
    }}
    .panel {{
      padding: 16px;
      overflow: hidden;
    }}
    .notice {{
      padding: 14px 16px;
      margin-bottom: 14px;
      border-left: 4px solid var(--gold);
      color: var(--muted);
      line-height: 1.45;
    }}
    .case-embed {{
      border: 1px solid var(--line);
      background: rgba(255,255,255,.88);
      box-shadow: var(--shadow);
      margin-bottom: 14px;
      overflow: hidden;
    }}
    .embed-head {{
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: center;
      padding: 12px 14px;
      border-bottom: 1px solid var(--line);
      color: var(--muted);
      font-size: 13px;
      line-height: 1.4;
    }}
    .embed-head strong {{
      color: var(--ink);
      display: block;
      font: 700 14px ui-monospace, SFMono-Regular, Menlo, monospace;
    }}
    .embed-head a {{
      color: var(--teal);
      font: 700 12px ui-monospace, SFMono-Regular, Menlo, monospace;
      text-decoration: none;
      white-space: nowrap;
    }}
    .case-embed iframe {{
      width: 100%;
      height: min(78vh, 820px);
      min-height: 620px;
      border: 0;
      display: block;
      background: var(--paper);
    }}
    .verdict {{
      color: var(--gold);
    }}
    .verdict.bad {{
      color: var(--rust);
    }}
    .dataset-tabs {{
      display: flex;
      gap: 8px;
      margin-bottom: 12px;
      flex-wrap: wrap;
    }}
    .summary-line {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 10px;
      margin-bottom: 14px;
    }}
    .mini {{
      border: 1px solid var(--line);
      padding: 10px;
      background: #fbfcfa;
      min-height: 72px;
    }}
    .mini strong {{
      display: block;
      margin-top: 6px;
      font: 700 18px ui-monospace, SFMono-Regular, Menlo, monospace;
      white-space: nowrap;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
    }}
    th, td {{
      border-bottom: 1px solid var(--line);
      padding: 9px 8px;
      text-align: right;
      vertical-align: middle;
    }}
    th:first-child, td:first-child {{
      text-align: left;
    }}
    th {{
      color: var(--muted);
      font: 700 11px ui-monospace, SFMono-Regular, Menlo, monospace;
      text-transform: uppercase;
      letter-spacing: .06em;
    }}
    .better {{ color: var(--teal); }}
    .worse {{ color: var(--rust); }}
    .flat {{ color: var(--muted); }}
    .bars {{
      display: grid;
      gap: 12px;
    }}
    .bar-row {{
      display: grid;
      grid-template-columns: 128px minmax(0, 1fr) 74px;
      gap: 10px;
      align-items: center;
    }}
    .track {{
      height: 22px;
      border: 1px solid var(--line);
      background: #eef1eb;
      position: relative;
      overflow: hidden;
    }}
    .fill {{
      height: 100%;
      min-width: 2px;
      background: var(--blue);
    }}
    .fill.candidate {{
      background: var(--teal);
    }}
    .legend {{
      display: flex;
      gap: 16px;
      margin: 0 0 12px;
      color: var(--muted);
      font-size: 12px;
    }}
    .dot {{
      display: inline-block;
      width: 10px;
      height: 10px;
      margin-right: 6px;
      background: var(--blue);
      vertical-align: -1px;
    }}
    .dot.candidate {{ background: var(--teal); }}
    code {{
      font: 12px ui-monospace, SFMono-Regular, Menlo, monospace;
      color: var(--ink);
      word-break: break-all;
    }}
    .paths {{
      margin-top: 14px;
      display: grid;
      gap: 8px;
      color: var(--muted);
      font-size: 12px;
      line-height: 1.4;
    }}
    @media (max-width: 960px) {{
      header, .grid {{
        grid-template-columns: 1fr;
      }}
      .stats, .summary-line {{
        grid-template-columns: 1fr 1fr;
      }}
    }}
    @media (max-width: 640px) {{
      .stats, .summary-line {{
        grid-template-columns: 1fr;
      }}
      .bar-row {{
        grid-template-columns: 1fr;
      }}
      th, td {{
        padding: 8px 5px;
        font-size: 12px;
      }}
    }}
  </style>
</head>
<body>
  <header>
    <div>
      <nav class="nav">
        <a href="../">Four-way G1 mesh</a>
        <a class="active" href="./">Replay eval</a>
        <a href="../replay_proto_cases/">Replay cases</a>
      </nav>
      <h1>replay_proto eval</h1>
      <p class="sub">Aggregate tracker evaluation for <code>{data["run"]["task"]}</code>. Candidate is <code>{CANDIDATE}</code>; baseline is <code>{BASELINE}</code>.</p>
    </div>
    <section class="stats" aria-label="headline stats">
      <div class="stat"><span>Taiji</span><strong>{data["run"]["taijiStatus"]}</strong></div>
      <div class="stat"><span>Verdict</span><strong class="verdict">Mixed</strong></div>
      <div class="stat"><span>Total motions</span><strong>{data["headline"]["totalMotions"]:,}</strong></div>
      <div class="stat"><span>Exact eval mesh replay</span><strong class="verdict bad">No</strong></div>
    </section>
  </header>
  <main>
    <div class="notice">
      {data["run"]["artifactNote"]} The aggregate metrics below are from that exact eval. The embedded G1 mesh comparison is the paired case-visualization run, used for concrete before/after inspection.
    </div>
    <section class="case-embed">
      <div class="embed-head">
        <div><strong>Concrete G1 case comparison</strong> Reference vs original ProtoMotions tracker vs replay-optimized tracker.</div>
        <a href="../replay_proto_cases/">Open full page</a>
      </div>
      <iframe src="../replay_proto_cases/" loading="lazy" title="Replay proto case-level G1 comparison"></iframe>
    </section>
    <div class="grid">
      <section class="panel">
        <div class="dataset-tabs" id="tabs"></div>
        <div id="summary"></div>
        <table>
          <thead>
            <tr>
              <th>Metric</th>
              <th>Base</th>
              <th>Replay</th>
              <th>Delta</th>
            </tr>
          </thead>
          <tbody id="metricRows"></tbody>
        </table>
      </section>
      <section class="panel">
        <p class="legend"><span><span class="dot"></span>Base</span><span><span class="dot candidate"></span>Replay</span></p>
        <div class="bars" id="bars"></div>
        <div class="paths" id="paths"></div>
      </section>
    </div>
  </main>
  <script>
    const DATA = {payload};
    let active = 0;

    function cls(row) {{
      return row.status === 'better' ? 'better' : row.status === 'worse' ? 'worse' : 'flat';
    }}

    function signedPct(value) {{
      return `${{(value * 100).toFixed(2)}} pp`;
    }}

    function renderTabs() {{
      const tabs = document.getElementById('tabs');
      tabs.innerHTML = '';
      DATA.datasets.forEach((dataset, index) => {{
        const button = document.createElement('button');
        button.textContent = dataset.name;
        button.className = index === active ? 'active' : '';
        button.onclick = () => {{
          active = index;
          render();
        }};
        tabs.appendChild(button);
      }});
    }}

    function renderSummary(dataset) {{
      const success = dataset.metrics.find(row => row.key === 'eval/success_rate');
      const gtErr = dataset.metrics.find(row => row.key === 'eval/gt_error/mean_mm');
      const jerk = dataset.metrics.find(row => row.key === 'eval/normalized_jerk_mean');
      document.getElementById('summary').innerHTML = `
        <div class="summary-line">
          <div class="mini"><span class="metric-label">Motions</span><strong>${{dataset.numMotions.toLocaleString()}}</strong></div>
          <div class="mini"><span class="metric-label">Success delta</span><strong class="${{cls(success)}}">${{signedPct(success.delta)}}</strong></div>
          <div class="mini"><span class="metric-label">GT err delta</span><strong class="${{cls(gtErr)}}">${{gtErr.deltaText}}</strong></div>
          <div class="mini"><span class="metric-label">Metric wins</span><strong>${{dataset.wins}} / ${{dataset.metrics.length}}</strong></div>
          <div class="mini"><span class="metric-label">Jerk delta</span><strong class="${{cls(jerk)}}">${{jerk.deltaText}}</strong></div>
          <div class="mini"><span class="metric-label">Dataset verdict</span><strong class="verdict">${{dataset.verdict.replaceAll('_', ' ')}}</strong></div>
        </div>`;
    }}

    function renderRows(dataset) {{
      const rows = document.getElementById('metricRows');
      rows.innerHTML = dataset.metrics.map(row => `
        <tr>
          <td>${{row.label}}</td>
          <td>${{row.baseText}}</td>
          <td>${{row.candidateText}}</td>
          <td class="${{cls(row)}}">${{row.deltaText}}</td>
        </tr>`).join('');
    }}

    function renderBars(dataset) {{
      const bars = document.getElementById('bars');
      bars.innerHTML = '';
      dataset.metrics.forEach(row => {{
        const max = Math.max(Math.abs(row.base), Math.abs(row.candidate), 1e-9);
        const baseWidth = Math.max(2, Math.min(100, Math.abs(row.base) / max * 100));
        const candWidth = Math.max(2, Math.min(100, Math.abs(row.candidate) / max * 100));
        const wrap = document.createElement('div');
        wrap.className = 'bar-row';
        wrap.innerHTML = `
          <div><span class="metric-label">${{row.label}}</span></div>
          <div>
            <div class="track"><div class="fill" style="width:${{baseWidth}}%"></div></div>
            <div class="track" style="margin-top:4px"><div class="fill candidate" style="width:${{candWidth}}%"></div></div>
          </div>
          <div class="${{cls(row)}}">${{row.deltaText}}</div>
        `;
        bars.appendChild(wrap);
      }});
    }}

    function renderPaths(dataset) {{
      document.getElementById('paths').innerHTML = `
        <div><strong>Summary:</strong> <code>${{dataset.path}}</code></div>
        <div><strong>Run log:</strong> <code>${{DATA.run.log}}</code></div>
        <div><strong>Instance:</strong> <code>${{DATA.run.instance}}</code></div>
      `;
    }}

    function render() {{
      const dataset = DATA.datasets[active];
      renderTabs();
      renderSummary(dataset);
      renderRows(dataset);
      renderBars(dataset);
      renderPaths(dataset);
    }}

    render();
  </script>
</body>
</html>
"""


def main() -> None:
    data = build_data()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_HTML.write_text(html_doc(data), encoding="utf-8")
    print(OUT_HTML)


if __name__ == "__main__":
    main()
