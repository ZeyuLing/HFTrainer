#!/usr/bin/env python3
"""Create a unified static index for ProtoMotions tracker dashboards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import sys

HERE = Path(__file__).resolve()
ROOT = HERE.parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.embodied.build_protomotions_ref_track_dashboard import _html_doc  # noqa: E402


DATASETS = [
    ("lafan1", "LAFAN1"),
    ("amass", "AMASS"),
    ("wild", "Wild"),
]


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def _write_dataset_page(root_dir: Path, key: str, label: str) -> dict[str, Any]:
    manifest_path = root_dir / key / "manifest.json"
    data = json.loads(manifest_path.read_text())
    data["sibling_links"] = [
        {"label": "Overview", "href": "../", "active": False},
        *[
            {"label": other_label, "href": f"../{other_key}/", "active": other_key == key}
            for other_key, other_label in DATASETS
            if (root_dir / other_key / "manifest.json").is_file()
        ],
    ]
    manifest_path.write_text(json.dumps(data, indent=2) + "\n")
    (root_dir / key / "index.html").write_text(_html_doc(data), encoding="utf-8")
    return {"key": key, "label": label, "data": data}


def _overview_html(items: list[dict[str, Any]]) -> str:
    cards = []
    for item in items:
        summary = item["data"].get("summary", {})
        cards.append(
            f"""
            <a class="card" href="./{item['key']}/">
              <span>{item['label']}</span>
              <strong>{item['data'].get('dataset', item['label'])}</strong>
              <dl>
                <div><dt>Cases</dt><dd>{summary.get('cases', 0)}</dd></div>
                <div><dt>Success</dt><dd>{_fmt(summary.get('success_rate'), 3)}</dd></div>
                <div><dt>Local MPJPE</dt><dd>{_fmt(summary.get('local_mpjpe_mm_mean'), 1)} mm</dd></div>
                <div><dt>Aligned Global</dt><dd>{_fmt(summary.get('aligned_global_mpjpe_mm_mean'), 1)} mm</dd></div>
              </dl>
            </a>
            """
        )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>ProtoMotions G1 Tracker Dashboard</title>
  <style>
    :root {{
      --paper: #f4f1ea;
      --ink: #191a17;
      --muted: #6c7169;
      --line: #c6c1b7;
      --panel: #fffefa;
      --track: #0a7c72;
      --shadow: 0 16px 38px rgba(36, 33, 25, .13);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      min-height: 100vh;
      color: var(--ink);
      background:
        linear-gradient(90deg, rgba(25,26,23,.035) 1px, transparent 1px) 0 0 / 28px 28px,
        linear-gradient(0deg, rgba(25,26,23,.028) 1px, transparent 1px) 0 0 / 28px 28px,
        var(--paper);
      font-family: Optima, Avenir Next, Segoe UI, sans-serif;
      letter-spacing: 0;
    }}
    header {{
      padding: 34px clamp(18px, 4vw, 58px) 22px;
      border-bottom: 1px solid var(--line);
    }}
    h1 {{
      margin: 0;
      font: 700 clamp(32px, 5vw, 62px)/.98 Georgia, Cambria, serif;
      max-width: 980px;
    }}
    p {{
      margin: 12px 0 0;
      color: var(--muted);
      max-width: 900px;
      line-height: 1.45;
      font-size: 15px;
    }}
    main {{
      padding: 24px clamp(18px, 4vw, 58px) 54px;
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 14px;
    }}
    .card {{
      min-height: 270px;
      display: flex;
      flex-direction: column;
      justify-content: space-between;
      padding: 18px;
      color: inherit;
      text-decoration: none;
      background: rgba(255,254,250,.92);
      border: 1px solid var(--line);
      box-shadow: var(--shadow);
    }}
    .card:hover {{ border-color: var(--track); }}
    .card span {{
      color: var(--track);
      font: 700 12px ui-monospace, SFMono-Regular, Menlo, monospace;
      text-transform: uppercase;
    }}
    .card strong {{
      display: block;
      margin-top: 10px;
      font: 700 28px/1.05 Georgia, Cambria, serif;
    }}
    dl {{
      margin: 24px 0 0;
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
    }}
    dt {{
      color: var(--muted);
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: .08em;
    }}
    dd {{
      margin: 5px 0 0;
      font: 700 16px ui-monospace, SFMono-Regular, Menlo, monospace;
    }}
    @media (max-width: 940px) {{
      main {{ grid-template-columns: 1fr; }}
      .card {{ min-height: 220px; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>ProtoMotions G1 tracker visual audit</h1>
    <p>Reference motion and actual tracker rollout are shown side by side with the same G1 mesh and fixed 30fps playback. Each dataset page contains case-level metrics and root trajectory plots.</p>
  </header>
  <main>
    {''.join(cards)}
  </main>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root-dir", type=Path, required=True)
    args = parser.parse_args()
    root_dir = args.root_dir.resolve()
    items = []
    for key, label in DATASETS:
        if (root_dir / key / "manifest.json").is_file():
            items.append(_write_dataset_page(root_dir, key, label))
    if not items:
        raise RuntimeError(f"No dataset manifests found under {root_dir}")
    (root_dir / "index.html").write_text(_overview_html(items), encoding="utf-8")
    print(root_dir / "index.html")
    for item in items:
        print(f"{item['key']}: {json.dumps(item['data'].get('summary', {}), sort_keys=True)}")


if __name__ == "__main__":
    main()
