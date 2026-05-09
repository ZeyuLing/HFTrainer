"""Parse partial results from run.log and emit a result.json compatible with the
eval_dashboard data_importer, so we can import results WHILE the eval run
is still executing.

Reads tools/eval_m2m_v2_all_tasks.py's `Results (n=50):` blocks from the log,
producing one JSON that mirrors what the driver would eventually dump, but only
for already-completed (model, setting) pairs.
"""
import argparse
import json
import re
import sys
from pathlib import Path


BLOCK_RE = re.compile(
    r"Model:\s+(\S+)\s+—.*?(?=Model:|\Z)",
    re.DOTALL,
)
TASK_RE = re.compile(
    r"Task:\s+(\S+_\S+)\s+—\s+(.*?)\s+\((.*?)\)\s*\n"
    r"\s+Loaded\s+(\d+)\s+samples.*?"
    r"Results\s+\(n=(\d+)\):(.*?)(?=\n\s{2}Task:|\n=+|\Z)",
    re.DOTALL,
)
METRIC_RE = re.compile(
    r"^\s+(\w+):\s+([-\d.]+)\s*±\s*([-\d.]+)\s*\(med=([-\d.]+)\)\s*$",
    re.MULTILINE,
)
CKPT_RE = re.compile(r"Loading checkpoint:\s+(\S+)")


def parse_log(log_text: str):
    out = {}
    blocks = list(BLOCK_RE.finditer(log_text))
    for mb in blocks:
        model = mb.group(1)
        block = mb.group(0)
        ckpt_match = CKPT_RE.search(block)
        checkpoint = ckpt_match.group(1) if ckpt_match else None
        tasks = {}
        for tm in TASK_RE.finditer(block):
            task_key = tm.group(1)  # e.g. E3_A
            task_name = tm.group(2).strip()
            setting_desc = tm.group(3).strip()
            n_samples = int(tm.group(5))
            body = tm.group(6)
            aggregated = {}
            for mm in METRIC_RE.finditer(body):
                name, mean, std, median = mm.groups()
                aggregated[name] = {
                    "mean": float(mean),
                    "std": float(std),
                    "median": float(median),
                    "count": n_samples,
                }
            if aggregated:
                task_id, setting = task_key.split('_', 1)
                tasks[task_key] = {
                    "task_id": task_id,
                    "setting": setting,
                    "num_samples": n_samples,
                    "aggregated": aggregated,
                    "per_sample": [],  # per-sample not needed for agg view
                }
        if tasks:
            out[model] = {
                "checkpoint": checkpoint,
                "model": model,
                "tasks": tasks,
            }
    return out


def dump_per_model_result_jsons(parsed: dict, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for model, data in parsed.items():
        # Wrap one model per file, structure mirrors driver's all_results[model]
        subdir = out_dir / model
        subdir.mkdir(parents=True, exist_ok=True)
        # Each task_key becomes its own json for importer, which reads
        # top-level fields: model, checkpoint, task_id, setting, aggregated, per_sample
        for task_key, task_payload in data["tasks"].items():
            rec = {
                "model": model,
                "checkpoint": data.get("checkpoint", ""),
                "task_id": task_payload["task_id"],
                "setting": task_payload["setting"],
                "num_prompts": task_payload["num_samples"],
                "aggregated": task_payload["aggregated"],
                "per_sample": task_payload["per_sample"],
            }
            path = subdir / f"{task_key}_partial.json"
            with open(path, "w") as f:
                json.dump(rec, f, indent=2)
            written.append(str(path))
    return written


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log_path")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    text = Path(args.log_path).read_text()
    parsed = parse_log(text)
    print("Parsed models:", list(parsed.keys()))
    for m, d in parsed.items():
        print(f"  {m}: ckpt={d.get('checkpoint')}, tasks={list(d['tasks'].keys())}")
    written = dump_per_model_result_jsons(parsed, Path(args.out_dir))
    print(f"Wrote {len(written)} partial JSON files:")
    for p in written:
        print("  ", p)


if __name__ == "__main__":
    main()
