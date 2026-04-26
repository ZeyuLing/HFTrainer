"""Generate a unified master list of all eval motions across E1-E16 datalists.

Output:
  data/eval/m2m_v2/motion_master_list.json : list of entries
    {
        "motion_path": "...",
        "caption": "...",           # best caption across datalists
        "tasks": ["eval_e2_inbetween", "eval_e3_keyframe", ...],
        "category": "",             # optional from source
        "prompt_id": "...",         # first prompt_id that references this motion
    }
  data/eval/m2m_v2/motion_master_list.csv  : flat CSV for easy review

Usage:
    python3 scripts/eval/build_motion_master_list.py
"""

import csv
import json
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATALIST_DIR = PROJECT_ROOT / "data" / "eval" / "m2m_v2"
OUT_JSON = DATALIST_DIR / "motion_master_list.json"
OUT_CSV = DATALIST_DIR / "motion_master_list.csv"

CAPTION_FIELDS = ("caption_en", "caption", "text_caption", "text")
MOTION_PATH_FIELDS = ("motion_path", "motion_a_path", "motion_b_path",
                      "target_motion_path")


def main():
    # motion_path -> {"captions": set, "tasks": set, "category": str, "prompt_id": str}
    motions: dict = defaultdict(lambda: {
        "captions": set(), "tasks": set(),
        "category": "", "prompt_id": "",
    })

    for f in sorted(DATALIST_DIR.glob("eval_e*.json")):
        if "_rewritten" in f.name:
            continue
        try:
            data = json.load(open(f))
        except Exception:
            continue
        items = data.get("data_list", [])
        task_stem = f.stem
        for it in items:
            for kmp in MOTION_PATH_FIELDS:
                mp = it.get(kmp)
                if not mp or not isinstance(mp, str):
                    continue
                entry = motions[mp]
                entry["tasks"].add(task_stem)
                if not entry["prompt_id"]:
                    entry["prompt_id"] = it.get("prompt_id", "")
                if not entry["category"]:
                    entry["category"] = it.get("category", "")
                for kc in CAPTION_FIELDS:
                    v = it.get(kc)
                    if isinstance(v, str) and v.strip():
                        if not v.strip().startswith("0"):  # filter bogus motion-name-as-caption
                            entry["captions"].add(v.strip())

    # Pick the longest caption as canonical
    out = []
    for mp, e in motions.items():
        captions_list = sorted(e["captions"], key=len, reverse=True)
        best_caption = captions_list[0] if captions_list else ""
        out.append({
            "motion_path": mp,
            "caption": best_caption,
            "all_captions": sorted(e["captions"]),
            "tasks": sorted(e["tasks"]),
            "num_tasks": len(e["tasks"]),
            "category": e["category"],
            "prompt_id": e["prompt_id"],
        })
    out.sort(key=lambda x: (-x["num_tasks"], x["motion_path"]))

    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"Wrote {OUT_JSON} with {len(out)} entries")

    with open(OUT_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["motion_path", "num_tasks", "tasks", "category",
                    "prompt_id", "caption"])
        for e in out:
            w.writerow([
                e["motion_path"], e["num_tasks"], ",".join(e["tasks"]),
                e["category"], e["prompt_id"], e["caption"],
            ])
    print(f"Wrote {OUT_CSV}")


if __name__ == "__main__":
    main()
