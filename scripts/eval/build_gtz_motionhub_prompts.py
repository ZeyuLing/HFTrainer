#!/usr/bin/env python3
"""Build GTZ (MotionMillion) MotionHub T2M prompts/ids using the ALL-ORIGINAL
caption protocol, matching scripts/eval/mdm_infer_hml3d263.py exactly.

For each MotionHub test entry (data_list key == smplx stem), the original caption
is the first item of the macro/meso/micro pool in the hierarchical caption JSON.
Outputs:
  ref_repo/MotionMillion-Codes/run_motionhub/prompts.txt  (one caption per line)
  ref_repo/MotionMillion-Codes/run_motionhub/ids.txt      (one id per line, aligned)
"""
import json
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ANNO = REPO / "data/annotation/test_motionhub_t2m.json"
DATA_DIR = REPO / "data/motionhub"
OUT_DIR = REPO / "ref_repo/MotionMillion-Codes/run_motionhub"
DEFAULT_FPS = 30.0


def load_caption_from_json(path: Path):
    try:
        data = json.loads(path.read_text())
    except Exception:
        return None
    pool = []
    if isinstance(data, dict) and all(isinstance(data.get(k), list) for k in ("macro", "meso", "micro")):
        for group in ("macro", "meso", "micro"):
            pool.extend(v.strip() for v in data[group] if isinstance(v, str) and v.strip())
    elif isinstance(data, dict) and isinstance(data.get("result"), list):
        for item in data["result"]:
            if not isinstance(item, dict):
                continue
            for key in ("short_caption_rewritten", "short caption_rewritten"):
                vals = item.get(key)
                if isinstance(vals, list):
                    pool.extend(v.strip() for v in vals if isinstance(v, str) and v.strip())
                    break
            else:
                for key in ("short_caption", "short caption"):
                    val = item.get(key)
                    if isinstance(val, str) and val.strip():
                        pool.append(val.strip())
                        break
    return pool[0] if pool else None


def main():
    raw = json.loads(ANNO.read_text())
    data_list = raw["data_list"]
    assert isinstance(data_list, dict), "expected dict data_list"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    ids, prompts = [], []
    skipped_nocap = skipped_len = 0
    for name, entry in data_list.items():
        cap = None
        hcp = entry.get("hierarchical_caption_path")
        if hcp:
            cap = load_caption_from_json(DATA_DIR / hcp)
        if not isinstance(cap, str) or not cap.strip():
            skipped_nocap += 1
            continue
        src_fps = float(entry.get("fps") or DEFAULT_FPS)
        length_src = int(entry.get("num_frames") or round(float(entry.get("duration", 0.0)) * src_fps))
        if length_src <= 0:
            skipped_len += 1
            continue
        cap = " ".join(cap.strip().split())  # single line
        ids.append(str(name))
        prompts.append(cap)

    (OUT_DIR / "ids.txt").write_text("\n".join(ids) + "\n")
    (OUT_DIR / "prompts.txt").write_text("\n".join(prompts) + "\n")
    # small smoke subset
    (OUT_DIR / "smoke_ids.txt").write_text("\n".join(ids[:8]) + "\n")
    (OUT_DIR / "smoke_prompts.txt").write_text("\n".join(prompts[:8]) + "\n")
    print(f"total data_list: {len(data_list)}")
    print(f"written: {len(ids)} (skipped no-caption={skipped_nocap}, bad-length={skipped_len})")
    print(f"-> {OUT_DIR/'prompts.txt'}")
    print(f"-> {OUT_DIR/'ids.txt'}")
    print("sample:")
    for i in range(min(3, len(ids))):
        print(f"  {ids[i]} | {prompts[i][:80]}")


if __name__ == "__main__":
    main()
