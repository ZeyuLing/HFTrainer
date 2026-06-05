#!/usr/bin/env python3
"""Prepare MBench prompts for HumanML3D-263 baseline inference scripts."""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str, payload) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def build_frame_map(eval_info_json: str) -> Dict[int, int]:
    frame_map: Dict[int, int] = {}
    for row in load_json(eval_info_json):
        motion_id = int(row["id"])
        frames = int(row["motion_duration"])
        old = frame_map.get(motion_id)
        if old is not None and old != frames:
            raise ValueError(f"Conflicting frames for id={motion_id}: {old} vs {frames}")
        frame_map[motion_id] = frames
    return frame_map


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt-json", default="ref_repo/ViMoGen/data/meta_info/MBench_final.json")
    parser.add_argument("--eval-info-json", default="ref_repo/ViMoGen/data/meta_info/MBench_eval_info.json")
    parser.add_argument("--out-anno", default="data/annotation/mbench_450_hml263_prompts.json")
    parser.add_argument("--out-captions", default="data/annotation/mbench_450_hml263_captions.json")
    parser.add_argument("--fps", type=float, default=20.0)
    args = parser.parse_args()

    frame_map = build_frame_map(args.eval_info_json)
    data_list = {}
    captions = {}
    for row in load_json(args.prompt_json):
        global_id = int(row.get("global_id", row.get("id")))
        frames = int(frame_map[global_id])
        key = str(global_id)
        prompt = str(row["prompt"]).strip()
        data_list[key] = {
            "subset": "MBench",
            "fps": float(args.fps),
            "num_frames": frames,
            "duration": frames / float(args.fps),
            "mbench_id": global_id,
            "caption": prompt,
        }
        captions[key] = prompt

    expected = [str(i) for i in range(len(data_list))]
    if sorted(data_list, key=int) != expected:
        raise ValueError("MBench prompt ids are not contiguous 0..N-1")

    annotation = {
        "meta_info": {
            "dataset": "MBench",
            "fps": float(args.fps),
            "source_prompt_json": args.prompt_json,
            "source_eval_info_json": args.eval_info_json,
        },
        "data_list": data_list,
    }
    write_json(args.out_anno, annotation)
    write_json(args.out_captions, captions)
    print(f"[prepare-mbench-hml263] wrote {args.out_anno} and {args.out_captions}")


if __name__ == "__main__":
    main()
