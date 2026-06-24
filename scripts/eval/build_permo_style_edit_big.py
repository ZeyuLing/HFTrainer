#!/usr/bin/env python3
"""Build a LARGER balanced PerMo neutral->style editing datalist for Table 10.

The shipped ``eval_e16_semantic_style_edit.json`` only has 120 pairs, far below
the Guo HumanML3D-263 motion encoder's 512-dim embedding, so the FID estimate is
rank-deficient/unstable. This script samples up to ``--per-style`` pairs per style
(balanced round-robin) from the full PerMo pool so the generated set comfortably
exceeds 512 clips, yielding a stable FID. Schema mirrors the 120-pair datalist
(``motion_path`` = target style clip, ``source_motion_path`` = neutral clip, paths
relative to ``data/hymotion_data`` via the legacy ``../hymotion_data`` prefix).
"""
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

_REPO = Path("/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer")
# Absolute canonical CephFS paths: the eval loader takes ``os.path.isabs`` paths
# verbatim, bypassing the symlinked ``data/hymotion_data`` join (whose ``../``
# legacy prefix does not resolve through the symlink).
PERMO_REL = str(_REPO / "data/hymotion_data/PerMo/PerMo/20260513/motions/train")
PERMO_ABS = _REPO / "data/hymotion_data/PerMo/PerMo/20260513/motions/train"

# "Style_Action_AXX_NNN.npz" -> (Style, Action, AXX, NNN)
PAT = re.compile(r"^([A-Za-z]+)_(.+?)_(A\d+)_(\d+)\.npz$")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-style", type=int, default=42,
                    help="max pairs per style (33 styles * 42 ~= 1300)")
    ap.add_argument("--out", default=str(
        _REPO / "data/eval/m2m_v2/eval_e16_semantic_style_edit_big.json"))
    args = ap.parse_args()

    neutral = set()
    style_files = defaultdict(list)
    for p in sorted(PERMO_ABS.glob("*.npz")):
        m = PAT.match(p.name)
        if not m:
            continue
        style, action, aidx, nnn = m.groups()
        if style == "Neutral":
            neutral.add((action, aidx, nnn))
        else:
            style_files[style].append((action, aidx, nnn, p.name))

    data_list = []
    pid = 0
    for style in sorted(style_files):
        kept = 0
        for action, aidx, nnn, fname in style_files[style]:
            if (action, aidx, nnn) not in neutral:
                continue  # require matching neutral source
            neutral_name = f"Neutral_{action}_{aidx}_{nnn}.npz"
            data_list.append({
                "prompt_id": f"style_{pid:05d}",
                "motion_path": f"{PERMO_REL}/{fname}",
                "source_motion_path": f"{PERMO_REL}/{neutral_name}",
                "caption": f"Perform {action} in a {style.lower()} style.",
                "caption_en": f"Perform {action} in a {style.lower()} style.",
                "source_caption": f"neutral source motion for {action}",
                "edit_setting": "style_edit",
                "edit_type": "style",
                "style_label": style,
                "action_name": action,
                "category": f"permo_style/{style}",
                "source": "PerMo neutral-to-style editing pairs (large balanced set)",
                "subset": "PerMo-editing-train",
                "fps": 30.0,
                "annotation_id": f"permo_editing_train_{Path(fname).stem}",
            })
            pid += 1
            kept += 1
            if kept >= args.per_style:
                break

    from collections import Counter
    dist = Counter(it["style_label"] for it in data_list)
    out = {
        "meta": {
            "task_id": "E16",
            "setting": "style_edit",
            "num_samples": len(data_list),
            "source": "Large balanced PerMo neutral-to-style audit set for stable FID (N>512).",
            "selection": f"up to {args.per_style} pairs per style, requiring a matching Neutral source.",
            "note": "Built by scripts/eval/build_permo_style_edit_big.py; not an official PerMo split.",
        },
        "data_list": data_list,
    }
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(f"[built] {len(data_list)} pairs over {len(dist)} styles -> {args.out}")
    print("per-style:", dict(dist))


if __name__ == "__main__":
    main()
