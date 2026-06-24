#!/usr/bin/env bash
# Repack/evaluate PRISM ablation-table generations and print LaTeX-ready rows.
set -euo pipefail

cd /apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer 2>/dev/null \
  || cd /apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false PYTHONPATH="$PWD:${PYTHONPATH:-}"

BASE=${BASE:-outputs/evaluation/prism_ablation_tables_20260616_e29}
BABEL_ROOT=${BABEL_ROOT:-outputs/evaluation/babel_seq/ktmodes_20260616_e29}
BABEL_SPECTRAL_JSON=${BABEL_SPECTRAL_JSON:-outputs/evaluation/babel_seq/ckpt_compare_20260615_m2/results/kt_latest_balanced_dedup.json}
EVAL_GPU=${EVAL_GPU:-0}
H3D_EXPECT=${H3D_EXPECT:-4200}
BABEL_EXPECT=${BABEL_EXPECT:-1100}

H3D_ANNO=data/annotation/test_hml3d.json
H3D_PREP="$BASE/h3d_ms272/prep"
H3D_RES="$BASE/h3d_ms272/results"
BABEL_RES="$BABEL_ROOT/results"
mkdir -p "$H3D_PREP" "$H3D_RES" "$BABEL_RES" "$BASE/logs" "$BABEL_ROOT/_logs"

count_npz() {
  python3 - "$1" <<'PY'
import sys
from pathlib import Path
p = Path(sys.argv[1])
print(sum(1 for x in p.glob("*.npz")) if p.exists() else 0)
PY
}

check_count() {
  local label="$1" dir="$2" expect="$3"
  local n
  n=$(count_npz "$dir")
  echo "[count] $label n=$n dir=$dir"
  if [ "$n" -lt "$expect" ]; then
    echo "[count] ERROR: $label has only $n npz, expected >= $expect" >&2
    return 1
  fi
}

repack_h3d() {
  local name="$1" src="$2" dst="$H3D_PREP/$name"
  check_count "$name/raw" "$src" "$H3D_EXPECT"
  if [ ! -f "$dst/_DONE" ]; then
    mkdir -p "$dst"
    python3 scripts/eval/repack_pred_to_272ids.py \
      --npz-dir "$src" --anno-file "$H3D_ANNO" --out-dir "$dst" --workers 16 \
      > "$BASE/logs/repack_${name}.log" 2>&1
    touch "$dst/_DONE"
  fi
  python3 - "$dst" <<'PY'
import sys
from pathlib import Path
split = Path("ref_repo/MotionStreamer/MotionStreamer/humanml3d_272/split/test.txt")
ids = {x.strip() for x in split.read_text().splitlines() if x.strip()}
pred = {x.stem for x in Path(sys.argv[1]).glob("*.npz")}
inter = len(ids & pred)
print(f"[sanity] {sys.argv[1]} pred={len(pred)} split_intersection={inter}")
if inter == 0:
    raise SystemExit("zero pred/GT split intersection")
PY
}

eval_h3d() {
  local name="$1" src="$2" oj
  oj="$H3D_RES/${name}.json"
  repack_h3d "$name" "$src"
  if [ ! -s "$oj" ]; then
    CUDA_VISIBLE_DEVICES="$EVAL_GPU" python3 scripts/eval/eval_motionstreamer_272.py \
      --pred-dir "$H3D_PREP/$name" --tag "$name" --also-refk --out-json "$oj" \
      > "$BASE/logs/eval_${name}.log" 2>&1
  fi
}

convert_babel() {
  local name="$1" gen="$2" f272
  f272="$BABEL_ROOT/${name}_272f"
  check_count "$name/raw" "$gen" "$BABEL_EXPECT"
  python3 scripts/eval/smpl_pred_to_272.py \
    --in-dir "$gen" --out-dir "$f272" --skip-existing \
    > "$BABEL_ROOT/_logs/${name}_to272.log" 2>&1
}

eval_babel() {
  local name="$1" gen="$2" oj
  oj="$BABEL_RES/${name}_balanced_dedup.json"
  convert_babel "$name" "$gen"
  if [ ! -s "$oj" ]; then
    CUDA_VISIBLE_DEVICES="$EVAL_GPU" python3 scripts/eval/eval_babel_seq_ms272.py \
      --manifest outputs/evaluation/babel_seq/common_valid_manifest.jsonl \
      --pred-dir "$BABEL_ROOT/${name}_272f" \
      --tag "${name}_balanced_dedup" \
      --out-json "$oj" \
      --max-total 360 \
      --mean-std humanml \
      --no-rewrite \
      --caption-template 'a person {cap}' \
      --dedup \
      --rprec-batching balanced \
      > "$BABEL_ROOT/_logs/${name}_balanced_dedup_eval.log" 2>&1
  fi
}

eval_h3d kafs_none "$BASE/h3d_kafs/none"
eval_h3d kafs_random "$BASE/h3d_kafs/random"
eval_h3d kt_seq "$BASE/h3d_kt/seq_depth"
eval_h3d kt_dfs "$BASE/h3d_kt/dfs_depth"

eval_babel kt_seq "$BABEL_ROOT/kt_seq_gen"
eval_babel kt_dfs "$BABEL_ROOT/kt_dfs_gen"

eval_h3d kafs_depth "$BASE/h3d_kafs/depth_driven"

python3 - "$BASE" "$BABEL_ROOT" "$BABEL_SPECTRAL_JSON" <<'PY'
import json
import sys
from pathlib import Path

base = Path(sys.argv[1])
babel_root = Path(sys.argv[2])
babel_spectral_json = Path(sys.argv[3])

def h3d(name):
    d = json.load(open(base / "h3d_ms272" / "results" / f"{name}.json"))
    p = d["pred"]
    return {
        "fid": float(p["fid_vs_gt_native"]),
        "r3": float(p["r_precision"][2]),
        "mm": float(p["matching_score"]),
    }

def babel(path):
    d = json.load(open(path))
    s = d["subseq"]
    return {
        "fid": float(s["fid"]),
        "r3": float(s["r3"]),
        "mm": float(s["mm_dist"]),
    }

res = {
    "kafs": {
        "none": h3d("kafs_none"),
        "uniform": h3d("kafs_none"),
        "random": h3d("kafs_random"),
        "depth": h3d("kafs_depth"),
    },
    "kt": {
        "seq": {"h3d": h3d("kt_seq"), "babel": babel(babel_root / "results" / "kt_seq_balanced_dedup.json")},
        "dfs": {"h3d": h3d("kt_dfs"), "babel": babel(babel_root / "results" / "kt_dfs_balanced_dedup.json")},
        "spectral": {"h3d": h3d("kafs_depth"), "babel": babel(babel_spectral_json)},
    },
}

out_json = base / "summary_tables.json"
out_json.write_text(json.dumps(res, indent=2) + "\n")

def fmt_h3d(m):
    return f"{m['fid']:.1f} & {m['r3']:.3f} & {m['mm']:.2f}"

def fmt_babel(m):
    return f"{m['r3']:.3f} & {m['fid']:.1f} & {m['mm']:.2f}"

rows = []
rows.append("% TABLE VI KAFS")
rows.append(f"None (baseline)          & {fmt_h3d(res['kafs']['none'])} \\\\")
rows.append(f"Uniform ($\\alpha{{=}}1.0$) & {fmt_h3d(res['kafs']['uniform'])} \\\\")
rows.append(f"Random ($\\alpha{{\\sim}}\\mathcal{{U}}$) & {fmt_h3d(res['kafs']['random'])} \\\\")
rows.append(f"Depth-driven (ours)      & {fmt_h3d(res['kafs']['depth'])} \\\\")
rows.append("")
rows.append("% TABLE VII KT-RoPE")
rows.append(f"Sequential RoPE ($\\rho{{=}}0.397$, baseline) & {fmt_h3d(res['kt']['seq']['h3d'])} & {fmt_babel(res['kt']['seq']['babel'])} \\\\")
rows.append(f"DFS Reindexing ($\\rho{{=}}0.628$)            & {fmt_h3d(res['kt']['dfs']['h3d'])} & {fmt_babel(res['kt']['dfs']['babel'])} \\\\")
rows.append(f"Projected Spectral ($r{{=}}4$, ours)         & {fmt_h3d(res['kt']['spectral']['h3d'])} & {fmt_babel(res['kt']['spectral']['babel'])} \\\\")
text = "\n".join(rows) + "\n"
(base / "summary_rows.tex").write_text(text)
print(text)
print(f"[summary] wrote {out_json}")
print(f"[summary] wrote {base / 'summary_rows.tex'}")
PY
