#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer}"
if [[ ! -d "$ROOT" ]]; then
  ROOT="/apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer"
fi
cd "$ROOT"

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-$ROOT/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"

NUM_GPUS="${NUM_GPUS:-${TJ_GPU_NUM:-8}}"
if [[ "$NUM_GPUS" -lt 1 ]]; then
  NUM_GPUS=1
fi

ANNO="data/annotation/test_hml3d_official272_gtlen.json"
DATA_DIR="."
RUN_ROOT="outputs/evaluation/t2m/humanml3d_official_test/_runs/table1_remaining_hml263_20260618"
LOG_DIR="$RUN_ROOT/logs"
PREP_DIR="$RUN_ROOT/prep"
AUDIT_DIR="$RUN_ROOT/audits"
mkdir -p "$LOG_DIR" "$PREP_DIR" "$AUDIT_DIR"

HML_BASE="outputs/evaluation/t2m/humanml3d_official_test/hml263"
M135_BASE="outputs/evaluation/t2m/humanml3d_official_test/motion135"
MS272_BASE="outputs/evaluation/t2m/humanml3d_official_test/ms272"

caption_file="$PREP_DIR/official_first_caption.json"
full_anno="$PREP_DIR/official_4042_anno.json"

echo "[start] $(date -Is) root=$ROOT num_gpus=$NUM_GPUS" | tee "$LOG_DIR/run.log"

ensure_python_deps() {
  local missing="$PREP_DIR/missing_python_deps.txt"
  python3 - <<'PY' > "$missing"
mods = {
    "einops": "einops",
    "ftfy": "ftfy",
    "regex": "regex",
    "tqdm": "tqdm",
    "chumpy": "chumpy>=0.70",
    "sentence_transformers": "sentence-transformers",
    "rotary_embedding_torch": "rotary-embedding-torch",
    "roma": "roma",
}
for mod, pkg in mods.items():
    try:
        __import__(mod)
    except Exception:
        print(pkg)
try:
    __import__("clip")
except Exception:
    print("git+https://github.com/openai/CLIP.git")
PY
  if [[ -s "$missing" ]]; then
    echo "[deps] installing $(tr '\n' ' ' < "$missing")" | tee -a "$LOG_DIR/run.log"
    python3 -m pip install -q \
      -i https://mirrors.tencent.com/pypi/simple \
      --trusted-host mirrors.tencent.com \
      -r "$missing"
  else
    echo "[deps] python deps ok" | tee -a "$LOG_DIR/run.log"
  fi
}

ensure_python_deps

python3 - <<'PY' "$ANNO" "$full_anno" "$caption_file" "$DATA_DIR"
import json
import sys
from pathlib import Path

anno_path = Path(sys.argv[1])
out_anno = Path(sys.argv[2])
out_caption = Path(sys.argv[3])
data_dir = Path(sys.argv[4])

raw = json.loads(anno_path.read_text())
data = raw.get("data_list", raw) if isinstance(raw, dict) else raw
if not isinstance(data, dict):
    data = {
        str(item.get("source_id") or item.get("motion_id") or idx): item
        for idx, item in enumerate(data)
    }

def first_caption(entry):
    rel = entry.get("hierarchical_caption_path")
    if not rel:
        return None
    path = data_dir / rel
    if not path.exists():
        return None
    try:
        obj = json.loads(path.read_text())
    except Exception:
        return None
    stack = [obj]
    while stack:
        cur = stack.pop(0)
        if isinstance(cur, str) and cur.strip():
            return cur.strip()
        if isinstance(cur, dict):
            for key in ("caption", "text", "sentence", "raw_caption"):
                val = cur.get(key)
                if isinstance(val, str) and val.strip():
                    return val.strip()
            for val in cur.values():
                if isinstance(val, (dict, list, str)):
                    stack.append(val)
        elif isinstance(cur, list):
            stack[:0] = cur
    return None

captions = {}
kept = {}
for sid, entry in data.items():
    sid = str(sid)
    cap = first_caption(entry)
    if not cap:
        # Last-resort stable prompt; should rarely be used for official annotations.
        cap = "a person is moving"
    captions[sid] = cap
    kept[sid] = entry

out_anno.parent.mkdir(parents=True, exist_ok=True)
out_anno.write_text(json.dumps({"meta": raw.get("meta", {}), "data_list": kept}, indent=2))
out_caption.write_text(json.dumps(captions, indent=2))
print(f"[prep] ids={len(kept)} captions={len(captions)} anno={out_anno} captions={out_caption}", flush=True)
PY

link_flat() {
  local src="$1"
  local dst="$2"
  local ext="$3"
  mkdir -p "$dst"
  if [[ -d "$src" ]]; then
    python3 - <<'PY' "$src" "$dst" "$ext"
import os
import sys
from pathlib import Path

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
ext = sys.argv[3]
linked = 0
for path in src.glob(f"*.{ext}"):
    out = dst / path.name
    if out.exists() or out.is_symlink():
        continue
    os.symlink(str(path.resolve()), out)
    linked += 1
print(f"[link] {src} -> {dst} ext={ext} linked={linked}", flush=True)
PY
  else
    echo "[link-skip] missing src=$src" | tee -a "$LOG_DIR/run.log"
  fi
}

write_method_meta() {
  local method_dir="$1"
  local method="$2"
  local native_src="$3"
  mkdir -p "$method_dir/logs" "$method_dir/metrics" "$method_dir/conversions/hml263_to_motion135" "$method_dir/conversions/motion135_to_ms272"
  cat > "$method_dir/command.txt" <<EOF
ROOT=$ROOT NUM_GPUS=$NUM_GPUS bash scripts/eval/run_table1_remaining_hml263_20260618.sh
EOF
  python3 - <<'PY' "$method_dir/run_config.json" "$method" "$native_src" "$ANNO" "$RUN_ROOT"
import json
import sys
from pathlib import Path
out = Path(sys.argv[1])
out.write_text(json.dumps({
    "method": sys.argv[2],
    "dataset": "humanml3d_official_test",
    "task": "t2m",
    "native_representation": "hml263",
    "annotation": sys.argv[4],
    "legacy_source": sys.argv[3],
    "runner": sys.argv[5],
    "created_by": "scripts/eval/run_table1_remaining_hml263_20260618.sh",
}, indent=2))
PY
}

# Formal HML263 directories plus legacy source directories.
declare -A HML_DIR=(
  [momask]="$HML_BASE/momask_official/predictions/hml263"
  [t2mgpt]="$HML_BASE/t2mgpt_official/predictions/hml263"
  [mdm]="$HML_BASE/mdm_official/predictions/hml263"
  [mld]="$HML_BASE/mld_official/predictions/hml263"
  [flowmdm]="$HML_BASE/flowmdm_official/predictions/hml263"
  [motionlab]="$HML_BASE/motionlab_official/predictions/hml263"
  [motiongpt3]="$HML_BASE/motiongpt3_official/predictions/hml263"
)

declare -A HML_LEGACY=(
  [momask]="$HML_BASE/momask_official/momask_263"
  [t2mgpt]="$HML_BASE/t2mgpt_official/t2mgpt_263"
  [mdm]="$HML_BASE/mdm_official/mdm_263"
  [mld]=""
  # Stats-fix canonical candidates. The older fixstats0605b dirs used the
  # small-root-std HumanML3D stats and cause systematic forward drift.
  [flowmdm]="$HML_BASE/flowmdm_officialstats_20260622/predictions/hml263"
  [motionlab]="$HML_BASE/motionlab_officialstats_cfg575_eval51_20260622/predictions/hml263"
  [motiongpt3]="outputs/evaluation/humanml3d/motiongpt_official_h3d263_repro_0605/pred"
)

declare -A M135_DIR=(
  [momask]="$M135_BASE/momask_official/predictions/motion135"
  [t2mgpt]="$M135_BASE/t2mgpt_official/predictions/motion135"
  [mdm]="$M135_BASE/mdm_official/predictions/motion135"
  [mld]="$M135_BASE/mld_official/predictions/motion135"
  [flowmdm]="$M135_BASE/flowmdm_official/predictions/motion135"
  [motionlab]="$M135_BASE/motionlab_official/predictions/motion135"
  [motiongpt3]="$M135_BASE/motiongpt3_official/predictions/motion135"
)

declare -A M135_LEGACY=(
  [momask]="$MS272_BASE/_suites/table1_exactlen_0617/prep/momask"
  [t2mgpt]="$MS272_BASE/_suites/table1_exactlen_0617/prep/t2mgpt"
  [mdm]="$MS272_BASE/mdm_repro/mdm_smpl135"
  [mld]=""
  [flowmdm]="$MS272_BASE/_suites/table1_exactlen_0617/prep/flowmdm"
  [motionlab]="$MS272_BASE/_suites/table1_exactlen_0617/prep/motionlab"
  [motiongpt3]=""
)

for method in momask t2mgpt mdm mld flowmdm motionlab motiongpt3; do
  write_method_meta "$(dirname "$(dirname "${HML_DIR[$method]}")")" "$method" "${HML_LEGACY[$method]}" 
  link_flat "${HML_LEGACY[$method]}" "${HML_DIR[$method]}" npy
  write_method_meta "$(dirname "$(dirname "${M135_DIR[$method]}")")" "$method" "${M135_LEGACY[$method]}" 
  link_flat "${M135_LEGACY[$method]}" "${M135_DIR[$method]}" npz
done

python3 - <<'PY' "$full_anno" "$AUDIT_DIR" \
  "momask=${HML_DIR[momask]}" "t2mgpt=${HML_DIR[t2mgpt]}" "mdm=${HML_DIR[mdm]}" \
  "mld=${HML_DIR[mld]}" "flowmdm=${HML_DIR[flowmdm]}" "motionlab=${HML_DIR[motionlab]}" \
  "motiongpt3=${HML_DIR[motiongpt3]}" \
  "momask_m135=${M135_DIR[momask]}" "t2mgpt_m135=${M135_DIR[t2mgpt]}" "mdm_m135=${M135_DIR[mdm]}" \
  "mld_m135=${M135_DIR[mld]}" "flowmdm_m135=${M135_DIR[flowmdm]}" "motionlab_m135=${M135_DIR[motionlab]}" \
  "motiongpt3_m135=${M135_DIR[motiongpt3]}"
import json
import sys
from pathlib import Path

anno = json.loads(Path(sys.argv[1]).read_text())
ids = list(anno["data_list"].keys())
out_dir = Path(sys.argv[2])
out_dir.mkdir(parents=True, exist_ok=True)
summary = {}
for spec in sys.argv[3:]:
    name, directory = spec.split("=", 1)
    directory = Path(directory)
    ext = ".npz" if name.endswith("_m135") else ".npy"
    stems = set()
    if directory.exists():
        for path in directory.glob(f"*{ext}"):
            stem = path.stem
            if stem.startswith("humanml3d_"):
                stem = stem.split("_")[-1]
            stems.add(stem)
    missing = [sid for sid in ids if sid not in stems]
    (out_dir / f"{name}_missing.txt").write_text("\n".join(missing) + ("\n" if missing else ""))
    # Annotation subset for tools that do not accept an id-file filter.
    if not name.endswith("_m135"):
        subset = {sid: anno["data_list"][sid] for sid in missing}
        (out_dir / f"{name}_missing_anno.json").write_text(json.dumps({
            "meta": anno.get("meta", {}),
            "data_list": subset,
        }, indent=2))
    summary[name] = {"present": len(ids) - len(missing), "missing": len(missing)}
Path(out_dir / "coverage_summary.json").write_text(json.dumps(summary, indent=2))
for name, item in summary.items():
    print(f"[coverage] {name} present={item['present']} missing={item['missing']}", flush=True)
PY

run_shards() {
  local phase="$1"
  local shards="$2"
  shift 2
  echo "[phase-start] $phase shards=$shards $(date -Is)" | tee -a "$LOG_DIR/run.log"
  local pids=()
  local shard gpu log
  for shard in $(seq 0 $((shards - 1))); do
    gpu=$((shard % NUM_GPUS))
    log="$LOG_DIR/${phase}_shard_${shard}.log"
    (
      set -euo pipefail
      export CUDA_VISIBLE_DEVICES="$gpu"
      "$@" "$shards" "$shard" > "$log" 2>&1
      echo "exit_code=0 finished_at=$(date -Is)" > "$LOG_DIR/${phase}_shard_${shard}.status"
    ) &
    pids+=("$!")
  done
  local rc=0
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      rc=1
    fi
  done
  if [[ "$rc" != 0 ]]; then
    echo "[phase-fail] $phase $(date -Is)" | tee -a "$LOG_DIR/run.log"
    return "$rc"
  fi
  echo "[phase-done] $phase $(date -Is)" | tee -a "$LOG_DIR/run.log"
}

run_momask() {
  local shards="$1" shard="$2"
  python3 scripts/eval/momask_infer_h3d_test.py \
    --momask_root ref_repo/Momask/momask-codes \
    --humanml3d_272 ref_repo/MotionStreamer/MotionStreamer/humanml3d_272 \
    --anno_file "$AUDIT_DIR/momask_missing_anno.json" \
    --data_dir "$DATA_DIR" \
    --caption_protocol original \
    --out_dir "${HML_DIR[momask]}" \
    --num_shards "$shards" \
    --shard_index "$shard" \
    --batch_size 32 \
    --device cuda \
    --gumbel_sample \
    --skip_existing
}

run_t2mgpt() {
  local shards="$1" shard="$2"
  python3 scripts/eval/t2mgpt_infer_hml3d263.py \
    --anno-file "$AUDIT_DIR/t2mgpt_missing_anno.json" \
    --caption-file "$caption_file" \
    --out-dir "${HML_DIR[t2mgpt]}" \
    --num-shards "$shards" \
    --shard-index "$shard" \
    --batch-size 32 \
    --device cuda \
    --skip-existing
}

run_mdm() {
  local shards="$1" shard="$2"
  python3 scripts/eval/mdm_infer_hml3d263.py \
    --model_path ref_repo/MDM/save/humanml_enc_512_50steps/model000750000.pt \
    --anno_file "$AUDIT_DIR/mdm_missing_anno.json" \
    --anno_data_dir "$DATA_DIR" \
    --caption_protocol original \
    --out_dir "${HML_DIR[mdm]}" \
    --num_shards "$shards" \
    --shard_index "$shard" \
    --batch_size 16 \
    --device 0 \
    --skip_existing
}

run_mld() {
  local shards="$1" shard="$2"
  python3 scripts/eval/mld_infer_hml3d263.py \
    --anno_file "$AUDIT_DIR/mld_missing_anno.json" \
    --anno_data_dir "$DATA_DIR" \
    --caption_protocol original \
    --out_dir "${HML_DIR[mld]}" \
    --num_shards "$shards" \
    --shard_index "$shard" \
    --batch_size 16 \
    --skip_existing
}

run_flowmdm() {
  local shards="$1" shard="$2"
  python3 scripts/eval/flowmdm_infer_hml3d263.py \
    --anno-file "$full_anno" \
    --caption-file "$caption_file" \
    --data-dir "$DATA_DIR" \
    --out-dir "${HML_DIR[flowmdm]}" \
    --only-ids "$AUDIT_DIR/flowmdm_missing.txt" \
    --num-shards "$shards" \
    --shard-index "$shard" \
    --device 0 \
    --min-length 1 \
    --max-length 196 \
    --skip-existing
}

run_motionlab() {
  local shards="$1" shard="$2"
  python3 scripts/eval/motionlab_infer_hml3d263.py \
    --anno-file "$full_anno" \
    --caption-file "$caption_file" \
    --data-dir "$DATA_DIR" \
    --out-dir "${HML_DIR[motionlab]}" \
    --source-id-file "$AUDIT_DIR/motionlab_missing.txt" \
    --num-shards "$shards" \
    --shard-index "$shard" \
    --device cuda \
    --min-length 1 \
    --max-length 196 \
    --stage eval \
    --no-cfg-from-checkpoint \
    --cfg configs/config_rfmotion.yaml \
    --skip-existing
}

run_motiongpt3() {
  local shards="$1" shard="$2"
  python3 scripts/eval/motiongpt3_infer_hml3d263.py \
    --anno_file "$AUDIT_DIR/motiongpt3_missing_anno.json" \
    --anno_data_dir "$DATA_DIR" \
    --caption_protocol original \
    --out_dir "${HML_DIR[motiongpt3]}" \
    --num_shards "$shards" \
    --shard_index "$shard" \
    --batch_size 8 \
    --guidance_scale 3.0 \
    --skip_existing
}

maybe_run_hml263() {
  local method="$1"
  local missing_file="$AUDIT_DIR/${method}_missing.txt"
  if [[ ! -s "$missing_file" ]]; then
    echo "[skip] $method hml263 complete" | tee -a "$LOG_DIR/run.log"
    return
  fi
  case "$method" in
    momask) run_shards "infer_${method}" "$NUM_GPUS" run_momask ;;
    t2mgpt) run_shards "infer_${method}" "$NUM_GPUS" run_t2mgpt ;;
    mdm) run_shards "infer_${method}" "$NUM_GPUS" run_mdm ;;
    mld) run_shards "infer_${method}" "$NUM_GPUS" run_mld ;;
    flowmdm) run_shards "infer_${method}" "$NUM_GPUS" run_flowmdm ;;
    motionlab) run_shards "infer_${method}" "$NUM_GPUS" run_motionlab ;;
    motiongpt3) run_shards "infer_${method}" "$NUM_GPUS" run_motiongpt3 ;;
  esac
}

for method in momask t2mgpt mdm flowmdm motionlab motiongpt3 mld; do
  maybe_run_hml263 "$method"
done

# Refresh missing files after native HML263 inference.
python3 - <<'PY' "$full_anno" "$AUDIT_DIR" \
  "momask=${HML_DIR[momask]}" "t2mgpt=${HML_DIR[t2mgpt]}" "mdm=${HML_DIR[mdm]}" \
  "mld=${HML_DIR[mld]}" "flowmdm=${HML_DIR[flowmdm]}" "motionlab=${HML_DIR[motionlab]}" \
  "motiongpt3=${HML_DIR[motiongpt3]}" \
  "momask_m135=${M135_DIR[momask]}" "t2mgpt_m135=${M135_DIR[t2mgpt]}" "mdm_m135=${M135_DIR[mdm]}" \
  "mld_m135=${M135_DIR[mld]}" "flowmdm_m135=${M135_DIR[flowmdm]}" "motionlab_m135=${M135_DIR[motionlab]}" \
  "motiongpt3_m135=${M135_DIR[motiongpt3]}"
import json
import sys
from pathlib import Path
anno = json.loads(Path(sys.argv[1]).read_text())
ids = list(anno["data_list"].keys())
out_dir = Path(sys.argv[2])
summary = {}
for spec in sys.argv[3:]:
    name, directory = spec.split("=", 1)
    directory = Path(directory)
    ext = ".npz" if name.endswith("_m135") else ".npy"
    stems = {p.stem.split("_")[-1] if p.stem.startswith("humanml3d_") else p.stem for p in directory.glob(f"*{ext}")} if directory.exists() else set()
    missing = [sid for sid in ids if sid not in stems]
    (out_dir / f"{name}_missing.txt").write_text("\n".join(missing) + ("\n" if missing else ""))
    summary[name] = {"present": len(ids) - len(missing), "missing": len(missing)}
(out_dir / "coverage_summary_after_hml263.json").write_text(json.dumps(summary, indent=2))
for name, item in summary.items():
    print(f"[coverage-after-hml263] {name} present={item['present']} missing={item['missing']}", flush=True)
PY

run_ik_method() {
  local method="$1"
  local shards="$2" shard="$3"
  python3 scripts/eval/hml263_to_smpl_ik.py \
    --in-dir "${HML_DIR[$method]}" \
    --out-dir "${M135_DIR[$method]}" \
    --ids "$AUDIT_DIR/${method}_m135_missing.txt" \
    --num-shards "$shards" \
    --shard-index "$shard" \
    --source-fps 20 \
    --target-fps 30 \
    --target-length-anno "$full_anno" \
    --device cuda \
    --batch-size 1 \
    --floor-align \
    --refine-iters 80 \
    --refine-lr 0.02 \
    --skip-existing
}

for method in momask t2mgpt mdm flowmdm motionlab motiongpt3 mld; do
  if [[ ! -s "$AUDIT_DIR/${method}_m135_missing.txt" ]]; then
    echo "[skip] $method motion135 complete" | tee -a "$LOG_DIR/run.log"
    continue
  fi
  run_shards "ik_${method}" "$NUM_GPUS" run_ik_method "$method"
done

python3 scripts/eval/audit_table1_lengths.py \
  --out-dir "$AUDIT_DIR/final_motion135_lengths" \
  --method "MoMask=${M135_DIR[momask]}" \
  --method "T2M-GPT=${M135_DIR[t2mgpt]}" \
  --method "MDM=${M135_DIR[mdm]}" \
  --method "MLD=${M135_DIR[mld]}" \
  --method "FlowMDM=${M135_DIR[flowmdm]}" \
  --method "MotionLab=${M135_DIR[motionlab]}" \
  --method "MotionGPT3=${M135_DIR[motiongpt3]}" \
  | tee "$LOG_DIR/final_motion135_length_audit.log"

convert_one_ms272() {
  local method="$1"
  local out_dir="$MS272_BASE/${method}_official_from_motion135/predictions/ms272"
  mkdir -p "$out_dir"
  python3 scripts/data/convert_motion135_to_h3d272.py \
    --in-dir "${M135_DIR[$method]}" \
    --out-dir "$out_dir" \
    --rotation-space local \
    --workers 8 \
    > "$LOG_DIR/ms272_${method}.log" 2>&1
}

for method in momask t2mgpt mdm flowmdm motionlab motiongpt3 mld; do
  echo "[ms272-start] $method $(date -Is)" | tee -a "$LOG_DIR/run.log"
  convert_one_ms272 "$method"
  echo "[ms272-done] $method $(date -Is)" | tee -a "$LOG_DIR/run.log"
done

echo "[done] $(date -Is)" | tee -a "$LOG_DIR/run.log"
touch "$RUN_ROOT/_DONE"
