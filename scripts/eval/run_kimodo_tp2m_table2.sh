#!/usr/bin/env bash
# KIMODO TP2M (prefix-pose + text) Table-2 pipeline on HumanML3D test.
#   gen SMPL-22 positions (prefix cond {1,5,9}) -> joints->272 prep
#   -> MS-272 evaluator (R@3 / FID-native / MM-Dist / Diversity)
#   -> physical metrics (FS / Float / Jitter / Dyn) via compute_phys_h3d gt272.
set -uo pipefail

cd /apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer 2>/dev/null \
  || cd /apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer
ROOT="$PWD"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}" PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false

OUT=${OUT:-outputs/evaluation/kimodo_tp2m}
GPU=${GPU:-0}
NUM_SHARDS=${NUM_SHARDS:-3}
MAX_SAMPLES=${MAX_SAMPLES:-600}
CONDS=${CONDS:-"1 5 9"}
WORKERS=${WORKERS:-12}
LOG="$OUT/logs"; RES="$OUT/results"; PREP="$OUT/prep"; PHYS="$OUT/phys272"
mkdir -p "$LOG" "$RES" "$PREP" "$PHYS"

# Cache MS-272 evaluator ckpt + GT/text to /dev/shm for speed.
bash scripts/eval/_cache_272_data.sh > "$LOG/cache.log" 2>&1 || true
MS_REL="ref_repo/MotionStreamer/MotionStreamer"
if [ ! -f /dev/shm/eval272_epoch99.ckpt ]; then
  cp "$MS_REL/MotionStreamer_HF/Evaluator_272/epoch=99.ckpt" /dev/shm/eval272_epoch99.ckpt 2>/dev/null || true
fi
# Prefer the /dev/shm test-set mirror (split+motion_data+texts) so the job loader
# reads clip lengths from RAM instead of scanning 4k CephFS files per shard.
H3D_ROOT=${H3D_ROOT:-/dev/shm/ms272_data}
if [ ! -f "$H3D_ROOT/split/test.txt" ]; then
  H3D_ROOT="$MS_REL/humanml3d_272"
fi
echo "[setup] H3D_ROOT=$H3D_ROOT" | tee -a "$LOG/run.log"

echo "[start] $(date) out=$OUT conds=$CONDS max=$MAX_SAMPLES shards=$NUM_SHARDS" | tee "$LOG/run.log"

for cond in $CONDS; do
  echo "[gen cond$cond] $(date)" | tee -a "$LOG/run.log"
  for i in $(seq 0 $((NUM_SHARDS - 1))); do
    CUDA_VISIBLE_DEVICES="$GPU" python3 scripts/eval/gen_kimodo_tp2m_positions.py \
      --humanml3d-272 "$H3D_ROOT" --out-dir "$OUT" --condition-num-frames "$cond" \
      --max-samples "$MAX_SAMPLES" --num-shards "$NUM_SHARDS" --shard-index "$i" \
      --skip-existing --device cuda \
      > "$LOG/gen_cond${cond}_s${i}.log" 2>&1 &
  done
  wait
  gen_dir="$OUT/cond${cond}"
  n_gen=$(ls "$gen_dir"/*.npy 2>/dev/null | wc -l)
  echo "[gen cond$cond done] n=$n_gen" | tee -a "$LOG/run.log"

  # joints -> 272 prep (for MS-272 eval)
  prep="$PREP/cond${cond}"
  python3 scripts/eval/joints_to_272_npz.py \
    --in-dir "$gen_dir" --out "$prep" --input-kind joints --src-fps 30 \
    --workers "$WORKERS" > "$LOG/conv_cond${cond}.log" 2>&1
  n_prep=$(ls "$prep"/*.npz 2>/dev/null | wc -l)
  echo "[conv cond$cond] n272=$n_prep" | tee -a "$LOG/run.log"

  # extract motion_272 -> (T,272) .npy dir for compute_phys_h3d gt272 mode
  phys="$PHYS/cond${cond}"
  python3 - "$prep" "$phys" <<'PY'
import numpy as np, glob, os, sys
src, dst = sys.argv[1], sys.argv[2]
os.makedirs(dst, exist_ok=True)
for f in glob.glob(os.path.join(src, "*.npz")):
    m = np.load(f)["motion_272"].astype(np.float32)
    np.save(os.path.join(dst, os.path.basename(f)[:-4] + ".npy"), m)
print("extracted", len(glob.glob(dst + "/*.npy")))
PY

  # MS-272 evaluator (R@3 / FID-native / MM-Dist / Diversity)
  echo "[eval-ms272 cond$cond] $(date)" | tee -a "$LOG/run.log"
  CUDA_VISIBLE_DEVICES="$GPU" python3 scripts/eval/eval_motionstreamer_272.py \
    --pred-dir "$prep" --tag "kimodo_c${cond}" --also-refk \
    --out-json "$RES/kimodo_c${cond}.json" \
    > "$LOG/eval_ms272_cond${cond}.log" 2>&1

  # Physical metrics (FS / Float / Jitter / Dyn)
  echo "[eval-phys cond$cond] $(date)" | tee -a "$LOG/run.log"
  python3 scripts/eval/compute_phys_h3d.py \
    --gt272-dir "$phys" --tag "kimodo_c${cond}" \
    --out-json "$RES/phys_kimodo_c${cond}.json" --workers "$WORKERS" \
    > "$LOG/eval_phys_cond${cond}.log" 2>&1
  grep "TABLE" "$LOG/eval_phys_cond${cond}.log" | tee -a "$LOG/run.log"
done

echo "[summarize] $(date)" | tee -a "$LOG/run.log"
python3 - "$OUT" $CONDS <<'PY'
import json, os, sys
out = sys.argv[1]; conds = sys.argv[2:]
print("\n==== KIMODO TP2M Table-2 ====")
hdr = f"{'cond':>4} {'n':>5} {'R-P_T3':>7} {'FID_nat':>8} {'MM-D':>7} {'Div':>7} "\
      f"{'FS_mm':>7} {'Float%':>7} {'Jit1e3':>7} {'Dyn1e3':>7}"
print(hdr)
for c in conds:
    mj = os.path.join(out, "results", f"kimodo_c{c}.json")
    pj = os.path.join(out, "results", f"phys_kimodo_c{c}.json")
    r3 = fid = mm = div = n = None
    if os.path.exists(mj):
        d = json.load(open(mj)).get("pred", {})
        rp = d.get("r_precision") or [None, None, None]
        r3 = rp[2]; fid = d.get("fid_vs_gt_native"); mm = d.get("matching_score")
        div = d.get("diversity"); n = d.get("nb")
    fs = fl = jit = dyn = pn = None
    if os.path.exists(pj):
        p = json.load(open(pj)).get(f"kimodo_c{c}", {})
        pn = p.get("n")
        fs = (p.get("Slide") or 0) * 1000
        fl = (p.get("Float") or 0) * 100
        jit = (p.get("Jitter") or 0) * 1000
        dyn = (p.get("Dynamic") or 0) * 1000
    def f(x, w=7, p=3):
        return (f"{x:>{w}.{p}f}" if isinstance(x, (int, float)) else f"{str(x):>{w}}")
    print(f"{c:>4} {str(n):>5} {f(r3,7,4)} {f(fid,8)} {f(mm)} {f(div)} {f(fs)} {f(fl)} {f(jit)} {f(dyn)}  (phys_n={pn})")
PY
echo "[done] $(date)" | tee -a "$LOG/run.log"
