#!/bin/bash
# Validation: regenerate a small BABEL sample with the FIX (grammatical caption
# rewrite + KAFS disabled, matching the proven text-only T2M setting) and report
# motion statistics vs GT. Confirms whether the "obviously wrong" ours was caused
# by OOD terse captions / KAFS before launching a full regeneration.
set -uo pipefail
ROOT=${ROOT:-/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer}
[ -d "$ROOT" ] || ROOT=/apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer
cd "$ROOT"
export PYTHONPATH=$PWD:${PYTHONPATH:-} PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
PY=${PY:-python3}

CONFIG=configs/prism/prism_1b_tp2m_multiframe_kt_spectral_unified.py
CKPT=${CKPT:-work_dirs/prism_1b_tp2m_multiframe_kt_spectral_unified_t5cached/checkpoint-epoch_19}
MAN=data/babel/babel_seq_val_manifest.jsonl
N=${N:-16}
OUT=${OUT:-outputs/evaluation/babel_seq/prism_gen_llm}

echo "[fix-val] ckpt=$CKPT N=$N out=$OUT"
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} "$PY" -u scripts/eval/gen_prism_babel_seq.py \
  --config "$CONFIG" --checkpoint "$CKPT" --manifest "$MAN" \
  --output-dir "$OUT" --kafs-mode none --rewrite-captions \
  --num-inference-steps 50 --guidance-scale 5.0 --max-episodes "$N"

echo "[fix-val] computing motion stats vs GT ..."
"$PY" - <<'PY'
import os,glob,numpy as np,torch,smplx
bm=smplx.create(model_path="checkpoints/smpl_models",model_type="smplx",gender="neutral",num_betas=10,use_pca=False);bm.eval()
def st(p):
    d=np.load(p,allow_pickle=True);T=d["poses"].shape[0]
    g=lambda k,n: torch.tensor(np.asarray(d[k][:T],np.float32)) if k in d.files else torch.zeros(T,n)
    with torch.no_grad():
        o=bm(global_orient=g("global_orient",3),body_pose=g("body_pose",63),transl=g("transl",3),betas=torch.zeros(T,10),
             left_hand_pose=g("left_hand_pose",45),right_hand_pose=g("right_hand_pose",45),
             jaw_pose=torch.zeros(T,3),leye_pose=torch.zeros(T,3),reye_pose=torch.zeros(T,3),expression=torch.zeros(T,10))
    J=o.joints[:,:22,:].numpy();v=np.linalg.norm(np.diff(J,axis=0),axis=-1);a=np.linalg.norm(np.diff(J,2,axis=0),axis=-1)
    tr=np.asarray(d["transl"])
    return dict(mv=round(float(v.mean()),4),mx=round(float(v.max()),3),jk=round(float(a.mean()),4),
                drift=round(float(np.ptp(tr[:,[0,2]],0).max()),3),y0=round(float(tr[0,1]),3))
out=os.environ.get("OUT","outputs/evaluation/babel_seq/prism_gen_llm")
for f in sorted(glob.glob(out+"/*.npz"))[:16]:
    sid=os.path.basename(f)[:-4]
    g="data/babel_272_stream/gt_smpl/%s/gt.npz"%sid
    b="outputs/evaluation/babel_seq/prism_gen/%s.npz"%sid
    line=f"{sid:10s} FIX {st(f)}"
    if os.path.exists(b): line+=f"  BROKEN {st(b)}"
    if os.path.exists(g): line+=f"  GT {st(g)}"
    print(line,flush=True)
PY
echo "[fix-val] DONE"
