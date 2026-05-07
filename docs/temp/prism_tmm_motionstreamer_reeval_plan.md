# PRISM TMM — MotionStreamer-Evaluator Re-evaluation Plan

**Status (2026-05-08)**: MotionCLIP evaluator ported into hftrainer with numerical
parity vs the original versatilemotion implementation
(`text_emb` diff = 1.9e-6, `motion_emb` diff = 0). Both evaluators pass GT-only
sanity (FID ≈ 0) on HumanML3D and (for MotionCLIP) on MotionHub. Baseline-side
inference + 272-dim conversion for non-GT comparisons still TODO.

---

## Why this re-evaluation

The PRISM TMM submission currently reports T2M numbers under our own
**SMPL-22 TMR** (re-trained on SMPL-22 pose features). Two reviewer-adjacent
concerns motivate redoing the evaluation:

1. **MoMask reproducibility gap.** Under the current SMPL-22 TMR, MoMask's
   R-Precision T1 is ${\approx}0.249$, far below the ${\approx}0.521$
   reported in the original paper. This is almost certainly a
   protocol/representation mismatch on our side, not a defect of MoMask. Tab. 1
   currently leaves the MoMask row blank with an explicit `[TODO]` paragraph in
   `sec:t2m`.

2. **External cross-check.** A second, widely-used evaluator —
   MotionStreamer's TMR trained on the 272-dim HumanML3D representation
   ([repo](https://github.com/zju3dv/MotionStreamer),
   [paper](https://arxiv.org/abs/2503.15451)) — should be used to verify
   that PRISM's reported T2M wins are not an artifact of our custom
   evaluator. Reviewer feedback explicitly asked for an
   "original HumanML3D TMR sanity check"; using MotionStreamer's TMR-272
   evaluator is one of the cleanest ways to do this.

---

## Assets already on disk (this commit)

| Asset | Location | Size | State |
|---|---|---|---|
| MoMask source code | `ref_repo/Momask/momask-codes/` | ~5MB | git clone @ 6e29...HEAD |
| MoMask paper PDF | `ref_repo/Momask/paper/MoMask_CVPR2024.pdf` | 6MB | arxiv 2312.00063 |
| MoMask t2m weights | `ref_repo/Momask/weights/t2m/{rvq_*, t2m_*, tres_*, length_estimator}/` | ~190MB | gdown ok |
| MoMask kit weights | `ref_repo/Momask/weights/kit/...` | ~280MB | gdown ok |
| MotionStreamer source | `ref_repo/MotionStreamer/MotionStreamer/` | ~7MB | git clone HEAD |
| MotionStreamer Causal TAE / TAE-t2m-babel | `MotionStreamer_HF/{Causal_TAE,Causal_TAE_t2m_babel}/net_last.pth` | ~250MB | HF |
| **MotionStreamer Evaluator-272** | `MotionStreamer_HF/Evaluator_272/epoch=99.ckpt` | ~50MB | HF, **the one we need** |
| MotionStreamer t2m model | `MotionStreamer_HF/Experiments/t2m_model/latest.pth` | ~1.5GB | HF |
| 272-dim HumanML3D dataset | `MotionStreamer/humanml3d_272/{motion_data,texts,split,mean_std}/` | ~6GB | HF dataset, unzipped (motion_data: 26846 files, texts: 29232 files) |
| 272-dim conversion utilities | `ref_repo/MotionStreamer/272-dim-Motion-Representation/` | ~5MB | git clone HEAD |
| **Standalone eval script** | `ref_repo/MotionStreamer/MotionStreamer/eval_with_motionstreamer_evaluator.py` | — | new in this commit |

### MotionCLIP (ours, SMPL-22, 135-dim) — port + parity check

The "our evaluator" referenced in the TMM paper is **MotionCLIP**
(originally trained in `versatilemotion/` on the MotionHub HQ caption split).
For TMM submission we reproduced training and inference inside
`hftrainer/`:

* `hftrainer/models/motion/motion_clip/` — model code (CLIP ViT-B/32-aligned
  text + motion encoders, 512-dim shared embedding, contrastive CLIP loss).
  Stripped of `mmotion`/`mmengine` deps; only `transformers + torch`.
* `MotionCLIPBundle` (HF `ModelBundle` pattern) wraps tokenizer + 135-dim
  SMPLPoseProcessor + the contrastive model.
* `MotionCLIPTrainer` / `MotionCLIPPipeline` — Accelerate-native train/infer.
* `tools/convert_motionclip_checkpoint.py` — converts the original mmengine
  `.pth` (`work_dirs/motionclip_base_1p_aug_hq/best_r_precision_top_3_epoch_840.pth`)
  to `checkpoints/motion_clip/motionclip_base_1p_aug_hq/{motionclip_model.safetensors, bundle_config.json}`.
* `tools/test_motionclip_parity.py` — bit-level parity check between the
  ported model and the original versatilemotion model on identical inputs.
  Loaded from the same converted checkpoint:
    * `motion_norm` diff = 0
    * `motion_emb` diff = 0 (exact)
    * `text_emb` diff = 1.9e-6 (FP32 noise from `padding=True` vs `'max_length'`)

### Standalone evaluators

Both evaluators expose the same protocol — chunk size 32, 20-repeat
shuffle, FID over the full set, R-P / MM-Dist averaged over chunks:

* `ref_repo/MotionStreamer/eval_with_motionstreamer_evaluator.py`
  for HumanML3D-272 motions.
* `tools/eval_with_motionclip_evaluator.py`
  for SMPL-22 (135-dim) motions; supports `--gt_only`,
  `--pred_dir`, `--pred_npz`, and HumanML3D *or* MotionHub anno files.

### GT-only sanity table (real-vs-real)

```bash
# MotionStreamer-272 evaluator
python3 ref_repo/MotionStreamer/eval_with_motionstreamer_evaluator.py \
    --evaluator_ckpt ref_repo/MotionStreamer/MotionStreamer/MotionStreamer_HF/Evaluator_272/epoch=99.ckpt \
    --data_root ref_repo/MotionStreamer/MotionStreamer/humanml3d_272 \
    --gt_only --n_repeats 20 \
    --out_json work_dirs/ms_eval/gt_sanity20.json

# MotionCLIP-135 evaluator (HumanML3D)
python3 tools/eval_with_motionclip_evaluator.py \
    --evaluator_ckpt checkpoints/motion_clip/motionclip_base_1p_aug_hq \
    --anno_file data/annotation/test_hml3d.json \
    --data_dir data/motionhub --gt_only --n_repeats 20 \
    --out_json work_dirs/mc_eval/full_gt_h3d.json

# MotionCLIP-135 evaluator (MotionHub)
python3 tools/eval_with_motionclip_evaluator.py \
    --evaluator_ckpt checkpoints/motion_clip/motionclip_base_1p_aug_hq \
    --anno_file data/annotation/test_motionhub_t2m.json \
    --data_dir data/motionhub --gt_only --n_repeats 20 \
    --out_json work_dirs/mc_eval/full_gt_motionhub_t2m.json
```

Results (real-vs-real, FID ≈ 0 verifies pipeline consistency):

| Evaluator | Dataset | Rep | n pairs | FID | R-P T1/T2/T3 | MM-Dist | Diversity |
|---|---|---|---|---|---|---|---|
| MotionStreamer-272 | HumanML3D test | 272-dim | 7392 | ${-3.1{\times}10^{-8}}$ | 0.706 / 0.857 / 0.911 | 15.01 | 27.34 |
| MotionCLIP-135 (ours) | HumanML3D test | SMPL-22 | 4269 | ${-2.1{\times}10^{-7}}$ | **0.918 / 0.962 / 0.971** | 37.57 | 45.43 |
| MotionCLIP-135 (ours) | MotionHub-T2M test | SMPL-22 | 1513 | ${-1.9{\times}10^{-7}}$ | **0.958 / 0.996 / 0.999** | 38.69 | 46.09 |

Reading the table:

* `FID ≈ 0` on every row confirms each evaluator's internal pipeline is
  self-consistent — necessary for any comparison vs predicted motions to be
  meaningful.
* The two evaluators sit on **incomparable absolute scales**: MotionStreamer's
  TMR latent is the VAE μ (low magnitude, ‖μ‖₂ ≈ 4–6), MotionCLIP's "embedding"
  is the un-normalized projection of a contrastive model (‖z‖₂ ≈ 30–50).
  Therefore MM-Dist and Diversity values are **not** comparable across
  evaluators; **only relative rankings of methods under a single, fixed
  evaluator are meaningful**.
* On HumanML3D, MotionCLIP gives a higher real-vs-real R-Precision ceiling
  (0.918) than MotionStreamer (0.706). This suggests the SMPL-22 body
  representation + contrastive CLIP-style training yields a tighter
  text-motion alignment than the 272-dim TMR-VAE on the same captions.
* On MotionHub the MotionCLIP ceiling is even higher (0.958 T1) — expected
  because the model was trained on MotionHub's HQ training split.
* The MotionStreamer evaluator can only see the HumanML3D-272 representation,
  so a direct evaluator-vs-evaluator comparison on MotionHub requires
  re-encoding MotionHub motions into 272-dim (Stage 4 below).

---

## Remaining TODOs (must run on lzy_debug_machine_1/2)

### Stage 1 — generate per-method 272-dim motions on HumanML3D test set

For every method we want in Tab. 1, generate one motion per test caption and
save as `<method_dir>/<test_id>.npy` with shape `(T, 272)` (already in the
**raw, un-standardized** units; the eval script handles standardization).

The 7392 (id, caption) pairs are produced by
`eval_with_motionstreamer_evaluator.py::load_test_pairs()` — call it once and
dump the prompt list to drive every baseline.

#### 1a) MoMask (highest priority — fixes the blank Tab. 1 row)

* Use `ref_repo/Momask/momask-codes/gen_t2m.py` with the released checkpoints
  in `ref_repo/Momask/weights/t2m/`.
* MoMask outputs HumanML3D-263 features. Convert HumanML3D-263 → SMPL-22
  joint positions via the standard `recover_from_ric` helper (in MoMask repo
  `common/skeleton.py` or our own copy).
* Convert joint positions + recovered rotations → 272-dim using
  `ref_repo/MotionStreamer/272-dim-Motion-Representation/representation_272.py`.
  The conversion expects:
  - `smpl_85_face_z_transform_joints/<name>.npy` — joint positions, shape (T, 22, 3)
  - `smpl_85_face_z_transform/<name>.npy` — SMPL params (axis-angle 22*3 + ...)
* Save each result as `<work_dir>/momask_pred_272/<test_id>.npy`.

#### 1b) PRISM (cross-check our own model)

* Load PRISM checkpoint via `hftrainer` config + run `tools/infer.py`-style
  inference on each test caption.
* PRISM already outputs SMPL params + FK-derived joint positions; reuse the
  same `representation_272.py` pipeline as for MoMask.
* Save each result as `<work_dir>/prism_pred_272/<test_id>.npy`.

#### 1c) Other baselines (optional, per Tab. 1)

* MotionStreamer itself: use `Experiments/t2m_model/latest.pth` directly via
  `MotionStreamer/eval_t2m.py` and save its 272-dim outputs.
* HY-Motion, Go-To-Zero, MDM, MLD, T2M-GPT, MotionGPT(3), ViMoGen: each has a
  different output format. Skip if time-constrained — the comparison is already
  meaningful with PRISM + MoMask + MotionStreamer.

### Stage 2 — run the standalone evaluator on each baseline

```bash
# For each method:
python3 eval_with_motionstreamer_evaluator.py \
    --evaluator_ckpt MotionStreamer_HF/Evaluator_272/epoch=99.ckpt \
    --data_root humanml3d_272 \
    --pred_dir <work_dir>/<method>_pred_272 \
    --n_repeats 20 \
    --out_json work_dirs/ms_eval/<method>.json
```

This produces a JSON with FID, R-P (T1/T2/T3), MM-Dist, Diversity, and their
standard deviations across 20 random shuffles.

### Stage 3 — fold the new numbers back into the paper

* Update `papers/PRISM_TMM2026/depds/tab_t2m_motionhub_h3d.tex`:
  - Fill in the MoMask row with the MotionStreamer-evaluator numbers.
  - Add a footnote (or a `\paragraph*` in `sec:t2m`) clarifying that the
    new MoMask numbers were obtained under MotionStreamer's TMR-272
    evaluator using the official `ericguo5513/momask-codes` checkpoint
    without retraining.
  - Keep the MotionCLIP numbers (now reproducibly trained inside
    `hftrainer/`) as the primary comparison; use MotionStreamer's TMR-272 as
    a "sanity column" for HumanML3D.
* Remove the `[TODO~--~MoMask re-evaluation]` paragraph in `sec:t2m`
  once the new numbers are in.

### Stage 4 — MotionStreamer-eval on MotionHub (extra cross-check)

To enable a same-evaluator MotionHub comparison, convert MotionHub
SMPL-22 motions into HumanML3D-272 features:

```bash
python3 ref_repo/MotionStreamer/272-dim-Motion-Representation/representation_272.py \
    --src data/motionhub/<subset>/smplx_55/<id>.npz \
    --joints_dir <work>/smpl_85_face_z_transform_joints \
    --params_dir <work>/smpl_85_face_z_transform \
    --out_dir <work>/motionhub_272/<subset>/<id>.npy
```

then run `eval_with_motionstreamer_evaluator.py --pred_dir <work>/motionhub_272`
together with MotionHub's caption `.txt` files (one caption per line, same
`#`-delimited format as HumanML3D). Both evaluators on MotionHub +
HumanML3D would give the cleanest 2-evaluator x 2-dataset comparison
matrix.

This stage is **not** required for the camera-ready (MotionCLIP is the
declared "our evaluator" of the paper), but is the cleanest way to answer
"do the two evaluators agree on the *ranking* of methods?".

---

## Operational notes for resuming this work

* Both `lzy_debug_machine_1` and `lzy_debug_machine_2` have 8x V100-32GB and
  share the same `/apdcephfs_cq11/share_1467498` filesystem as the local
  cluster, so all artifacts above are visible from both.
* `setsid nohup ... &disown` is required to keep long-running jobs alive when
  launched via `tools/taiji_exec.py` (a plain `nohup ... &` gets killed by
  SIGHUP when the launcher's PTY closes).
* Use `export HF_ENDPOINT=https://hf-mirror.com` before any `huggingface-cli`
  / `from_pretrained` call — direct Hugging Face is firewalled on the
  cluster.
* Disk usage: 272-dim HumanML3D + MotionStreamer HF + MoMask weights together
  occupy ~7GB of the shared filesystem. Avoid duplicating in user home.

---

## Quick reference

* Paper: `papers/PRISM_TMM2026/`, Overleaf
  `https://www.overleaf.com/project/69fb08d1092d1ecf5d89f341` (master).
* MoMask blank-row commit: `fe0091b` ("TMM: blank MoMask row pending evaluator re-run").
* MoMask repo HEAD: see `ref_repo/Momask/momask-codes/`.
* MotionStreamer repo HEAD: see `ref_repo/MotionStreamer/MotionStreamer/`.
* Eval script: `ref_repo/MotionStreamer/MotionStreamer/eval_with_motionstreamer_evaluator.py`.
* Standalone eval: see "Sanity check" above; replace `--gt_only` with
  `--pred_dir <work_dir>/<method>_pred_272` for real comparisons.
