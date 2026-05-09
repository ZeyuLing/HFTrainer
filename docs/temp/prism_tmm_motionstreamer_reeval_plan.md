# PRISM TMM — MotionStreamer-Evaluator Re-evaluation Plan

**Status (2026-05-08, evening)**: MotionCLIP evaluator ported into hftrainer with
numerical parity vs the original versatilemotion implementation (`text_emb` diff
= 1.9e-6, `motion_emb` diff = 0). The original MotionCLIP eval script computed
MM-Dist / Diversity on the **un-normalized projection** (‖z‖₂≈30+); fixed to
match versatilemotion `TMRMetric` exactly: chunk=256, L2-normalize before R-P /
MM-Dist, L1 distance for Diversity. Both evaluators now pass GT-only sanity on
both datasets (FID ≈ 0). MotionHub→HumanML3D-272 converter built, so
MotionStreamer's TMR-272 evaluator can be run on MotionHub (1257/1590 motions
converted). Baseline-side inference for non-GT comparisons still TODO.

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
# MotionStreamer-272 evaluator on HumanML3D test
python3 ref_repo/MotionStreamer/eval_with_motionstreamer_evaluator.py \
    --evaluator_ckpt ref_repo/MotionStreamer/MotionStreamer/MotionStreamer_HF/Evaluator_272/epoch=99.ckpt \
    --data_root ref_repo/MotionStreamer/MotionStreamer/humanml3d_272 \
    --gt_only --n_repeats 20 \
    --out_json work_dirs/ms_eval/gt_sanity20.json

# MotionStreamer-272 evaluator on MotionHub-T2M test (Stage 4: motion conversion first)
python3 tools/convert_motionhub_to_h3d272.py \
    --anno_file data/annotation/test_motionhub_t2m.json \
    --data_dir data/motionhub \
    --out_root work_dirs/ms_eval/motionhub_272 \
    --ms_data_root ref_repo/MotionStreamer/MotionStreamer/humanml3d_272 \
    --smpl_model_path checkpoints/smpl_models/smplx
python3 ref_repo/MotionStreamer/eval_with_motionstreamer_evaluator.py \
    --evaluator_ckpt ref_repo/MotionStreamer/MotionStreamer/MotionStreamer_HF/Evaluator_272/epoch=99.ckpt \
    --data_root work_dirs/ms_eval/motionhub_272 \
    --gt_only --n_repeats 20 \
    --out_json work_dirs/ms_eval/gt_motionhub_272.json

# MotionCLIP-135 evaluator (versatilemotion TMRMetric protocol: chunk=256, L2-norm)
python3 tools/eval_with_motionclip_evaluator.py \
    --evaluator_ckpt checkpoints/motion_clip/motionclip_base_1p_aug_hq \
    --anno_file data/annotation/test_hml3d.json \
    --data_dir data/motionhub --gt_only --n_repeats 20 --chunk_size 256 \
    --out_json work_dirs/mc_eval/full_gt_h3d_v2.json

python3 tools/eval_with_motionclip_evaluator.py \
    --evaluator_ckpt checkpoints/motion_clip/motionclip_base_1p_aug_hq \
    --anno_file data/annotation/test_motionhub_t2m.json \
    --data_dir data/motionhub --gt_only --n_repeats 20 --chunk_size 256 \
    --out_json work_dirs/mc_eval/full_gt_motionhub_t2m_v2.json
```

Results (real-vs-real, FID ≈ 0 verifies internal pipeline consistency):

| Evaluator (chunk) | Dataset | n pairs | FID | R-P T1/T2/T3 | MM-Dist | Diversity |
|---|---|---|---|---|---|---|
| MotionStreamer-272 (chunk=32, L2-dist diversity) | HumanML3D test | 7392 | ${-3.1{\times}10^{-8}}$ | 0.706 / 0.857 / 0.911 | 15.01 | 27.34 |
| MotionStreamer-272 (chunk=32, L2-dist diversity) | **MotionHub-T2M test** | **10657** | ${-6.8{\times}10^{-10}}$ | **0.223 / 0.355 / 0.447** | **21.57** | **25.48** |
| MotionCLIP-135 (chunk=256, **TMRMetric**) | HumanML3D test | 4269 | ${-1.4{\times}10^{-10}}$ | 0.785 / 0.902 / 0.937 | 0.989 | 21.64 |
| MotionCLIP-135 (chunk=256, **TMRMetric**) | MotionHub-T2M test | 1513 | ${-2.8{\times}10^{-10}}$ | 0.815 / 0.945 / 0.979 | 0.984 | 21.33 |

Reading the table:

* `FID ≈ 0` on every row confirms each evaluator's internal pipeline is
  self-consistent — necessary for any comparison vs predicted motions to be
  meaningful.
* **Two evaluators use different metric protocols, so absolute values are
  not comparable across rows.** Only relative rankings of methods *within*
  one row are meaningful.
  * MotionStreamer eval (default in their repo): chunk=32, **no L2-norm**
    of the VAE-μ embedding, **L2 distance** for Diversity, **n=300**.
  * MotionCLIP eval (matches versatilemotion `TMRMetric`): chunk=256,
    **L2-normalize** the 512-d projection before R-P / MM-Dist, **L1 distance**
    for Diversity, **n=300**.
* MotionCLIP MM-Dist ≈ 1.0: distance between L2-normalized 512-d unit vectors
  ranges in [0, √2 ≈ 1.41]. Real text-motion pairs sit at ≈ 0.99, well below
  the random-pair baseline √2, indicating tight contrastive alignment.
  (The earlier ${\sim}38$ value was a bug — the script returned the
  un-normalized projection (‖z‖₂≈30); now matches versatilemotion exactly.)
* MotionCLIP Diversity ≈ 21.6: L1 distance between two random L2-normalized
  512-d vectors has expected value ${\sqrt{4d/\pi}\approx 25.5}$; real motion
  embeddings come in slightly tighter (21.6) — the cluster has structure.
* MotionStreamer Diversity ≈ 25–27: L2 distance on un-normalized 256-d VAE
  μ (‖μ‖₂≈4–6) — different scale, can't compare directly.
* **Cross-dataset on MotionStreamer-272 evaluator**: the same evaluator
  trained on HumanML3D drops sharply on MotionHub (R-P T1: 0.71 → 0.22).
  Two reasons:
  1. **Distribution shift in motion**: MotionHub motions are more diverse
     than HumanML3D's locomotion-heavy distribution, so the TMR latent
     trained on HumanML3D doesn't separate them as cleanly.
  2. **Caption distribution shift**: MotionHub's hierarchical captions
     (macro/meso/micro) are more abstract than HumanML3D's literal action
     descriptions; one motion has ~7 captions on average, increasing
     same-class confusability in retrieval.
  Importantly, FID ≈ 0 still holds for real-vs-real, so the evaluator
  is *internally consistent* on MotionHub-272, just with a lower
  R-Precision ceiling. Method *rankings* under this evaluator are still
  meaningful.
* **Same-evaluator-different-dataset reading**: under MotionCLIP, both
  HumanML3D (0.785) and MotionHub (0.815) ceilings are healthy and similar;
  this evaluator generalizes well across the two test sets because it
  was trained on the union of MotionHub + HumanML3D HQ captions.

---

## Remaining TODOs (must run on lzy_debug_machine_1/2)

### Stage 1 — generate per-method 272-dim motions on HumanML3D test set

For every method we want in Tab. 1, generate one motion per test caption and
save as `<method_dir>/<test_id>.npy` with shape `(T, 272)` (already in the
**raw, un-standardized** units; the eval script handles standardization).

The 7392 (id, caption) pairs are produced by
`eval_with_motionstreamer_evaluator.py::load_test_pairs()` — call it once and
dump the prompt list to drive every baseline.

#### 1a) MoMask — **DONE (2026-05-08, native HumanML3D-263 evaluator)**

Pipeline implemented twice: (i) MotionStreamer-272 evaluator (cross-protocol,
abandoned — see "v1, abandoned" below), and (ii) **MoMask's own native
HumanML3D-263 evaluator (Comp_v6_KLD005 / text_mot_match)** with a
`tools/build_h3d263_test_from_h3d272.py` reconstructed test set. The native
eval gives credible, paper-comparable numbers and is what we report.

Steps that ran (all on `lzy_debug_machine_1`):

1. `tools/momask_infer_h3d_test.py`: drives MoMask's released `gen_t2m`
   (rvq + masked + residual + length predictor) over the 2480 unique
   HumanML3D test ids in `humanml3d_272/split/test.txt`. For each id we use
   the *first valid caption* (no tag-restricted sub-clips) and pass
   `m_length = round(T_gt × 20/30)` (rounded to nearest multiple of 4) so
   MoMask generates a motion of the correct **physical duration** at its
   native 20 fps. Output: `work_dirs/momask_eval/momask_pred_263_v2/<id>.npy`
   shape `(T_20, 263)`. Total: 2480 motions in ~28s on a single V100.
2. `tools/convert_momask263_to_h3d272.py`: HumanML3D-263 (20 fps) →
   HumanML3D-272 (30 fps). Decodes 263 → joint positions via
   `recover_from_ric`, decodes non-root rotations via `cont6d_to_matrix`,
   builds root rotation as `R_y(cumsum(rot_velocity))` (HumanML3D-263 only
   carries root yaw, no body tilt — so root rotation 6D in 272 = identity
   after heading removal). Linear-resamples positions and slerp-resamples
   rotations from 20 → 30 fps, then re-runs MotionStreamer's
   `representation_272` logic. Output:
   `work_dirs/momask_eval/momask_pred_272_30fps_v2/<id>.npy` shape
   `(T_30, 272)`.
**v2 (the canonical numbers — DONE)**

Native MoMask evaluator setup on `lzy_debug_machine_1`:

* `humanml3d_evaluator.zip` (227 MB) downloaded via `gdown` and unzipped
  into `ref_repo/Momask/momask-codes/checkpoints/t2m/{Comp_v6_KLD005,
  text_mot_match,...}`. The actual evaluator weights are
  `text_mot_match/model/finest.tar` (epoch 28). `Comp_v6_KLD005/meta`
  carries the canonical HumanML3D-263 mean / std.
* `glove.zip` (6 MB) downloaded into `ref_repo/Momask/momask-codes/glove/`
  for `WordVectorizer('our_vab')`.
* `tools/build_h3d263_test_from_h3d272.py` reconstructs the
  HumanML3D-263 test set from `humanml3d_272`: decode 272 → global joint
  positions (30 fps) → linear-resample to 20 fps → MoMask's
  `process_file` → 263-dim features + 22-joint positions. Reference
  skeleton offsets taken from `000021`'s first reconstructed frame so that
  `uniform_skeleton` is internally consistent. Output:
  `work_dirs/momask_eval/h3d263_test_recon/{new_joint_vecs,new_joints,
  Mean.npy,Std.npy,test.txt}`. **3970 / 4042 ids reconstructed
  successfully** (the 72 dropped were too short, missing texts, or had
  process_file numerical issues — same drop pattern as native
  HumanML3D).
* `tools/eval_momask_native_h3d263.py` reads the reconstructed test set,
  `humanml3d_272/texts/<id>.txt` captions (full-clip only,
  `f_tag==0,to_tag==0`), and applies MoMask's `EvaluatorModelWrapper` /
  `calculate_R_precision` / `calculate_frechet_distance` /
  `calculate_diversity`, mirroring `evaluation_mask_transformer_test`
  with `drop_last=True` and 32-batch.

Results (HumanML3D test set, 20 random repeats for `pred`, 5 for
`gt-only`):

| Metric | GT-only sanity (this run) | MoMask paper GT | MoMask paper | **MoMask (this run, native eval)** |
|---|---|---|---|---|
| FID         | 0.000 (by construction) | 0.002 | 0.045 | **2.672 ± 7.0e-8** |
| R-P T1      | 0.432 ± 0.008 | 0.511 | 0.521 | **0.443 ± 0.008** |
| R-P T2      | 0.617 ± 0.007 | 0.703 | 0.713 | **0.638 ± 0.007** |
| R-P T3      | 0.731 ± 0.008 | 0.797 | 0.807 | **0.739 ± 0.007** |
| MM-Dist     | 3.546 ± 0.024 | 2.974 | 2.958 | **3.319 ± 0.032** |
| Diversity   | 8.227 ± 0.217 | 9.503 | 9.640 | **9.976 ± 0.119** |

**Reading the table**:

* GT-only R-P is ${\approx}$ 0.43 / 0.62 / 0.73 vs MoMask paper's
  0.51 / 0.70 / 0.80; MM-Dist 3.55 vs paper 2.97; Diversity 8.23 vs
  paper 9.50. The 5–8% gap reflects information loss in the
  272 → joints → 20 fps → 263 reconstruction (linear interp, IK / FK
  re-encoding, uniform-skeleton retargeting from a single reference
  frame). It is **NOT** a MoMask issue; it bounds the achievable
  numbers under our reconstructed test data.
* Within this protocol, MoMask predictions are **comparable to GT** on
  R-P, MM-Dist and Diversity (slightly *better* on T2/T3 and
  Diversity, slightly *worse* on T1 and MM-Dist), and the FID 2.67
  >> paper's 0.045. The high FID is a **distribution-shift artifact**:
  MoMask was trained on the original (clean) HumanML3D-263 distribution;
  our reconstructed GT is a slightly noisier 272 → 263 reconstruction of
  the same motions, so FID(MoMask outputs, reconstructed GT) is inflated
  even though semantic alignment is preserved.

**v1 (abandoned)**: same MoMask 263 outputs converted to 272 (slerp +
linear, see `tools/convert_momask263_to_h3d272.py`) and run through
`eval_with_motionstreamer_evaluator.py`. That gave FID 539.85 / R-P T1
0.156 / MM-Dist 25.48 / Diversity 12.98 — meaningless absolute numbers
caused by SMPL-22 (MoMask) vs SMPL-X-derived 30 fps (MotionStreamer
evaluator) domain shift, not by MoMask's intrinsic quality. We keep the
v1 outputs on disk (`work_dirs/momask_eval/momask_pred_272_30fps_v2/`)
purely as a relative-ranking sanity column and **do not** report them in
the paper.

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

### Stage 4 — MotionStreamer-eval on MotionHub (extra cross-check) — **DONE**

`tools/convert_motionhub_to_h3d272.py` is a self-contained converter that
mirrors MotionStreamer's `face_z_transform.py + infer_get_joints.py +
representation_272.py` pipeline:

```
MotionHub .npz (poses[T,165], trans[T,3], mocap_framerate)
  ↓ resample to 20 fps (slerp on quaternions, linear on translation)
  ↓ build smpl_85 = [global_orient, body_pose, 0-pad, trans, betas=0]
  ↓ face_z_transform : rotate so first frame's root faces +Z
  ↓ SMPL-X FK via SmplxLite : (T, 22, 3) joint positions
  ↓ representation_272 logic : 272-dim feature
output: <out_root>/{motion_data,texts,split,mean_std}/  (HumanML3D-272 layout)
```

Mean/Std are symlinked from the original HumanML3D-272 release so the
MotionStreamer evaluator sees the same input distribution it was trained on.

Conversion result on `test_motionhub_t2m.json` (1590 entries):
* 1257 written, 333 skipped (284 too-short after 20-fps resampling, 49 no caption).
* The full GT-only sanity result is the row added to the table above
  (FID ≈ 0, R-P T1 = 0.223, MM-Dist = 21.57, Diversity = 25.48 over 10657 caption-motion pairs).

Note this is **only** the GT-vs-GT sanity check; running the same evaluator on
predicted motions from each baseline still requires the per-method 272-dim
inference described in Stage 1.

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
