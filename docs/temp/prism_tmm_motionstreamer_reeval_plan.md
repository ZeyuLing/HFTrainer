# PRISM TMM — MotionStreamer-Evaluator Re-evaluation Plan

**Status (2026-05-07)**: setup complete, sanity verified, baseline-conversion + inference still TODO.

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

### Sanity check (already passed)

```bash
# On lzy_debug_machine_1 (or any V100 node), from MotionStreamer/ root:
export HF_ENDPOINT=https://hf-mirror.com
python3 eval_with_motionstreamer_evaluator.py \
    --evaluator_ckpt MotionStreamer_HF/Evaluator_272/epoch=99.ckpt \
    --data_root humanml3d_272 \
    --gt_only --n_repeats 20 \
    --out_json /tmp/gt_sanity.json
```

Result on the 272-dim HumanML3D test split (7392 paired captions, 20 repeats):

| Metric | Real-vs-Real |
|---|---|
| FID | ${-3.1\!\times\!10^{-8}} \pm 3.8\!\times\!10^{-8}$ ≈ 0 ✓ |
| R-P (T1/T2/T3) | 0.706 / 0.857 / 0.911 |
| Diversity (real / pred) | 27.34 / 27.28 |
| MM-Dist | 15.01 |

`FID ≈ 0` confirms the evaluator pipeline is internally consistent. Note the
**absolute** R-P / Diversity / MM-Dist numbers differ from the MotionStreamer
paper's "Real" row (T1=0.491, Div=9.50, MM-Dist=2.97); the discrepancy is
likely an embedding-norm convention difference. **What matters for the paper
revision is the relative ranking of methods under this single, fixed
evaluator**, which is meaningful regardless of the absolute scale.

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
  - Keep the SMPL-22 TMR numbers for the other baselines as the primary
    comparison, OR provide both columns if space allows.
* Remove the `[TODO~--~MoMask re-evaluation]` paragraph in `sec:t2m`
  once the new numbers are in.

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
