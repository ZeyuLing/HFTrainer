# HyMotion M2M — Motion Task Stack Documentation

> Sub-document of `../../../CLAUDE.md`. See root for framework overview.

## 🚨 CRITICAL CONSTRAINT — READ BEFORE MODIFYING EVAL CODE

**ALL motion fed into the M2M model during inference MUST be in a frame the model
has seen during training. Failing this produces foot skating, jitter, jump artifacts
and broken transitions — the single most common bug in this stack.**

### Two distribution regimes the model was trained on

1. **Standard single-motion tasks** (E1/E2/E3/E5/E6/E10/E13) — `sample['motion']`
   from the eval datalist is **yaw-augmented around ±180°** during training
   (see `hftrainer/datasets/motion/motionhub/transforms/load_smplx.py`
   `transl_aug_yaw_deg=180.0`). Any arbitrary starting yaw + XZ offset is
   within the training distribution. **No runtime canonicalization is needed.**

2. **Transition/loop tasks** (E14/E15/E16/E8-D/E9) — the model input is a
   *constructed* segment (e.g. `[A_tail | pad | B_head]`) where the two
   segments may sit at wildly different positions/yaws. This is **strictly
   out-of-distribution** unless you canonicalize:
   - Place the anchor frame (the last cond frame before generation) at origin
   - Align its yaw to +Z
   - Preserve anchor Y (pelvis height) — do NOT zero it
   - Apply the same rigid yaw+XZ transform to **every** frame and to body
     rotations if `rotation_space='global'` (see transition_utils.py v2 bug)
   After inference, `decanonicalize_segment` maps back to world coords.

   The canonical tools are in `hftrainer/pipelines/motion/transition_utils.py`:
   - `canonicalize_segment(motion, anchor_frame, rotation_space)`
   - `decanonicalize_segment(motion_canon, R_canon, offset_canon, rotation_space)`
   - `place_b_after_a(motion_a, motion_b, forward_step, yaw_offset_deg)`

### Mandatory checklist when modifying a transition-style eval branch

Before committing any change that alters how E14/E15/E16/E8-D/E9 build the
`motion_raw` fed to the pipeline:

| # | Check | How |
|---|---|---|
| 1 | Model input passes through `canonicalize_segment`? | grep for the call in the branch; the last `motion_135 = canon_segment_t.numpy()` line |
| 2 | Anchor frame is the frame immediately before generation? | `anchor_frame = N_cond_a - 1` for E14; `0` for E15/E16 (P or target anchor) |
| 3 | Full user-facing motion preserved for viz? | Store in `_transition_canon_info.motion_a_full` / `motion_b_world_full`; do NOT truncate just for N_cond ablation |
| 4 | N_cond frames are within training distribution? | `sample_tier2_inbetween` uses ≤5 per side; >15 is mild OOD, >30 is collapse risk |
| 5 | Smoke test: run 2 samples, verify `jitter_pos < 1000` and `foot_skating < 0.35` for uncond_local? | If either blows up, a canonicalization or truncation bug was introduced |
| 6 | `decanonicalize_segment` is called with the **same `rotation_space`** as `canonicalize_segment`? | Check pipeline output path (usually around line 2800 in eval_m2m_v2_all_tasks.py) |

### Failure modes seen in 2026-04-23 N_cond ablation

- **Teleport at cond→gen boundary (3x boundary_accel_jump)** — bug was in
  `apply_rigid_transform_to_motion`: when `rotation_space='global'`, body
  joints were left rotated in the original (pre-canon) yaw while pelvis
  had the new yaw. Fix: yaw-rotate ALL 22 joint matrices, not just pelvis
  (see `transition_utils.py` line 180-185).

- **Severe foot skating + 30% jitter increase on E15** — caused by
  truncating `motion_a` down to 5 frames before feeding to model. Training
  E15-style prepend always sees the full motion A as cond. Fix: truncate
  only the **model input** (`motion_a_model`), preserve full motion for
  decanonicalize + viz (`motion_b_world_full` in canon_info).

- **Near-origin Y pelvis (pushed ~1m into ground)** — earlier bug in
  `canonicalize_segment` that subtracted full anchor_pos including Y.
  Fix: subtract only XZ, preserve Y (v2 mean Y ≈ 1.09m).

### Quality-signal thresholds (50 samples, uncond_local baseline)

After an eval run, if any of these exceed their threshold, suspect a
canonicalization or truncation bug rather than model capacity:

| Task/Setting | jitter_pos (max ok) | foot_skating (max ok) | boundary_accel_jump (max ok) |
|---|---|---|---|
| E14 A/C (N_cond=5) | 1000 | 0.35 | 60 |
| E14 B (N_cond=15) | 800 | 0.35 | 70 |
| E15 A/B/C | 800 | 0.30 | 8 |
| E8-D (full GT) | 500 | 0.30 | 10 |

If jitter_pos > 1000 or foot_skating > 0.40 → open `docs/temp/m2m_canonical_ood_solution.md`
and audit the eval branch against checklist items 1-6 above.

### Post-Processing Inventory For M2M / KIMODO Eval

Treat post-processing as part of the evaluated method unless it is explicitly
visualization-only. Current code has three different classes:

| Scope | Entry Point | What It Changes | Notes |
|---|---|---|---|
| HyMotion M2M output | `scripts/eval/eval_m2m_v2_all_tasks.py` | Hard-pastes condition frames from GT after denormalization; optional boundary Gaussian blending, accel-spike median filter, Savitzky-Golay smoothing, E9 adaptive post-hoc replacement, long-window overlap blending | These change `motion_135` before metrics/import. They can hide binary-mask boundary discontinuities. |
| MoGenDIT repair | `scripts/repair/postprocess_hymotion_with_mogendit.py` | Converts HyM `motion_135` ↔ SMPL-H, runs MoGenDIT `denoise` / `ada_denoise` / `trans_regen` / `impute`, then recomputes positions + QC metrics | `trans_regen` regenerates root translation only; `impute` may hard-preserve obs frames after MoGenDIT to avoid normalization drift. |
| Official KIMODO postprocess | `ref_repo/KIMODO/kimodo/kimodo/postprocess.py` | Runs foot/contact cleanup and constraint cleanup through `motion_correction.correct_motion`, using contacts, root margin, and original constraint frames as targets | This is part of upstream KIMODO's default SOMA CLI path. `scripts/kimodo/run_kimodo_all_tasks.py` should keep it enabled by default; use `KIMODO_POST_PROCESSING=0` only for ablations. |
| KIMODO E14/E15 visualization | `scripts/kimodo/append_kimodo_context_soma77.py`, `scripts/kimodo/append_kimodo_e15_context_soma77.py` | Appends SOMA-77 prefix/suffix mesh data and writes `layout_json`; E14 can replace main condition frames with exact source-condition SOMA; E14 blends prefix/suffix seams over 30 frames; E15 blends the first suffix frames over up to 15 frames | This is display metadata for full-timeline rendering, but it can visually smooth KIMODO context seams. It must not be used as evidence that raw KIMODO generated boundaries are continuous. |
| score_m2m web stitching | `motion_annot_web/score_m2m/score_m2m_web.py` | For SMPL-frame methods, stitches E14/E15 source context around main output and ground-aligns stitched frames | Web-only timeline alignment so pair A/B have the same frame count. KIMODO mesh-sequence files are expected to be pre-stitched offline. |
| eval_dashboard source overlay | `motion_annot_web/eval_dashboard/app.py` | Rebuilds E14/E15 source motion overlays with the same placement / mesh type as eval output | Visualization-only; useful for debugging, but not a substitute for checking saved NPZ metrics. |

Debug rule: if a case looks smooth on E15 but not E14, first check whether
`append_kimodo_e15_context_soma77.py` suffix blending or web stitching is
hiding the seam. If the artifact occurs inside the generated red segment
(for example leg twist in E15 frame 40-50), suspect KIMODO inference /
SMPL→SOMA retarget / canonicalization rather than suffix post-processing.

KIMODO conditioning rule: match the official `FullBodyConstraintSet`
semantics unless running an explicit ablation. The official full-body
constraint observes joint positions, root planar trajectory, and heading,
but does **not** pin every condition-frame global rotation. Pinning
`global_joints_rots` and then hard-pasting the condition features can create
large one-frame wrist / knee snaps at E14/E15 condition boundaries: the
condition frame is exact, while the adjacent generated frame remains on the
model trajectory. `KIMODO_PIN_COND_ROT=1` is therefore ablation-only; the
default eval path should keep it disabled and validate saved NPZ boundary
deltas numerically before relying on the web visualization.

KIMODO full-body rotation continuity depends on MotionCorrection, not on the
denoiser mask alone. `FullBodyConstraintSet.update_constraints()` does not
feed `global_joints_rots` to the denoiser; it stores them so
`post_process_motion()` can correct constrained keyframes through the official
`motion_correction` IK pass. If `motion_correction` is missing or
`KIMODO_POST_PROCESSING=0`, boundary palms/knees can visibly jump even when
canonicalization and root height are correct.

SMPL↔KIMODO SOMA conversion reference: the eval bridge lives in
`scripts/kimodo/run_kimodo_all_tasks.py` (`smpl22_to_soma30_retarget`,
`soma30_to_soma77`, `soma77_to_smpl22`). It is documented in
`ref_repo/KIMODO/CLAUDE.md` §“SMPL ↔ SOMA 转换逻辑（我方评测桥接）”. The
E14/E15 visualization appenders reuse the same chain to write SOMA-77
`posed_joints` / `global_rot_mats` for source context frames.

---

## Overview

HyMotion M2M (Motion-to-Motion) is a **universal motion completion** model based on HunyuanMotion MMDiT.

**Goal**: Given any frames x any joints as condition (`src_mask (T, D)`, 0=known, 1=generate), complete all masked positions.
- All mask=0 -> identity
- All mask=1 -> unconditional generation (degenerates to T2M)
- Arbitrary mask -> arbitrary-granularity completion

Uses flow matching directly in motion space (no VAE latent), with VACE conditioning for mask-based editing.

**Architecture**: `HunyuanMotionMMDiT` (0.46B / 1.5B) — dual-stream + single-stream transformer blocks, narrowband mask attention, text conditioning via qwen3 (4096-dim) + clipl (768-dim).

---

## VACE Conditioning

Reference: [VACE: All-in-One Video Creation and Editing](https://arxiv.org/abs/2503.07598)

Model input = `[x_t, inactive, reactive, src_mask]`, total dim = 4 x motion_dim.

| Channel | mask=0 (keep) | mask=1 (generate/edit) |
|---------|---------------|------------------------|
| **inactive** | normalized motion | **0** |
| **reactive** | **0** | **depends on paradigm** |
| **mask** | 0 | 1 |

- inactive and reactive are **complementary**: each is 0 where the other has values.
- **Completion vs Editing**: the ONLY difference is whether reactive is 0 or has pre-edit values in mask=1 regions.

### Implementation

```python
# prepare_vace_input (universal for both paradigms):
inactive = src_motion * (1 - mask)  # preserved regions
reactive = src_motion * mask        # Completion: 0; Editing: pre-edit values
vace_context = cat([inactive, reactive, mask], dim=-1)  # (B, L, 3*D)

# Final model input:
x_input = cat([x_t, vace_context], dim=-1)  # (B, L, 4*D)
```

### ⚠️ Critical Constraints

1. **Completion**: `src_motion` MUST be zeroed in mask=1 regions before `prepare_vace_input`.
   Flow: `normalize(motion)` -> `motion * (1 - mask)` -> `prepare_vace_input(zeroed, mask)`.
   If not zeroed, reactive leaks target answers.

2. **Editing**: `src_motion` keeps pre-edit (LQ) values in mask=1 regions.
   Flow: `normalize(LQ_motion)` -> `prepare_vace_input(LQ_norm, mask)`.

---

## Supported Tasks

### Completion Paradigm (reactive=0)

| Task | mask pattern | Training mask strategy |
|------|-------------|----------------------|
| T2M / Unconditional | All 1 | M5: full_mask |
| Motion In-Between | middle=1, start/end=0 | M3: temporal_contiguous |
| Motion Prediction | tail=1 | M3: temporal_contiguous |
| Motion Prefix | head=1 | M3: temporal_contiguous |
| Joint Completion | specific joints=1 (all frames) | M4: joint_contiguous |
| Trajectory Completion | translation=1 (all frames) | M4: joint_contiguous |
| Sparse Keyframe Interpolation | non-keyframes=1 | M6: keyframe_sparse |
| Arbitrary Completion | any frame x joint | M1: random_cell / M2: random_block |
| Scattered Repair (Completion) | sparse (frame,joint) spots | M7: scattered_joint |
| Repair (Completion) | abnormal frames/joints=1 | Inference: checker/adaptive/manual |

### Editing Paradigm (reactive=pre-edit values)

| Task | reactive (mask=1) | Training data | Status |
|------|-------------------|---------------|--------|
| Repair (Editing) | degraded motion | corruptor-generated (15% online corruption) | ✅ Training |
| Motion Editing | original motion | general editing pairs (not ready) | 📋 Planned |

### Training Mask Strategies (7 types)

| Strategy | Description | Coverage | Weight |
|----------|-------------|----------|--------|
| **M1: Random cell** | Bernoulli(p) per cell, p~U[0.01,0.95] | B1,B4,B5,E2 + sparse repair generalization | 20% |
| **M2: Random block** | Random [t1,t2] x random joints | B2,B3,B6,C6 | 12% |
| **M3: Temporal contiguous** | Contiguous frame segments masked | A1-A5, F3 | 23% |
| **M4: Joint contiguous** | Random joints masked (all or partial frames) | C1-C5, D1-D3 | 15% |
| **M5: Full mask** | All masked | F1, F2 | 5% |
| **M6: Keyframe sparse** | K random keyframes kept | E1-E4, D4 | 15% |
| **M7: Scattered joint** | Scattered (frame,joint) spots + temporal dilation, no transl | Checker/adaptive repair patterns | 10% |

#### 2026-03 Coverage Patch (M1 range + M4 temporal + M7)

Three blind spots were identified via 50K-sample analysis and fixed:

1. **M1 density floor lowered**: `p ~ U[0.3, 0.95]` → `U[0.01, 0.95]`. M1 now reaches ratio ~1%, covering sparse random repair where checkers flag only a few scattered cells.
2. **M4 temporal_partial sub-mode** (30% prob): selected joints now sometimes mask only 10-80% of frames (via random frame subset), not always all frames. Covers "specific frames × specific joints" patterns from quality checkers.
3. **M7 scattered_joint**: new strategy. Samples N scattered (frame, joint) flag-points with 1-8 frame temporal dilation. Never masks translation (col 0). Grid ratio typically 1-20%. Directly simulates checker/adaptive mask distributions.

See `docs/figures/mask_patterns_m1_m7.png` for visual comparison of all 7 strategies.

#### 2026-04 Universal Rank-K Boolean Tensor Prior (v3 sampler)

`PrepareM2Mv2Condition(sampler_version='v3')` switches to a
mathematically universal mask prior that replaces the v2 template
mixture with a Boolean Rank-K decomposition: `M = ⋁_k (t_k ⊗ d_k)`.
Every eval-task signature E1-E15 lies in the support of the prior by
construction. Design and coverage audit:
`docs/design/mask_prior_rank_k.md`.

Coverage audit (N=10k, `scripts/eval/sampler_coverage_audit.py`): v3 reaches
**≥0.1% effective coverage on 21/25 eval settings** (v2 10/25). The
remaining 4 sub-0.1% v3 settings (E4.B/C/E, E2.mid60) still see tens of
hits per epoch at realistic dataset sizes. v2 remains the default for
backward compatibility; flip `sampler_version='v3'` in config to
activate.

Components (`condition_sampler_v3.py`):
- πK: K ∈ {0..4} with weights (0.10, 0.55, 0.25, 0.07, 0.03).
- πT: 6 temporal primitives (all, empty, interval, periodic, renewal,
  markov). `interval` uses a 40/30/30 short/mid/long length mixture
  with prefix/suffix/interior position bias.
- πD: 5 dimensional kinds (rot_only, pos_only, trans_only, mixed,
  all_dim). `all_dim` is essential for E2/E3/E7/E8/E15 full-frame locks.
- Anatomical joint dictionary: 17 weighted groups matching SMPL-22
  topology (all, upper_body, lower_body, arms, legs, left/right_*,
  ankles, feet, wrists, hands_feet, end_effectors, spine_chain, head).

Tests: `tests/unit/test_condition_sampler_v3.py` (34 assertions),
`tests/unit/test_prepare_m2m_v2_sampler_switch.py` (drop-in integration).

---

## Data Flow (Training + Inference)

```
Completion (current training):
  1. Dataset: LoadSMPLX -> raw motion (T, 135)
  2. Dataset: RandomCropPadding -> pad to 360 frames
  3. Dataset: PrepareM2MUniversalMask -> src_motion, tgt_motion, src_mask
  4. Trainer: normalize(src_motion), normalize(tgt_motion)
  5. Trainer: src_motion = src_motion * (1 - src_mask)  <- ZERO mask regions
  6. Trainer: zero padding frames
  7. Trainer: prepare_vace_input(src_motion, src_mask)
  8. Flow matching: x_t = (1-t)*noise + t*tgt_motion, predict velocity

Editing (training):
  4. Trainer: normalize(LQ_motion), normalize(HQ_motion)
  5. Trainer: DO NOT zero - src_motion keeps LQ values
  7. Trainer: prepare_vace_input(LQ_norm, mask) -> reactive=LQ[edit]

Inference (Completion):
  1. Load NPZ -> raw motion (T, 135)
  2. Build mask (checker / adaptive / manual)
  3. normalize(motion)
  4. motion_norm = motion_norm * (1 - mask)
  5. Pad to max_frames
  6. prepare_vace_input(zeroed_motion_norm, mask)
  7. ODE integration: odeint(fn, y0, t, method='midpoint')
  8. Denormalize, blend with original in unmasked regions
```

---

## Padding & Sequence-Length Convention (MUST FOLLOW)

Clips shorter than `clip_len` (default 360) are right-padded to a fixed length
inside `RandomCropPadding` so the batch can be stacked. Padded frames are
**book-keeping only** — they must never influence training loss, attention,
or inference metrics. Every transform/trainer/pipeline in this stack must
obey the following invariants.

### 1. The single source of truth is `num_frames`

`RandomCropPadding` always writes `results['num_frames']` = the **pre-pad**
valid frame count (= `min(original_T, clip_len)`). Downstream code must
read `num_frames` (or a value derived from it) to recover the real length;
the post-pad tensor shape `(clip_len, D)` does **not** tell you how many
frames are real.

### 2. `tgt_length` / `src_length` must equal `num_frames`

These two fields, produced by `PrepareM2Mv2Condition` / `PrepareM2MUniversalMask`
/ `PrepareM2Mv2FullMask`, are the only handle downstream batching has on the
original clip length. They **must** be set to `int(results.get('num_frames', T))`,
**not** `motion.shape[-2]`. The fallback to `T` exists only for datasets that
skip `RandomCropPadding` (e.g. variable-length batches without padding).

Consequences of setting them to the padded length instead:
- Trainer's zero-out block (`tgt_motion[:, tgt_len:] = 0`) becomes a no-op.
- `tgt_padding_mask` degenerates to all-`True`, disabling masked attention.
- All loss terms (velocity, x1, keypoints3d, smoothness, translation) treat
  the replicated last frame as ground truth; for a 30-frame clip, 91.7% of
  the supervision signal becomes "hold last frame static", teaching the
  model to freeze late in the clip.

### 3. `tgt_padding_mask` is `[True] * num_frames + [False] * (L - num_frames)`

Built by `HyMotionM2MBundle.prepare_padding` from `tgt_length`. It is the
canonical attention mask for the transformer and the canonical frame mask
for every loss term. Anything that needs a "valid frames" mask must derive
it from `tgt_padding_mask`, never re-derive from shapes.

Dual usage inside the trainer (`HyMotionM2MTrainer._prepare_and_forward`):

- `x_mask_temporal=tgt_padding_mask` → MMDiT attention. Inside
  `HyMotionMMDiT.forward`, `_canonical_mask` converts it to an additive
  `(0, -inf)` mask, then `_build_dmm_attn_mask_shared` /
  `_build_smm_attn_mask_shared` adds it to `base` along the **key**
  dimension (`base += key_padding_mask.view(B, 1, 1, total_len)`). That
  forbids any query (motion or text) from attending to a padded motion
  key, so padded frames cannot contaminate valid-frame representations.
  **Attention runs in `mode="torch"` (hard-coded); the `"flash"` branch in
  `network/attention.py` ignores `attn_mask` and must not be enabled
  without first adding `cu_seqlens` varlen packing.**
- `data_mask_temporal=tgt_padding_mask` → every per-frame loss term
  (velocity / x1 / keypoints3d / smoothness / translation / SOAR velocity).

Both usages depend on `tgt_length` being the real pre-pad length (see §2).
Setting `tgt_length = padded_L` silently disables **both** at once.

On query-side padding: the attention mask only masks the key dimension;
padded-query rows still produce non-zero outputs against valid+text keys.
This is fine because those rows are masked out again at the loss stage
via `data_mask_temporal`. It is not a bug, just slightly wasted compute.
Never repurpose the key_padding_mask into a 2D `(L, L)` attn_mask to try
to also zero the query rows — doing so can make some query rows all-`-inf`
(when combined with causal or narrowband masks) and produce `NaN` from
softmax.

### 4. Padded frames are zeroed after normalization

In `HyMotionM2MTrainer._prepare_and_forward`, once `tgt_length` is correct,
padded positions of `src_motion`, `tgt_motion`, and `src_mask` are set to 0.
This keeps VACE `inactive/reactive` zero outside valid frames and lets the
attention mask and loss mask do the rest.

### 5. Loss aggregation respects both masks

`M2MLoss` asserts `data_mask_temporal is not None` (= `tgt_padding_mask`)
and multiplies every per-frame term by it. When `generation_mask` (= `src_mask`
aggregated over joints) is also present, the loss is double-masked:
`loss = Σ (per-frame-loss · data_mask · generation_mask) / Σ (data_mask · generation_mask + eps)`.
`SOAR`'s `_masked_velocity_loss` follows the same convention.

### 6. Inference mirrors training

`HyMotionM2MPipeline.generate` builds `keep_mask = (src_mask == 0) & valid_frame_mask`
where `valid_frame_mask` comes from the per-sample `tgt_length` / `src_length`
passed by the caller. Padded tail frames are therefore always **free to evolve
during ODE integration** (never pinned to a padded value), and output
metrics are computed on the truncated `[:T]` window. `scripts/eval/eval_m2m_v2_all_tasks.py`
passes the actual motion length `T` as both `src_length` and `tgt_length`.

### 7. Adding a new condition transform? Checklist

When you write a new `Prepare*Condition` transform that produces
`src_motion` / `tgt_motion` / `src_mask`:

- [ ] Read `num_frames` via `int(results.get('num_frames', motion.shape[-2]))`.
- [ ] Set `tgt_length` and `src_length` to that value.
- [ ] If your transform clones `motion` into `src_motion` / `tgt_motion`,
      the replicated-pad tail is fine — the trainer will zero it.
- [ ] If your transform generates `src_mask` from rules that depend on
      "valid frames", build it against `num_frames`, not the padded length.
- [ ] Add a smoke test similar to `scripts/debug/smoke_test_m2m_padding_fix.py`
      covering T ∈ {short, exactly-clip_len, long}.

---

## Known-Region Conditioning: Cross-Project Comparison

Motion completion 的核心问题：**模型怎么知道哪些帧/关节是已知的，以及它们的值是什么？** 四个项目给出了完全不同的答案。这些设计差异直接决定了：(1) 推理时能否逐步替换已知区域来消除边界跳变；(2) 能否精确控制特定身体部位。

---

### 1. KIMODO（NVIDIA）— Imputation：直接往 x_t 里塞 GT 值

**核心思路**：最直觉的做法——既然我知道某些帧/关节的 GT 值，就在每步 denoise 前直接用 GT 覆盖 x_t 的对应位置。模型看到的 x_t 是"一部分干净 + 一部分 noisy"的混合体，concat 一个 binary mask 告诉模型哪里是干净的。

**动作表示**：333 维，使用**世界坐标系** global joint rotation (6D) + global joint position (xyz) + smooth root + heading + foot contact。27 个关节。这是关键——因为 rotation 和 position 都在世界坐标系下，想约束"右手在世界坐标 (1.2, 0.8, 0.3)"可以直接写入对应维度，不需要做 IK。

**训练**：
- Phase 1（500k steps）：纯 Text-to-Motion，模型不知道约束的存在
- Phase 2（500k steps）：随机采样约束（keyframe、end-effector、trajectory 等），在每步前 impute：
  ```
  x̃_t = mask * x_target + (1-mask) * x_t      ← 已知维度用 GT 覆盖
  model_input = concat([x̃_t, mask])             ← 333+333 = 666 维
  model 预测 x_0（完整的 clean motion）
  ```
  模型在 Phase 2 学会了：看 mask=1 的位置去读 GT 信息，看 mask=0 的位置去做去噪生成。

**推理**：每步 denoise 前都做 imputation：
```
for each denoising step:
    x_t[mask=1] = GT_values                      ← 硬替换
    model_input = concat([x_t, mask])
    x_pred = model(model_input, t)
    x_{t-1} = scheduler.step(x_pred)             ← 下一步又会被替换
```

**身体部位控制**：
- ✅ 全身关键帧：所有关节的 global position + rotation 都 impute
- ✅ End-effector：只 impute 手/脚关节的 global position（rotation 不约束）→ 模型自己推断合理的 rotation，但 FK 后的 position 和约束值有 cm 级误差
- ✅ 轨迹：impute root 的 2D 平面坐标
- ✅ Foot contact：impute 4 维 foot contact 标志
- 约束用的是 **global position**，不是 local rotation——这使得不需要做 IK 就能控制世界空间坐标

**关键特性**：
- 推理时逐步替换 = 训练时逐步 impute，**分布完全一致**，所以 boundary 过渡平滑
- 已知帧精确保持（零误差）
- 但：Phase 1 → Phase 2 的两阶段训练增加复杂度；global rotation 表示与 SMPL 生态不兼容

---

### 2. MoGenDiT（内部）— Mask-Aware Noise：已知帧在 x_t 中直接保持 clean

**核心思路**：和 KIMODO 类似，也是让模型在 x_t 中直接看到已知帧的 clean 值。但实现方式不同：不是在每步手动 impute，而是在**加噪过程 (q_sample) 中就跳过已知帧**——已知帧根本不加噪，保持原始 clean 值。

**动作表示**：201 维（`OccamMotionRep`），local rotation 6D (column-major, 22×6=132) + local joint position (22×3=66) + translation (3)。和 KIMODO 不同，MoGenDiT 用**局部坐标系**。注：旧版用 `HM263XRep` (263 维，含 stationary + padding)，已弃用。

**训练**：单阶段，但 50% batch 会做 motion degradation：
```python
# q_sample（加噪）—— mask-aware
x_noise = sqrt_α * x0 + sqrt(1-α) * noise       # 标准 DDPM 加噪
noise_mask[obs_mask >= 1] = False                 # ← 关键：obs_mask=1 的位置不加噪
x_t = x0.clone()
x_t[noise_mask] = x_noise[noise_mask]             # x_t[已知] = clean，x_t[未知] = noisy

model_input = concat([x_t, obs_mask])             # 201+201 = 402 维
model 预测 x_0（完整 clean motion）
loss = MSE(pred_x0, original_clean_motion)
```

50% 的 batch 额外做 motion degradation（8 种合成缺陷：关节跳变、扭曲、冻帧、位移漂移等），loss target 依然是原始干净 motion，所以模型同时学会去噪和修复。

**推理**：保存 clean motion → 纯噪声出发 → 每步替换已知区域：
```python
x_clean = x_wrap["x_t"].clone()                  # 保存 clean motion
x_wrap["x_t"] = randn_like(...)                   # 从纯噪声开始
for each DDIM step:
    x_wrap["x_t"][obs_mask] = x_clean[obs_mask]   # 已知区域替换为 clean
    x_wrap["x_t"] = ddim_sample(model, x_wrap, t)
# 三种模式：all（每步都替换+最后一步），skip_last（默认），none
```

**身体部位控制**：
- ✅ Per-joint per-dim mask：obs_mask 是 (T, 201)，可以 mask 任意维度组合
- 实际使用以 **temporal mask** 为主（keyframe、contiguous segments），joint-level mask 也支持
- 包含 translation 维度，可以约束位移

**关键特性**：
- 训练时 x_t 中已知帧 = clean，推理时替换为 clean，**分布完全一致**
- 同时学去噪 + 修复（motion degradation 训练），是唯一能处理"输入运动本身有缺陷"的方案
- 但：不能做 world-space position control（local 坐标系）；推理用 10 步 DDIM 已经很快

---

### 3. UMO（Brown/MIT/Meta）— Adapter Add：通过 0.207M 轻量适配器注入已知信息

**核心思路**：完全不碰 x_t。冻住预训练好的 T2M backbone（HY-Motion-Lite，460M 参数），额外加一个极小的 MLP adapter（E_ctx，0.207M 参数）。已知 motion 经过 adapter 编码后，通过 **element-wise add** 叠加到 backbone 的 input embedding 上。模型通过 self-attention 自己消化这个"叠加信号"。

**动作表示**：201 维，local rotation 6D + local joint position + global translation。与 HY-Motion 完全一致。

**条件粒度**：**帧级 whole-body**。每帧分配一个 meta-operation 标签 τ ∈ {Preserve, Generate, Edit}：
- **[P] Preserve**：这帧应该保持原样（in-between 的首尾帧）
- **[G] Generate**：这帧需要从头生成（中间的补全帧）
- **[E] Edit**：这帧需要在 source 基础上按文本指令修改（editing 任务）

注意：一帧只能是 P/G/E 其中之一，**不能对同一帧的不同关节分别指定**。这是 UMO 的核心局限。

**训练**：
```python
# backbone 冻结，只训练 E_ctx + 3 个 meta-op embeddings
s̃_i = source_motion_i + Emb(τ_i)                 # source + meta-op embedding
source_emb = E_ctx(s̃)                             # 0.207M adapter
x_input = E_in(x_t) + source_emb                  # element-wise add 到 input embedding
# x_t 全程均匀加噪（和标准 T2M 一样），已知信息只通过 source_emb 传入
# backbone 的 self-attention、cross-attention 全都不改
```

**推理**：和训练完全一致，没有任何 per-step 替换。模型自己学着把 [P] 帧的值"写回"到输出中，但不是精确的——[P]-MPJPE ≈ 0.95mm，接近但不等于零。

**身体部位控制**：
- ❌ **无法做 part-level control**（论文 Limitation §5 明确指出）
- 一帧只有一个 τ_i 标签，不能"这帧的上半身 preserve、下半身 generate"
- 这是帧级 meta-operation 设计的固有限制

**关键特性**：
- 极轻量（0.207M 参数），backbone 完全不动，T2M 能力完整保留
- ✅ 支持 [Edit] 语义——这是其他三个方案都没有的：提供 source motion + editing instruction text，模型修改运动风格/幅度
- ❌ 不支持 per-joint 控制
- ❌ 推理时替换 x_t 无效（模型从 adapter 信号读已知信息，不从 x_t 读）

---

### 4. HyMotion M2M（我方）— VACE：已知信息走独立 concat 通道

**核心思路**：不碰 x_t（和 UMO 一样），但不是用 add 而是用 **channel-wise concat**。把 x_t、已知运动值（inactive）、待编辑运动值（reactive）、binary mask 四个通道拼起来作为输入。模型通过独立通道读取"什么位置已知"和"已知值是什么"。

**动作表示**：135 维，absolute translation (3D) + local rotation 6D (SMPL, row-major, 22 joints × 6)。**没有 joint position**，只有 rotation + translation。

**条件粒度**：**逐帧逐维度 (T, 135)**，实际操作以 joint group 为最小单位（23 组：1 translation + 22 joints），每组 all-or-nothing。

**训练**：
```python
# Flow matching：所有区域均匀加噪
x_t = (1-t) * noise + t * x_clean                 # x_t[已知区域] 也是 noisy 的

# VACE 构建（已知信息进入独立通道）
src_motion = normalize(motion) * (1 - mask)        # mask=1 区域清零
inactive = src_motion * (1 - mask)                 # 已知区域的 clean 值（completion 模式）
reactive = src_motion * mask                       # completion 模式下 = 0
model_input = concat([x_t, inactive, reactive, mask])  # 135×4 = 540 维

model 预测 velocity (v = x1 - x0) 或 x1
loss = SmoothL1(pred, target)
```

训练时 x_t 的已知区域 = noisy（和生成区域一样加了噪），模型**不从 x_t 读已知信息**，而是从 inactive 通道读。

**推理**：标准 ODE 积分，完成后做 post-hoc blend：
```python
trajectory = odeint(fn, noise, t=[0,1], method='midpoint')  # 50 步
output = trajectory[-1]
final = original * (1-mask) + denormalize(output) * mask     # 硬拼接
```

**身体部位控制**：
- ✅ 任意 (frame, joint) 组合的 mask：M4 策略支持关节组 mask（上半身、下半身、左臂、右臂等）
- 约束用的是 **local rotation 6D (SMPL)**：mask=0 的关节保持 SMPL 相对旋转不变
- ❌ 无法直接约束 world-space xyz position（表示中没有 joint position 维度）
- ✅ 可以约束 translation（mask 的 col 0 = translation group）

**关键特性**：
- ✅ 最细粒度的 mask（per-joint per-dim），7 种 mask 策略覆盖多种场景
- ✅ 支持 editing 模式（reactive 通道传入 LQ/pre-edit motion）
- ✅ 文本条件（Qwen3 + CLIP，双编码器）
- ❌ 推理时逐步替换 x_t 对**标准模型**无效——模型不读 x_t 的已知区域，训练时那里就是 noisy 的
- ✅ `_man` 变体：推理时逐步替换有效（训练时 x_t[known]=clean，imputation 是 train-consistent）
- ❌ 标准模型 Post-hoc blend 导致边界跳变（generated 区域的第一帧和 original 的最后一帧不连续）

---

### 对比总结

#### 已知区域如何进入模型

| 方案 | 已知信息在哪里 | 训练时 x_t 已知区域 | 推理时逐步替换 | 已知帧精度 |
|------|-------------|-------------------|-------------|----------|
| **KIMODO** | 直接在 x_t 中（imputation）+ mask concat | Clean（Phase 2 impute） | ✅ 有效 | 精确（零误差） |
| **MoGenDiT** | 直接在 x_t 中（不加噪）+ mask concat | Clean（q_sample 跳过） | ✅ 有效 | 精确（零误差） |
| **UMO** | 侧通道 adapter（E_ctx element-wise add） | Noisy | ❌ 无效 | 近似（~0.95mm） |
| **M2M** | 侧通道 VACE（inactive channel concat） | Noisy（标准）/ Clean（`_man`） | ❌ 标准 / ✅ `_man` | Post-hoc blend / Imputation |

**规律**：已知信息直接写入 x_t 的方案（KIMODO、MoGenDiT、M2M `_man`），推理时替换有效、已知帧精确。走侧通道且 x_t 全程 noisy 的方案（UMO、M2M 标准），推理时替换无效、已知帧不精确。

#### 身体部位控制能力

| 方案 | Part-level | 约束坐标系 | 约束内容 | World-space position |
|------|-----------|----------|---------|---------------------|
| **KIMODO** | ✅ Per-joint | **Global**（世界坐标系） | Global rotation + global position | ✅ 直接 impute xyz |
| **MoGenDiT** | ✅ Per-joint | Local（parent-relative） | Local rotation + local position + translation (201 维) | ❌ 需要 FK |
| **UMO** | ❌ Frame-level | Local | Local rotation + local position | ❌ 不支持 |
| **M2M** | ✅ Per-joint | Local（SMPL） | Local rotation + absolute translation | ❌ 需要 FK |

**KIMODO 独特优势**：global 坐标系让它可以直接 impute"右手在世界坐标 (x,y,z)"，其他三个方案都需要 FK/IK 转换才能做到 world-space position control。

#### 任务覆盖

| Task | KIMODO | MoGenDiT | UMO | M2M |
|------|--------|----------|-----|-----|
| In-Between | ✅ full-body keyframe impute | ✅ temporal mask | ✅ [P] first/last | ✅ M3 temporal |
| Prediction | ✅ keyframe impute | ✅ prefix mask | ✅ [P] prefix | ✅ M3 temporal |
| Joint Completion | ✅ per-joint impute (global) | ✅ per-dim mask | ❌ frame-level only | ✅ M4 joint |
| End-Effector | ✅ **直接 impute global pos** | ❌ | ❌ | ❌ |
| Trajectory | ✅ root impute | ❌ | ✅ text description | ✅ translation mask (M4 body part group) |
| Motion Editing | ❌ | ❌ (repair only) | ✅ [Edit] + instruction | ✅ reactive channel |
| Repair / Denoise | ❌ | ✅ primary task | ❌ | ✅ edit-repair mode |
| Text-Conditioned | ✅ separated CFG | ❌ | ✅ | ✅ Qwen3+CLIP |

---

### M2M 的改进方向

**标准模型的边界跳变根因**：M2M 的 VACE 设计让已知信息走 inactive 侧通道，x_t 全程 noisy。ODE 积分完成后做 hard blend（`original*(1-mask) + generated*mask`），generated 区域的第一帧和 original 的最后一帧没有连续性保证。

**`_man` 变体（已实现）**：通过 mask-aware noise 训练 + imputation 推理解决此问题。详见上方 §Imputation Inference。

**V4 方案：Mask-Aware Flow Matching**（✅ 已实现，`_man` 变体）：
```python
# 训练：x_t[keep] = x_clean[keep]       （已知区域保持 clean）
#       x_t[gen]  = (1-t)*noise + t*x_clean  （生成区域正常加噪）
# 推理：y0[keep] = clean_motion          （初始化训练一致）
#       每步：x[keep] = clean_motion      （imputation）
```
让模型在训练时就看到 x_t 中"已知区域 = clean + 生成区域 = noisy"的分布，推理时逐步替换就变得 train-consistent。VACE inactive 通道保留，作为冗余信号。推理侧的 replacement guidance 代码已在 pipeline 中实现，`clean_motion` batch key 由调用端提供。

⚠️ `adaptive_mask_to_dense` 已修复 trans_mask 丢弃问题（2026-04）：现在正确将 MoGenDiT 的 `trans_mask` 写入 combined grid 第 0 列。

---

## Motion Representation

### 135-dim Layout (smpl_22, rot6d + abs transl)

```
dims [0:3]     — translation (3 absolute)
dims [3:9]     — joint 0:  Pelvis (root orientation, rotation_6d)
dims [9:15]    — joint 1:  L_Hip
dims [15:21]   — joint 2:  R_Hip
dims [21:27]   — joint 3:  Spine1
dims [27:33]   — joint 4:  L_Knee
dims [33:39]   — joint 5:  R_Knee
dims [39:45]   — joint 6:  Spine2
dims [45:51]   — joint 7:  L_Ankle
dims [51:57]   — joint 8:  R_Ankle
dims [57:63]   — joint 9:  Spine3
dims [63:69]   — joint 10: L_Foot
dims [69:75]   — joint 11: R_Foot
dims [75:81]   — joint 12: Neck
dims [81:87]   — joint 13: L_Collar
dims [87:93]   — joint 14: R_Collar
dims [93:99]   — joint 15: Head
dims [99:105]  — joint 16: L_Shoulder
dims [105:111] — joint 17: R_Shoulder
dims [111:117] — joint 18: L_Elbow
dims [117:123] — joint 19: R_Elbow
dims [123:129] — joint 20: L_Wrist
dims [129:135] — joint 21: R_Wrist
```

### Mask Granularity

Minimum unit is **joint group** (not per-dim):
- Translation: dims [0:3] — all-or-nothing
- Each joint: dims [3+j*6 : 3+(j+1)*6] — 6 rot6d dims all-or-nothing
- Mask defined on (T, 23) grid (1 transl + 22 joints), expanded via `expand_grid_to_mask()` to (T, 135)

### ⚠️ Rotation 6D Convention (Critical)

Two different conventions coexist:

| Convention | Usage | 6D layout | Code |
|-----------|-------|-----------|------|
| **Column-major** | `rotation_convert` math functions | `[R00,R10,R20,R01,R11,R21]` | `rotation_convert.py` |
| **Row-major** | Training data, model I/O, checkpoints | `[R00,R01,R10,R11,R20,R21]` | `load_smplx.py` line 93 |

**Data flow**:
```
axis_angle
  -> rotation_convert.axis_angle_to_rotation_6d()  -> column-major
  -> load_smplx.py: out[:,:,[0,3,1,4,2,5]]         -> row-major  <- training/model use this
  -> model training and inference
  -> inverse: rot6d[:,[0,2,4,1,3,5]]                -> column-major
  -> rotation_convert.rotation_6d_to_axis_angle()    -> axis_angle
```

**Rules**:
- Encoding (axis_angle -> rot6d): always via `process_smplx_pose` (handles reorder internally)
- Decoding (rot6d -> axis_angle): MUST reorder `[0,2,4,1,3,5]` first, then `rotation_6d_to_axis_angle`
- NEVER mix `geometry.py` and `rotation_convert.py` rot6d functions

### ⚠️ Per-Dimension Normalization

Controlled by `HyMotionM2MBundle.mean_std_dir`:
- `None` -> mean=0, std=1 (no normalize)
- `'data/hymotion_m2m_data/_stats'` -> load Mean.npy/Std.npy (135-dim)

**Inference MUST use same mean_std_dir as training.** Check via `work_dirs/<exp>/*/config.py`.

### ⚠️ Relative Translation Reconstruction

When using `transl_type='rel'`, first frame's displacement is `[0,0,0]`.
Must provide original `abs_trans[0]` as starting point for reconstruction.

### Global vs Local Rotation Space (Ablation V5, 2026-03-29)

**背景**：M2M 默认使用 **local rotation**（SMPL 父节点相对旋转）。V5 消融实验引入 **global rotation**（世界坐标系绝对旋转），训练时用 global、推理 decode 时转回 local 输出 SMPL-compatible NPZ。

**核心假说**：Global rotation 下，被 mask 的关节更容易从邻居推断——因为所有关节的旋转在同一坐标系下，可以直接几何插值；local rotation 下邻居处于不同参考系，无法直接插值。

**实证验证**（5363 帧真实数据）：邻居均值插值预测 masked 关节的 MAE：

| 关节 | 运动链深度 | Local MAE | Global MAE | 改善 |
|------|----------|-----------|------------|------|
| Pelvis (root) | 0 | 0.169 | 0.032 | +81% |
| Spine1 | 1 | 0.107 | 0.029 | +73% |
| L_Foot | 4 | 0.122 | 0.006 | +95% |
| L_Elbow | 6 | 0.422 | 0.196 | +54% |
| L_Wrist | 7 | 0.336 | 0.196 | +42% |
| L_Collar | 4 | 0.113 | 0.168 | **-48%** |
| **Overall (21 joints)** | — | **0.183** | **0.108** | **+41%** |

**结论**：Global rotation 在 19/21 个关节上更可预测（平均 +41%）。仅 L/R_Collar 例外——因为 Collar 同时连接 Spine3 和 Shoulder，两个邻居的全局旋转差异大于局部旋转差异（肩胛骨运动特性）。

#### 转换精度：实质无损

| 场景 | 最大误差 | 说明 |
|------|---------|------|
| Float64 round-trip | 4.8e-7 | Gram-Schmidt 重投影引入的数值噪声 |
| Float32 round-trip | 3.9e-7 | 训练/推理实际精度 |
| Float32 + normalize/denormalize | 4.1e-7 | 完整 train→infer 管线 |
| Float16 accumulation | **3.2e-4** | ⚠️ 混合精度不可接受 |
| 10 次迭代 round-trip | 1.1e-6 | 误差线性累积，不发散 |

**结论**：Float32 下转换实质无损（误差 < 1e-6，远小于模型预测噪声）。**禁止在 float16/bf16 下做 FK 转换**——误差放大 1000 倍。当前 M2M 训练使用 float32（`mixed_precision='no'`），安全。

#### 优势

1. **邻居可预测性 +41%**：masked 关节可从已知邻居直接几何插值推断，不需要 IK
2. **统一坐标系**：所有关节在同一参考系下，模型可直接学习关节间的空间关系
3. **与 KIMODO 对齐**：KIMODO 选择 global rotation 的理由相同——imputation 时直接在世界坐标系操作
4. **误差不放大**：FK 链的误差放大因子 ~1-2x（实测），不会导致远端关节爆炸

#### 劣势

1. **方差膨胀**：远端关节累积祖先旋转，Std 显著增大（Spine3: 6.1x, L_Wrist: 2.5x），对 normalization 更敏感
2. **信息冗余**：子关节的 global rotation 包含了所有祖先信息，表示冗余度高。模型需要学习这种冗余结构
3. **局部运动难表达**：纯局部运动（如手腕旋转不改变上臂）在 global space 中影响从该关节到所有后代的值，增加建模难度
4. **生态不兼容**：SMPL/SMPL-X 使用 local rotation，推理时必须转回 local，增加 decode 开销
5. **L/R_Collar 例外**：肩胛骨区域邻居预测性反而下降 48%，说明 global 不是所有拓扑结构的最优选择

#### 方差膨胀详情

| 关节 | 运动链深度 | Local Std | Global Std | 膨胀倍数 |
|------|----------|-----------|------------|---------|
| Pelvis | 0 | 0.302 | 0.249 | 0.8x |
| Spine2 | 2 | 0.073 | 0.281 | **3.9x** |
| Spine3 | 3 | 0.048 | 0.297 | **6.1x** |
| L_Wrist | 7 | 0.198 | 0.497 | **2.5x** |

Spine 链方差膨胀最严重——因为 Spine 关节本身的 local rotation 很小（Std 0.05-0.07），但累积了 root 的大旋转后 global Std 暴增。远端关节（Wrist）由于自身 local rotation 较大，膨胀相对温和。

#### 实现

- **训练**：`LocalToGlobalRotation` transform 插入 `LoadSmplx55` 之后，使用 `_stats_global_rot/` 的 Mean/Std
- **推理**：`HyMotionM2MBundle.decode_motion_from_latent()` 在 denormalize 后调用 `global_to_local_rot6d_torch()` 转回 local
- **Config**：`model.rotation_space='global'` + `model.mean_std_dir='data/hymotion_m2m_data/_stats_global_rot'`
- **转换代码**：`hftrainer/datasets/motion/motionhub/transforms/fk_utils.py`

#### 开放问题

1. **长训练后 global 是否真的优于 local？** 短期 loss 相当，需要完整训练 + 评估 motion quality（FID/diversity）才能下结论
2. **混合方案可行性**：对不同关节用不同坐标系（如 spine 用 global、limb 用 local）是否更优？
3. **Collar 问题**：是否需要对肩胛骨区域做特殊处理？

---

## Cross-Project Convention Table

| Project | motion_dim | transl_type | smpl_type | Layout |
|---------|-----------|-------------|-----------|--------|
| **HyMotion M2M** | **135** | `abs` (3d) | smpl_22 | `[abs_transl(3), rot6d(132)]` |
| **HyMotion T2M** | **201** | `rel` (3d) | smpl_33 | `[rel_transl(3), rot6d(198)]` |
| **PRISM / MCM / VerMo** | **138** | `abs_rel` (6d) | smpl_22 | `[abs_rel_transl(6), rot6d(132)]` |

Normalization sources:
- HyMotion M2M: `data/hymotion_m2m_data/_stats/{Mean,Std}.npy` (135-dim)
- HyMotion T2M: `checkpoints/HY-Motion-1.0/stats/{Mean,Std}.npy` (201-dim)
- PRISM/VerMo: `data/statistic/smplx55_stats_hymotion_aug.json` (JSON, assembled by `SMPLPoseProcessor`)

### Common Pitfalls

| Scenario | Consequence | Prevention |
|----------|-------------|------------|
| PRISM config + M2M checkpoint | dim mismatch (138 vs 135) | Check `transl_type` and `_motion_dim` |
| M2M mean/std for PRISM | broadcast error | Each project uses own stats |
| `rotation_6d_to_axis_angle` without reorder | rotation error > 3 radians | Reorder `[0,2,4,1,3,5]` first |
| New config + old checkpoint (normalize mismatch) | wrong output range | Use training config for inference |
| VACE reactive leak | model cheats, loss artificially low | `src_motion *= (1-mask)` after normalize |
| VACE operator precedence bug | `a * 1 - b` != `a * (1 - b)` | Always use parentheses |
| Missing normalize in trainer | loss ~100x too high | Call `bundle.normalize_motion()` |
| Training on unfiltered data | model learns from LQ motion, limits quality ceiling | Use `high_quality.json` filtered annotation (see §Training Data Quality Issue) |

---

## Model Variants & Config Mapping

### Text-Conditioned (0.46B, pretrained from HunyuanMotion T2M 1.0-Lite)

Config 目录: `configs/hymotion_m2m/`

使用 `HunyuanMotionMMDiT`（dual-stream + single-stream），从 T2M 1.0-Lite 加载预训练权重。支持文本条件。

| 变体 | Config | Work Dir | Epoch | 说明 |
|------|--------|----------|-------|------|
| uncond_fm_man | `hymotion_m2m_completion_uncond_fm_man_046b.py` | `hymotion_m2m_completion_uncond_fm_man_046b` | 1000 | **最优 checkpoint** |
| uncond_fm_man_globalrot | `..._globalrot_046b.py` | `..._globalrot_046b` | 527 | GlobalRot 消融 |
| caption_fm_man | `..._caption_fm_man_046b.py` | `..._caption_fm_man_046b` | — | 带文本 |
| 等 | ... | ... | — | FM/JiT × MAN × caption/uncond × local/global 组合 |

### Text-Free (DiT, 从零训练)

> **⚠️ Config 目录**: `configs/hymotion_dit/`（**不是** `configs/hymotion_m2m/*textfree*`）
>
> `configs/hymotion_m2m/*textfree*` 是早期重复实现，与 `configs/hymotion_dit/` 功能完全一致（同架构、同参数量、同训练数据、同 loss），仅 work_dir 不同。**应使用 `configs/hymotion_dit/` 作为标准配置。**

使用 `HunyuanMotionDiT`（纯 single-stream），无文本编码器，从零训练。

| 变体 | 参数量 | Config | Work Dir | Epoch | 状态 |
|------|-------|--------|----------|-------|------|
| dit_fm_man_s | 49M | `hymotion_dit_fm_man_s.py` | `hymotion_dit_fm_man_s` | 762 | ✅ 训练中 |
| dit_fm_man_b | 288M | `hymotion_dit_fm_man_b.py` | `hymotion_dit_fm_man_b` | 806 | ✅ 训练中 |
| dit_fm_man_l | 383M | `hymotion_dit_fm_man_l.py` | `hymotion_dit_fm_man_l` | 19 | ❌ 需重新启动 |
| dit_fm_man_globalrot_s | 49M | `hymotion_dit_fm_man_globalrot_s.py` | `hymotion_dit_fm_man_globalrot_s` | 757 | ✅ 训练中 |
| dit_fm_man_globalrot_b | 288M | `hymotion_dit_fm_man_globalrot_b.py` | `hymotion_dit_fm_man_globalrot_b` | 833 | ✅ 训练中 |
| dit_fm_man_globalrot_l | 383M | `hymotion_dit_fm_man_globalrot_l.py` | `hymotion_dit_fm_man_globalrot_l` | 21 | ❌ 需重新启动 |
| dit_fm_man_m / dit_jit_* | — | 有 config | — | — | ❌ 未启动训练 |

**注意**: M-size (162M) 和所有 JiT 变体有 config 但从未训练。

---

## SOAR Post-Training (2026-04)

**SOAR** (Self-Correction for Optimal Alignment and Refinement, arXiv 2604.12617) 是一种 reward-free 后训练方法，专门解决 rectified flow / flow matching 模型的 **exposure bias**：训练时 `x_t` 来自 GT forward process（on-trajectory），推理时 `x_t` 来自模型自身 ODE 积分（off-trajectory），分布 mismatch 导致 50 步累积误差。

M2M 的 exposure bias 在生成区域（`src_mask=1`）尤其严重——`_man` (mask-aware noise) 只解决了 known regions 的分布匹配，generated regions 仍有完整的 off-trajectory 误差累积。**SOAR 和 `_man` 正交互补**：`_man` 修 known 区域，SOAR 修 generated 区域。

### Trainer 架构（B1 继承方案）

```
HyMotionM2MTrainer (SFT 基线)
  ├── _prepare_and_forward(batch)  # 负责 padding/text/VACE/forward，返回 ctx dict
  ├── _compute_base_loss(ctx)       # 纯 loss 计算（velocity/x1/smoothness/fk）
  └── train_step(batch)             # 薄封装：ctx=prepare → base_loss=compute → return

HyMotionM2MSoarTrainer(HyMotionM2MTrainer)
  └── train_step(batch):
        ctx = self._prepare_and_forward(batch)   # 复用父类
        base_losses = self._compute_base_loss(ctx)
        corr_loss = self._soar_correction_loss(ctx)  # SOAR-only: rollout + re-noise + correction forward
        total = sum(base_losses) + lambda * corr_loss
```

**关键设计原则**：
1. **父类零改动风险**：`HyMotionM2MTrainer` 的 `train_step` 行为严格等价（已通过 v2 smoke 验证），所有 v1/v2 config 照常跑
2. **共享 x0**（shared noise）：base loss 和 SOAR re-noise 使用**同一个** x0，保持 re-noised states 在原 transport ray 附近
3. **Mask-aware 在 4 处应用**：`x_t0`（父类已有）、`x_hat`（rollout 后）、`z_re`（re-noise 后）、loss 权重（`generation_mask=src_mask`）
4. **Correction target (FM velocity 形式)**：`v_corr = (x1 - z_re) / (1 - t').clamp_min(0.05)`——给定 off-trajectory 点 `z_re` 在时间 `t'`，这是把它引导回 `x1` 所需的 FM 速度（等价于 SOAR 论文在 sigma 参数化里的 `(z_t - x_clean)/sigma_t`，符号翻转后对应 velocity 参数化）
5. **CFG 简化**：v1 实现强制 `soar_cfg_scale=1.0`（直接用 `v_pred.detach()` 作为 rollout velocity）。文本条件模型的 CFG rollout 作为 TODO（需额外无条件 forward pass）

### 配置结构

**新增文件**：
- `hftrainer/trainers/motion/hymotion_m2m_soar_trainer.py` — 新 trainer（~200 行）+ 3 个内置单元测试
- `configs/hymotion_m2m_v2/soar/hymotion_m2m_v2_{uncond,caption}_{local,global}_046b_soar.py` — 4 份后训练 config

**重构文件**：
- `hftrainer/trainers/motion/hymotion_m2m_trainer.py` — 把 `train_step` 拆为 `_prepare_and_forward` + `_compute_base_loss` + 薄 `train_step`，**行为不变**

### 推荐超参（plan §5 默认值）

| 参数 | 值 | 说明 |
|------|---|------|
| `soar_lambda` | 0.1 | correction loss 权重（motion 空间 loss scale 与图像不同，保守起步） |
| `soar_num_aux` | 1 | 每样本 N 个 auxiliary 点；N=1 时计算开销最低 |
| `soar_K` | 50 | rollout 步长 = 1/K，匹配推理 50 步 |
| `soar_cfg_scale` | 1.0 | v1 仅支持 1.0（其他值会 raise NotImplementedError） |
| `soar_sigma_clamp` | 0.05 | `(1 - t').clamp_min(0.05)` 避免数值爆炸 |
| `lr` | 2e-5 | SFT 的 1/5，post-training 用 |
| `max_iters` | 5000 | 约 SFT step 数的 2-5%，可根据 loss 曲线决定是否续训到 10K |
| batch size | SFT 的 1/2 | SOAR 每 step 约 2× forward，VRAM 紧张减半 |

### 已启动的 Taiji 任务（2026-04-17）

| Config | SFT load_from | Taiji task_flag | GPU | Status |
|--------|--------------|-----------------|-----|--------|
| `hymotion_m2m_v2_uncond_local_046b_soar.py` | `uncond_local_046b/checkpoint-epoch_485` | `m2m_v2_uncond_local_soar` | 4×8 V100 | 🟢 running |
| `hymotion_m2m_v2_caption_local_046b_soar.py` | `caption_local_046b/checkpoint-epoch_498` | `m2m_v2_caption_local_soar` | 4×8 V100 | 🟢 running |
| `hymotion_m2m_v2_uncond_global_046b_soar.py` | `uncond_global_046b/checkpoint-epoch_544` | — | — | 待提交 |
| `hymotion_m2m_v2_caption_global_046b_soar.py` | `caption_global_046b/checkpoint-epoch_548` | — | — | 待提交 |

### 验证现状（本地 400 iter quickcheck，单卡 V100）

| 指标 | first100 | last100 | Δ |
|------|---------|---------|---|
| `loss_velocity` | 0.0258 | 0.0255 | **-1.1%** |
| `loss_soar_corr` | 0.0427 | 0.0419 | **-1.9%** |
| `loss_total` | 0.0304 | 0.0300 | **-1.2%** |

第 200-400 步斜率 `-3.8e-5/step`，全程无 NaN/Inf。`loss_velocity` 量级与 SFT epoch 487 基线（0.02-0.04）一致，加载成功。

### 续训（若 5K 后仍在下降）

`load_from.load_scope='model'` 会重置 optimizer/global_step，所以 5K 跑完后想再延长：

```python
# 方案 A：改 max_iters=10000 重新提交（等价于一次跑 10K，因 lr_scheduler=None 恒定 LR）
_base_ = '../hymotion_m2m_v2_uncond_local_046b.py'
# ...
load_from = dict(
    path='work_dirs/hymotion_m2m_v2_uncond_local_046b/checkpoint-epoch_485',  # 仍从 SFT
    load_scope='model',
)
train_cfg = dict(max_iters=10000, ...)

# 方案 B：从 5K 的 SOAR checkpoint 续训
load_from = dict(
    path='work_dirs/hymotion_m2m_v2_uncond_local_046b_soar/checkpoint-iter_5000',
    load_scope='model',
)
train_cfg = dict(max_iters=5000, ...)  # 再跑 5K
```

### ⚠️ 注意事项

1. **只支持 `pred_type='velocity'`**：`HyMotionM2MSoarTrainer.train_step` 会 raise `NotImplementedError` 对 `pred_type='x1'`。所有 v2 config 都是 velocity，不受影响
2. **caption config 的 SOAR 初始 loss 量级高于 uncond**（`loss_velocity ≈ 0.2` vs uncond 的 `≈ 0.02`）——这是 caption SFT 本身的基线水平，不是 SOAR 引入的问题。对比时请用相对变化而非绝对值
3. **ref_pose 兼容**：`x1 = cat([ref_pose, tgt_motion])` 的场景下，`ref_pose` 区域天然在 `src_mask=0`（known），mask-aware 4 处都会保留它
4. **日志频率**：当前 SOAR config 用 `LoggerHook.interval=1`（每步一行），5000 step 约 5000 行 log，正常可接受；若 Taiji web UI 监控延迟，直接 `tail -f work_dirs/.../train.log` 看文件

### 相关文档

- 方法学习文档：`ref_repo/SOAR/CLAUDE.md` — SOAR 论文分析 + 对 M2M 的适用性论证
- 实施方案：`docs/temp/soar_m2m_v2_post_training_plan.md` — 完整 §§1-11 设计方案（算法、超参、ablation、compute budget）
- Trainer 实现：`hftrainer/trainers/motion/hymotion_m2m_soar_trainer.py` — 含 3 个单元测试，运行 `python3 -c "from hftrainer.trainers.motion.hymotion_m2m_soar_trainer import *; _test_mask_aware_preserves_known_regions(); _test_cfg_scale_validation(); _test_soar_shapes_and_finiteness()"`

---

## Training Configuration

### ⚠️ Training Data Quality Issue (Critical, 2026-04)

**现状**：所有 M2M 配置（包括所有 `_man`、`globalrot`、`caption` 变体）都使用 `data/annotation/train_hymotion_400h.json` 作为训练数据，共 **549,130 条**，**没有任何质量过滤**。

**问题**：这 549K 条数据中包含约 **85,191 条低质量数据**（foot_sliding、jitter、joint_jump 等）和 **69,172 条 borderline 数据**。低质量数据直接参与训练，会拉低模型的 motion 质量上限——模型从有缺陷的数据中学习，生成/修复时会复现类似缺陷。

**正确做法**：使用 `motion_annot_web/m2m_database` 当前判定的高质量数据进行训练。

| 数据源 | 路径 | 数量 | 状态 |
|--------|------|------|------|
| 当前训练集（全量） | `data/annotation/train_hymotion_400h.json` | 549,130 | ❌ 包含低质量 |
| 高质量子集 | `data/hymotion_m2m_refine_data/data_quality_list/high_quality.json` | 456,530 | ✅ 可用但未接入 |
| 低质量 | `data/hymotion_m2m_refine_data/data_quality_list/low_quality.json` | 85,191 | 应排除 |
| Borderline | `data/hymotion_m2m_refine_data/data_quality_list/borderline_quality.json` | 69,172 | 可选排除 |

**实现方案**：
1. 在 `MotionhubMultiTaskMultiAgentDataset` 中添加 `quality_filter_list` 参数
2. 在 `load_data_list()` 中根据 path 过滤，只保留高质量数据
3. 生成新的 annotation JSON（仅包含高质量 path），避免运行时过滤开销
4. 影响：训练数据从 549K → 456K（-17%），但质量显著提升

**注意**：高质量列表会随 checker 更新和修复进度变化。应定期从 `motion_annot_web/m2m_database` 重新导出。

### Loss Types

| pred_type | Loss | Config |
|-----------|------|--------|
| `velocity` | SmoothL1(pred_v, x1-x0) | `hymotion_m2m_completion_velocity_046b.py` |
| `x1` (JiT) | SmoothL1(reparam_v, gt_v) + SmoothL1(pred_x1, x1) | `hymotion_m2m_completion_x1_046b.py` |

### Weight Initialization

From HunyuanMotion T2M 1.0-Lite (0.46B, motion_dim=201):
- **Loaded**: 18 transformer blocks (feat_dim=1024), text encoders, timestep_encoder, text_refiner (305/308 params)
- **Random init**: input_encoder (201->540), final_layer (201->135) (3 params)

---

## ⚠️ Inference Practical Guide — MUST READ Before Writing Inference Code

这个章节总结了多次踩坑的经验教训。任何新的推理代码都**必须**遵循以下规则。

### 1. Mask Pattern 必须匹配训练分布

**规则**：`src_mask` 的 pattern 必须是训练 7 种策略（M1-M7）能产生的 pattern。

**训练策略的粒度**：所有 7 种策略都在 `(T, 23)` joint-group grid 上操作，然后 `expand_grid_to_mask()` 扩展到 `(T, 135)`。即：
- Group 0 = translation (dims 0:3)，3 个 dim 同时 mask/keep
- Group 1-22 = 22 个关节（每个 6 dims rot6d），6 个 dim 同时 mask/keep
- **从不会出现同一关节内部分 dim mask、部分不 mask 的情况**

**常见错误**：
```python
# ❌ 错误：单独保护 translation dims，其他 dims mask
src_mask[:, :3] = 0.0   # translation observed
src_mask[:, 3:] = 1.0   # pose generated
# → 模型从未见过这种 per-dim 不一致的 pattern，会输出垃圾

# ✅ 正确：以 joint-group 为单位，整帧 mask 或整帧 keep
src_mask[ki] = 0.0       # keypose 帧：所有 135 dim 都 observed
src_mask[other] = 1.0    # 其他帧：所有 135 dim 都 generate
```

**如果需要保护 translation 但不保护 pose**：不要通过 mask 实现，用 **post-hoc 替换**：
```python
# 模型用 joint-group 粒度的 mask 正常推理
final = composite * (1 - mask) + model_output * mask
# 推理完成后，post-hoc 替换 translation
final[:, :, :3] = before_motion[:, :, :3]
```

### 2. src_motion 在 mask=1 区域必须置零（Completion 模式）

**规则**：`src_motion` 传入模型前，mask=1 区域必须为零。

```python
# ✅ 正确顺序
motion_norm = bundle.normalize_motion(motion)
vace_input = motion_norm * (1 - src_mask)  # 先零化 mask=1 区域
batch = {"src_motion": vace_input, "src_mask": src_mask, ...}
```

`prepare_vace_input()` 内部会构造：
- `inactive = src_motion * (1 - mask)` → mask=0 处有值，mask=1 处为零
- `reactive = src_motion * mask` → **completion 模式下必须全为零**

如果 mask=1 区域的 src_motion 没有置零，reactive 会泄露信息，模型会直接拷贝而非生成。

### 3. Replacement Guidance — Imputation Inference for _man Variants

**For `_man` (mask-aware noise) variants**, replacement guidance implements
train-consistent imputation. During training, `x_t[known] = x1` (clean),
so the pipeline:

1. Initializes `y0[known] = clean_motion` (not noise) — matching training
2. At each ODE step, replaces known regions with `clean_motion` — maintaining
   the clean signal the model expects to see in `x_t`

**Modes**: `'skip_last'` (recommended), `'all'`, or `'flow_interp'` (train-consistent).

- `skip_last`: Replace known regions with `clean_motion` every step except last
- `all`: Replace every step including last
- `flow_interp`: Replace known regions with flow-matching interpolation `(1-t)*z0 + t*clean_motion` — at each ODE time `t`, known regions follow the exact interpolation path the model was trained on. Yields ~40-60% boundary smoothness improvement over `skip_last` for MAN models.

**Requires**: `clean_motion` key in batch — full normalized motion `(B, T, D)`
**without** masked-region zeroing.

**For standard (non-MAN) variants**, replacement guidance has limited effect
because training uses uniform noise on all regions. Use `'none'`.

**Data flow for `_man` + `skip_last`**:
```python
# Caller prepares batch:
motion_norm_full = bundle.normalize_motion(motion[:T].unsqueeze(0))
motion_norm_zeroed = motion_norm_full * (1 - mask)  # for VACE inactive
batch = {
    "src_motion": motion_norm_zeroed,
    "src_mask": mask,
    "clean_motion": motion_norm_full,  # NOT zeroed
    ...
}

# Pipeline internally:
y0 = where(keep_mask, clean_motion, noise)    # t=0: known=clean
for step in ode_steps:
    v = model(t, x);  x = x + v * dt
    if not is_last_step:
        x = where(keep_mask, clean_motion, x)  # imputation
```

### 4. SDEdit — Removed (2026-04)

SDEdit (`sdedit_strength` parameter) has been removed from the pipeline.
It was never correctly implemented for the _man training distribution and
is conceptually redundant with imputation inference. The `_eval_globalrot_single_v3.py`
script implements its own denoise-from-near-clean logic outside the pipeline.

### 5. Keypose 动作编辑（Hybrid Blend + Boundary Polish）

**任务**：给定原动作 + 1-2 个 keypose 帧的目标姿态，修改原动作使其经过目标 keypose，同时保持动作自然。

**最优方案（2026-04 验证）**：

**Step 1 — Pure Correction Blend**（无模型）：
- 计算 keypose 校正向量 `correction = after[ki] - before[ki]`
- 双权重传播到全帧：temporal proximity（余弦衰减）+ pose similarity（周期动作支持）+ temporal smoothing
- 结果：全帧都接受了适度的 keypose 校正，效果已经很好

**Step 2 — Boundary-only Model Polish**（M2M 模型）：
- 在 blend 权重衰减到 0 的边界位置（±8帧窄带），用 M2M `skip_last` 生成平滑过渡
- 其余区域保持纯 blend 不动——模型只做边界平滑，不破坏 blend 质量
- 结果：比纯 blend smooth -14%, bnd_smooth -14%, foot -7%, global -3%

**为什么纯模型方案不如 Hybrid**：
- SDEdit from before：模型去噪后≈before，keypose 校正无法传播
- SDEdit from blended（全局）：模型偏离 blend 太多，反而破坏效果
- 全帧 imputation（local_edit/anchor_inbetween）：mask 边界跳变，比 blend 差
- **模型只适合做边界平滑，不适合做全局校正传播**

**评估脚本**：
```bash
# 纯 blend 基线
python3 scripts/run_pure_blend_baseline.py

# Hybrid 最优方案
python3 scripts/run_hybrid_blend_polish.py --gpu 0

# 网站查看
python3 motion_annot_web/keypose_eval/app.py --port 8080
```

**数据**：`data/PeacekeeperElite_MB/PeacekeeperElite_part4_{before,after}_MB/`（155 pairs）

**旧约束传导问题**（已被 Hybrid 方案替代）：如果只标记 1 帧为 observed（mask=0），模型的 141 帧中 140 帧都是 before_motion 的 pattern。模型在去噪时会把邻居帧恢复回 before。Hybrid 方案不依赖模型做校正传播，从根本上规避了此问题。

### 6. MoGenDIT vs M2M 的关键差异

| 方面 | MoGenDIT | HyMotion M2M |
|------|----------|-------------|
| **去噪框架** | DDPM | Flow Matching |
| **Mask 训练** | mask-aware noise（observed 帧不加噪） | 标准：uniform noise；`_man`：mask-aware flow matching |
| **Model 条件输入** | `x = cat([x_t, mask], dim=-1)` | `x = cat([x_t, inactive, reactive, mask], dim=-1)` |
| **Known 信息来源** | 模型从 x_t 中直接读（因为 observed 位置是 clean 的） | 标准：从 inactive channel 读；`_man`：x_t + inactive 双通道 |
| **Per-step replacement** | 非常有效（和训练一致） | 标准：效果有限；`_man`：有效（train-consistent imputation） |
| **Mask 粒度** | per-frame per-dim (201-dim) | per-frame per-joint-group (T×23 grid→T×135) |
| **表示维度** | 201-dim (pose + joint + trans) | 135-dim (trans + rot6d) |

### 7. 保存 NPZ 的 roundtrip 问题

`motion_135_to_npz` 将 rot6d→axis-angle→NPZ。这个转换在 axis-angle 角度接近 π 时有不连续性。

**规则**：保存后必须验证 roundtrip：
```python
saved = motion_135_to_npz(combined, ...)
reloaded = load_npz_as_motion(saved_path)
assert (combined - reloaded).abs().max() < 0.1  # roundtrip error 应该很小
```

如果输出帧数（T）大于原始数据帧数（orig_data["poses"].shape[0]），`full_poses[:, :66] = axis_angle` 必须用所有 T 帧，不能用 `T_save = min(T, orig_T)` 截断。

---

## Repair Pipeline Comparison: MoGenDIT ada_denoise vs HyMotion M2M (2026-04)

### Adaptive Mask (shared)

Both pipelines use the **same adaptive mask** computed by MoGenDIT's `compute_adaptive_mask`:
- `joint_mask (T, 22)`: per-joint per-frame, flagged if axis-angle change > 0.15 rad after light denoise
- `trans_mask (T,)`: per-frame, flagged if translation change > 0.05m after light denoise
- Percentile cap: if >15% of joints flagged, threshold raised; if >50% of frames have trans change, trans threshold raised

### ⚠️ Critical Difference: Translation Handling

**MoGenDIT `ada_denoise`** does NOT use the per-joint adaptive mask for imputation. Source: `motion_refiner.py` lines 326-373.

```
Phase 1: Standard denoise
  - mask = zeros(1, T, 201)
  - mask[:, :1] += 1          ← ONLY first frame is keep_mask=True
  - Everything else (including all translation) is freely regenerated

Phase 2: Compute change = |original - denoised|
  - high_change = change > 0.1 (per-dim threshold)
  - new_keep_mask = low_change regions (keep) + original mask

Phase 3: Re-denoise with new_keep_mask
  - Translation often has change > 0.1 → NOT protected → regenerated again
```

**HyMotion M2M `completion`** uses the full per-joint adaptive mask expanded to 135 dims:
```
mask_135 (T, 135):
  - col 0-2 (translation): masked by trans_mask per-frame
  - col 3+j*6 to 3+j*6+5 (rot6d per joint): masked by joint_mask[:, j]

Imputation (replacement_guidance='skip_last'):
  - Every ODE step: x_t[mask=0] = clean_motion[mask=0]
  - trans_mask=0 frames → translation preserved exactly
```

### Result

| Aspect | MoGenDIT `ada_denoise` | HyMotion M2M `completion` |
|--------|----------------------|--------------------------|
| **Translation preservation** | ❌ Regenerated (only first frame protected) | ✅ Preserved where `trans_mask=0` |
| **Mask granularity for imputation** | Per-dim change threshold (0.1), not per-joint adaptive mask | Per-joint adaptive mask expanded to 135-dim |
| **Imputation protocol** | DDIM skip_last (obs_mask from change analysis) | ODE skip_last (obs_mask from adaptive mask) |
| **Eval success rate** | 72.1% (111/154) | 62.3% (96/154) |
| **Best at** | knee_x, small_wobble | arm_penetration, spine |
| **Worst at** | joint_twist, arm_penetration | jitter |

### Implication

When comparing repair results visually, MoGenDIT will show **different translation trajectories** from original even in unmasked frames. This is NOT from the adaptive mask — it's because MoGenDIT's `refine(mode='ada_denoise')` only protects the first frame during denoise. The adaptive mask is only used for the **mask computation phase** (comparing original vs denoised), not as the actual imputation mask during repair.

To make MoGenDIT respect the adaptive mask for imputation, one would need to modify `refine()` or use `impute_with_obs_mask()` instead, passing the full `(T, 201)` adaptive mask as `obs_mask`.

---

## Historical Bug Record

### 2026-03-27: Bundle-level Parameters not trained, not saved, not synced (FRAMEWORK BUG)

**Severity**: Critical — affected ALL bundles with direct `nn.Parameter` or `register_buffer`.

**Root cause**: `ModelBundle.trainable_parameters()` only iterated `_trainable_modules` (registered sub-modules like `motion_transformer`). Direct bundle attributes like `null_vtxt_feat`, `null_ctxt_input` (nn.Parameter) and `mean`, `std` (register_buffer) were invisible to:
1. **Optimizer** — never trained (stayed at pretrained init values)
2. **Checkpoint save** — lost on save (not in `model.pt`)
3. **Checkpoint load** — re-initialized randomly on each load
4. **DDP gradient sync** — gradients not all_reduced across ranks

**Symptom**: M2M inference produced completely invalid motion (rot6d norm 4-5 instead of ~1.0, output range [-10, 13] instead of [-1, 1]). The `null_vtxt_feat` parameter, used as unconditional text embedding in every inference call, was randomly initialized on each load instead of using the T2M pretrained value.

**Debugging path**: training loss=0.015 in log but model.pt weights gave loss=25 when loaded → discovered model.pt and model.safetensors were identical (302/302 keys match) → realized `null_vtxt_feat` was **not** in either file → traced to `_save_ckpt_modules` only saving sub-module state_dicts → confirmed `trainable_parameters()` also excluded bundle-level params.

**Fix** (4 files):
- `base_model_bundle.py`: `trainable_parameters()` + `trainable_named_parameters()` now include `self.named_parameters(recurse=False)`; `state_dict_to_save()` saves `'__bundle_params__'`; `load_state_dict_selective()` restores `'__bundle_params__'`
- `accelerate_runner.py`: `_state_dict_to_save()` saves `'__bundle_params__'`; `_sync_orphan_param_grads()` all_reduces bundle-level param gradients after backward; both train loops call it
- `hymotion_m2m/bundle.py`, `hymotion_t2m/bundle.py`, `hymotion_umo/bundle.py`: null embeddings init changed from `torch.randn` to `torch.zeros`; M2M/UMO null text embeddings frozen (`requires_grad=False`) since they should keep T2M pretrained values

**Affected bundles**: HyMotionM2MBundle, HyMotionT2MBundle, HyMotionUMOBundle (null_vtxt_feat, null_ctxt_input, mean, std), PrismBundle, PrismMCMBundle (buffers).

**Backward compatibility**: old checkpoints (without `'__bundle_params__'`) load normally. For M2M inference with old checkpoints, `null_vtxt_feat` must be loaded from T2M pretrained checkpoint (`checkpoints/HY-Motion-1.0/HY-Motion-1.0-Lite/latest.ckpt`) as fallback.

### 2026-03-25: VACE reactive leaked target motion

**Root cause**: `PrepareM2MUniversalMask` returned full `src_motion` (mask regions not zeroed), trainer passed it directly to `prepare_vace_input`. `reactive = src_motion * mask` contained target values.

**Impact**: Training loss artificially low (~0.0003), model learned to copy from reactive. All checkpoints trained with this bug must be retrained.

**Fix**: Trainer now does `src_motion = src_motion * (1 - src_mask)` after normalize for Completion. Old repo's `build_src_mask()` already zeroed mask regions.

### 2026-03-23: VACE operator precedence bug

**Root cause**: `inactive = src_motion * 1 - src_mask` (Python: `*` binds tighter than `-`) instead of `src_motion * (1 - src_mask)`.

**Impact**: All old repo (`hymotion_1.0_train`) M2M checkpoints produce divergent ODE output. Not recoverable, must retrain.

### 2026-03-23: Rotation 6D convention mismatch

**Root cause**: `axis_angle_to_rotation_6d()` outputs column-major, but training data uses row-major.

**Fix**: `load_smplx.py` line 93: `out = out[:, :, [0, 3, 1, 4, 2, 5]]`.

### 2026-03-23: Missing motion normalization in trainer

**Root cause**: Old repo normalizes in dataset, new repo doesn't — but trainer also didn't.

**Fix**: `HyMotionM2MTrainer.train_step()` calls `bundle.normalize_motion()` before `prepare_padding()`.

### 2026-03-23: Translation type vs Mean/Std mismatch

**Root cause**: M2M uses `transl_type='abs'`, but config was temporarily changed to `'rel'`.

**Fix**: All M2M configs use `transl_type='abs'`.

---

## Related Documentation

- MoGenDIT integration: see `ref_repo/MoGenDiT/CLAUDE.md`
- KIMODO/UMO baseline comparison: see `ref_repo/CLAUDE.md`
- Physics RL enhancement: see `docs/design/physics_rl_motion.md`
- Ablation experiments: see `ref_repo/m2m_ablation_experiments.md`
- **Canonical Pose OOD in Transition**: see `docs/temp/m2m_canonical_ood_solution.md` — runtime canonicalization + training augmentation for transition tasks
- **SOAR method analysis**: see `ref_repo/SOAR/CLAUDE.md` — SOAR 论文分析 + 对 M2M 的适用性论证
- **SOAR implementation plan**: see `docs/temp/soar_m2m_v2_post_training_plan.md` — 完整设计方案（算法、超参、ablation、compute budget）

> Note: All temporary documents, solution proposals, evaluation plans should be stored in `docs/temp/`.
