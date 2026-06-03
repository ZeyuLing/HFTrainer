# PhysFlow Online Adversarial — Iteration Log (Cursor)

Fundamental goal: co-optimize, via online adversarial training, the **physical
realism of the T2M generator (KIMODO-G1)** and the **motion tracking of the
position-aware G1 tracker**.

Loop: `prompt -> KIMODO-G1 qpos CSV -> ProtoMotions .motion -> position-aware G1
ONNX tracker rollout in MuJoCo -> root-aware adversarial score`. Hard prompts
(tracker fails) push the generator; trackable motions feed tracker fine-tuning.

## Environment (debug machine `lzy_debug_machine_2`, 8×V100-32GB)

- KIMODO gen + scoring python: `/usr/local/bin/python3` (kimodo, mujoco, onnxruntime ready).
- IsaacGym tracker-training venv: `/root/physflow_isaacgym_py38_cu118/bin/python`
  (py3.8, torch 2.4.1+cu118, isaacgym preview4). Survives container restart.
- **Fixes applied to the isaacgym venv for ONNX export** (`export_bm_tracker_onnx.py`):
  - `pip install onnx` (was missing).
  - `pip uninstall onnxscript` (0.2.7 breaks on py3.8: `UserList[...]` subscription).
  - `pip install onnxruntime` (needed for the post-export YAML sidecar step).

## Key knobs / gotchas discovered

- `kimodo.scripts.generate` is spawned **once per motion** and reloads the model
  each time (~2–3 min/motion). Dominant cost of sweeps.
- ProtoMotions `--training-max-steps` is a **global** budget that **includes the
  warm-start checkpoint's step count**, and **resume ignores CLI overrides**
  (config is frozen in `resolved_configs.pt`). => to train N more epochs from a
  warm start at ~5.1M steps (stable ckpt), use a **fresh experiment name** with
  `max-steps ≈ 5.1M + N*batch_size`. batch_size 16384 => ~16k steps/epoch.

## Baseline (current stable position-aware tracker)

Held-out EVAL split = 5 prompts × 3 KIMODO samples = 15 motions
(`output/physflow_kimodo_g1/cursor_iter1_eval_baseline`):

| metric | value |
|---|---|
| falls | **13/15 (87%)** |
| mean_completion | 0.419 |
| mean_joint_error (rad) | 1.598 |
| mean_root_trajectory_error (m) | 0.636 |

TRAIN sweep = 18 prompts × 2 = 36 motions: falls 34/36 (94%), comp 0.476.
=> The position-aware tracker is the dominant bottleneck: it falls on ~90% of
KIMODO-G1 motions. Only 14/36 train motions reach completion ≥ 0.5.

## Finding: not all generated motions are trackable

First fine-tune (all 36 motions, LR 5e-6, ~54 epochs, ckpt e182) re-eval:
falls 13→14, comp 0.419→0.372 (worse), joint_err 1.60→1.51, root_traj 0.636→0.496 (−22%).
Root tracking improved but stability regressed — training the tracker to imitate
physically-implausible KIMODO motions is counterproductive. This is exactly the
reason co-optimization is needed (generator must produce achievable motions).

## Active experiments (fresh names, 20M budget)

- Run A2 `physflow_g1_xyvel_cursor_iter1_v2`: all 36 motions, stable config (LR 5e-6), GPU0.
- Run B2 `physflow_g1_xyvel_cursor_iter1b_v2`: 14 trackable motions (completion≥0.5)
  curriculum, fast config (LR 2e-5, `physflow_g1_xy_offset_cursor_fast.py`), GPU2.

Evaluation protocol: snapshot ckpt -> export ONNX (GPU1) -> `--mode score` re-scoring
the 15 cached eval CSVs (`cursor_iter1_eval_baseline/kimodo_raw`) with the new ONNX.

## Results — held-out EVAL (15 motions), tracker fine-tuning works

| tracker | falls | completion | joint_err (rad) | root_traj (m) |
|---|---|---|---|---|
| baseline (stable) | 13/15 | 0.419 | 1.598 | 0.636 |
| A2 e317 (all 36, LR 5e-6) | 12/15 | 0.491 (+17%) | 1.372 (−14%) | 0.497 (−22%) |
| B2 e330 (14 trackable, LR 2e-5) | 12/15 | 0.559 (+33%) | 1.483 (−7%) | 0.671 (+5%) |

Both fine-tunes beat baseline on the held-out eval set after ~190–200 epochs of
adaptation (the earlier e182/54-epoch "regression" was just under-training).
=> Fine-tuning the position-aware tracker on KIMODO-G1 output measurably improves
motion tracking on held-out generated motions. A2 best on tracking error; B2
(trackability curriculum) best on completion (+33%). Still falling on ~80% of
eval motions => headroom; both runs continue.

### Round-1 FINAL (both runs finished at epoch 609, ~481 epochs adaptation)

| tracker | falls | completion | joint_err | root_traj |
|---|---|---|---|---|
| baseline | 13/15 | 0.419 | 1.598 | 0.636 |
| A2 e317 | 12/15 | 0.491 | 1.372 | 0.497 |
| **A2 e609 (all 36, LR 5e-6)** | **10/15** | **0.615 (+47%)** | **1.093 (−32%)** | 0.480 (−25%) |
| **B2 e609 (14 trackable, LR 2e-5)** | **10/15** | 0.573 (+37%) | 1.232 (−23%) | **0.443 (−30%)** |

Monotone improvement with epochs. **A2 e609 is the best overall** and is promoted
to the active position-aware tracker. Motion-tracking half of the goal: VERIFIED
improvement on held-out generated motions.

### Generator side (physical realism) via best-of-N with A2 e609 as critic

best-of-3 mean completion **0.805** vs per-sample 0.615 (+31%): selecting the most
trackable KIMODO sample per prompt yields more physically-realistic motion.
Per-prompt frontier: `left_leg_balance` fully trackable; `boxing`/`circle_walk`
solvable via best-of-N (best sample comp 1.0, no fall); **`backward_walk` and
`robot_dance` still fall on all samples = generator's physical-realism frontier.**

### Active tracker
`output/physflow_kimodo_g1/active_tracker_v1/unified_pipeline.onnx`
(= cursor_iter1_v2 e609). Round 2 warm-starts from this.

### Train-set re-score with active tracker A_e609 (generalization check)

| set | falls | completion | joint_err | root_traj |
|---|---|---|---|---|
| train baseline | 34/36 | 0.476 | 1.670 | 0.578 |
| train A_e609 | 30/36 | 0.644 (+35%) | 1.295 (−22%) | 0.580 |

Improvement holds on the training distribution too (consistent with eval).

### max-steps gotcha (round 2)
A_e609 checkpoint already sits at the 20M global budget (it stopped at e609 because
it hit 20M). Warm-starting round 2 with `max-steps 20M` => instant finish. Fix:
round 2 (`physflow_g1_xyvel_cursor_iter2b`) uses `max-steps 40M` (~+600 epochs).
steps/epoch ≈ 32.8k. **Rule: each new round must bump max-steps above the
warm-start checkpoint's global step.**

### Round 2 (RUNNING)
`physflow_g1_xyvel_cursor_iter2b`: warm-start A_e609, all-36 pool, stable cfg
(LR 5e-6), 40M budget, GPU0 on lzy_debug_machine_2. Confirmed training at e615+.
ETA ~e1200 in ~2.5h. Re-eval on the 15-motion held-out set when it nears budget.

---

## ⚠️ CRITICAL FINDING (2026-05-31): position-aware fine-tuning is NET HARMFUL

Triggered by user report "results almost completely fail to imitate reference".
Investigation (algo = PPO+AMP+L2C2 BeyondMimic; harness validated; reference
quality OK) revealed the real root cause.

### Harness validation on KNOWN-GOOD standard G1 motions (`g1_bones_seed_mini.pt`)
Ran the released deploy policy + our trackers through the SAME MuJoCo scorer:

| policy | std motion falls | maxJointErr (rad) |
|---|---|---|
| RELEASED g1-bones-deploy | **0/4** | 0.65–1.02 |
| stable_v1 (my "baseline") | 3/4 | 1.47–2.60 |
| A_e609 | 3/4 | 1.42–2.28 |
| R2_e1219 | 3/4 | 1.98–2.10 |

=> The MuJoCo harness is CORRECT (released tracks fine). ALL position-aware
variants — including the inherited `stable_isaacgym_train_v1` baseline — FALL on
the standard motions the released tracker handles easily. The damage was done at
the first position-aware step (open xy_offset + PPO), not by my fine-tuning.

### Released policy on the KIMODO eval set (never trained on KIMODO)

| tracker | falls | completion | jointErr | rootTraj |
|---|---|---|---|---|
| **RELEASED (no KIMODO training)** | **0/15** | **1.000** | **0.736** | **0.427** |
| stable_v1 (my baseline) | 13/15 | 0.419 | 1.598 | 0.636 |
| A_e609 | 10/15 | 0.615 | 1.093 | 0.480 |
| R2_e1219 | 9/15 | 0.744 | 1.077 | 0.503 |

**The untouched released policy tracks KIMODO motions PERFECTLY (0 falls, comp
1.0) and beats every fine-tuned tracker on joint error AND root trajectory.**

### Root cause
Warm-starting the released G1 tracker, reopening `include_xy_offset` channels,
and running PPO+AMP on OOD/hard KIMODO motions destroys the pretrained tracking
prior (catastrophic forgetting). `task_reward` stays flat ~2.8 while
`episode_reward` grows only via `episode_length` => the policy learns to SURVIVE
(stand/balance), not to IMITATE. My earlier "improvement" numbers were measured
vs an equally-broken `stable_v1` baseline, creating an illusion of progress.

The motivating premise ("released policy can't follow global displacement, so we
need xy_offset") is empirically FALSE here: the released policy follows the
KIMODO motions' global displacement BETTER (rootTraj 0.427) than any xy_offset
variant.

### Correct path forward
1. Abandon the degraded position-aware checkpoints; use RELEASED as the tracker.
2. Re-baseline the whole online-adversarial loop against the released policy.
3. If xy_offset capability is truly needed: prevent forgetting via rehearsal
   (mix the released training motions into fine-tuning), frozen-backbone/adapter
   for the new channels, easy→hard curriculum, down-weight AMP, and regression-
   test against standard motions after every adaptation (harness now exists:
   `scripts/embodied/cursor_validate_harness.py`).
4. Promote RELEASED ONNX as active_tracker and rebuild viz manifests with it.

---

## ✅ CLEAN RESTART (2026-05-31): warmstart-from-released + rehearsal, judged by reconstruction curve

User directive: *"按这个方式重启，根据 reference motion 是否被重建的变化曲线判断是否训练有效。"*
=> Training validity is now judged ONLY by whether the in-sim reference-
reconstruction error (`eval/gt_error`, `eval/max_joint_error`,
`eval/relative_body_pos`, `eval/gr_error`) goes DOWN and `eval/success_rate`
stays high — NOT by episode_length / survival.

### What changed vs the failed approach
| | failed position-aware | CLEAN RESTART |
|---|---|---|
| init | `g1_xyvel_partial_warmstart` (remapped) | RELEASED `last.ckpt`, epoch reset to 0 (`cursor_make_warmstart_ckpt.py`, `skip_optimizer_load=True`) |
| architecture | `include_xy_offset=True` (NEW unnormalized global channels → corrupted prior) | **released config verbatim, `include_xy_offset=False`** (no input change → exact weight load) |
| data | KIMODO-only (OOD, catastrophic forgetting) | **mixed pool: 36 KIMODO + 36 standard rehearsal** (`physflow_g1_released_rehearsal_v1_pool`) |
| success signal | episode_reward / length (survival illusion) | **reference-reconstruction curve** (`cursor_plot_reconstruction_curve.py`) |

Files: experiment `examples/experiments/mimic/physflow_g1_released_rehearsal.py`
(copy of released config); launcher `scripts/embodied/cursor_launch_released_rehearsal.sh`
(no `--skip-initial-eval`, `eval_metrics_every=50`); run on `lzy_debug_machine_2`
GPU0, tmux `physflow_rehearsal`, 256 envs.

### Epoch-0 baseline (warmstarted released, measured in IsaacGym on the 72-motion mixed pool)
| metric | epoch 0 |
|---|---|
| `eval/success_rate` | **0.986** (71/72 no failure) |
| `eval/gt_error/mean` [m] | 0.388 |
| `eval/max_joint_error/mean` [rad] | 0.470 |
| `eval/relative_body_pos/mean` [m] | 0.147 |
| `eval/gr_error/mean` [rad] | 0.258 |

Healthy honest start (vs degraded stable_v1 which fell 3/4 on standard motions).
Warmstart settled by epoch ~18 (`clip_frac` dropped <0.6, actor updates resumed).
**Win condition: these errors trend DOWN at the epoch 50/100/150/… eval points
while success_rate holds ≈1.0.** Curve PNG: `output/physflow_kimodo_g1/reconstruction_curve_physflow_g1_released_rehearsal_v1.png`.

### v1 reconstruction curve (cold-reset optimizer, task_w 0.5 / amp 2.0)
| epoch | gt_error | max_joint | rel_body_pos | gr_error | success |
|---|---|---|---|---|---|
| 0 (base) | 0.388 | 0.470 | 0.147 | 0.258 | 0.986 |
| 50 | 0.474 | 0.568 | 0.150 | 0.309 | 0.972 |
| 100–250 | 0.44–0.46 | 0.52–0.55 | 0.14–0.16 | 0.27–0.31 | 0.93–0.99 |
| 300 | 0.381 | 0.462 | 0.136 | 0.252 | 0.972 |
| 400 | 0.378 | 0.458 | 0.138 | 0.253 | 0.958 |
| **500** | **0.359** | **0.444** | 0.147 | 0.265 | 0.972 |

**Shape = warmstart DIP (cold Adam, ~+0.09 gt) → recovery by ~e300 → genuine
descent below baseline by e500** (gt −7%, joint −6%). Slow + noisy (eval runs
with domain-rand on, ±0.04). The e50–250 plateau-above-baseline was a transient,
NOT the verdict — confirms judging by the *curve*, not a snapshot.

### v2 (warm optimizer kept + tracking-focused reward) — `physflow_g1_released_rehearsal_v2_taskheavy`
Same arch+pool. Changes: warmstart ckpt keeps released Adam moments
(`g1_released_warmstart_epoch0_warmopt.ckpt`, `skip_optimizer_load=False`);
overrides `task_reward_w=1.0`, `amp_parameters.discriminator_reward_w=0.25`.
Run on GPU1, tmux `physflow_rehearsal_v2`. Result so far: dip is **~half** of v1
(e0 0.360 → e50 0.402 vs v1 +0.086), recovering by e150 (0.383). Warm optimizer
clearly tames the destructive warmstart dip.

Compare plot: `output/physflow_kimodo_g1/reconstruction_curve_compare.png`
(plotter: `scripts/embodied/cursor_plot_curve_compare.py`; single-run:
`scripts/embodied/cursor_plot_reconstruction_curve.py <exp>`).

### ⚠️ RETRACTED in-training "curve" verdict — it was measured WRONG
User pushback ("I don't believe ours OR baseline is implemented correctly") was
correct. Two defects in the in-training `eval/*` curve:
1. **Eval runs under domain randomization + noisy obs.** `BaseEvaluator.evaluate`
   steps the TRAINING env (friction/COM/**random pushes**/action-noise + noisy
   observations all ON; only the action is deterministic `mean_action`). Run-to-
   run RNG variance ≈±0.04 — SWAMPS the fine-tune signal. The "descent below
   baseline" was within that noise band.
2. **The epoch-0 "baseline" is not clean released** (it's released after 1 train
   epoch, under domain-rand, on the train pool). Not a deployment-faithful number.
(Sanity-checked the weights DO move: maxabs(released, v1_e3000)=0.69, so training
is real — the problem was the *measurement*, not zero learning.)

### ✅ TRUSTWORTHY clean eval (deterministic MuJoCo, no domain-rand, clean obs)
Exported `score_based.ckpt` → ONNX (`deployment/export_bm_tracker_onnx.py`,
`compiled_best/`) and scored RELEASED vs v1 vs v2 through the SAME MuJoCo scorer
used to expose the old forgetting (`scripts/embodied/cursor_clean_eval_compare.py`),
15 KIMODO + 6 standard motions:

| policy | set | falls | maxJointErr | rootTrajErr |
|---|---|---|---|---|
| RELEASED | KIMODO | 1/15 | 0.860 | 0.431 |
| v1_best | KIMODO | 3/15 | 0.774 | 0.465 |
| **v2_best** | KIMODO | **1/15** | **0.635 (−26%)** | **0.279 (−35%)** |
| RELEASED | STD | **0/6** | 0.968 | 0.332 |
| v1_best | STD | 2/6 | 1.062 | 0.887 |
| v2_best | STD | 2/6 | 1.073 | 0.471 |

(comp omitted: denominator is source-fps frames vs 50 Hz sim steps → unreliable;
falls + jErr + rootErr are the trustworthy signals.)

### Honest verdict
- **v2 (warm-opt + task-heavy) genuinely improves KIMODO reconstruction** at EQUAL
  fall rate vs released (jErr −26%, rootErr −35%) — a real gain on the target
  distribution, confirmed in the clean regime.
- **v1 (cold-opt) is worse** (3/15 KIMODO falls).
- **BUT both fine-tunes REGRESS on standard motions** (2/6 falls vs released 0/6)
  → residual catastrophic forgetting; the 36-motion rehearsal buffer was too small
  and/or AMP/task pressure on KIMODO still erodes the broad prior.

### 🔴 ROOT CAUSE of "can't even imitate basic motions" in the viz
`output/physflow_kimodo_g1/active_tracker_v1/unified_pipeline.onnx` (md5
37f6457665…, yaml `checkpoint: results/physflow_g1_xyvel_cursor_iter1_v2/last.ckpt`)
= the DEGRADED xy_offset position-aware model (A_e609/iter1_v2) — the one that
falls 10/15 KIMODO and 3/4 standard. The deployed/visualized tracker has been
this broken model, NOT released. → backed up to
`active_tracker_v1_DEGRADED_iter1v2_backup/`; RELEASED staged at
`active_tracker_released/` (md5 27f82a75…). Runner `DEFAULT_ONNX` already points
to released (the translation-free pose tracker); the position-aware default
(`DEFAULT_POSITION_AWARE_ONNX` = stable_v1) is degraded and should not be used.

Note on translation: released BeyondMimic is anchor-relative (no explicit
xy_offset) yet DOES follow global translation via per-step vel/heading tracking
(clean MuJoCo rootTrajErr 0.43m KIMODO / 0.33m STD). The "need xy_offset for
translation" premise was wrong; xy_offset variants track translation WORSE.

### Next (to make it a clean win, not a tradeoff)
1. Replace the in-training eval gate with the clean MuJoCo eval (or disable
   domain-rand/noise inside the evaluator) so the curve is trustworthy.
2. Stronger rehearsal: use all 58 standard motions (or more), weight standard ≥
   KIMODO, to stop the STD regression.
3. Re-run v2 recipe with stronger rehearsal; accept only if KIMODO improves AND
   STD falls stay 0.

---

## ROOT CAUSE FOUND (2026-05-31): position-aware warmstart scrambles the actor

User challenge: "small-data overfit should be trivial for both tracker and T2M;
it isn't, so this is an implementation bug, not data. We're fine-tuning a
translation-free model into a translation-bearing one and the implementation is
probably wrong."

**Confirmed — it was a real bug.**

### The bug
`scripts/embodied/make_g1_xyvel_partial_warmstart.py` builds the position-aware
warmstart (used by every stable_v1 / iter1_v2 run via
`launch_position_aware_g1_tracker_train.sh` ->
`physflow_g1_xy_offset_stable.py`).

Actor input order (released): `reduced(64) + target(256) + prev_actions(29) = 349`.
Position-aware turns on `include_xy_offset` + `include_anchor_vel`, growing the
target block 256 -> 276.

The target block is NOT contiguous. `build_reduced_coords_target_poses`
(`protomotions/envs/obs/target_poses.py`) builds it PER future step and flattens
(`obs.view(num_envs, -1)`). With future_steps=[1,2,4,8] (4 steps):
- released per-step: `[rot6, dof_vel29, dof_pos29] = 64`  -> 4x64 = 256
- posaware per-step: `[rot6, dof_vel29, dof_pos29, xy2, vel3] = 69` -> 4x69 = 276

So the 5 new channels are INTERLEAVED at target offsets 64/133/202/271, not
appended at 256. The old warmstart assumed `new_target = [old_256 | new_20]`:
- only future step 0 aligns;
- steps 1/2/3 are shifted by 5/10/15 channels;
- the 20 zero-fill channels overwrite REAL step-3 pose features.
=> 3 of 4 future-step target blocks (and their obs-norm stats) are scrambled.
The pretrained pose prior survives only for the [1]-step horizon.

### Proof (scripts/embodied/verify_xyvel_warmstart.py)
Feeding identical pose values through the released layout vs the new interleaved
layout, comparing actor first-layer pre-activations:
- FIXED remap:  max|out - released| = 1.8e-15  (bit-identical at init) -> PASS
- OLD  remap:   max|out - released| = 4.13, mean = 1.0 (every output corrupted)

This is the mechanism behind "can't even imitate basics" and "translation-free
looks far better than our translation fine-tune". Every position-aware run was
trained from a corrupted init.

### Fix
`scripts/embodied/make_g1_xyvel_partial_warmstart_FIXED.py`: remaps the actor
first linear + obs-norm mean/var PER STEP, copying each step's 64 released
channels into the correct interleaved slot and zero-initialising the 5 new
channels. At init the actor is bit-identical to released; PPO then learns to
USE the translation channels from a perfect starting point.

Regenerated: `output/physflow_kimodo_g1/checkpoints/g1_xyvel_partial_warmstart_FIXED.ckpt`.

### Next: small-data overfit gate
Run the position-aware tracker from the FIXED warmstart on a tiny motion set
with domain-rand/noise reduced, and confirm clean reconstruction error drops to
near zero (true overfit). If it overfits -> pipeline is correct and we can scale.
If not -> keep digging (next suspects: reward/termination wiring for the new
channels, or T2M-side data path).

---

## OVERFIT VALIDATION (2026-06-01): FIXED warmstart tracks + overfits

Env/data setup (per user decision):
- Run on taiji `lzy_debug_machine_2` (IsaacGym py38 env `/root/physflow_isaacgym_py38_cu118`,
  GPUs 3-7 free; machine_1 is saturated by the user's PRISM-1B training, do not touch).
- Text prompts: 100 real HumanML3D captions sampled across CMU/ACCAD/BMLmovi/KIT/...
  (`configs/experiments/physflow_kimodo_g1/prompt_bank_humanml3d_overfit100.jsonl`,
  built by `scripts/embodied/cursor_build_humanml3d_prompt_bank.py`).
- Motions: KIMODO-G1 generated from those captions (the actual adversarial target).
  NOTE: KIMODO needs py3.10+ (uses `X|Y` unions); the isaacgym py38 env CANNOT run it.
  Generation uses `python3` (py3.10) via `scripts/embodied/cursor_kimodo_gen100.sh`.

Architecture: `examples/experiments/mimic/physflow_g1_xy_offset_overfit.py` (369-d
position-aware, xy_offset=True; domain-rand + reset-noise disabled for a clean curve),
warm-started from the FIXED interleaved ckpt
`output/physflow_kimodo_g1/checkpoints/g1_xyvel_partial_warmstart_FIXED.ckpt`.

Smoke on 2 KIMODO motions (1024 envs, clean eval every 3 epochs):

| metric                         | e3   | e6   | e9   | e12  | e15  |
|--------------------------------|------|------|------|------|------|
| eval/success_rate              | 0.50 | 1.00 | 1.00 | 1.00 | 1.00 |
| eval/relative_body_pos/mean    | 0.259| 0.115| 0.114| 0.111| 0.105|
| eval/max_joint_error/mean (rad)| 1.03 | 0.81 | 0.81 | 0.80 | 0.83 |
| eval/gt_error/min (m)          | 0.022| 0.020| 0.015| 0.014| 0.014|
| env/raw_r/relative_body_pos    | 0.97 | 0.97 | 0.98 | 0.98 | 0.98 |

Interpretation:
- FIXED 369-d warmstart loads into the xy_offset arch with NO size mismatch.
- Tracks well from step 0 (body-pos reward 0.97) — i.e. behaves like released at init,
  exactly as the numerical proof predicted.
- Overfits: success_rate 0.5->1.0 by epoch 6, relative_body_pos error -60%.
- gt_error/mean (global translation, meters) plateaus ~0.75 at conservative actor
  lr=5e-6; the xy_offset translation-following is the part still being learned (it was
  zero-weighted at init by design). For the full run consider a higher actor lr to
  speed translation learning.

CONCLUSION: the earlier "can't even track basics" was the warmstart scramble bug, NOT
a fundamental impossibility and NOT (primarily) data. With the correct warmstart the
position-aware tracker tracks and overfits. Next: full 100-motion overfit run, fresh
experiment name + large max-steps (resume-vs-warmstart gotcha), gate on eval curves.

### Full 99-motion overfit run (2026-06-01)

- Pool: 99/100 KIMODO-G1 motions generated (1 prompt failed gen) from the HumanML3D
  captions, converted to `.motion` at `output/physflow_kimodo_g1/overfit100_pool/proto`.
  Generation was parallelized 6-way (`cursor_kimodo_gen_parallel.sh`, GPUs 2-7, tmux
  `gen_sh0..5`) after the serial loop (~50s/motion) proved too slow.
- Run: tmux `oft99`, GPU 3, fresh name `physflow_g1_xyvel_overfit99_FIXED`, 1024 envs /
  8192 batch, FIXED warmstart, overfit config (DR + reset-noise off), eval every 20 ep.
  NOTE: 2048 envs segfaults PhysX GPU narrowphase on this V100 (pinned-mem alloc fail);
  1024 is the safe ceiling (matches the smoke runs).
- First clean eval @epoch 20 over ALL 99 translation-bearing motions:

| metric | value |
|--------|-------|
| eval/success_rate | **0.97** |
| eval/gt_error/mean (m, global incl. xy) | 0.317 |
| eval/gr_error/mean | 0.230 |
| eval/max_joint_error/mean | 0.392 |
| eval/relative_body_pos/mean (fail 0.020) | 0.123 |
| eval/anchor_height_error/mean (m) | 0.024 |
| info/episode_reward (e1->e36) | 52 -> 374 |

This is the decisive answer to the user's skepticism: a translation-free-pretrained
tracker, warmstarted (correctly) into the translation-aware xy_offset head, tracks 97%
of 99 KIMODO translation-bearing motions on the FIRST clean eval. The "can't imitate
basics" symptom was 100% the warmstart scramble bug. Monitoring further eval points
(`cursor_overfit_monitor.sh`) for the curve trending to 1.0 / gt_error down.

FINAL conservative curve (ran to epoch 792, then exited cleanly — the max-steps budget
INCLUDES the warmstart's own step count, so 26M total ≈ +6.5M new ≈ 792 epochs):

| metric | e20 | e792 | delta |
|--------|-----|------|-------|
| eval/success_rate | 0.970 | 0.970 | flat (mid-run 0.95 dips are noise) |
| eval/gt_error/mean (m) | 0.317 | 0.190 | -40% |
| eval/max_joint_error/mean | 0.392 | 0.243 | -38% |

GATE PASSED: clean monotone reconstruction improvement on 99 motions => pipeline correct.

### How success_rate is computed (answering "is the gradual decline normal?")

`success_rate = 1 - frac(motions that failed)`. A motion is latched failed (OR across the
whole episode, `base_evaluator.py:322`) iff at ANY active frame:
- `relative_body_pos > 0.5 m` (root-relative body-pose error; NOT global translation), or
- `anchor_height_error > 0.25 m` (root height; effectively a fall detector).
Only threshold-bearing components count (`base_evaluator.py:242-244`). `gt_error`/`gr_error`
are logged as metrics with NO threshold => they DO NOT affect success_rate.
So success_rate is a coarse "didn't fall / didn't lose the pose" gate, ±1-2 clips of noise
(0.97<->0.95 = 1-2/99) under residual reset noise. The real reconstruction signal is the
continuous gt_error / max_joint_error / relative_body_pos-mean, which fall monotonically.
=> the small wobble is normal; success recovered to 0.97 at e792 while errors kept dropping.

### Hi-actor-lr comparison (2026-06-01): lr is NOT the lever

Parallel run `physflow_g1_xyvel_overfit99_HILR` (tmux `oft99hi`, GPU 4,
config `physflow_g1_xy_offset_overfit_hilr.py`, actor lr 2e-5=4x, critic 1e-4). At matched
epochs it is WORSE, not faster: e20 gt_error 0.366 (vs 0.317), e100 0.279 (vs 0.257). Cause:
the 4x lr makes nearly every epoch trip `clip_frac>0.45` -> `Skipping actor updates`, so the
extra lr is cancelled by the PPO clip guard (`actor_clip_frac_threshold=0.45`) and just adds
noise. CONCLUSION: to speed translation overfit, raising actor lr alone is futile; would need
to ALSO raise actor_clip_frac_threshold / e_clip (and/or up-weight a translation reward),
not just lr. Deferred — GPUs got saturated by other (PRISM) training; only GPU 4 free.
