# HumanML3D-263 to SMPL Retargeting Pipeline

This note documents the current conversion used for HumanML3D-263 baseline
outputs, including FlowMDM and MotionLab, when we need SMPL-style motions or
MotionStreamer-272 evaluation.

## Scope

The conversion is generic for any un-normalized HumanML3D-263 sequence with
shape `(T, 263)` and the standard HumanML3D feature layout.  It is not tied to
FlowMDM or MotionLab.  The only model-specific requirement is that the baseline
output must already be in the HumanML3D-263 physical feature scale.  If a model
stores normalized features, pass the matching `Mean.npy` and `Std.npy` through
`--input-normalized`.

The conversion is approximate rather than exact.  HumanML3D-263 provides a
22-joint kinematic signal, but it does not uniquely determine SMPL twist,
shape, mesh details, or pose-blendshape deformation.  The retargeter therefore
saves an IK fit error (`fit_mpjpe_mm`) for every output clip.  This number is
the main quality diagnostic.

## Stage A: HML263 to SMPL-style `motion_135`

Script:

```bash
python3 scripts/eval/hml263_to_smpl_ik.py \
  --in-dir outputs/evaluation/humanml3d_hml3d263_fixed_stats/flowmdm \
  --out-dir outputs/evaluation/humanml3d_smpl135_fixed_stats/flowmdm \
  --model-dir ref_repo/MDM/body_models \
  --source-fps 20 \
  --target-fps 30 \
  --floor-align \
  --refine-iters 0 \
  --skip-existing
```

Implementation:

1. Decode HumanML3D-263 features to canonical 22-joint positions with
   `recover_from_ric`.
2. Resample the joint trajectory from HumanML3D's 20 fps to the SMPL/evaluator
   30 fps target.
3. Optionally floor-align the recovered joints.
4. Estimate local SMPL rotations by hierarchical bone alignment on the neutral
   SMPL rest skeleton.
5. Solve root translation by matching the SMPL root joint to the recovered root
   trajectory.
6. Optionally refine SMPL pose and translation with differentiable joint fitting.
7. Save one `.npz` per clip with:
   - `motion_135`: `(T, 135)` = `transl(3) + 22 * local_rot6d(6)`.
   - `transl`, `global_orient`, `body_pose`.
   - `target_joints`, `fitted_joints`.
   - `fit_mpjpe_mm`, `source_fps`, `target_fps`, `refine_iters`.

Current FlowMDM run:

```text
input:  outputs/evaluation/humanml3d_hml3d263_fixed_stats/flowmdm
output: outputs/evaluation/humanml3d_smpl135_fixed_stats/flowmdm
files:  3810
mean IK fit MPJPE: 26.3 mm
```

## Stage B: SMPL-style `motion_135` to MotionStreamer 272

Script:

```bash
python3 scripts/data/convert_motion135_to_h3d272.py \
  --in-dir outputs/evaluation/humanml3d_smpl135_fixed_stats/flowmdm \
  --out-dir outputs/evaluation/humanml3d/flowmdm_smpl272 \
  --workers 8
```

Implementation:

1. Load `motion_135` from each retargeted `.npz`.
2. Run forward kinematics with
   `scripts/eval/motionstreamer_272_encoder.py::motion135_to_272`.
3. Use the default `canon272` skeleton, which is extracted from the GT
   MotionStreamer 272 HumanML3D body.  This avoids the known SMPL-H-vs-SMPL-X
   rest-skeleton mismatch that inflates 272-evaluator FID.
4. Save one `(T, 272)` `.npy` per clip.

Current FlowMDM run:

```text
input:  outputs/evaluation/humanml3d_smpl135_fixed_stats/flowmdm
output: outputs/evaluation/humanml3d/flowmdm_smpl272
files:  3810
```

## Stage C: MotionStreamer 272 Evaluation

Command:

```bash
python3 ref_repo/MotionStreamer/eval_with_motionstreamer_evaluator.py \
  --evaluator_ckpt ref_repo/MotionStreamer/MotionStreamer/MotionStreamer_HF/Evaluator_272/epoch=99.ckpt \
  --data_root ref_repo/MotionStreamer/MotionStreamer/humanml3d_272 \
  --pred_dir outputs/evaluation/humanml3d/flowmdm_smpl272 \
  --n_repeats 20 \
  --batch_size 32 \
  --out_json outputs/evaluation/humanml3d/flowmdm_smpl272/eval_flowmdm_smplretarget_rep20.json
```

Current FlowMDM result:

```text
FID:      101.975
R@1/2/3:  0.393 / 0.555 / 0.637
MM-Dist:  20.189
Div:      25.651
```

These are the numbers currently written into
`papers/HYMotionM2M_ICLR2027/depds/tab_t2m.tex`.

## MotionLab inference-stage check

MotionLab smoothing is not an official operation and must not be used as the
reported MotionLab result.  The jitter we observed is primarily caused by
using the released checkpoint's fast evaluation step count for visualization:

- The released checkpoint stores `cfg.model.scheduler.num_eval_steps = 21`.
- The official demo path uses `cfg.model.scheduler.num_demo_steps = 201`.
- The official `demo_text` path calls `diffusion_reverse(stage="demo",
  condition_type="text", instructions=...)`, so the visual demo should be
  generated with 201 steps rather than the 21-step eval path.

```bash
python3 scripts/eval/motionlab_infer_hml3d263.py \
  --stage demo \
  --out-dir outputs/evaluation/humanml3d_hml3d263_fixed_stats/motionlab_demo201 \
  --batch-size 16 \
  --device cuda \
  --skip-existing
```

Quick jitter check on the first 400 common HumanML3D test ids, measured after
decoding HML263 to 22-joint trajectories, shows that 21-step eval output is the
problem.  The Savitzky-Golay row is diagnostic only:

```text
name             vel       acc       jerk      acc_p95   jerk_p95
GT               0.021975  0.005976  0.004674  0.019562  0.015371
FlowMDM          0.021972  0.006207  0.005265  0.021088  0.019027
MotionLab raw    0.028917  0.024414  0.042575  0.044600  0.076070
MotionLab smooth 0.022630  0.006157  0.006577  0.015672  0.012801
```

After fixing the wrapper to support the official `--stage demo` path, a 16-case
201-step smoke run without smoothing gives:

```text
name                 vel       acc       jerk      acc_p95   jerk_p95
GT                   0.029906  0.006927  0.005612  0.020661  0.016790
FlowMDM              0.022770  0.006211  0.005310  0.022274  0.021062
MotionLab eval-21    0.037480  0.024981  0.042795  0.046396  0.077166
MotionLab demo-201   0.030702  0.007554  0.007519  0.021632  0.019199
```

Current viewers:

- Raw 21-step output: `http://21.6.58.73:8216/`.
- 16-case official demo 201-step smoke: `http://21.6.58.73:8217/`.

MotionLab should remain `--` in the paper table until the full 201-step output
is generated, visually checked, retargeted, and evaluated.

## Practical checks

For any new HML263 baseline:

1. Visualize the raw HML263 output before retargeting.
2. Retarget to SMPL and inspect `fit_mpjpe_mm`.
3. Visualize the retargeted SMPL output.
4. Encode to 272 only after the SMPL visualization is plausible.
5. Run the MotionStreamer evaluator and write the exact JSON path into the
   table comment.

This order avoids confusing three different failure modes: bad baseline
inference, bad retargeting, and evaluator-domain conversion mismatch.
