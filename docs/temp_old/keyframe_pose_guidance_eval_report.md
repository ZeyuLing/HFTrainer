# Keyframe Pose Guidance Evaluation Report

**Date**: 2026-03-30 13:41
**Test samples**: 20 (from test_motionhub_recon)
**Variants evaluated**: 60 (10 models × 3 imputation modes × 2 replacement modes)

## Background

This evaluation tests **keyframe pose imputation** — the ability of
mask-aware noise (MAN) trained models to preserve a target pose at a
specified keyframe while generating natural surrounding motion.

### Imputation Strategies

| Strategy | Description |
|----------|-------------|
| `keyframe_only` | All frames masked except keyframe (hardest) |
| `anchor_inbetween` | First + keyframe + last preserved (practical) |
| `local_edit` | Only ±30 frames around keyframe masked (easiest) |

### Replacement Guidance

| Mode | Description |
|------|-------------|
| `none` | Standard ODE, no per-step replacement |
| `flow_interp` | Replace known regions with flow-matching interpolation each step |

## Results

### Top 10 MAN Models (by MPJPE)

| # | Model | Imp Mode | Rep | Rot | MPJPE↓ | Bnd Smooth↓ | Foot Skate↓ |
|---|-------|----------|-----|-----|--------|-------------|-------------|
| 1 | uncond_jit_man | local_edit | **flow_interp** | local | **2.987** | **3.202** | 0.124 |
| 2 | uncond_jit_man | keyframe_only | **flow_interp** | local | **3.002** | 3.293 | 0.074 |
| 3 | uncond_jit_man | anchor_inbetween | **flow_interp** | local | **3.006** | **2.908** | 0.137 |
| 4 | uncond_jit_man | keyframe_only | none | local | 3.331 | 5.502 | 0.070 |
| 5 | uncond_jit_man | anchor_inbetween | none | local | 3.439 | 5.298 | 0.153 |
| 6 | uncond_fm_man | local_edit | **flow_interp** | local | 3.486 | 3.699 | 0.153 |
| 7 | caption_jit_man | anchor_inbetween | **flow_interp** | local | 3.507 | 3.551 | 0.186 |
| 8 | caption_jit_man | keyframe_only | **flow_interp** | local | 3.509 | 3.990 | 0.113 |
| 9 | uncond_fm_man | keyframe_only | **flow_interp** | local | 3.531 | 4.268 | 0.050 |
| 10 | uncond_jit_man | local_edit | none | local | 3.554 | 5.127 | 0.131 |

## Key Findings

### 1. `flow_interp` replacement guidance dramatically improves boundary smoothness

For **ALL MAN models**, `flow_interp` reduces boundary smoothness by ~40-60%:

| Model | Rep=none bnd | Rep=flow_interp bnd | Improvement |
|-------|-------------|---------------------|-------------|
| uncond_jit_man (anchor) | 5.298 | **2.908** | 45% ↓ |
| uncond_jit_man (kf_only) | 5.502 | **3.293** | 40% ↓ |
| uncond_fm_man (anchor) | 6.163 | **3.653** | 41% ↓ |
| uncond_jit_man_globalrot (anchor) | 7.580 | **3.151** | 58% ↓ |

**flow_interp is essential for MAN models.** This validates the V4 design hypothesis.

### 2. JIT loss outperforms FM velocity loss (~14%)

- uncond_jit_man best: **2.987** mpjpe
- uncond_fm_man best: **3.486** mpjpe

### 3. Local rotation > Global rotation (with epoch caveat)

- Best local: **2.987** mpjpe (362 epochs)
- Best global: **4.604** mpjpe (70 epochs, ~5x fewer)

### 4. Best configuration: `uncond_jit_man` + `anchor_inbetween` + `flow_interp`

MPJPE: 3.006, Boundary Smoothness: 2.908, Foot Skating: 0.137

### 5. Non-MAN baselines show lower MPJPE but misleadingly

Non-MAN models achieve mpjpe 0.8-1.6 by copying from VACE context — they cannot
generate novel motion through a novel target pose.

## Web Visualization

Results: `http://<host>:8096/`
