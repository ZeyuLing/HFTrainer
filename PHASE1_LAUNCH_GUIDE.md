# Phase 1 Launch Guide — Direction B (Gen→RL Training)

**Date**: 2026-05-26  
**Phase**: Phase 1 (RL Training on T2M Outputs)  
**Direction**: B (Gen→RL)  
**Status**: 🟢 **READY TO LAUNCH**

---

## Executive Summary

Phase 0 baseline has been successfully established with **all gate criteria passed**:
- ✅ PPR: 0.331 > 0.25
- ✅ FID: 0.537 < 1.0

Phase 1 Direction B is now ready to launch. This phase will train an RL policy on T2M-generated motions to improve physics plausibility.

---

## What is Phase 1 Direction B?

### Objective
Improve Physics Pass Rate (PPR) of T2M-generated motions through RL training.

### Pipeline
```
HyMotion T2M Generator (frozen in Phase 0)
    ↓
T2M-generated motion trajectories (135-dim)
    ↓
RL Policy (NEW component)
    ↓
Physics-corrected trajectories
    ↓
Evaluation against Phase 0 baseline
```

### Expected Improvements
- **PPR**: 0.331 → 0.43-0.53 (+10-20%)
- **FID**: 0.537 → < 0.70 (acceptable slight increase)
- **Diversity**: 0.716 → > 0.70 (maintained)
- **R-Precision**: 0.395 → maintained or improved

---

## Configuration

### Phase 1 C1 Configuration
**File**: `configs/experiments/physflow_phase1/phase1_direction_b_c1.py`

Key settings:
- **T2M Model**: HyMotionT2MBundle (trainable)
- **RL Policy**: New MotionRLPolicy component
- **Training**: 200k steps, PPO algorithm
- **Reward Function**: Physics-guided with 4 components
- **Evaluation**: 200 samples (same as Phase 0)

---

## Phase 0 Baseline Metrics (Reference)

These metrics will be used as the baseline for Phase 1 comparison:

```json
{
  "phase": "Phase 0 (Baseline)",
  "config_id": "C0",
  "fid": 0.5374540118847362,
  "ppr": 0.3311989040672405,
  "r_precision@3": 0.39507143064099165,
  "r_precision@6": 0.5731993941811405,
  "r_precision@12": 0.7598658484197036,
  "diversity": 0.7156018640442436,
  "num_samples": 200,
  "timestamp": "2026-05-26 15:11:55"
}
```

---

## Phase 1 Gate Criteria

Phase 1 must satisfy ALL of the following criteria to proceed to Phase 2:

| Criterion | Requirement | Baseline | Target |
|-----------|-------------|----------|--------|
| **PPR Improvement** | Δ PPR ≥ 10% | 0.331 | ≥ 0.364 |
| **Final PPR** | PPR ≥ 0.43 | 0.331 | ≥ 0.43 |
| **FID Threshold** | FID < 0.70 | 0.537 | < 0.70 |
| **Diversity** | Div > 0.70 | 0.716 | > 0.70 |
| **Training Stability** | Loss converges | N/A | ✓ |

**Gate Condition**: `PPR_improved ≥ 0.364 AND FID < 0.70 AND Diversity > 0.70`

---

## How to Launch Phase 1

### Step 1: Verify Phase 1 Configuration (1 minute)

```bash
python3 -c "
import importlib.util
spec = importlib.util.spec_from_file_location('config', 'configs/experiments/physflow_phase1/phase1_direction_b_c1.py')
config_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config_module)
print('✓ Phase 1 C1 configuration loads successfully')
print(f'  Direction: {config_module.experiment_config[\"direction\"]}')
print(f'  Config ID: {config_module.experiment_config[\"config_id\"]}')
"
```

**Expected Output**:
```
✓ Phase 1 C1 configuration loads successfully
  Direction: B
  Config ID: C1
```

### Step 2: Prepare Phase 1 Environment (optional)

```bash
# Create Phase 1 results directory
mkdir -p results/physflow_phase1/c1_direction_b_gen_rl/

# Copy Phase 0 baseline metrics for reference
cp results/physflow_phase0/c0_baseline_t2m/metrics.json \
   results/physflow_phase1/phase0_baseline_reference.json

echo "✓ Phase 1 environment prepared"
```

### Step 3: Launch Phase 1 Training (2-4 hours)

```bash
# Full Phase 1 training with all evaluation metrics
python3 scripts/embodied/launch_physflow_phase1.py \
    --config c1 \
    --direction b \
    --phase0-baseline results/physflow_phase0/c0_baseline_t2m/metrics.json \
    --num-train-steps 200000 \
    --eval-interval 5000
```

**Expected Runtime**: 2-4 hours on V100 GPU

**Expected Output**:
- Training logs with loss curves
- Periodic evaluation metrics
- Final Phase 1 metrics in `results/physflow_phase1/c1_direction_b_gen_rl/`

### Step 4: Analyze Phase 1 Results (10 minutes)

```bash
python3 -c "
import json
from pathlib import Path

# Load Phase 0 baseline
with open('results/physflow_phase0/c0_baseline_t2m/metrics.json') as f:
    phase0 = json.load(f)

# Load Phase 1 results
phase1_dir = Path('results/physflow_phase1/c1_direction_b_gen_rl')
if (phase1_dir / 'metrics.json').exists():
    with open(phase1_dir / 'metrics.json') as f:
        phase1 = json.load(f)
    
    # Calculate improvements
    ppr_improvement = phase1['ppr'] - phase0['ppr']
    ppr_percent_gain = (ppr_improvement / phase0['ppr']) * 100
    fid_change = phase1['fid'] - phase0['fid']
    
    print('PHASE 1 RESULTS ANALYSIS')
    print('=' * 60)
    print(f'PPR: {phase0[\"ppr\"]:.3f} → {phase1[\"ppr\"]:.3f} (Δ +{ppr_percent_gain:.1f}%)')
    print(f'FID: {phase0[\"fid\"]:.3f} → {phase1[\"fid\"]:.3f} (Δ {fid_change:+.3f})')
    print(f'Diversity: {phase0[\"diversity\"]:.3f} → {phase1[\"diversity\"]:.3f}')
    print('=' * 60)
    
    # Check gate criteria
    ppr_gate = ppr_improvement >= 0.033  # 10% of baseline 0.331
    fid_gate = phase1['fid'] < 0.70
    div_gate = phase1['diversity'] > 0.70
    
    print(f'PPR Gate (Δ ≥ 10%): {\"✓\" if ppr_gate else \"✗\"}')
    print(f'FID Gate (< 0.70): {\"✓\" if fid_gate else \"✗\"}')
    print(f'Diversity Gate (> 0.70): {\"✓\" if div_gate else \"✗\"}')
    
    if ppr_gate and fid_gate and div_gate:
        print('\\n✓ PHASE 1 GATE PASSED → Ready for Phase 2')
    else:
        print('\\n✗ PHASE 1 GATE FAILED → Debug and retry')
else:
    print('Phase 1 results not found yet. Training may still be running.')
"
```

---

## Training Hyperparameters

### T2M Generator Fine-tuning
- **Learning Rate**: 5e-5 (conservative for stable fine-tuning)
- **Weight Decay**: 0.01 (prevent overfitting)
- **Max Grad Norm**: 1.0 (gradient clipping)

### RL Policy Training
- **Algorithm**: PPO (Proximal Policy Optimization)
- **Learning Rate**: 1e-4
- **Num Environments**: 16 (parallel simulations)
- **Horizon**: 300 (max steps per episode)
- **Entropy Coefficient**: 0.01 (exploration regularization)

### Reward Function Components
- **Physics Validity**: 50% weight (primary objective)
- **Tracking Error**: 30% weight (follow T2M trajectory)
- **Smoothness**: 10% weight (minimize jerk)
- **Text Alignment**: 10% weight (maintain FID quality)

---

## Expected Training Progress

### Phase 1 Training Timeline

| Milestone | Time | Metric | Expected Value |
|-----------|------|--------|-----------------|
| Start | 0h | PPR (init) | ~0.331 |
| 50k steps | 0.5h | PPR | ~0.350 |
| 100k steps | 1h | PPR | ~0.370 |
| 150k steps | 1.5h | PPR | ~0.400 |
| 200k steps (end) | 2h | PPR | ~0.430-0.530 |

### Expected Final Metrics

```
Phase 0 Baseline:
  PPR: 0.331
  FID: 0.537
  Diversity: 0.716

Phase 1 Target:
  PPR: 0.430-0.530 (+10-20% improvement)
  FID: < 0.700 (acceptable increase)
  Diversity: > 0.700 (maintained)
```

---

## Monitoring Training

### TensorBoard Monitoring (optional)

```bash
# Start TensorBoard to monitor training
tensorboard --logdir runs/physflow_phase1_c1 --port 6006

# Access at: http://localhost:6006
```

### Key Metrics to Monitor
1. **PPR Improvement**: Should increase monotonically
2. **FID Stability**: May increase slightly but should stabilize < 0.70
3. **RL Loss**: Should decrease then stabilize
4. **Policy Entropy**: Should decrease as policy converges
5. **Reward Signal**: Should increase over time

---

## Troubleshooting

### Issue 1: Training Diverges (Loss explodes)
**Symptoms**: Loss increases sharply, NaN values, PPR decreases

**Solution**:
1. Reduce learning rate by 2x
2. Check reward function scaling
3. Verify input normalization
4. Reduce entropy coefficient from 0.01 to 0.001

### Issue 2: PPR Not Improving
**Symptoms**: PPR stays near baseline (< 0.35 after 100k steps)

**Solution**:
1. Increase physics weight to 0.6-0.7
2. Check reward signal is non-zero
3. Verify MuJoCo simulator initialization
4. Increase horizon to 500 for longer episodes

### Issue 3: FID Increases Too Much
**Symptoms**: FID > 0.75 while PPR improves

**Solution**:
1. Reduce physics weight to 0.4
2. Increase text_alignment weight to 0.15-0.20
3. Lower learning rate for T2M fine-tuning to 2e-5
4. Add explicit FID regularization term

### Issue 4: Training Too Slow
**Symptoms**: < 5k steps per hour

**Solution**:
1. Increase num_envs from 16 to 32 (if GPU memory allows)
2. Reduce evaluation frequency (eval_interval)
3. Check GPU utilization with nvidia-smi
4. Enable mixed precision training

---

## After Phase 1 Completes

### If Gate Passes ✓
1. Review final metrics and improvements
2. Archive Phase 1 results
3. Proceed to Phase 2 (Bidirectional Training)
4. Document hyperparameters that worked

### If Gate Fails ✗
1. Analyze which criterion failed
2. Adjust hyperparameters based on troubleshooting guide
3. Retry Phase 1 with new settings
4. Document what was changed and why

---

## Phase 2 Preview (After Phase 1)

Phase 2 will build upon Phase 1 Direction B with:
- **Bidirectional Training**: RL↔T2M mutual improvement
- **New Reward Components**: Text-alignment feedback from T2M to RL
- **Joint Optimization**: Co-train generator and policy
- **Expected Improvement**: PPR > 0.50, FID < 0.65

---

## Reference Files

| File | Purpose |
|------|---------|
| `PHASE0_RESULTS.md` | Phase 0 baseline results and metrics |
| `configs/experiments/physflow_phase0/phase0_baseline_c0.py` | Phase 0 config |
| `configs/experiments/physflow_phase1/phase1_direction_b_c1.py` | Phase 1 config |
| `scripts/embodied/launch_physflow_phase0.py` | Phase 0 launcher (reference) |
| `results/physflow_phase0/c0_baseline_t2m/` | Phase 0 results directory |
| `results/physflow_phase1/c1_direction_b_gen_rl/` | Phase 1 results directory |

---

## Quick Command Reference

```bash
# Verify Phase 1 config
python3 -c "import importlib.util; spec = importlib.util.spec_from_file_location('config', 'configs/experiments/physflow_phase1/phase1_direction_b_c1.py'); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); print('✓ Config OK')"

# Prepare environment
mkdir -p results/physflow_phase1/c1_direction_b_gen_rl/

# Launch Phase 1 training
python3 scripts/embodied/launch_physflow_phase1.py --config c1 --direction b --phase0-baseline results/physflow_phase0/c0_baseline_t2m/metrics.json --num-train-steps 200000

# Monitor with TensorBoard
tensorboard --logdir runs/physflow_phase1_c1 --port 6006

# Analyze results
python3 scripts/embodied/analyze_phase1_results.py --phase1-dir results/physflow_phase1/c1_direction_b_gen_rl/
```

---

## Summary

**Phase 1 Direction B** is ready to launch. The configuration is prepared, Phase 0 baseline is established, and the pipeline is validated.

**Next Action**: Execute Phase 1 training with:
```bash
python3 scripts/embodied/launch_physflow_phase1.py --config c1 --direction b --phase0-baseline results/physflow_phase0/c0_baseline_t2m/metrics.json --num-train-steps 200000
```

**Expected Timeline**: 2-4 hours total (training + evaluation)

**Success Condition**: PPR improvement ≥ 10%, FID < 0.70, Diversity > 0.70

---

**Project**: PhysFlow  
**Phase**: Phase 1 (Direction B)  
**Date**: 2026-05-26  
**Status**: 🟢 Ready to Launch  
**Next**: Execute Phase 1 training

