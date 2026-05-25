# Next Steps: MuJoCo Self-Collision Fix Testing Guide

## ✅ What Was Completed

The MuJoCo self-collision disabling feature has been **fully implemented** in the ProtoMotions framework:

1. **Method Added:** `_disable_self_collisions()` in `protomotions/simulator/mujoco/simulator.py` (lines 1191-1210)
2. **Configuration Check Added:** Integrated into `_create_simulation()` (lines 320-322)
3. **Documentation Created:** Implementation report + session summary

**Files Modified:**
- `ref_repo/ProtoMotions/protomotions/simulator/mujoco/simulator.py` (26 lines total)

**Documentation:**
- `MUJOCO_SELF_COLLISION_FIX_IMPLEMENTATION.md` (290 lines)
- `SESSION_SUMMARY_20260525.md` (313 lines)

---

## 🧪 Testing Phase (What to Do Next)

### Step 1: Verify Configuration
Before testing, ensure the fix is properly configured:

```bash
# Check method exists
grep -n "def _disable_self_collisions" \
  ref_repo/ProtoMotions/protomotions/simulator/mujoco/simulator.py

# Check configuration integration
grep -A3 "_override_joint_properties" \
  ref_repo/ProtoMotions/protomotions/simulator/mujoco/simulator.py | \
  grep -B1 -A1 "_disable_self_collisions"

# Verify Python syntax
python3 -m py_compile \
  ref_repo/ProtoMotions/protomotions/simulator/mujoco/simulator.py
```

**Expected Output:**
```
Line 1191: def _disable_self_collisions(self) -> None:
Lines 320-322: Configuration check visible
✓ Syntax OK
```

---

### Step 2: Set Configuration in Your Experiment

Create a test script or modify your experiment to disable self-collisions:

**Option A: Command Line Override**
```bash
python protomotions/train_agent.py \
    --robot-name g1 \
    --simulator mujoco \
    --experiment-path examples/experiments/mimic/mlp.py \
    --motion-file data/motion_for_trackers/g1_bones_seed_mini.pt \
    --num-envs 1 \
    --batch-size 16 \
    --overrides "robot_config.asset.self_collisions=False"
```

**Option B: Code Configuration**
```python
# In your training script:
from protomotions.robot_configs.base import load_robot_config

robot_config = load_robot_config("g1")
robot_config.asset.self_collisions = False  # Disable self-collisions

# Continue with simulator and training initialization...
```

**Option C: Experiment File Modification**
```python
# In examples/experiments/mimic/mlp.py:
def configure_robot_and_simulator(robot_config, simulator_config, ...):
    # ... existing config code ...
    
    # Add this line:
    robot_config.asset.self_collisions = False
    
    return robot_config, simulator_config
```

---

### Step 3: Run Training with MuJoCo Backend

**Simple Test (Single Environment):**
```bash
cd /path/to/ref_repo/ProtoMotions

python protomotions/train_agent.py \
    --robot-name g1 \
    --simulator mujoco \
    --experiment-path examples/experiments/mimic/mlp.py \
    --motion-file data/motion_for_trackers/g1_bones_seed_mini.pt \
    --num-envs 1 \
    --batch-size 16 \
    --overrides "robot_config.asset.self_collisions=False"
```

**Expected Behavior:**
- ✅ Training starts without errors
- ✅ No crashes related to collision
- ✅ Reward increases smoothly over episodes
- ✅ No "robot falls" or "unstable" error messages

---

### Step 4: Monitor Training Stability

**What to Look For (Good Signs):**
1. **Smooth Reward Curve:** Tracking reward should increase monotonically
2. **No Crash Events:** No sudden spikes in error or loss
3. **Stable Episode Returns:** Rewards consistent across episodes
4. **No Collision Messages:** No "self-collision detected" or fall warnings

**What to Look For (Problems):**
1. **Noisy Reward:** Tracking reward fluctuates wildly
2. **Crash Events:** Sudden large errors
3. **Unstable Training:** Loss spikes unexpectedly
4. **Contact Warnings:** Messages about collision forces

### Recommended Logging Commands

```bash
# Monitor training output
tail -f output.log | grep -i "reward\|error\|collision\|fall"

# Check for stability
grep "episode\|reward" output.log | head -50

# Verify no crashes
grep -i "error\|exception\|crash" output.log
```

---

### Step 5: Compare With/Without Fix

**Test 1: WITH self-collision disabling (should work)**
```bash
# Set self_collisions = False
python train_agent.py ... --overrides "robot_config.asset.self_collisions=False"
# Expected: Stable training, no falls
```

**Test 2: WITHOUT self-collision disabling (may show original problem)**
```bash
# Set self_collisions = True (default)
python train_agent.py ... --overrides "robot_config.asset.self_collisions=True"
# Expected: Training may be unstable (original behavior)
```

Compare the two training runs to verify the fix improves stability.

---

## 🔍 Detailed Verification Checklist

Run through this checklist during testing:

### Code Verification
- [ ] `_disable_self_collisions()` method exists and has correct signature
- [ ] Configuration check properly integrated in `_create_simulation()`
- [ ] Python syntax validation passes
- [ ] No import errors when loading MuJoCo simulator
- [ ] `robot_config.asset.self_collisions` flag recognized

### Functional Verification  
- [ ] Simulator initializes without warnings
- [ ] Method is called when `self_collisions=False`
- [ ] Method is NOT called when `self_collisions=True`
- [ ] No exceptions during model initialization
- [ ] MuJoCo data structures properly populated

### Training Verification
- [ ] RL training starts successfully
- [ ] Episodes run without crashes
- [ ] Rewards increase over time (learning signal present)
- [ ] No uncontrolled falls or unstable behavior
- [ ] Training loss decreases smoothly

### Regression Verification
- [ ] Other simulators still work (IsaacGym, Newton if available)
- [ ] Non-collision aspects of training unaffected
- [ ] Existing experiment configs still work
- [ ] Command-line overrides still work

---

## 🐛 Troubleshooting

### Issue: "Module not found" or Import Error
```
Error: ModuleNotFoundError: No module named 'protomotions'
```
**Solution:**
```bash
cd ref_repo/ProtoMotions
pip install -e .
```

### Issue: Syntax Error in simulator.py
```
SyntaxError: invalid syntax at line 1191
```
**Solution:**
```bash
# Verify file integrity
python3 -m py_compile protomotions/simulator/mujoco/simulator.py

# If error, check for indentation:
sed -n '1185,1215p' protomotions/simulator/mujoco/simulator.py | cat -A
```

### Issue: AttributeError: 'robot_config' has no attribute 'asset'
```
Error: AttributeError: robot_config has no attribute 'asset'
```
**Solution:**
```python
# Verify robot_config structure
print(robot_config.__dict__)
print(hasattr(robot_config, 'asset'))
print(hasattr(robot_config.asset, 'self_collisions'))
```

### Issue: Training Crashes Immediately
```
Error: ... during MuJoCo initialization
```
**Solution:**
1. Check error logs: `tail -f output.log`
2. Verify MJCF file exists and is valid
3. Run with `--num-envs 1` (simpler setup)
4. Check MuJoCo version: `python -c "import mujoco; print(mujoco.__version__)"`

---

## 📊 Expected Results

### Training Without Fix (Before/Baseline)
```
Episode 1-10: Random initialization, reward ≈ -5.0 to 0.0
Episode 10-20: Unstable learning, reward varies wildly (-10.0 to 1.0)
Episode 20-30: Training crashes or becomes unstable
ERROR: Robot falls due to self-collision forces
```

### Training With Fix (After)
```
Episode 1-10: Random initialization, reward ≈ -3.0 to -1.0
Episode 10-20: Smooth learning curve, reward ≈ -1.5 to -0.5
Episode 20-50: Consistent improvement, reward → 0.0
Episode 50+: Converged or close to convergence, stable rewards
```

**Key Metric:** Reward should increase smoothly without discontinuities or crashes.

---

## 📝 Recommended Test Cases

### Test 1: Minimal Configuration
```bash
# Smallest possible setup to verify fix works
python protomotions/train_agent.py \
    --robot-name g1 \
    --simulator mujoco \
    --experiment-path examples/experiments/mimic/mlp.py \
    --motion-file data/motion_for_trackers/g1_bones_seed_mini.pt \
    --num-envs 1 \
    --num-steps 100  # Just 100 steps for quick test
    --overrides "robot_config.asset.self_collisions=False"
```
**Expected:** Completes without errors in < 1 minute

### Test 2: Longer Training
```bash
# Run for multiple episodes to verify stability
python protomotions/train_agent.py \
    --robot-name g1 \
    --simulator mujoco \
    --experiment-path examples/experiments/mimic/mlp.py \
    --motion-file data/motion_for_trackers/g1_bones_seed_mini.pt \
    --num-envs 1 \
    --num-steps 1000  # 1000 steps = more episodes
    --overrides "robot_config.asset.self_collisions=False"
```
**Expected:** Rewards increase smoothly, no crashes

### Test 3: Comparison Test
```bash
# Run both with and without fix to compare
# WITH fix:
python train_agent.py ... --overrides "robot_config.asset.self_collisions=False"
# Save output to: test_with_fix.log

# WITHOUT fix:
python train_agent.py ... --overrides "robot_config.asset.self_collisions=True"
# Save output to: test_without_fix.log

# Compare:
diff <(grep "reward" test_with_fix.log) <(grep "reward" test_without_fix.log)
```

---

## 📖 Documentation References

For more details, see:

1. **Full Implementation Report:** `MUJOCO_SELF_COLLISION_FIX_IMPLEMENTATION.md`
   - Complete technical details
   - MuJoCo collision mechanics explained
   - Verification commands

2. **Session Summary:** `SESSION_SUMMARY_20260525.md`
   - Overview of all work done
   - Design decisions explained
   - Feature parity comparison

3. **Root Cause Analysis:** `mujoco_self_collision_fix.md`
   - Why the problem occurred
   - How other simulators handle it
   - Why this solution was chosen

---

## 🚀 Success Criteria

The fix is **successfully validated** when:

1. ✅ Training runs without MuJoCo crashes
2. ✅ Reward increases smoothly over episodes  
3. ✅ No "self-collision" or "falls" error messages
4. ✅ SMPL humanoid tracks motion without instability
5. ✅ Comparison shows improvement vs. without fix
6. ✅ Other simulators continue to work normally

---

## 📞 When to Stop Testing

**Stop testing (you're done) when:**
- Training runs successfully for 50+ episodes
- Rewards show learning (trend is positive)
- No crashes or instability observed
- Other simulators verified to still work

**Continue investigating if:**
- MuJoCo still crashes during init
- Rewards don't improve at all
- New error messages appear
- Training becomes unstable after N episodes

---

## 🎯 Next Decision Point

**After testing, you should:**

1. ✅ **If Fix Works:** 
   - Integrate into main training pipeline
   - Add to default configuration for MuJoCo
   - Update training documentation
   - Close related issues

2. ⚠️ **If Issues Found:**
   - Check error logs for specific errors
   - Verify configuration was applied correctly
   - Run manual verification checklist
   - Consult root cause analysis document

---

**Estimated Testing Time:** 30-60 minutes
**Difficulty Level:** Low (configuration-based testing)
**Requirements:** ProtoMotions installed, MuJoCo Python package, motion data files

Good luck with testing! 🎉
