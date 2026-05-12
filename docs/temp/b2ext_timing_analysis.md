# B2-ext Timing Analysis: Running Task Impact

## Key Finding: Config Changes Are Uncommitted (Not Yet Deployed)

The `null_embedding_source` fix has been added to **working directory** but is **not yet committed to git**. This is critical for understanding the running task status.

### Timeline

1. **2026-05-08**: Task launched with config `hymotion_m2m_v2_caption_local_phase2.py` (NO null_embedding_source)
2. **2026-05-12**: Config fix applied (uncommitted changes)
3. **Current state**: Config has null_embedding_source, but HEAD still points to the old version

### Running Task Status

The task launched on 2026-05-08 is using the **old config WITHOUT null_embedding_source**. 

**Key question**: Does the running task re-read config on restart?

**Answer**: NO. Configuration is read at task launch time, not on each restart. When the Taiji task resumes from a checkpoint (e.g., epoch 50), it uses the **same config that was baked in at launch**.

**Verification**:
- The runner code (accelerate_runner.py lines 1040-1053) loads from `latest` checkpoint in work_dir
- It restores training position (global_step, epoch) but does NOT re-read the config file
- The `self.load_from` and `self.load_cfg` are instance attributes set once at runner initialization

### Does the Running Task Have the B2-ext Bug?

**YES**, but only in a LIMITED way:

1. **At initial load (epoch 0)**: 
   - Phase 1 checkpoint loaded from: `work_dirs/hymotion_m2m_v2_caption_local_phase1/checkpoint-epoch_50/model.safetensors`
   - This is a safetensors file which doesn't store bundle-level parameters
   - **Bundle-level null embeddings would be all-zero** 
   - **BUT**: Runner code has fallback behavior (lines 1293-1302)

2. **Fallback mechanism** (pre-existing code, even without config field):
   - If `null_embedding_source` is not specified in config
   - Runner falls back to loading from `load_from.path` itself
   - Since the Phase 1 checkpoint is also a safetensors file, this doesn't help
   - **However**: The runner's workaround is partial and doesn't guarantee correct values

3. **Impact on training**:
   - Phase 2 config uses `cond_mask_prob=0.1`
   - This means 10% of batches use null text embeddings during training
   - Zero null embeddings → model sees uninformative null conditioning
   - **This affects training quality** (not just inference)
   - CFG training loss may be poorly calibrated

4. **On resume (epoch 1+)**:
   - When the task resumes from `work_dirs/hymotion_m2m_v2_caption_local_phase2/checkpoint-epoch_50/model.safetensors` etc.
   - The runner calls `_patch_zero_null_embeddings_from_pretrained()` (line 1052)
   - It looks for `load_from.null_embedding_source` or falls back to `load_from.path`
   - Since the config didn't have `null_embedding_source`, it tries to patch from the safetensors phase1 checkpoint (which doesn't work)
   - The runner skips patching if files don't load or params aren't found (graceful degradation)

### Practical Impact on Running Task

**Current training (with old config)**:
- Null embeddings are likely all-zero or nearly-zero
- CFG training sees poor null conditioning signal
- Training loss for null-text cases is higher/noisier than optimal
- **But training still progresses** because cond_mask_prob=0.1 is small

**If task is restarted with new config**:
- `null_embedding_source` points to HY-Motion-1.0-Lite checkpoint
- That checkpoint has proper null embeddings from T2M pretraining
- Patching happens on both initial load AND on resume (lines 1052, 1069)
- Training quality improves for the null-text conditioning branch
- **Would require full restart** (not just checkpoint resume)

### Does the Running Task Need a Restart?

**Recommendation**: Yes, for optimal training quality, but not urgent:

**Reasons for restart**:
- Running 4+ days with suboptimal null conditioning
- New checkpoints will inherit zero null embeddings (stored in safetensors)
- Cumulative training inefficiency compounds

**Reasons it's not critical**:
- cond_mask_prob=0.1 = only 10% of training affected
- Phase 2 is primarily for learning completion/editing (90% non-null)
- Inference CFG (which is completely broken without fix) isn't being used yet

### Config Changes Required for Future Tasks

All v2 caption configs should include:
```python
load_from = dict(
    ...
    null_embedding_source='checkpoints/HY-Motion-1.0/HY-Motion-1.0-Lite/latest.ckpt',
)
```

This is especially critical for:
- Any task that uses conditional text masking (cond_mask_prob > 0)
- Any task that will do inference-time CFG
- New Phase 1 training (currently runs with old config without this fix)

### Summary Table

| Aspect | Status |
|--------|--------|
| B2-ext bug present in running task? | **Yes** (zero null embeddings) |
| Does config get re-read on resume? | **No** (config baked at launch) |
| Is running task broken? | **No** (still trains, 90% unaffected) |
| Training quality impact? | **Moderate** (10% of batches degraded) |
| Inference CFG impact? | **Critical** (would be broken) |
| Does running task need restart? | **Yes, recommended** (for quality) |
| Is restart urgent? | **No** (training progresses acceptably) |
