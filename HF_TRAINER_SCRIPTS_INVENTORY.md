# HFTrainer Repository - Complete Scripts & Tools Inventory

**Generated**: 2026-05-10  
**Repository Path**: `/apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer`

---

## Summary Statistics

| Category | Count | Total Lines |
|----------|-------|------------|
| **ROOT scripts** | 9 | ~1000 |
| **tools/ scripts** | 127 | ~29,811 |
| **scripts/ scripts** | 68 | ~50,000+ |
| **Subdirectories in tools/** | - | analysis_tools/, robot_sim/ |
| **Subdirectories in scripts/** | - | eval/ |
| **TOTAL** | **204** | **~80,000+** |

---

## 1. ROOT DIRECTORY SCRIPTS (9 files)

### Training & Testing Scripts
| Script | Type | Purpose |
|--------|------|---------|
| `train.py` | Python | Main training entry point for hftrainer - supports config-based training |
| `infer.py` | Python (in tools/) | Inference entry point for motion pipelines |
| `setup.py` | Python | Package setup and installation |

### Testing & Validation
| Script | Type | Purpose |
|--------|------|---------|
| `_test_train_textfree.sh` | Shell | Quick 1-GPU training test |
| `_test_dist_train_textfree.sh` | Shell | 8-GPU distributed training test with Medium config |
| `_test_dit.sh` | Shell | DiT (text-free) model testing |

### Taiji (Job Scheduling) & Infrastructure
| Script | Type | Purpose |
|--------|------|---------|
| `_taiji_run.sh` | Shell | Taiji job execution wrapper (Usage: task_flag instance_id script_path args) |
| `.taiji_test.sh` | Shell | Taiji integration testing |
| `run_setup.sh` | Shell | Environment/dependency setup |

### Utilities
| Script | Type | Purpose |
|--------|------|---------|
| `_remote_cmd.sh` | Shell | Remote command execution helper |
| `E14_DEBUG_VERIFICATION.py` | Python | Verification and debugging for E14 metrics bug fix |

---

## 2. TOOLS/ DIRECTORY SCRIPTS (127 files + 2 subdirectories)

### A. Core Training & Inference (3 files)

| Script | Purpose |
|--------|---------|
| `train.py` | **Main training entry point** - Loads configs and orchestrates training |
| `infer.py` | **Inference entry point** - Runs trained models for motion generation |
| `dist_train.sh` | Launch distributed training with accelerate |

### B. Data Conversion & Format Bridging (16 files)

Converting between different motion representation formats (HumanML3D-263, H3D272, SMPL-85, SMPL-135, etc.)

| Script | Purpose |
|--------|---------|
| `convert_hml263_to_h3d272.py` | Bridge MoMask HumanML3D-263 outputs to H3D272 format |
| `convert_momask263_to_h3d272.py` | Re-implementation bridging FPS gap between MoMask (20fps) and H3D272 (30fps) |
| `convert_motionclip_checkpoint.py` | Convert MotionClip checkpoint from mmengine .pth format |
| `convert_motionhub_to_h3d272.py` | Convert MotionHub features for MotionStreamer's TMR-272 evaluator |
| `fbx_to_motion_json.py` | Convert FBX files to motion JSON (handles PreRotation, IK, multi-layer animations) |
| `smpl85_to_repr272.py` | Convert SMPL-85 to 272-dim representation |
| `momask263_to_smpl85.py` | Pipeline: momask263 → joints → SMPL-85 fitting |
| `momask263_to_smpl85_sharded.py` | Distributed version of momask263→SMPL-85 conversion (1/N files per shard) |
| `validate_smpl85_to_272.py` | Verify SMPL-85 to 272-dim conversion correctness |
| `build_h3d263_test_from_h3d272.py` | Extract H3D263 subset from MotionStreamer's H3D272 (30fps) |
| `_recover_orphan_tmp_smpl85.py` | Recovery script for files where np.save auto-appended .npy |
| `_parallel_momask263_to_smpl85.sh` | Parallel SMPL fitting across 8 GPUs |

### C. Data Building & Preparation (13 files)

Building evaluation datasets and data splits for different tasks (E1-E15)

| Script | Purpose |
|--------|---------|
| `build_e14_hq400h_data.py` | Build E14 Transition Stitching evaluation data (HQ400h test set) |
| `build_e15_prepend_v2_data.py` | Build E15 v2 (2026-04-27) prepend task data |
| `build_e2_inbetween_v2_data.py` | Build E2 inbetween task v2 (2026-04-25) rewritten from scratch |
| `build_e3_keyframe_v2_data.py` | Build E3 keyframe task v2 (2026-04-25) |
| `build_e8_loop_v2_data.py` | Build E8 loop task v2 (2026-04-26) |
| `build_e9_repair_v2.py` | Build E9 repair task data with high-confidence real defect sampling |
| `build_m2m_v2_eval_data.py` | Select diverse actions for M2M v2 evaluation from HYMotion npz_split |
| `rebuild_e2_from_scan.py` | Rebuild E2 inbetween from raw scan with on-the-fly pose-difference scoring |
| `rebuild_e2_inbetween_datalist.py` | Rebuild E2 datalist (settings A/B/C with different keep_start/keep_end) |
| `rebuild_e5_from_scan.py` | Rebuild E5 trajectory following task from raw scan |
| `rebuild_e5_trajectory_datalist.py` | Rebuild E5 datalist - filter to quality cases where root follows trajectory |
| `precompute_bone_offsets.py` | Precompute relative bone offsets for SMPL-22 skeleton |
| `prepare_motionfix_hymotion.py` | Prepare motion-fix data for HyMotion processing |

### D. Quality Checking & Analysis (11 files)

Analyzing motion quality, detecting defects, and auditing data integrity

| Script | Purpose |
|--------|---------|
| `check_original_quality.py` | Detect jitter, joint jumps, frozen frames in ORIGINAL (pre-repair) FBX data |
| `diag_smpl85_fit_quality.py` | Diagnose quality of SMPL-85 fitting (fitting error, joint reconstruction error) |
| `diagnose_kimodo_e14_boundary_jumps.py` | Report rotation-angle and joint-position jumps at KIMODO E14 boundaries |
| `diag_audit_orphan.py` | Scan all work_dirs/latest ckpt and report bundle-level Parameter/buffer storage state |
| `sampler_coverage_audit.py` | For each E-task, define mask signature checkers to verify sampler coverage |
| `scan_pelvis_pathlen.py` | Scan all NPZ files, compute pelvis path length, output JSON sorted by path_len |
| `diag_caption_forward.py` | Debug whether models actually depend on caption embedding |
| `diag_caption_v2.py` | Caption-related diagnostics (run on debug machine) |
| `diag_converter_only.py` | Bypass process_file to test converter pipeline in isolation |
| `diag_h3d263_to_272_roundtrip.py` | Test roundtrip conversion H3D263 → processing → 272-dim |
| `repair_eval_cjgame.py` | Scan CJGame NPZ, filter quality-problematic samples, evaluate repair methods |

### E. Null/Zero Embedding Diagnostics (4 files)

Investigating null embedding issues in caption-conditioned models

| Script | Purpose |
|--------|---------|
| `diag_null_embed_pipeline.py` | Checkpoint-by-checkpoint diagnosis to identify where null_vtxt_feat becomes zero |
| `diag_null_embed_fix_verify.py` | Multi-stage verification of null embedding fix |
| `diag_null_embed_e2e_train.py` | Full AccelerateRunner flow test for null embedding issues |
| `smoke_test_m2m_padding_fix.py` | Verify M2M padding fix correctness after RandomCropPadding |

### F. Evaluation & Model Testing (18 files)

Comprehensive evaluation across different models and tasks

| Script | Purpose |
|--------|---------|
| `eval_m2m_checkpoints.py` | Test evaluation script (mirror of eval_with_motionstreamer_evaluator.py) |
| `eval_m2m_v2_all_tasks.py` | Evaluate 4 M2M v2 model variants across all tasks |
| `eval_m2m_v2_t2m.py` | M2M v2 text-to-motion evaluation (two modes) |
| `eval_manual_repair.py` | Comprehensive JSON + Markdown report for manual repair evaluation |
| `eval_momask_native_h3d263.py` | Load MoMask's evaluator (Comp_v6_KLD005 + glove) |
| `eval_with_motionclip_evaluator.py` | Mirror of MotionStreamer's eval_with_motionstreamer_evaluator.py |
| `eval_result_watcher.py` | Watch and monitor evaluation results in progress |
| `eval_m2m_repair.py` | Comprehensive repair evaluation for M2M models (lower-level) |
| `momask_infer_h3d_test.py` | Drive MoMask's released checkpoints (rvq + t2m_transformer + tres + length_estimator) |
| `test_m2m_repair.py` | Load low_quality.json, test M2M completion in refine mode |
| `test_motionclip_parity.py` | Test parity between MotionClip and other implementations |
| `debug_m2m_padding_real_data.py` | Debug M2M padding on real v2 training config with full dataset |
| `m2m_v2_v3_mask_density.py` | Compare mask density generation under v2 vs v3 sampler |
| `m2m_v3_loss_curve.py` | Analyze M2M v3 loss curves |
| `test_vermo_tasks.py` | Test various VERMO pipeline tasks |
| `launch_eval_batch_20260506.sh` | Batch launch evaluation jobs (2026-05-06) |
| `run_dit_eval.sh` | Run DiT (text-free) checkpoint evaluation |
| `run_e14_hq400h_eval.sh` | Run E14 Transition Stitching eval on HQ400h test set |

### G. Merging & Aggregation (4 files)

Merge sharded/distributed outputs into consolidated results

| Script | Purpose |
|--------|---------|
| `merge_kimodo_shards_simple.py` | Merge KIMODO shard directories for one task/setting |
| `merge_kimodo_e14_shards.py` | Merge sharded KIMODO E14 outputs into dashboard-visible run dirs |
| `refresh_e3_e8_score_selective_20260506.py` | Selectively refresh score_m2m for E3/E8 pair cases (2026-05-06 reruns) |
| `selective_update_score_m2m_from_m2m_v2_rerun.py` | Update score DB from M2M v2 reruns without deleting old NPZ files |

### H. Patching & Post-Processing (7 files)

Bug fixes and output corrections

| Script | Purpose |
|--------|---------|
| `patch_bundle_orphan_params.py` | Fix bundle-level Parameter/buffer orphan issues |
| `patch_e2_v2_meta_distribution.py` | Fix E2 v2 meta distribution for dashboard compatibility |
| `patch_kimodo_y_anchor.py` | Fix KIMODO ~10-30cm upward Y drift by anchoring to first frame |
| `patch_nvml.py` | Fix PyTorch 2.5.0 nvmlDeviceGetNvLinkRemoteDeviceType assertion |
| `render_fbx_comparison.py` | Generate stick figure MP4 comparison from FBX bone positions |
| `check_original_quality.py` | (Also listed above) Detect original motion defects |

### I. Checkpoint & Model Analysis (4 files)

| Script | Purpose |
|--------|---------|
| `compute_198dim_stats.py` | Compute statistics for 198-dim motion representation |
| `compute_201dim_stats.py` | Compute statistics for 201-dim motion representation |
| `compute_global_rot_stats.py` | Compute global rotation statistics (LoadSmplx55 → LocalToGlobalRotation) |
| `visualize_mask_patterns.py` | Generate visualization figures for mask patterns |

### J. Taiji/Distributed Job Management (9 files)

Scripts for submitting and managing distributed training/eval on Taiji platform

| Script | Purpose |
|--------|---------|
| `taiji_submit.py` | Submit jobs to Taiji with full nested config format |
| `taiji_submit_eval.py` | Submit evaluation jobs to Taiji (1 host × 1 GPU V100 per job) |
| `taiji_exec.py` | Execute Taiji commands (handles TTY requirement) |
| `taiji_exec_host.py` | Handle Taiji multi-host MPI execution |
| `taiji_dist_train.sh` | Distributed training launcher for Taiji |
| `submit_v2_eval.py` | Submit M2M v2 evaluation jobs |
| `submit_v2_eval_taiji.py` | Submit M2M v2 evaluation to Taiji |
| `submit_v2_eval_taiji.sh` | Shell wrapper for Taiji M2M v2 eval submission |
| `submit_globalrot_tasks.py` | Submit all 4 global rotation model variants |

### K. Evaluation Job Launchers (10 files)

Shell scripts that orchestrate large-scale evaluation runs

| Script | Purpose |
|--------|---------|
| `run_m2m_eval.sh` | Run M2M checkpoint evaluation on Taiji GPU node |
| `run_m2m_v2_eval_latest.sh` | Run M2M v2 all-tasks evaluation with latest checkpoints |
| `run_e3_kimodo_adaptive_20260430.sh` | E3 KIMODO adaptive evaluation (2026-04-30) |
| `run_e3_m2m_latest_20260430.sh` | E3 M2M latest evaluation (2026-04-30) |
| `run_e8d_kimodo_fixed_20260430.sh` | E8-D KIMODO fixed evaluation (2026-04-30) |
| `run_e13_rerun_eval.sh` | E13 Multi-Prompt Autoregressive eval rerun |
| `run_e15_v2_sweep_and_full.sh` | E15 v2 (2026-04-27) sweep + full evaluation |
| `run_m2m_v2_latest_selective_rerun_20260429.sh` | M2M v2 selective rerun (2026-04-29) |
| `start_m2m_v2_latest_selective_rerun_20260429.sh` | Start selective M2M v2 rerun |
| `run_kimodo_e14_rotfix_batch.sh` | KIMODO E14 rotation fix batch evaluation |

### L. KIMODO/Robot Simulation (2 files)

KIMODO humanoid robot control and related simulations

| Script | Purpose |
|--------|---------|
| `run_kimodo_all_tasks.py` | Bridge SMPL-22 eval data ↔ KIMODO SOMA-30 skeleton via rotation-based conversion |
| `_run_kimodo_debug.sh` | Mini debug run on KIMODO aux loss verification |

### M. Caption & Label Management (3 files)

| Script | Purpose |
|--------|---------|
| `batch_rewrite_captions.py` | Run `/api/rewrite_caption` endpoint on all items |
| `rewrite_caption_file.py` | Call deployed caption rewriter service |
| `rewrite_e2_v2_captions.py` | Rewrite captions for E2 v2 eval dataset (built 2026-04-25) |

### N. Data Pair Analysis (2 files)

| Script | Purpose |
|--------|---------|
| `analyze_cjgame_pairs.py` | Analyze CJGame_MB original/cleaned NPZ pairs |
| `analyze_fbx_pairs.py` | Compute bone world-position differences at FBX level (pre-SMPL) |

### O. FBX & Skeleton Processing (1 file)

| Script | Purpose |
|--------|---------|
| `append_kimodo_context_soma77.py` | Add mesh visualization context to eval dashboard |
| `append_kimodo_e15_context_soma77.py` | Add mesh visualization context for E15 (v2) eval dashboard |

### P. Batch Inference (1 file)

| Script | Purpose |
|--------|---------|
| `batch_infer_vermo.py` | Batch inference across multiple VERMO task configurations |

### Q. Reference Repository Management (1 file)

| Script | Purpose |
|--------|---------|
| `ref_repo_commit_by_project.sh` | Split add/commit by ref_repo sub-project (respects .gitignore) |

### R. Subdirectory: `tools/analysis_tools/` (2 files)

| Script | Purpose |
|--------|---------|
| `get_flops.py` | Compute FLOPs for model analysis |
| `print_config.py` | Pretty-print model configuration |

### S. Subdirectory: `tools/robot_sim/` (3 files)

ASAP (humanoid robot) simulation and control

| Script | Purpose |
|--------|---------|
| `__init__.py` | Package initialization |
| `setup_asap.py` | Set up external dependencies for ASAP robot simulation |
| `text_to_g1.py` | Full pipeline for driving Unitree G1 humanoid from text commands |

---

## 3. SCRIPTS/ DIRECTORY SCRIPTS (68 files + 1 subdirectory)

Primary evaluation and post-processing scripts for motion quality assessment and repair.

### A. Evaluation Workers (4 files)

Single-worker evaluation scripts called by parallel launchers

| Script | Purpose |
|--------|---------|
| `_eval_m2m_single.py` | Compute MoGenDiT adaptive mask + M2M repair inline per sample |
| `_eval_globalrot_single.py` | Global rotation repair evaluation (single worker) |
| `_eval_globalrot_single_v2.py` | Global rotation v2 (MoGenDiT aligned) single worker |
| `_eval_globalrot_single_v3.py` | Global rotation v3 (MoGenDiT mask + M2M denoise-impute) strategy |

### B. Parallel Evaluation & Repair (13 files)

Distributed evaluation across multiple GPUs

| Script | Purpose |
|--------|---------|
| `eval_m2m_repair.py` | Comprehensive M2M repair evaluation (lower-level single script) |
| `eval_m2m_repair_parallel.py` | Parallel M2M repair: one process per (config, mode) pair on separate GPU |
| `eval_globalrot_repair_parallel.py` | Global rotation repair: split data into 4 shards (1 per GPU) |
| `eval_globalrot_repair_parallel_v2.py` | Global rotation v2: 2 configs × 4 GPUs = 8 workers |
| `eval_globalrot_repair_parallel_v3.py` | Global rotation v3: MoGenDIT mask + M2M denoise-impute parallel |
| `eval_cjgame_repair.py` | Evaluate ALL candidate models on CJGame_MB npz_split data |
| `eval_mogendit_repair.py` | MoGenDIT baseline repair evaluation |
| `eval_mogendit_t7.py` | Run MoGenDIT ada_denoise on T7 eval cases for comparison |
| `mogendit_repair.py` | MoGenDIT motion repair CLI |
| `mogendit_multigpu_repair.py` | MoGenDIT multi-GPU parallel repair script (中文) |
| `mogendit_pipeline_multigpu.py` | MoGenDIT pipeline multi-GPU parallel repair (中文) |
| `mogendit_cjgame_eval.py` | MoGenDIT CJGame MB repair evaluation (中文) |
| `sdedit_repair.py` | SDEdit-style motion repair using HyMotion T2M model |

### C. Comprehensive Evaluation Suites (10 files)

Multi-task evaluation for models across different motion completion scenarios

| Script | Purpose |
|--------|---------|
| `eval_m2m_all_tasks.py` | **Single entry point** for comprehensive M2M model evaluation (4 tasks) |
| `eval_m2m_completion.py` | Evaluate all 8 MAN model variants on 3 completion tasks |
| `eval_m2m_ablation.py` | Evaluate M2M on 4 completion tasks (ablation studies) |
| `eval_m2m_transition.py` | Evaluate M2M transition: stitch different motion pairs |
| `eval_m2m_checkpoint_report.py` | Evaluate all 8 converged M2M models (4 tasks + replacement guidance) |
| `eval_m2m_completion.py` | (See above) MAN model variants |
| `eval_repair_benchmark.py` | Benchmark repair methods on M2M database's low-quality data (~85k items) |
| `eval_keypose_guidance.py` | Evaluate keypose-conditioned completion using HyMotion M2M + MoGenDIT |
| `eval_keyframe_pose_guidance.py` | Evaluate keypose-conditioned SDEdit on real before/after motion pairs |
| `eval_sparse_keyframe_mib.py` | Test MAN configs at multiple keyframe densities (5/15/30 fps) |

### D. Quality & Completion Analysis (5 files)

| Script | Purpose |
|--------|---------|
| `eval_man_repair_quick.py` | Sample 10 items per failure category, compute MoGenDiT quick metrics |
| `eval_m2m_completion.py` | (See above) |
| `run_quality_check_m2m.py` | Run MotionQualityChecker on all eval_results M2M outputs |
| `postprocess_quality.py` | Write quality.json with comprehensive metrics |
| `lq_overlay_clean_frames.py` | Overlay clean frames from HyM v6 → trans_regen → ada_denoise pipeline (2026-04-27) |

### E. Post-Processing & Polish (8 files)

Motion quality improvement after model generation

| Script | Purpose |
|--------|---------|
| `postprocess_e14_antislide.py` | Dampen sliding when foot is detected on ground |
| `postprocess_e14_foot_contact.py` | Foot contact-aware post-processing for E14 transitions |
| `postprocess_e14_footpin.py` | Pin pelvis position based on grounded foot positions |
| `postprocess_e14_veldamp.py` | Dampen pelvis XZ velocity when foot is grounded and sliding |
| `postprocess_hymotion_with_mogendit.py` | Post-process HyMotion M2M v4 with MoGenDIT (goal: 43.6% → higher qc_pass) |
| `run_blend_then_polish.py` | Run polish post-processing after pure_blend |
| `run_hybrid_blend_polish.py` | Best-of-both: hybrid approach between blending and polishing |
| `run_pure_blend_baseline.py` | Apply correction propagation without model inference (baseline) |

### F. Data Building & Preparation (8 files)

Build evaluation datasets and data splits

| Script | Purpose |
|--------|---------|
| `build_eval_datalists.py` | Generate evaluation datalists |
| `build_keypose_eval_data.py` | Create (src_motion, keyposes, target_motion) triplets for keypose guidance eval |
| `preprocess_keypose_eval_for_web.py` | Preprocess keypose eval data for web dashboard (requires GPU) |
| `precompute_t7_masks.py` | Precompute MoGenDIT adaptive masks for T7 eval datalist |
| `process_cjgame_npz.py` | CJGame MB NPZ quality check, slicing, coordinate normalization (中文) |
| `process_cjgame_phase2.py` | CJGame MB Phase 2+3: slicing, coord norm, quality check (中文) |
| `normalize_npz_split.py` | Normalize NPZ splits (first frame faces Z+) |
| `extract_eval_caption_embeddings.py` | Extract caption embeddings for evaluation |

### G. Adaptive Mask Computation (2 files)

| Script | Purpose |
|--------|---------|
| `compute_adaptive_masks.py` | Lightweight script: run MoGenDiT light-denoise (10 steps) only |
| `compute_adaptive_masks_for_eval.py` | Save hierarchical masks to eval_results/.../adaptive_masks/ |

### H. Repair & Blending Pipelines (5 files)

| Script | Purpose |
|--------|---------|
| `repair_and_evaluate.py` | Batch repair low-quality motion data using HyMotion M2M |
| `run_kimodo_base_pose_edit.py` | Edit KIMODO output HyMotion 135-dim SMPL representation |
| `run_keypose_imputation.py` | Keypose-based imputation (best config from eval report) |
| `lq_overlay_clean_frames.py` | (See above) Overlay clean frames |
| `multiseed_e14_best_of_n.py` | Pick best multi-seed result by lowest foot_skating_ratio |

### I. A/B Testing & Debugging (5 files)

| Script | Purpose |
|--------|---------|
| `ab_test_e14_ground_anchor.py` | A/B test ground anchor on worst-sliding cases |
| `ab_test_e14_pad_strategies.py` | A/B test padding strategies (zeros vs alternatives) |
| `debug_e14_decanon.py` | Debug E14 decanonicalization pipeline stage-by-stage |
| `demo_position_constraint.py` | Demonstrate world-space position constraints on motion |
| `diagnose_stablemotion_roundtrip.py` | Test round-trip lossy-ness of StableMotion encoding |

### J. Monitoring & Progress Tracking (5 files)

| Script | Purpose |
|--------|---------|
| `live_progress_monitor.py` | 实时进度监控 - Real-time progress monitor for full repair tasks (中文) |
| `monitor_full_repair.py` | 全量修复进度监控 - Full repair progress monitor (中文) |
| `monitor_repair_eval.sh` | Shell script for monitoring repair eval progress |
| `eval_result_watcher.py` | (Also in tools/) Watch evaluation result progress |
| `eval_m2m_checkpoint_report.py` | Generate comprehensive evaluation report |

### K. Report & Result Consolidation (8 files)

| Script | Purpose |
|--------|---------|
| `eval_m2m_checkpoint_report.py` | (See above) Comprehensive report for all converged models |
| `stablemotion_to_dashboard.py` | Post-process StableMotion outputs for dashboard import |
| `relabel_dashboard_json.py` | Relabel dashboard JSON with better identifiers |
| `merge_multiseed_results.py` | Merge multi-seed results (use best for high skating cases) |
| `select_e9_best_qc_candidate.py` | Lightweight ensemble picker for E9 best candidate |
| `rewrite_eval_captions.py` | Rewrite captions in datalists (caption-like items) |
| `serve_kf_eval_results.py` | Serve keyframe eval results via API |
| `visualize_mogendit_results.py` | Visualize MoGenDIT repair result comparisons (中文) |

### L. Baseline & Reference Methods (3 files)

| Script | Purpose |
|--------|---------|
| `run_stablemotion_e9.py` | Strict open-source StableMotion pipeline (NO tricks) |
| `test_replacement_guidance.py` | Test replacement guidance feature |
| `test_m2m_all_masks.py` | Test M2M across all mask variations with quality report |

### M. Taiji Job Management (6 files)

| Script | Purpose |
|--------|---------|
| `submit_eval.py` | Submit evaluation jobs (usage-based) |
| `submit_custom_task.py` | Submit custom command task to Taiji (adapted from tools/taiji_submit.py) |
| `submit_ablation_taiji.sh` | Submit ablation experiments to Taiji |
| `run_multiseed.sh` | Multi-seed training/evaluation launcher |
| `run_multiseed_part1.sh` | Part 1: Run multi-seed on first half of bad PIDs (machine 1) |
| `run_multiseed_part2.sh` | Part 2: Run multi-seed on second half of bad PIDs (machine 2) |

### N. Local Evaluation Launchers (12 files)

Shell scripts for local GPU machine evaluation

| Script | Purpose |
|--------|---------|
| `run_m2m_repair_eval.sh` | Run M2M repair evaluation on Taiji GPU node |
| `run_eval_m2m_repair.sh` | Run M2M repair eval (alternative launcher) |
| `run_keypose_eval.sh` | Run keypose guidance evaluation on Taiji debug machine |
| `launch_ablation_experiments.sh` | Launch ablation experiment suite |
| `launch_repair_eval_final.sh` | Final phase: quality checking + report generation |
| `launch_repair_eval_machine1.sh` | CJGame repair eval on debug machine 1 |
| `launch_repair_eval_machine2.sh` | CJGame repair eval on debug machine 2 |
| `launch_keypose_eval_web.sh` | Preprocess data + run MoGenDIT eval + start web server |
| `monitor_repair_eval.sh` | Monitor repair eval progress (shell) |
| `run_cjgame_eval_gpu0.sh` | CJGame repair eval - Debug Machine 1 |
| `run_cjgame_eval_gpu1.sh` | CJGame repair eval - Debug Machine 2 |
| `run_cjgame_report.sh` | Generate final evaluation report after repair jobs |

### O. Specialized Evaluation (4 files)

| Script | Purpose |
|--------|---------|
| `run_e9_all_models_variant.sh` | Run E9 evaluation across all model variants |
| `run_e9_full_gpu_taiji_20260428.sh` | E9 full GPU run on Taiji (2026-04-28) |
| `run_e9_jitter50_v1_taiji.sh` | E9 jitter=50 v1 on Taiji |
| `run_e9_lowq_expand_v3_corrected_taiji.sh` | E9 low-quality expand v3 (corrected) on Taiji |
| `run_e9_lowq_expand_v3_full_taiji.sh` | E9 low-quality expand v3 (full) on Taiji |
| `run_e9_lowq_expand_v4_taiji.sh` | E9 low-quality expand v4 on Taiji |
| `run_e9_stable_mogendit_refresh_20260428.sh` | E9 StableMotion + MoGenDIT refresh (2026-04-28) |

### P. Test & Validation (2 files)

| Script | Purpose |
|--------|---------|
| `find_bad_pids.sh` | Find problematic PIDs (best-of-5 E14-M with skating > 10%) |
| `run_pad_test.sh` | Padding strategy testing |

### Q. Baseline Comparison (2 files)

| Script | Purpose |
|--------|---------|
| `run_base_pose_anchor_regen_sweep.py` | Evaluate anchor-window regeneration variants (model-centric) |
| `run_quality_check_m2m.py` | (See above) Run MotionQualityChecker on M2M outputs |

### R. Subdirectory: `scripts/eval/` (4 files)

High-level evaluation orchestration

| Script | Purpose |
|--------|---------|
| `build_motion_master_list.py` | Build comprehensive motion master list for evaluation |
| `compare_baseline_vs_latest.py` | Compare baseline vs latest model variants |
| `run_e3_e8d_e14_e15_latest_v2.sh` | Re-run E3/E8-D/E14/E15 with latest M2M v2 checkpoints |
| `split_and_import_eval_v2.py` | Split and import eval v2 data to dashboard DB (port 8082) |

---

## 4. SUMMARY BY FUNCTIONAL DOMAIN

### Training & Core Infrastructure
- **Main Entry Points**: `tools/train.py`, `tools/infer.py`
- **Distributed Training**: `tools/dist_train.sh`, `tools/taiji_dist_train.sh`, `tools/taiji_submit.py`
- **Infrastructure**: Taiji job submission (9 scripts in tools/)

### Motion Representation & Format Conversion (16 scripts)
Converting between HumanML3D-263, H3D272, SMPL-85, SMPL-135, FBX, etc.
- Largest group: Model format bridging pipelines
- Key files: `convert_momask263_to_h3d272.py`, `smpl85_to_repr272.py`, `momask263_to_smpl85_sharded.py`

### Evaluation & Repair (48 scripts)
- **Parallel Workers**: 4 single-worker scripts called by parallel orchestrators
- **Parallel Launchers**: 13 scripts coordinating multi-GPU evaluation
- **Comprehensive Suites**: 10 scripts for multi-task evaluation
- Focus: M2M, MoGenDIT, KIMODO, Taiji job coordination

### Post-Processing & Quality (20 scripts)
- **E14 Polish**: 4 scripts specific to transition stitching task
- **General Post-Processing**: Blending, overlay, baseline propagation
- **Quality Metrics**: MotionQualityChecker, repair rates, skating detection

### Data Pipeline & Building (25 scripts)
- **Task-Specific**: E1-E15 task data builders (13 scripts in tools/)
- **Eval Prep**: 8 scripts in scripts/ for eval data preparation
- **Adaptive Masks**: 2 scripts for MoGenDIT mask precomputation

### Analysis & Diagnostics (20 scripts)
- **Quality Diagnostics**: 11 scripts in tools/ (jitter, jumps, embeddings)
- **A/B Testing**: 5 scripts for comparison studies
- **Monitoring**: 5 scripts for progress tracking

### Robot Simulation & KIMODO (5 scripts)
- ASAP humanoid setup and control
- KIMODO SOMA-30 skeleton bridging
- Text-to-robot command pipelines

### Taiji Platform Integration (15 scripts total)
Distributed across both tools/ and scripts/ for job submission and execution

### Report & Dashboard Integration (8 scripts)
Consolidate results, generate markdown/JSON reports for eval_dashboard

---

## 5. KEY STATISTICS

### File Distribution
```
ROOT:                 9 files
tools/:             127 files (including 2 subdirectories)
  - tools/analysis_tools/: 2 files
  - tools/robot_sim/: 3 files
scripts/:            68 files (including 1 subdirectory)
  - scripts/eval/: 4 files
---
TOTAL:              204 files
```

### Code Volume
- **Total Python/Shell Scripts**: 204
- **Total Lines of Code**: ~80,000+
- **tools/ alone**: ~29,811 lines
- **Largest components**: Evaluation (48 scripts), Conversion (16), Data Pipeline (25)

### Organization Patterns
1. **Modular evaluation**: Single-worker scripts called by parallel orchestrators
2. **Task stratification**: E1-E15 specialized task builders
3. **Format bridging**: Heavy investment in format conversion pipelines
4. **Quality-centric**: Extensive post-processing and quality checking
5. **Platform support**: Taiji GPU scheduling + local GPU support

---

## 6. CRITICAL FILES FOR NEW DEVELOPERS

**Start Here** (in order):
1. `tools/train.py` - Main training orchestration
2. `tools/infer.py` - Model inference
3. `scripts/eval_m2m_all_tasks.py` - Comprehensive evaluation entry point
4. `scripts/eval_repair_benchmark.py` - Repair evaluation pipeline

**Core Pipelines**:
- Conversion: `tools/convert_momask263_to_h3d272.py`, `tools/smpl85_to_repr272.py`
- Repair: `scripts/eval_m2m_repair_parallel.py`, `scripts/mogendit_repair.py`
- Evaluation: `scripts/eval_m2m_all_tasks.py`, `scripts/eval_m2m_checkpoint_report.py`
- Post-Processing: `scripts/postprocess_e14_*.py` (suite of 4)

---

## 7. SCRIPT EXECUTION ENVIRONMENT

### Common Dependencies
- HuggingFace Transformers & Datasets
- PyTorch with CUDA support
- Accelerate (distributed training)
- Motion representation libraries (motion data processing)
- FBX SDK (for FBX parsing)
- Taiji client (for distributed job submission)

### Typical Execution Patterns
1. **Local single-GPU**: `python tools/train.py config.py`
2. **Distributed training**: `accelerate launch tools/train.py config.py`
3. **Taiji submission**: `python tools/taiji_submit.py --config config.json`
4. **Batch evaluation**: `python scripts/eval_m2m_all_tasks.py --config config.json`
5. **Parallel repair**: `python scripts/eval_m2m_repair_parallel.py --num-gpus 8`

---

## Notes

- Scripts use task codes (E1-E15) corresponding to different motion completion scenarios
- Heavy integration with HyMotion motion database and evaluation dashboards
- Extensive KIMODO (humanoid) simulation support
- Strong emphasis on foot skating detection and velocity damping (E14 focus)
- Multi-representation format support (crucial for compatibility across different models)

