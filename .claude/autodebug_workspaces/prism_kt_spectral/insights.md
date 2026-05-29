status: running

# 活跃结论
- PRISM `kt_spectral` / `spectral_unified` 的 0-step/父 checkpoint 重保存问题：`model.pt` 漏保存被 `load_from` 恢复过的 frozen VAE/SMPL stats，会导致回放/eval 使用 config-default latent space。
- 但 2026-05-29 复跑证明：对已保存的 `checkpoint-epoch_199`，推理侧只从非 KT parent `checkpoint-iter_15000` 补 `vae` / `smpl_pose_processor` / latent stats 不能修复结果。
- 因此当前 epoch199 overfit 推理坏，不能再单独归因于 VAE 缺失；还存在 saved transformer / FSDP checkpoint 等价性或训练时/保存时代码不一致问题。
- 2026-05-29 clean rerun 已证明：当前代码 + frozen module 自包含保存后，正式训练 hook 写出的新 KT spectral/toporesid checkpoint 可以独立加载并正常回放/推理。

# 证据
- 原始 `work_dirs/prism_1b_tp2m_multiframe/checkpoint-iter_15000`：t999 loss≈0.345，pred_rms≈0.937。
- 0 step FSDP 只导出 transformer：t999 loss≈9.78，pred_rms≈2.69。
- prepare 前 `bundle.state_dict_to_save()` 只含 transformer：t999 loss≈9.86。
- prepare 前额外保存 frozen modules：t999 loss≈0.345。
- 修复 config 后 FSDP runner 实际 `_state_dict_to_save()`：t999 loss≈0.345。
- `checkpoint-epoch_199` + `--frozen-module-checkpoint checkpoint-iter_15000` 推理：3x50 MPJPE≈938.614mm，和旧结果≈938.624mm 基本一致。
- `checkpoint-epoch_199` + frozen parent loss scan：t999 loss≈10.14，random loss≈0.75-0.97。
- clean rerun `work_dirs/prism_overfit100_kt_toporesid_savefix_0529/checkpoint-epoch_19`：
  - `model.pt` 顶层包含 `transformer`、`vae`、`smpl_pose_processor`、`__bundle_params__`。
  - 独立 loss scan：4-sample random loss≈0.132-0.143，t999 loss≈0.3423。
  - 3x50 cached-T5 inference：MPJPE≈366.7mm。
- clean rerun `checkpoint-epoch_39`：
  - 独立 loss scan：4-sample random loss≈0.111-0.124，t999 loss≈0.3425。
  - 3x50 cached-T5 inference：MPJPE≈144.0mm。
- clean rerun `checkpoint-epoch_59`：
  - 独立 loss scan：4-sample random loss≈0.100-0.115，t999 loss≈0.3384。
  - 3x50 cached-T5 inference：MPJPE≈143.0mm。
- clean rerun `checkpoint-epoch_79`：
  - 独立 loss scan：4-sample random loss≈0.091-0.106，t999 loss≈0.3387。
  - 3x50 cached-T5 inference：MPJPE≈123.0mm。
- 对比旧坏 checkpoint：同口径 3x50 MPJPE≈938.6mm；新 run 在 epoch39 已明显恢复，且训练还未完成。

# 已做修复
- `configs/prism/prism_1b_tp2m_multiframe_kt_spectral.py`：设置 `vae.save_ckpt=True`、`smpl_pose_processor.save_ckpt=True`。
- `configs/prism/prism_1b_tp2m_multiframe_kt_spectral_unified.py`：设置 `vae.save_ckpt=True`、`smpl_pose_processor.save_ckpt=True`。
- `configs/prism/prism_1b_tp2m_multiframe_kt_spectral_unified_t5cached_overfit100_toporesid_savefix_0529.py`：clean rerun config，隔离旧坏 work_dir，保存间隔 80 iter。
- 新增 `tools/diagnose_prism_pre_prepare_save.py` 用于保存 prepare 前状态并验证 frozen module 缺失。
- 扩展 `tools/diagnose_prism_fsdp_save_methods.py`，增加 `runner_state_dict_to_save` 路径验证真实 runner 保存行为。

# 仍然有效的历史结论
- 旧 `spectral_unified` 的 L2-norm scalarization 会造成左右关节 RoPE position 碰撞，已经用 signed/topology residual 修复。
- 100-sample overfit 必须固定 T5 caption variant；当前有效 annotation 为 `data/annotation/train_overfit_prism_100_valid.json`。
- V100 上旧 `use_fp16_autocast=True` 曾在 epoch36 step2 触发 NaN；当前稳定路径是 no-fp16 autocast + dtype alignment。

# 下一步建议
- 旧的错误 `model.pt` 若已经缺少 VAE/processor，补齐 frozen 模块只解决 latent-space 缺失；若 transformer 保存本身已经不等价，必须重新从健康 in-memory run 保存或重训。
- 当前 clean rerun 可继续跑到 `checkpoint-epoch_99+` / full run 后再评估最终指标；截至 epoch79 趋势保持，根因闭环为“旧 checkpoint 历史坏产物 + config 保存不自包含”，不是当前网络实现/FSDP 导出仍坏。
- 后续可在 runner 加 warning：当 `load_from(load_scope='model')` 加载了 frozen module 但该 module `save_ckpt=False` 时，提示 checkpoint 将不可自洽。
