## Iteration 1 - 2026-05-27

### 预注册
假设：异常不是 buffer persistence 或训练 crash，而是 KT spectral positional encoding 本身丢失 joint identity，导致 overfit 训练 loss 可下降但生成/eval 不正常。

### 观察
1. `work_dirs/prism_overfit_100/20260526_212303/train.log` 到 epoch 1224 仍稳定，末段 loss 约 0.049-0.06，loss_rot 约 0.09-0.12。
2. `work_dirs/prism_overfit_100/eval_epoch*.json` 的生成误差仍明显偏大：epoch49 mean_l2≈1.53，epoch74≈2.35，epoch99≈2.28，epoch299_nocfg≈2.17。
3. `spectral_unified` 将 spectral coords 取 L2 norm 后，左右对称关节产生完全相同 RoPE position。

### 关键数值
重复位置包括：L/R Hip、L/R Knee、L/R Ankle、L/R Foot、L/R Collar、L/R Shoulder、L/R Elbow、L/R Wrist。22 个 body joints 只有 14 个唯一标量位置。

### 结论
根因定位到 `spectral_unified` 的 scalarization：L2 norm 去掉了谱坐标方向和符号，破坏左右/分支 identity。

## Iteration 2 - 2026-05-27

### 预注册
假设：修复 KT-RoPE 后，100-sample overfit 需要先保证实验目标本身确定，包括固定 caption variant、无随机裁剪、T5 cache 完整、dtype 路径与 PRISM 预训练配置一致。

### 行动
1. 将 `spectral_unified` 从 L2-norm scalarization 改为 signed spectral projection，并用 DFS tie-break 避免碰撞。
2. 新增 `prism_1b_tp2m_multiframe_kt_spectral_unified_t5cached_overfit100.py`，使用 100 条有效样本、关闭 prompt/frame condition dropout、关闭 translation augmentation。
3. 给 `LoadPreExtractedT5Feature` 增加 `select_idx`，overfit 配置固定为 `select_idx=0`，避免同一 motion 随机换 caption variant。
4. 提交 Taiji 1x8 V100 训练任务。第一个任务在首个 forward 失败，原因是 `use_fp16_autocast=False` 时 fp32 residual stream 进入 bf16 Linear，触发 `mat1 Float / mat2 BFloat16`。
5. 将 overfit 配置改为 `use_fp16_autocast=True` 并重新提交任务 `prism_kt_rope_overfit100_proj_0527_v2`。

### 观察
- 100 条样本 `num_frames` 最大 300，小于 `clip_len=360`，不会随机裁剪，只会 replicate padding。
- 通过 transform 映射检查，100 条样本对应 T5 cache 全部存在。
- 15000 checkpoint 能加载；RoPE buffer missing keys 预期可接受，因为 buffer 由新 KT-RoPE 重新生成。


## Iteration 2026-05-28 no-fp16 overfit stability check

### 假设与预注册
- 假设：旧 overfit 任务 epoch36 step2 的 NaN 由 `use_fp16_autocast=True` 在 V100 上触发的 fp16 数值溢出/不稳定导致，而不是 100 条样本、T5 缺失或 KT-RoPE 投影本身导致。
- 预期：保持同一 100 valid annotation、固定 cached T5、同一 batch/crop，只关闭 fp16 autocast 并修正 bf16 Linear 输入 dtype 后，应能跨过旧故障点 epoch36 step2 且 loss finite。

### 执行结果
- 旧 run `work_dirs/prism_overfit100_kt_projected_t5cached/20260527_223526/train.log`：epoch36 step1 `loss=0.5051`，epoch36 step2 起 `loss=nan/loss_transl=nan/loss_rot=nan`。
- 新 run `work_dirs/prism_overfit100_kt_projected_t5cached_nofp16/20260528_005440/train.log`：epoch36 step2 `loss=0.3178, loss_transl=0.2281, loss_rot=0.4074`；epoch40 后仍无 `nan/inf/Traceback`。
- Taiji task `prism_kt_rope_overfit100_nofp16_0528` 状态 `TRAINING_RUNNING`。

### 结论
- 根因高置信定位为 fp16 autocast 数值不稳定。正确训练策略是 `use_fp16_autocast=False`，并在 transformer block/proj_out 对 bf16 Linear 输入做显式 dtype 对齐。
- 训练继续等待 checkpoint 后运行 cached-T5 MPJPE/MPJRE overfit 评估。


## Iteration 2026-05-28 cached-T5 overfit evaluation setup

### 目标
- 在 no-fp16 overfit 训练 checkpoint 上直接评估训练条件下的 text/GT/model output gap。
- 评估必须使用 cached T5 embedding，不走在线 tokenizer/text_encoder，因为 overfit config 中 tokenizer/text_encoder=None。

### 产出
- 新增 `tools/eval_prism_overfit_cached_t5.py`：加载同一 dataloader 的 `t5_text_embeds/t5_text_mask`，按 PRISM denoising 生成 motion，输出 transl_l2、rot6d_l2、MPJRE、MPJPE 以及 caption 样例。
- 新增 `tools/prism_overfit_eval_watch.sh`：监控新 checkpoint 并自动评估 3 条 50-step cached-T5 样本。
- 已挂 watcher：`work_dirs/prism_overfit100_kt_projected_t5cached_nofp16/eval_watch/`，跳过了已评估的 `checkpoint-epoch_49`。

### 评估观察
- `checkpoint-epoch_49` 训练 loss 约 0.27，3 条 50-step 评估结果：MPJRE≈21.54 deg，MPJPE≈1.36，说明当前 checkpoint 尚未 overfit 到 GT；这与用户预期“loss 到 0.1 左右才应基本一致”并不矛盾。
- 训练仍稳定运行，继续等 loss 降到 0.1 附近的 checkpoint 复测。


## Iteration 2026-05-28 FSDP/checkpoint equivalence root cause

### 假设与预注册
- 用户反馈：PRISM `kt_spectral` overfit 结果完全错误，怀疑 FSDP 保存实现或网络实现。
- 预注册检查顺序：
  1. 比较 FSDP 内存中模型与导出 `model.pt` 的固定 timestep loss。
  2. 在 0 step 情况下排除 optimizer/update 影响。
  3. 若 FSDP 导出坏，继续检查 prepare 前后的保存边界。

### 关键实验
- 0 step FSDP 内存模型健康：raw-first-batch seeded t999 loss≈0.335。
- 0 step 只保存 transformer 的 `accelerator.get_state_dict` / `summon_full_params` / DCP `get_model_state_dict` 回放均坏：t999 loss≈9.78，pred_rms≈2.69。
- 原始 `work_dirs/prism_1b_tp2m_multiframe/checkpoint-iter_15000` 直接回放健康：t999 loss≈0.345。
- prepare 前调用 `bundle.state_dict_to_save()` 保存出的 checkpoint 若只含 transformer，同样坏：t999 loss≈9.86。
- prepare 前强制额外保存 frozen `vae` 和 `smpl_pose_processor` 后，回放完全恢复：t999 loss≈0.345，与原始 15000 checkpoint 一致。
- 修复 config 让 KT spectral/unified 的 `vae.save_ckpt=True`、`smpl_pose_processor.save_ckpt=True` 后，FSDP runner 的真实 `_state_dict_to_save()` 回放健康：t999 loss≈0.345。

### 结论
- 不是 FSDP 参数展开/合并错误，也不是网络 forward 在内存中错误。
- 根因是 KT config 通过 `load_from` 从 sequential PRISM checkpoint 恢复了 frozen VAE/SMPL stats，但保存 `model.pt` 时只保存 trainable transformer，丢掉了被 checkpoint 覆盖过的 frozen latent-space 组件。
- 之后 eval/replay 从 config-default `checkpoints/vermo_vae` 和默认 stats 重建 frozen 模块，导致 transformer 与 VAE latent/target 空间不一致，高噪声 timestep 误差被放大到 t999≈9.8。

### 行动
- 修改 `configs/prism/prism_1b_tp2m_multiframe_kt_spectral.py` 和 `configs/prism/prism_1b_tp2m_multiframe_kt_spectral_unified.py`：保存 `vae` 与 `smpl_pose_processor`。
- 新增/扩展诊断脚本：
  - `tools/diagnose_prism_pre_prepare_save.py`
  - `tools/diagnose_prism_fsdp_save_methods.py`

### 后续建议
- 旧 checkpoint 若已经缺少 VAE/processor，不能只靠重新 eval 修复；需要从健康父 checkpoint 重新补齐 frozen 模块，或用修复后的 config 重新保存/训练。
- 可以进一步把“load_from 覆盖过 frozen module 但 save_ckpt=False”的情况做成 runner warning，避免其它模型复现同类问题。


## Iteration 2026-05-29 KT inference rerun with external frozen modules

### 背景修正
- 用户指出 `work_dirs/prism_1b_tp2m_multiframe/checkpoint-iter_15000` 不是 KT spectral checkpoint。
- 澄清：该 checkpoint 只能作为 frozen `vae` / `smpl_pose_processor` / latent stats 来源，KT transformer 必须来自 KT overfit checkpoint。

### 执行
- 修改 `tools/eval_prism_overfit_cached_t5.py`，新增 `--frozen-module-checkpoint`，加载顺序为：
  1. 用 KT config 构建模型；
  2. 从 KT overfit checkpoint 加载 transformer；
  3. 从非 KT parent checkpoint 只过滤加载 `vae`、`smpl_pose_processor`、`__bundle_params__`。
- 修改 `tools/diagnose_prism_loss_scan.py`，加入同样的 frozen-module-only 加载选项。

### 结果
- 对 `work_dirs/prism_overfit100_kt_toporesid_fixed_t5cached_nofp16/checkpoint-epoch_199` 重跑 3-sample 50-step 推理：
  - 旧结果 MPJPE≈938.624mm；
  - 补 `checkpoint-iter_15000` frozen modules 后 MPJPE≈938.614mm，基本不变。
- 同一 checkpoint 的 fixed-timestep loss scan 仍坏：
  - t999 loss≈10.14；
  - random loss≈0.75-0.97。

### 结论修正
- “补 VAE/processor”可以解释并修复 0-step/父 checkpoint 重保存时的 latent-space mismatch，但不能修复这个已保存的 epoch199 KT checkpoint。
- 对 epoch199 这类旧 checkpoint，问题已经进入 saved transformer / FSDP checkpoint 等价性或训练时/保存时代码不一致范畴；推理侧只加载非 KT parent 的 frozen modules 不够。


## Iteration 2026-05-29 clean KT savefix rerun

### 假设与预注册
- 假设：旧 `checkpoint-epoch_199` 的坏结果来自历史训练/保存产物，当前代码加上 frozen module 自包含保存后，正式训练 hook 写出的新 KT spectral/toporesid checkpoint 应该能独立加载并正常推理。
- 预期：
  1. 新 run 的 `model.pt` 应包含 `transformer`、`vae`、`smpl_pose_processor`、`__bundle_params__`。
  2. 新 checkpoint 不需要 `--frozen-module-checkpoint`，固定 timestep loss 不应出现旧 checkpoint 的 t999≈10。
  3. 新 checkpoint 的 cached-T5 50-step 推理应明显优于旧 3x50 MPJPE≈938.6mm。

### 执行
- 新增 clean rerun config：`configs/prism/prism_1b_tp2m_multiframe_kt_spectral_unified_t5cached_overfit100_toporesid_savefix_0529.py`。
- 在 `lzy_debug_machine_2` 启动 8xV100 训练：
  - work dir: `work_dirs/prism_overfit100_kt_toporesid_savefix_0529`
  - log: `logs/prism_savefix_0529/train.out`
  - parent load_from: `work_dirs/prism_1b_tp2m_multiframe/checkpoint-iter_15000`
- 第一保存点：`checkpoint-epoch_19`。

### 观察
- 模型摘要确认 `vae.save_ckpt=True`、`smpl_pose_processor.save_ckpt=True`。
- `checkpoint-epoch_19/model.pt` 顶层 keys：
  - `transformer` 825 keys
  - `vae` 246 keys
  - `smpl_pose_processor` 2 keys
  - `__bundle_params__`: `latents_mean`、`latents_std`
- 训练 loss：
  - epoch1 loss≈0.2880
  - epoch20 loss≈0.2010
  - epoch39 loss≈0.1708
  - epoch44 loss≈0.1543
- `checkpoint-epoch_19` 独立回放：
  - single sample t999 loss≈0.2866
  - 4-sample scan：random loss≈0.132-0.143，t999 loss≈0.3423
  - 3x50 cached-T5 inference MPJPE≈366.7mm；其中样本 0/2 约 100/158mm，样本 1 尚未 overfit，约 842mm。
- `checkpoint-epoch_39` 独立回放：
  - 4-sample scan：random loss≈0.111-0.124，t999 loss≈0.3425
  - 3x50 cached-T5 inference MPJPE≈144.0mm；三条分别约 86/259/87mm。
- `checkpoint-epoch_59` 独立回放：
  - 4-sample scan：random loss≈0.100-0.115，t999 loss≈0.3384
  - 3x50 cached-T5 inference MPJPE≈143.0mm；三条分别约 84/258/87mm。
- `checkpoint-epoch_79` 独立回放：
  - 4-sample scan：random loss≈0.091-0.106，t999 loss≈0.3387
  - 3x50 cached-T5 inference MPJPE≈123.0mm；三条分别约 82/194/93mm。

### 结论
- 当前代码 + savefix config 的正式训练 checkpoint 是自包含且可回放的；没有复现旧 `checkpoint-epoch_199` 的 t999≈10 或 3x50≈938mm 灾难。
- 旧 `checkpoint-epoch_199` 不应继续作为当前 KT spectral 实现正确性的证据；它更像历史坏 checkpoint / 历史代码状态产物。
- 训练仍在继续，下一步可观察 `checkpoint-epoch_99+` / full run，用更接近 overfit 的 checkpoint 确认最终推理指标；但当前 epoch39/59/79 已足以排除旧 938mm 灾难在新保存路径中复现。


## Iteration 2026-05-29 (INFRA + 验证) epoch_260 full 100-sample eval + 网站可视化
### 类型
基础设施（多卡分片 eval + 根对齐指标 + overfit_viewer）+ 最终 overfit 正确性验证

### 目标
savefix 训练已跑满（max_epochs=260，checkpoint-epoch_260 已落盘）。对最终 checkpoint 在
全部 100 个 overfit 样本上做 cached-T5 推理，量化 MPJPE/MPJRE 并可视化，回答"savefix 后
overfit 推理是否正确"。

### 执行
- `tools/eval_prism_overfit_cached_t5.py` 新增 `--start-index`，支持按样本分片（多 GPU 并行）。
  - 修正 `safe_sample_key` 用全局 index 命名，避免分片间 key 碰撞。
- 新增 `scripts/eval/run_prism_overfit_eval_sharded.sh`：8×V100 分片并行（每片 13 样本），
  全部子进程为同一前台进程的子进程，靠保持 taiji_exec 会话存活防止被杀。
  - 关键教训：taiji_exec 的 PTY 会话关闭会杀掉所有后代进程，nohup/setsid 都救不回来；
    必须让 taiji_exec 前台命令一直运行（本地 Shell 后台保活）才能跑长任务。
- 新增 `scripts/eval/aggregate_prism_overfit_positions.py`：除存量 full MPJPE 外，额外计算
  **根对齐 MPJPE**（逐帧减 pelvis），分离"局部姿态误差"与"全局轨迹漂移"。
- 启动 `motion_annot_web/overfit_viewer/app_prism.py --port 8096 --eval-dir <epoch260 positions>`。

### 执行结果（100/100 样本，epoch_260，50 步，decode 360）
- MPJRE: mean 5.0°, median 5.0°, p90 6.4°, max 8.4°（全样本，无离群）。
- 根对齐 MPJPE: mean 36.1mm, median 33.6mm, p90 52.1mm, max 94.4mm（全样本，无离群）。
- full MPJPE: median 68.1mm，但 mean 被拉到 240.7mm，全因 1 个极端离群
  `0024_CMP002832_2` full=15876.8mm(15.9m)，而其根对齐仅 94.4mm、MPJRE 6.65°。
- worst case 全是长序列（T=254~300）：full 175~283mm 但根对齐 24~55mm、MPJRE 4~8°。
- best case 是短序列（T=78~135）：full 21~25mm。
- 重算 full 与 NPZ 内 stored mpjpe_mm 完全一致（240.657≈240.657），确认计算无误。
- 对照旧坏 checkpoint：同口径 full MPJPE≈938mm。

### 预测 vs 实际
- 预测：savefix 后正式 checkpoint 推理应明显优于旧 938mm 灾难。
- 实际：局部姿态/旋转已完全 overfit（MPJRE≈5°、根对齐 MPJPE≈36mm，全样本无离群）；
  残余误差几乎全在全局根轨迹漂移（abs_rel 平移 rollout 积分），长序列累积、个别样本爆掉。

### 结论
- **savefix 修复正确，闭环成立**：旧 938mm 是历史坏 checkpoint + frozen 模块未自包含保存导致
  推理用了 config-default latent space；当前自包含 checkpoint 上 overfit 已成立——模型记住了
  100 条样本的旋转与局部姿态。
- 残余 full MPJPE 不是网络/latent 实现 bug，而是根平移表示（abs_rel + rollout 积分）在长序列上的
  全局轨迹漂移；个别样本（0024）出现速度尖峰导致积分爆炸 15.9m，但其局部姿态仍正确。

### 回退
无（仅新增工具脚本 + eval 脚本加 --start-index，未改被调试的模型代码）。
