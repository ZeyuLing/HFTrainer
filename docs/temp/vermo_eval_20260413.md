# VerMo 16k SFT 全任务评测报告 (2026-04-13)

## 模型信息

| Model | Config | Checkpoint | Base LM |
|-------|--------|------------|---------|
| **Llama-1B** | `vermo_sft_16k_llama1b_wavtokenizer.py` | `checkpoint-epoch_9` (step≈62000) | Llama-3.2-1B-Instruct |
| **Qwen-1.7B** | `vermo_sft_16k_qwen1.7b_wavtokenizer.py` | `checkpoint-epoch_5` (step≈41000) | Qwen3-1.7B |

评测在 GPU 7 (V100 32GB) 上运行，每任务 5 样本，`do_sample=False`, `max_new_tokens=4096`。

---

## 结果汇总

### Llama-1B (epoch_9)

| Task | Motion NPZ | Caption TXT | Audio WAV | Error | 备注 |
|------|-----------|-------------|-----------|-------|------|
| t2m_1p | 2 | — | — | 2 | torchaudio crash; 1 text fallback (motion tokens 未结尾) |
| t2m_2p | 1 | — | — | 2 | 同上；生成的 NPZ 只有单人 |
| m2t_1p | — | 5 ✅ | — | 0 | |
| m2t_2p | — | 5 ✅ | — | 0 | |
| m2d | — | 5 ⚠️ | 0 | 0 | 2/5 生成了 audio tokens 但未解码；3/5 输出 caption/genre text |
| d2m | 2 | 3 | — | 0 | 3/5 输出 caption 而非 motion (任务混淆) |
| s2g | 5 ✅ | — | — | 0 | |
| pred | 4 | 1 | — | 0 | 1/5 motion tokens 被 max_new_tokens 截断 |
| inbetween | 1 | 4 ⚠️ | — | 0 | **4/5 输出 caption** (严重任务混淆) |

### Qwen-1.7B (epoch_5)

| Task | Motion NPZ | Caption TXT | Audio WAV | Error | 备注 |
|------|-----------|-------------|-----------|-------|------|
| t2m_1p | 2 | — | — | 3 | torchaudio crash |
| t2m_2p | 4 | — | — | 1 | NPZ 均为单人 |
| m2t_1p | — | 3 | — | 2 | torchaudio crash |
| m2t_2p | — | 5 ✅ | — | 0 | |
| m2d | — | 5 ⚠️ | 0 | 0 | 5/5 生成 audio tokens 但均未解码为 WAV |
| d2m | **5** ✅ | — | — | 0 | **Qwen 完胜 Llama** |
| s2g | 4 | — | — | 1 | torchaudio crash |
| pred | 3 | 2 | — | 0 | 2/5 输出 caption (任务混淆) |
| inbetween | 2 | 2 ⚠️ | — | 1 | 2/5 motion tokens 未解析 |

### 两模型对比

| 任务 | Llama-1B | Qwen-1.7B | 胜者 |
|------|---------|-----------|------|
| m2t | 10/10 ✅ | 8/10 (2 env crash) | Llama (env luck) |
| d2m | 2/5 | **5/5** | **Qwen** |
| s2g | 5/5 | 4/5 (1 env crash) | 相当 |
| pred | 4/5 | 3/5 | Llama 略好 |
| inbetween | 1/5 | 2/5 | **Qwen** (但都差) |
| t2m_2p | 1/5 | 4/5 | **Qwen** |

---

## Bug 分析：推理问题 + 训练问题

### P0-1: 推理 TASK_PROMPTS 全部 OOD（8/9 任务）

**文件**: `hftrainer/pipelines/motion/vermo_pipeline.py` line 11-21

训练时从 `task.templates`（每个任务 40+ 模板）中随机选取；推理时 `TASK_PROMPTS` 是手写的固定字符串，**8/9 不在训练 templates 中**。

| 推理 task | 推理 prompt | 训练中匹配？ |
|-----------|------------|-----------|
| t2m_1p | "Generate motion sequence from the given caption." | ❌ OOD |
| t2m_2p | "Generate multi-person motion sequence from the given caption." | ❌ OOD |
| m2t_1p | "Describe the given motion." | ❌ OOD |
| m2t_2p | "Describe the given multi-person motion." | ❌ OOD |
| **m2d** | **"Dance to the given music."** | **✅ EXACT** (唯一匹配) |
| d2m | "Add music to this dance." | ❌ OOD |
| s2g | "Add body movements to speech." | ❌ OOD |
| pred | "Predict future motion from past motion." | ❌ OOD |
| inbetween | "Interpolate between two motion segments." | ❌ OOD |

**影响**：OOD prompt 导致模型对有歧义的 task（相同 input modality、不同 output modality）做错误的 output modality 选择。

### P0-2: 评测脚本 inbetween 输入构造错误

**文件**: `tools/batch_infer_vermo.py` `get_task_inputs()`

inbetween 任务需要 `past_motion` + `future_motion`，但代码中两者都指向 `entry['motion_path']`——**同一段完整 motion 同时作为 past 和 future**。正确做法是将 motion 按比例拆分成前/中/后三段，past 取前段，future 取后段。

### P1-1: 训练侧——SFT label mask 削弱 template 消歧

**文件**: `hftrainer/models/motion/vermo/processor.py` line 1256-1275

SFT 阶段 `instruction_stage=True`，label mask 将 `<|begin_of_output|>` 之前的所有 token 设为 -100，**包括 template 文本**。这意味着 SFT 不对 template→output modality 的映射计算 loss，模型仅通过 attention 读取 template。

这导致 pretrain 阶段学到的 "template X → output modality Y" 关联在 SFT 后被削弱。当推理时 prompt 是 OOD 的，模型回退到高频 prior（m2t 的 caption 输出），造成任务混淆。

**关键歧义对**：

| 任务 A | 任务 B | 共享 input modality | 不同 output | 训练频率 |
|--------|--------|-------------------|-------------|---------|
| d2m | m2t | `motion` | music vs caption | d2m << m2t |
| d2m | g2s | `motion` | music vs audio | d2m << g2s |
| pred | inbetween | `past_motion` | future_motion vs middle_motion | pred ≈ inbetween |

当 template OOD 时，模型默认输出训练中更高频的 modality (caption)。

### P1-2: 训练侧——任务频率严重不均衡

训练数据 `train_hq_motionhub_hymotion.json`（824K 条）中：
- **几乎所有**条目有 motion + caption → t2m/m2t 可选
- **仅 ~51K** 条 (aist++) 有 music_path → d2m/m2d 可选
- **仅 ~2K** 条 (ted_db) 有 audio_path → s2g/g2s 可选
- pred/inbetween 由 transform 从每条 motion 拆分

`task_mode='auto'` 按数据可用 modality 自动选 task，没有均衡采样。结果：
- m2t/t2m 训练样本量 >> d2m/m2d >> s2g
- 模型对 m2t 形成强 prior，遇到歧义输入时默认输出 caption

### P1-3: M2D audio output 解析路径不完整

**文件**: `tools/batch_infer_vermo.py` line 402-424 + `vermo_backend.py` line 335-363

两个层面的问题：

1. **backend `text_to_output_modal()`**：`locate_modal()` 从 raw text 提取各 modality substring，然后对 Audio 类型调用 `string2audio()` → `audio_tokenizer.decode()`。Qwen 5/5 都生成了 audio tokens，说明模型输出正确，但 decode 后的 tensor 在 `batch_infer_vermo.py` 的 `torchaudio.save()` 处 crash（channel_layout=0x0）。

2. **batch_infer 的 save 逻辑**：`for key, value in output.items()` 遍历 output_dict 时，`key` 是 modality class（如 `Music`），`getattr(key, 'name', None)` 拿到 `'music'`，但保存 branch 只检查了 `modal_name == 'audio'`，**没有检查 `'music'`**。所以即使 audio tensor decode 成功，M2D 的 music output 也不会被保存为 WAV。

### P2: torchaudio FFmpeg 环境 bug

`RuntimeError: Failed to create input filter: "time_base=1/24000:sample_rate=24000:sample_fmt=flt:channel_layout=0x0"`

WavTokenizer decode 输出的 tensor 的 channel 信息为 0，torchaudio FFmpeg backend 不接受。这是纯环境问题，不影响模型质量。

### P2: T2M_2P 只生成单人 motion

T2M_2P 的 NPZ 中没有 `<|next_person|>` separator，说明模型只输出了单人 motion token sequence。根因是双人训练数据不足——`ComposeMultiPerson` 的 `compose_prob=0.2` 意味着只有 20% 的样本被合成为多人，且仅限于 motion-only 样本（`skip_with_audio=True` 跳过有音频的数据）。

---

## 修复优先级

### P0（立即修复，不需要重训）

1. **`vermo_pipeline.py` TASK_PROMPTS**：从 `task.templates` 中选取，而非手写 OOD prompt
2. **`batch_infer_vermo.py` inbetween 输入**：对输入 motion 做 split（前 20% + 后 20%）
3. **`batch_infer_vermo.py` M2D 保存逻辑**：增加 `modal_name == 'music'` 分支
4. **torchaudio save**：在 save 前对 1D tensor unsqueeze / 设置 channel layout

### P1（下轮训练改进）

1. **SFT label mask 策略**：考虑在 SFT loss 中包含 template tokens（或至少 task_bos..task_eos 段），强化模型对 task template 的敏感性
2. **任务采样均衡**：在 `MotionhubMultiTaskMultiAgentDataset` 中增加 per-task sampling weight，确保低频任务（d2m/m2d/s2g）有足够训练量
3. **增加双人数据比例**：提高 `compose_prob` 或引入真实双人数据集

### P2（后续优化）

1. 升级 torchaudio/FFmpeg 兼容性
2. 增加 `max_new_tokens` 到 8192 避免长 motion 被截断
3. 增加评测 metrics（FID, foot sliding, boundary smoothness）

---

## 文件

- 评测脚本: `tools/batch_infer_vermo.py` (已更新 MODELS + annotation 加载)
- 推理 pipeline: `hftrainer/pipelines/motion/vermo_pipeline.py` (TASK_PROMPTS 需修复)
- 推理 backend: `hftrainer/pipelines/motion/vermo_backend.py`
- 训练 processor: `hftrainer/models/motion/vermo/processor.py` (SFT label mask 逻辑)
- Task 定义: `hftrainer/models/motion/vermo/task_utils/task_lib/`
- Modality 定义: `hftrainer/models/motion/vermo/task_utils/modality.py`
- 训练数据: `data/annotation/train_hq_motionhub_hymotion.json` (824K, 无任务均衡)
- 结果目录: `work_dirs/vermo_eval/{llama1b,qwen1.7b}/{task}/`
