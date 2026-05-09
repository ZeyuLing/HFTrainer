# 编辑信息传输方式对比：Reactive Channel vs SDEdit (加噪去噪)

## 1. 问题定义

在动作/图像编辑任务中，模型需要知道**编辑前的内容**才能做出有针对性的修改。核心问题是：**编辑前的信息如何传递给模型？**

当前有两种主流范式：

| 范式 | 代表 | 核心思路 |
|------|------|---------|
| **显式通道传输** | VACE (HyMotion M2M reactive), InstructPix2Pix, Flux Kontext | 编辑前的信息通过独立通道/token 显式提供给模型 |
| **隐式噪声传输** | SDEdit, MoGenDIT | 对编辑前的内容加噪声，再去噪，信息通过噪声残留隐式传递 |

## 2. 显式通道传输

### 2.1 Channel Concatenation（InstructPix2Pix, VACE）

**原理**：将编辑前的图像/动作编码后，与噪声 latent 在 channel 维度拼接，作为模型输入。

```
InstructPix2Pix: model_input = cat([z_t, z_source], channel_dim)  # 8ch -> UNet
VACE (M2M):      model_input = cat([x_t, inactive, reactive, mask], dim=-1)  # 4*D -> DiT
```

**InstructPix2Pix (2023)**：
- Source image 通过 VAE 编码为 latent，与 noisy target latent 拼接（4+4=8 channels）
- UNet 的 input conv 扩展为 8ch
- 模型同时看到 "编辑前的是什么" + "当前要去噪的是什么" + "文本说要怎么改"
- 支持 CFG 双轴调控：image guidance scale + text guidance scale

**VACE / HyMotion M2M (2025)**：
- reactive 通道携带编辑前的动作（mask=1 区域的 pre-edit 值）
- inactive 通道携带保持不变的动作（mask=0 区域）
- 模型看到完整的 "编辑前状态" + "哪里需要改" + "当前噪声状态"

### 2.2 Token/Sequence Concatenation（Flux Kontext, 2025）

**原理**：将 reference image 编码为 token 序列，与 target noisy tokens 在序列维度拼接。

```
Flux Kontext: sequence = [target_tokens, ref_image_tokens]  # 序列维度拼接
              3D RoPE: t=0 for target, t=1 for reference
```

**Flux Kontext** (Black Forest Labs, 2025)：
- Reference image 通过 frozen VAE 编码 → patchify → 得到 token 序列
- 与 target noisy latent tokens 在序列维度拼接
- 3D RoPE 位置编码：`(t, h, w)`，t=0 为 target，t≥1 为 reference images
- Dual-stream + Single-stream transformer blocks 联合处理
- 优点：支持多张 reference image、不同分辨率、零样本编辑

### 2.3 显式传输的共同特点

**优点**：
- **信息无损**：编辑前的完整信息直接传给模型，不经过任何噪声损坏
- **精确控制**：模型可以精确地 "看到" 编辑前每个位置的值，决定保留什么、修改什么
- **可训练的编辑语义**：通过训练数据（编辑前→编辑后 pair），模型学会 "怎么根据 source 做编辑"
- **不依赖噪声强度超参**：不需要调 SDEdit 的 noise level

**缺点**：
- **需要训练数据**：必须有（source, edited, instruction）三元组训练数据
- **增加模型输入维度**：channel concat 增加 input channels，token concat 增加序列长度
- **训练成本更高**：模型需要从头或 finetune 学习利用额外通道

## 3. 隐式噪声传输（SDEdit）

### 3.1 原理

SDEdit (Meng et al., 2021) 的核心思路：
1. 对 source image/motion 加噪到时间步 t（前向扩散）
2. 从加噪后的 x_t 出发，用新的 text prompt 去噪
3. 噪声强度 t 控制 "保留多少 source 信息"：t 小→保留多、t 大→修改多

```python
# SDEdit
x_noisy = sqrt(α_t) * x_source + sqrt(1-α_t) * noise   # 加噪
x_edited = denoise(x_noisy, t_start=t, prompt="new prompt")  # 从 t 开始去噪
```

### 3.2 MoGenDIT 的去噪修复

MoGenDIT 的 denoise/ada_denoise 模式本质上就是 SDEdit：
- 对整个动作加少量噪声（step=10，约 10/1000 的噪声）
- 去噪后，有问题的帧/维度被模型 "修正"，没问题的帧基本保持不变
- 不需要显式的编辑通道——编辑前的信息通过噪声残留传递

### 3.3 SDEdit 的特点

**优点**：
- **零训练成本**：不需要编辑 pair 数据，不需要修改模型架构，直接用预训练模型
- **模型输入不变**：不增加通道数或序列长度
- **通用性强**：任何预训练的生成模型都可以做 SDEdit，不需要专门训练

**缺点**：
- **信息有损**：加噪过程不可逆地损坏了 source 信息，噪声越大损失越多
- **精度-编辑力度权衡**：noise level 是关键超参，调太小→编辑不够，调太大→source 信息丢失
- **无法做精细局部编辑**：噪声是全局加的，不能指定 "只改这里、保留那里"（除非配合 mask imputation）
- **不可控**：模型不知道 "用户想要什么编辑"，只能靠 text prompt 引导去噪方向
- **不稳定**：对同一 source + prompt，不同噪声强度可能得到完全不同的结果

## 4. 商业级图像/视频编辑模型的选择

### 4.1 趋势总结

| 模型 | 年份 | 编辑信息传输方式 | 说明 |
|------|------|----------------|------|
| SDEdit | 2021 | 加噪去噪 | 开创性工作，零训练 |
| InstructPix2Pix | 2023 | Channel concat | 首个端到端训练的图像编辑模型 |
| Prompt-to-Prompt | 2023 | Cross-attention manipulation | 零训练，操纵 attention map |
| IP-Adapter | 2023 | Cross-attention injection | 通过 adapter 注入 reference image features |
| VACE (Wan Video) | 2025 | Channel concat (inactive+reactive+mask) | 统一视频生成与编辑 |
| Flux Kontext | 2025 | Token/sequence concat | 最新 SOTA，in-context editing |
| Gemini/Imagen | 2025 | Token concat (multimodal) | Google 的 native image generation/editing |

**明确的趋势**：商业级产品**全部选择显式传输**（channel concat 或 token concat），没有用 SDEdit 作为主要编辑方案的。SDEdit 仅作为快速原型或 fallback。

### 4.2 为什么商业产品不用 SDEdit？

1. **可控性差**：用户说 "把头发变红"，SDEdit 不能保证只改头发——噪声是全局的
2. **精度不够**：在保持 source 大部分内容不变的同时做精确局部编辑，SDEdit 的 noise-fidelity tradeoff 很难调好
3. **不可复现**：同样的输入，noise level 稍有变化结果差异很大
4. **不支持复杂编辑指令**：SDEdit 只能靠 text prompt 隐式引导，不能处理结构化的编辑指令

### 4.3 最新趋势：从 Channel Concat → Token Concat

- **Channel concat**（InstructPix2Pix, VACE）：source 和 target 必须空间对齐（同分辨率），通过 channel 维度拼接
- **Token concat**（Flux Kontext, Gemini）：source 和 target 编码为独立 token 序列，通过序列维度拼接 + positional encoding 区分。不要求空间对齐，支持多个 reference、不同分辨率

Token concat 正在取代 channel concat 成为主流，因为它更灵活（任意数量的 reference image）且与 DiT/Transformer 架构天然兼容。

## 5. 对 HyMotion M2M 的分析

### 5.1 当前设计

HyMotion M2M 的 VACE 使用 channel concat：
```
model_input = cat([x_t, inactive, reactive, mask], dim=-1)  # 4 * 135 = 540 维
```

- **Completion 模式**：reactive = 0（无编辑前信息）
- **Editing 模式**：reactive = 编辑前的 LQ 动作值（在 mask=1 区域）

### 5.2 Reactive 通道是否必要？

**对 Completion 任务**：reactive 始终为 0，**完全不必要**（这就是我们 no_inactive 消融的动机）。

**对 Editing/Repair 任务**：答案取决于我们选择哪种编辑范式：

#### 如果用 SDEdit 替代 reactive：

```python
# 不需要 reactive 通道
# 对 LQ motion 加噪到 t=0.3，然后从 t=0.3 开始去噪
x_noisy = (1 - 0.3) * noise + 0.3 * LQ_motion_norm  # flow matching
x_edited = odeint(model, x_noisy, t=[0.3, 1.0])      # 从 t=0.3 开始积分
```

**可行**，但问题：
- 需要调 noise level（0.3 是一个超参），不同类型的缺陷需要不同的 noise level
- 全局加噪——好的部分也被加了噪，可能引入新的 artifact
- 没有 mask 信息——模型不知道 "哪里是坏的需要修"
- 与 MAN 训练不兼容——MAN 训练时 known 区域是 clean 的，SDEdit 把它们弄脏了

#### 如果保留 reactive 通道：

```python
# reactive 告诉模型 "编辑前这里的值是什么"
# mask 告诉模型 "这里需要修改"
# 模型可以精确地决定保留什么、修改什么
```

**优势**：
- 模型知道 "原来是什么" + "哪里需要改"，可以做最小化修改
- 不需要调 noise level 超参
- 与 mask-based workflow 完全兼容
- 支持 "repair"（修复质量问题）和 "edit"（按指令修改风格/动作）两种任务

### 5.3 结论

**Reactive 通道对 Editing/Repair 任务是有价值的**，但对当前实际使用的 Completion 任务是冗余的。

理论上 SDEdit 可以替代 reactive 做修复，但实际中：
1. SDEdit 的 noise level 很难调——不同缺陷类型需要不同强度
2. SDEdit 全局加噪会损坏好的区域
3. SDEdit 不能精确指定 "修哪里"
4. 商业产品全部选择显式传输（channel/token concat），而非 SDEdit

**推荐**：
- 短期（当前消融）：既然只做 Completion + Repair，reactive 通道在 MAN 训练下确实可以去掉（repair 也可以用 MAN imputation 而非 reactive 传输 LQ 值）
- 长期（如果要做 motion editing）：保留 reactive 或迁移到 token concat 架构（类似 Flux Kontext），通过 reference motion tokens 传输编辑前信息

## 6. 各方案对比总结

| 维度 | Reactive Channel (VACE) | SDEdit (加噪去噪) | Token Concat (Kontext) |
|------|------------------------|-------------------|----------------------|
| **信息保真度** | ✅ 无损 | ❌ 有损（噪声损坏） | ✅ 无损 |
| **编辑精度** | ✅ mask 级精确控制 | ❌ 全局，无法精确局部编辑 | ✅ 模型自学 |
| **超参敏感度** | ✅ 无超参 | ❌ noise level 关键超参 | ✅ 无超参 |
| **训练数据需求** | 需要 edit pair | 不需要（零训练） | 需要 edit pair |
| **模型改动** | 增加 input channels | 无改动 | 增加序列长度 |
| **灵活性** | 固定空间对齐 | 通用 | 任意 reference 数量/分辨率 |
| **商业采用** | InstructPix2Pix, VACE | 无（仅原型） | Flux Kontext, Gemini |
| **适合 M2M** | ✅ 当前架构 | ⚠️ 可用但不如显式 | 🔮 未来方向 |

Sources:
- [Flux Kontext Paper (arXiv:2506.15742)](https://arxiv.org/abs/2506.15742)
- [VACE: All-in-One Video Creation and Editing (arXiv:2503.07598)](https://arxiv.org/abs/2503.07598)
- [Flux Kontext Technical Analysis](https://indolte.com/posts/flux-1-kontext-flow-matching)
