# HyMotion M2M v3: Dual-Stream Condition Fusion with Timestep-Adaptive Gating

**日期**: 2026-05-08
**问题**: Caption 模型几乎不理解文本输入；motion condition 的 VACE shortcut 完全淹没了 text gradient
**目标**: 设计新方案同时理解 motion condition 和 text condition，支持任意 condition-target pattern
**Debug 环境**: Taiji `lzy_debug_machine_1` / `lzy_debug_machine_2`

---

## 0. Executive Summary

本方案提出 **Dual-Stream Condition Fusion (DSCF)**：将 motion condition 和 text condition 从 "全部塞在同一个 input concat" 的设计，升级为 **两个独立 encoding stream + timestep-adaptive fusion gate**。核心洞察来自 Seedance 2.0 的 Dual-Branch DiT 和 Chat Image 2 的 Multi-Level Condition Injection：

> **Text 告诉模型"做什么动作"（语义 intent），Motion condition 告诉模型"在哪些帧/关节做"（空间约束）。两者的作用层级不同，不应在同一维度竞争梯度。**

关键创新点：
1. **Condition Decoupling**: Motion condition 不再通过 VACE concat 直接进入 x_t 同维度空间（这会形成 shortcut），而是通过独立的 motion-condition encoder 压缩为结构化 token sequence
2. **Timestep-Adaptive Fusion Gate**: 在 early timestep（noise 大）时加强 text influence，在 late timestep（接近 clean）时加强 motion condition influence——模拟人类"先理解语义再精确执行"的过程
3. **Role Tokens**: 引入 3 类 learnable role tokens [GEN]/[KEEP]/[EDIT] 标注每帧的角色，让 attention 显式区分不同区域的生成策略

这是一个兼顾 **学术新颖性**（Dual-Stream + Timestep Gating 的组合在 motion 领域首创）和 **工程可行性**（基于现有 MMDiT 架构增量修改，可复用 T2M 预训练权重）的方案。

---

## 1. 根因分析（Why Caption Fails）

### 1.1 VACE Shortcut 的数学本质

当前训练 loss 分解:

```
L_total = E_t,mask [ ||v_pred - v_gt||_mask_region ]
```

模型可以通过两条路径降低 loss:
- **Path A (Motion Condition)**: 从 VACE inactive/reactive + x_t[known]=clean 读取已知区域 → 几何插值/外推 → 预测生成区的 velocity
- **Path B (Text)**: 从 cross-attention 读取语义 → 理解动作语义 → 预测全身 motion velocity

在当前训练中:
- 84% 样本有 motion condition (Path A available)
- Path A 的梯度信号远强于 Path B（直接几何约束 vs 高维语义映射）
- 优化器会自然走阻力最小路径 → **Path A 主导，Path B 退化**

这不是单纯 "T2M 样本太少" 的问题。即使把 pure_gen 提到 30%，Path B 只在那 30% 被使用，在其余 70% 里 Path A 仍然足够 → **text encoder 的表示被 undertrained**。

### 1.2 Current Architecture 的结构性问题

```
Current: x_input = [x_t(198), reactive(198), mask(198)] = 594-dim
         + cross-attention to ctxt_input (text tokens)
         + vtxt_input add to timestep embedding (adaptive layer norm)
```

问题:
1. **Motion condition 在 input 层，Text 在 attention 层**: motion condition 直接修改每个 token 的输入表示（第一层就能看到），text 需要通过 cross-attention 间接影响（信息传递效率低）
2. **Mask-aware noise 让 x_t 自身携带 condition 值**: `x_t[known] = clean_motion`，模型可以直接从 noisy input 中抽取已知信息，完全绕过 text 和 VACE
3. **Single-path generation**: 无论是 pure T2M 还是 complex completion，走的是同一套 forward path，没有显式的 task routing

### 1.3 对比其他领域的解决方案

| 方案 | 领域 | 如何平衡 structural condition 和 semantic condition |
|------|------|--------------------------------------------------|
| **Seedance 2.0** | Video | Dual-Branch DiT 并行处理 visual/audio + cross-modal attention bridge |
| **Chat Image 2** | Image Editing | Multi-Level injection: structure (ControlNet-style), semantics (cross-attn), task (instruction tokens) |
| **Nano Banana** | Image Inpainting | Noise-aware conditioning: condition injection strength varies with timestep |
| **InstructPix2Pix** | Image | Dual condition: image condition (concat) + text instruction (cross-attn), trained with CFG on both |
| **BrushNet** | Image Inpainting | Separate condition branch (full copy of UNet encoder), adds to main UNet via residual |
| **UMO** | Motion | Frozen backbone + tiny adapter, text through normal cross-attn, motion via add |

---

## 2. 方案设计: Dual-Stream Condition Fusion (DSCF)

### 2.1 Architecture Overview

```
                    ┌─────────────────────────────────────┐
                    │         Text Condition Stream        │
                    │  Qwen3 (frozen) → ctxt tokens       │
                    │  CLIP  (frozen) → vtxt embedding    │
                    └──────────────┬──────────────────────┘
                                   │ cross-attention
                                   ▼
┌──────────────┐     ┌────────────────────────────────┐     ┌──────────────┐
│  x_t (noisy) │────▶│    Shared MMDiT Backbone       │────▶│  v_pred      │
│  + role_emb  │     │  (18 layers, from T2M 1.0)     │     │  (198-dim)   │
└──────────────┘     └────────────────────────────────┘     └──────────────┘
                                   ▲
                                   │ cross-attention (new!)
                                   │ + timestep-gated residual
                    ┌──────────────┴──────────────────────┐
                    │      Motion Condition Stream         │
                    │  CondEncoder: motion[known] → tokens │
                    │  + mask pattern encoding             │
                    │  + positional encoding (frame/joint) │
                    └─────────────────────────────────────┘
```

### 2.2 Core Components

#### Component 1: Motion Condition Encoder (CondEncoder)

**目的**: 将 "scattered known frames/joints + mask" 编码为一组 structured tokens，通过 cross-attention 注入 backbone，而不是直接 concat 在 input 里。

**关键**: 这打破了 "motion condition 直接在 input layer" 的 shortcut，让 backbone 必须通过 attention 同时消化 text tokens 和 motion-condition tokens。

```python
class MotionCondEncoder(nn.Module):
    """Encode known motion regions into condition tokens.
    
    Unlike VACE which concatenates condition in the input dimension (forming
    a shortcut), CondEncoder compresses known-region information into a set
    of tokens that interact with the backbone through cross-attention — at
    the SAME level as text tokens.
    """
    def __init__(self, motion_dim=198, feat_dim=1024, num_layers=4, 
                 num_heads=8, max_tokens=128):
        super().__init__()
        # Input: known motion (B, T, D) + mask (B, T, D)
        # We aggregate per-frame info into condition tokens
        self.input_proj = nn.Linear(motion_dim * 2, feat_dim)  # [motion, mask] -> feat
        
        # Learnable frame-position + joint-group queries
        # These learn to "ask" for specific spatial-temporal patterns
        self.cond_queries = nn.Parameter(torch.randn(max_tokens, feat_dim))
        
        # Self-attention layers to aggregate condition info
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(feat_dim, num_heads) 
            for _ in range(num_layers)
        ])
        
        # Temporal position encoding (shared with backbone)
        self.temporal_pe = nn.Embedding(360, feat_dim)
        
        # Mask density encoding: tells the model how much is known
        self.density_emb = nn.Sequential(
            nn.Linear(1, feat_dim // 4),
            nn.SiLU(),
            nn.Linear(feat_dim // 4, feat_dim),
        )
    
    def forward(self, known_motion, mask, num_frames):
        """
        Args:
            known_motion: (B, T, D) - clean motion values in known regions, 0 elsewhere
            mask: (B, T, D) - binary, 1=generate, 0=known
            num_frames: (B,) - valid frame count
        Returns:
            cond_tokens: (B, N_cond, feat_dim) - condition tokens for cross-attn
            cond_mask: (B, N_cond) - validity mask
        """
        B, T, D = known_motion.shape
        
        # Frame-level aggregation: [known_motion, mask] per frame
        frame_input = torch.cat([known_motion, mask], dim=-1)  # (B, T, 2D)
        frame_feat = self.input_proj(frame_input)  # (B, T, feat_dim)
        
        # Add temporal position
        pos_ids = torch.arange(T, device=frame_feat.device)
        frame_feat = frame_feat + self.temporal_pe(pos_ids)
        
        # Add mask density as global conditioning
        density = (1 - mask).mean(dim=-1, keepdim=True).mean(dim=1, keepdim=True)  # (B, 1, 1)
        frame_feat = frame_feat + self.density_emb(density)
        
        # Cross-attend from learnable queries to frame features
        # This compresses T×D info into N_cond tokens
        queries = self.cond_queries.unsqueeze(0).expand(B, -1, -1)
        for layer in self.layers:
            queries = layer(queries, frame_feat, frame_feat)
        
        return queries  # (B, N_cond, feat_dim)
```

**新颖性**: 不同于 ControlNet（copy entire encoder）或 UMO（单层 MLP adapter），我们用 **learnable queries cross-attend 到 frame features** 来压缩 condition。这等效于一个 "condition tokenizer"，让任意 mask pattern 都能被编码为固定数量的 tokens。

#### Component 2: Timestep-Adaptive Fusion Gate

**目的**: 在 denoising 的不同阶段，动态调整 text vs motion-condition 的影响力。

**直觉**: 
- **Early timesteps (t→0, high noise)**: 模型需要 text 来建立全局语义方向 → text influence 高
- **Late timesteps (t→1, low noise)**: 模型需要 motion condition 来精确执行空间约束 → motion-cond influence 高

这模拟了 Nano Banana 的 noise-aware conditioning 思想，但应用于 motion 领域的双条件融合。

```python
class TimestepAdaptiveFusionGate(nn.Module):
    """Dynamically balance text vs motion-condition influence based on timestep.
    
    Inspired by Nano Banana's noise-aware conditioning and Seedance 2.0's
    cross-modal attention bridge. At early timesteps (high noise), text 
    semantics dominate; at late timesteps (low noise), spatial constraints
    from motion condition dominate.
    """
    def __init__(self, feat_dim=1024):
        super().__init__()
        # Timestep → gate values
        self.gate_mlp = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.SiLU(),
            nn.Linear(feat_dim, 2),  # [text_gate, motion_gate]
            nn.Sigmoid(),
        )
        # Learnable bias: initially equal (0.5, 0.5)
        self.gate_bias = nn.Parameter(torch.tensor([0.0, 0.0]))
    
    def forward(self, timestep_emb, text_attn_out, motion_cond_attn_out):
        """
        Args:
            timestep_emb: (B, feat_dim) - from timestep encoder
            text_attn_out: (B, L, feat_dim) - cross-attn output from text
            motion_cond_attn_out: (B, L, feat_dim) - cross-attn output from motion cond
        Returns:
            fused: (B, L, feat_dim) - gated combination
        """
        gates = self.gate_mlp(timestep_emb) + self.gate_bias  # (B, 2)
        text_gate = gates[:, 0:1].unsqueeze(1)      # (B, 1, 1)
        motion_gate = gates[:, 1:2].unsqueeze(1)    # (B, 1, 1)
        
        fused = text_gate * text_attn_out + motion_gate * motion_cond_attn_out
        return fused
```

#### Component 3: Role Tokens

**目的**: 让 backbone 在 attention 阶段就知道每帧的"角色"（generate / keep / edit），而不是只通过 mask 数值间接推断。

**灵感来自 UMO 的 meta-operation tokens**，但扩展到 per-frame（UMO 是 per-frame 但 frame-level only；我们支持 per-joint-group 的 role）。

```python
class RoleEmbedding(nn.Module):
    """Per-frame role tokens injected into x_t input embeddings.
    
    Extends UMO's meta-operation tokens to per-joint-group granularity.
    Each frame receives a role embedding based on its mask pattern.
    """
    def __init__(self, feat_dim=1024, num_joint_groups=23):
        super().__init__()
        # 3 role types per joint group
        self.role_emb = nn.Embedding(3, feat_dim // num_joint_groups)
        # 0 = KEEP (mask=0), 1 = GENERATE (mask=1, completion), 2 = EDIT (mask=1, has reactive)
        
        # Project to full feat_dim
        self.proj = nn.Linear(feat_dim, feat_dim)
    
    def forward(self, mask_grid, edit_flags):
        """
        Args:
            mask_grid: (B, T, 23) - per-joint-group mask (0=keep, 1=generate)
            edit_flags: (B,) - whether this sample is edit mode
        Returns:
            role_emb: (B, T, feat_dim) - to add to x_t input embeddings
        """
        B, T, J = mask_grid.shape
        # Assign roles: 0=KEEP, 1=GEN, 2=EDIT
        roles = mask_grid.long()  # (B, T, 23), 0 or 1
        # For edit samples, mask=1 regions are EDIT (role=2)
        if edit_flags is not None:
            edit_mask = edit_flags.view(B, 1, 1).expand(B, T, J)
            roles = torch.where((roles == 1) & edit_mask, 
                               torch.full_like(roles, 2), roles)
        
        # Embed each joint group's role
        emb = self.role_emb(roles)  # (B, T, 23, feat_dim//23)
        emb = emb.reshape(B, T, -1)  # (B, T, feat_dim)
        return self.proj(emb)
```

### 2.3 Modified MMDiT Block

关键修改：在每个 transformer block 中增加一个 motion-condition cross-attention。

```python
class DualCondMMDiTBlock(nn.Module):
    """Modified MMDiT block with dual condition streams.
    
    Original: self-attn(motion) → cross-attn(text) → FFN
    New:      self-attn(motion) → cross-attn(text) → cross-attn(motion_cond) 
              → timestep-gated-fusion → FFN
    """
    def __init__(self, feat_dim, num_heads, mlp_ratio=4.0):
        super().__init__()
        # Original components (loaded from pretrained)
        self.self_attn = MultiHeadAttention(feat_dim, num_heads)
        self.cross_attn_text = MultiHeadAttention(feat_dim, num_heads)
        self.ffn = MLP(feat_dim, int(feat_dim * mlp_ratio))
        self.norm1 = nn.LayerNorm(feat_dim)
        self.norm2 = nn.LayerNorm(feat_dim)
        self.norm3 = nn.LayerNorm(feat_dim)
        
        # NEW: motion condition cross-attention (randomly initialized)
        self.cross_attn_motion_cond = MultiHeadAttention(feat_dim, num_heads)
        self.norm_motion_cond = nn.LayerNorm(feat_dim)
        
        # NEW: timestep-adaptive gate
        self.fusion_gate = TimestepAdaptiveFusionGate(feat_dim)
    
    def forward(self, x, text_tokens, motion_cond_tokens, timestep_emb,
                x_mask=None, text_mask=None, cond_mask=None):
        # Self-attention (unchanged)
        x = x + self.self_attn(self.norm1(x), mask=x_mask)
        
        # Text cross-attention (unchanged, from pretrained)
        text_out = self.cross_attn_text(self.norm2(x), text_tokens, text_tokens, 
                                         mask=text_mask)
        
        # Motion condition cross-attention (NEW)
        cond_out = self.cross_attn_motion_cond(
            self.norm_motion_cond(x), motion_cond_tokens, motion_cond_tokens,
            mask=cond_mask
        )
        
        # Timestep-adaptive fusion (NEW)
        fused = self.fusion_gate(timestep_emb, text_out, cond_out)
        x = x + fused
        
        # FFN (unchanged)
        x = x + self.ffn(self.norm3(x))
        return x
```

### 2.4 x_t 输入简化

**关键改变**: 由于 motion condition 已经通过 cross-attention 注入，x_t 的输入不再需要 VACE concat。

```
Before (v2): x_input = [x_t(198), reactive(198), mask(198)] = 594-dim
After (v3):  x_input = [x_t(198) + role_emb(198)] = 198-dim (or with minimal mask signal)
```

这意味着:
1. Input projection 从 594→1024 简化为 198→1024（**可以直接复用 T2M 预训练权重的 input projection 的前 198-dim slice！**）
2. motion condition 信息**必须**通过 cross-attention 获取，和 text 在同一层级竞争，**消除 shortcut**
3. MAN (mask-aware noise) 保留：`x_t[known]=clean` 仍然给模型微弱的位置 hint，但不再是主要 condition pathway

**Option**: 保留一个轻量 mask signal 在 input 层:
```
x_input = [x_t(198), mask(198)] = 396-dim
```
mask 仅告诉模型"哪里要生成"，但不泄露已知区域的**值**。

### 2.5 Training Strategy: Progressive Curriculum

受 KIMODO 两阶段训练和 Seedance 2.0 五层 Pipeline 启发，但统一为渐进式课程:

```
Phase 0 (Warmup, 50 epochs): 
  - 冻结 backbone, 只训练新增模块 (CondEncoder, cross_attn_motion_cond, fusion_gate, role_emb)
  - 50% pure T2M (text only) + 50% simple completion (inbetween/prefix)
  - 目的: 让新模块学会 encoding，不破坏 backbone 的 T2M 能力

Phase 1 (Joint Training, 500 epochs):
  - 解冻全部，learning rate = 5e-5 (backbone) + 1e-4 (new modules)
  - 30% pure T2M + 30% text+completion + 25% completion-only + 15% editing
  - Timestep gate 从均匀初始化 → 自然学习 early/late 分工
  - 目的: 让 backbone 学会同时处理两个 condition stream

Phase 2 (Hard-case Fine-tune, 100 epochs):
  - 加入 E4/E9/E14/E15 的 hard mask patterns (5% each)
  - SOAR post-training correction on the final model
  - 目的: 覆盖边缘 mask 分布
```

### 2.6 Inference: Dual-CFG

推理时可以分别控制 text 和 motion-condition 的强度:

```python
# Classifier-Free Guidance with dual conditions
v_uncond = model(x_t, text=null, motion_cond=null)          # 全无条件
v_text = model(x_t, text=caption, motion_cond=null)          # 仅文本
v_full = model(x_t, text=caption, motion_cond=cond_tokens)   # 全条件

# Dual-CFG (可独立调节两个 scale)
v_guided = v_uncond + s_text * (v_text - v_uncond) + s_cond * (v_full - v_text)
```

这允许:
- T2M: `s_text=7.5, s_cond=0` → 纯文本生成
- Completion: `s_text=0, s_cond=1` → 纯空间约束
- Text+Completion: `s_text=3.0, s_cond=5.0` → 平衡两者
- 精细调节: 不同任务用不同 scale 组合

---

## 3. 与现有方法的对比与新颖性分析

### 3.1 vs VACE (当前)

| 维度 | VACE | DSCF (ours) |
|------|------|-------------|
| Motion condition pathway | Input concat (first layer) | Cross-attention (all layers) |
| Text condition pathway | Cross-attention (all layers) | Cross-attention (all layers) |
| 竞争公平性 | Motion >> Text | Motion = Text (same mechanism) |
| T2M 预训练兼容 | 需要随机初始化 I/O proj | 可复用 T2M 的 input proj |
| Dual-CFG | 不支持 | 原生支持 |
| Timestep awareness | 无 | Adaptive gate |

### 3.2 vs UMO

| 维度 | UMO | DSCF (ours) |
|------|-----|-------------|
| Backbone 训练 | 冻结 | Phase 0 冻结 → Phase 1 解冻 |
| Condition granularity | Per-frame (P/G/E) | Per-joint-group |
| Condition pathway | Element-wise add (single MLP) | Cross-attention (multi-layer encoder) |
| Condition capacity | 0.207M params | ~50M params (CondEncoder) |
| Arbitrary mask support | No (frame-level only) | Yes (any frame×joint pattern) |

### 3.3 vs KIMODO

| 维度 | KIMODO | DSCF (ours) |
|------|--------|-------------|
| Known region信息 | Imputation into x_t | Cross-attention from encoded tokens |
| Representation | Global rotation (333-dim) | Local rotation + position (198-dim) |
| Text + Motion | Separate CFG | Unified Dual-CFG |
| Training | Two-phase (500K+500K) | Progressive curriculum (3 phases) |
| Inference replacement | Every step (train-consistent) | Not needed (cross-attn is inherently consistent) |

### 3.4 新颖性总结 (论文 Contribution Points)

1. **First motion model using cross-attention for spatial condition injection** — 所有现有方案要么 concat (VACE/MoGenDiT) 要么 add (UMO) 要么 imputation (KIMODO)。Cross-attention 方式首次在 motion 中使用。
2. **Timestep-Adaptive Fusion Gate for multi-modal condition balancing** — 受 noise-aware conditioning 启发，将 "timestep 决定 condition 强度" 的思想引入 motion diffusion。
3. **Dual-CFG for motion generation** — 首次提出对 text 和 spatial constraint 分别做 CFG 的范式。
4. **Progressive training curriculum that preserves T2M capability** — 解决 multi-task 训练中 catastrophic forgetting 的具体方案。

---

## 4. 实现计划与代码结构

### 4.1 新增/修改文件

```
hftrainer/models/motion/hymotion_m2m/
├── network/
│   ├── motion_cond_encoder.py     # NEW: MotionCondEncoder
│   ├── timestep_gate.py           # NEW: TimestepAdaptiveFusionGate
│   ├── role_embedding.py          # NEW: RoleEmbedding  
│   ├── hymotion_mmdit_v3.py       # NEW: DualCondMMDiT (modified from mmdit.py)
│   └── dual_cfg_sampler.py        # NEW: DualCFG inference logic
├── bundle_v3.py                   # NEW: HyMotionM2Mv3Bundle
└── ...

hftrainer/trainers/motion/
├── hymotion_m2m_v3_trainer.py     # NEW: with progressive curriculum

hftrainer/pipelines/motion/
├── hymotion_m2m_v3_pipeline.py    # NEW: Dual-CFG inference

configs/hymotion_m2m_v3/
├── _base_hymotion_m2m_v3_046b.py  # NEW: base config
├── hymotion_m2m_v3_phase0.py      # Phase 0: warmup
├── hymotion_m2m_v3_phase1.py      # Phase 1: joint training
├── hymotion_m2m_v3_phase2.py      # Phase 2: hard-case finetune
└── hymotion_m2m_v3_debug.py       # Single-GPU debug config
```

### 4.2 预训练权重利用策略

```python
# 从 T2M 1.0-Lite 加载的部分 (305/308 keys):
# - 18 transformer blocks (self_attn, cross_attn_text, FFN, norms)
# - timestep_encoder
# - text_refiner
# - vtxt_encoder, ctxt_encoder
# - input_encoder: 前 198-dim 可复用! (T2M 的 input 是 201-dim, 取前 198 维度)

# 随机初始化的部分:
# - MotionCondEncoder (~50M)
# - cross_attn_motion_cond (per block, 18 × ~4M = ~72M)
# - TimestepAdaptiveFusionGate (per block, 18 × ~2M = ~36M)
# - RoleEmbedding (~0.5M)
# - Final layer (output projection)

# Total new params: ~158M
# Total pretrained: ~304M (backbone reused)
# Total model: ~462M
```

### 4.3 Input Dimension 设计选项

**Option A (推荐): Minimal mask in input**
```
x_input = [x_t(198) + role_emb, scalar_mask(1)] = 199-dim → input_proj(199, 1024)
```
- Pro: 最接近 T2M 的 input 分布，最大化预训练复用
- Pro: motion condition 完全通过 cross-attention，消除 shortcut
- Con: MAN 时 x_t[known]=clean 仍给微弱 hint（可控，且 train-consistent）

**Option B: Mask in input (兼容性)**
```
x_input = [x_t(198), mask(198)] = 396-dim → input_proj(396, 1024)
```
- Pro: mask 信息明确，backbone 知道哪里要生成
- Con: 需要重新初始化 input_proj

**Option C: Pure x_t only (最激进)**
```
x_input = x_t(198) + role_emb → input_proj(198, 1024)  [直接复用 T2M!]
```
- Pro: 100% T2M 预训练复用，甚至 input_proj 权重也能用
- Con: backbone 完全不知道 mask pattern，全靠 cross-attention + role_emb

**选择 Option A**: 平衡预训练复用和信息完整性。scalar_mask 维度仅 +1，不影响权重复用。

---

## 5. Debug 计划 (lzy_debug_machine_1/2)

### 5.1 Phase 0: Sanity Check (1-2 天)

**目标**: 确认新架构能正常前向传播，loss 能下降。

```bash
# Step 1: 在 debug_machine_1 上跑单卡 smoke test
ssh lzy_debug_machine_1
cd /apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer

# 用 debug config (bs=4, 10 iterations, 无 text encoder)
python tools/train.py configs/hymotion_m2m_v3/hymotion_m2m_v3_debug.py

# 验证:
# 1. Loss 应该 ~0.5-1.0 (随机初始化, 比 pretrained 高)
# 2. 无 NaN/Inf
# 3. GPU 内存 < 32GB
# 4. CondEncoder 输出 shape 正确 (B, 128, 1024)
# 5. FusionGate 的 gate values 初始约 0.5
```

### 5.2 Phase 0.5: T2M Preservation Check (1 天)

**目标**: 确认 Phase 0 (冻结 backbone) 不破坏 T2M 能力。

```bash
# 在 debug_machine_2 上评测冻结的 backbone
# 先跑 Phase 0 训练 50 epochs (freeze backbone, train new modules only)
python tools/train.py configs/hymotion_m2m_v3/hymotion_m2m_v3_phase0.py \
    --cfg-options train_cfg.max_epochs=50

# 然后 eval E1 (T2M):
python tools/eval_m2m_v2_all_tasks.py \
    --config configs/hymotion_m2m_v3/hymotion_m2m_v3_phase0.py \
    --checkpoint work_dirs/hymotion_m2m_v3_phase0/checkpoint-epoch_50 \
    --task E1 \
    --save-npz

# 预期: E1 T2M 质量应该接近 T2M 1.0 baseline (因为 backbone 冻结)
```

### 5.3 Phase 1: Joint Training Pilot (3-5 天)

```bash
# 在 debug_machine_1 (8 GPU) 上跑 Phase 1 small-scale pilot
bash tools/dist_train.sh configs/hymotion_m2m_v3/hymotion_m2m_v3_phase1.py 8

# 监控:
# 1. loss_velocity 应该在 100 epochs 后 < 0.05
# 2. gate values 应该分化: early timesteps text_gate > motion_gate
# 3. 每 50 epochs 跑一次 E1 (T2M) + E3 (inbetween) 对比

# 关键检查点:
# - 100 epochs: loss 是否在下降? gate 是否在分化?
# - 200 epochs: E1 是否退化? (如果是 → 降 backbone lr)
# - 500 epochs: E1 + E3 + E4 全面评测
```

### 5.4 Dual-CFG 标定 (1 天)

```bash
# Phase 1 训练到 500 epochs 后, 在 debug_machine_2 上做 CFG scale sweep
for s_text in 1.0 3.0 5.0 7.5; do
  for s_cond in 0.0 1.0 3.0 5.0; do
    python tools/eval_m2m_v2_all_tasks.py \
        --config configs/hymotion_m2m_v3/hymotion_m2m_v3_phase1.py \
        --checkpoint work_dirs/hymotion_m2m_v3_phase1/checkpoint-epoch_500 \
        --task E1,E3,E4 \
        --text-guidance-scale $s_text \
        --motion-cond-guidance-scale $s_cond \
        --save-npz \
        --output-dir output/v3_cfg_sweep/s${s_text}_m${s_cond}
  done
done
```

### 5.5 Full Training (Taiji 提交)

```bash
# Phase 1 在 debug machine 验证成功后, 提交到 Taiji 做完整训练
python3 tools/taiji_submit.py m2m_v3_phase1 \
    configs/hymotion_m2m_v3/hymotion_m2m_v3_phase1.py \
    --host_num 4
```

---

## 6. 风险评估与 Fallback

### 6.1 Risk: Cross-attention 容量不够

**症状**: Motion condition 信息丢失，completion 任务质量大幅下降

**缓解**: 
- 增加 CondEncoder 层数 (4→8)
- 增加 max_tokens (128→256)
- 考虑在最后 6 layers 额外加入 "imputation residual"（将 known-region 的 clean 值直接加到 backbone 对应位置，但仅在后 6 层）

### 6.2 Risk: Timestep gate 不收敛

**症状**: gate values 始终 ~0.5，没有分化

**缓解**:
- 初始化 gate bias: text_bias=0.3, motion_bias=-0.3（soft prior: early text, late motion）
- 加入 gate regularization: `L_gate = ||gate_text(t=0) - 1||^2 + ||gate_motion(t=1) - 1||^2`

### 6.3 Risk: 新模块 (~158M) 训练不稳定

**症状**: gradient explosion/vanishing in new modules

**缓解**:
- 用 smaller learning rate for new modules (Phase 0 验证)
- Gradient clipping per-module
- Xavier initialization for cross-attention
- Warmup 1000 steps with lr ramp

### 6.4 Fallback Plan (如果 DSCF 完全失败)

退回到改进版 VACE + 以下 quick fix:
1. `cond_mask_prob` 0.1→0.3（已验证有帮助）
2. `pure_gen` 16%→35%
3. 加入 **text-conditioned generation mask**: 在 completion 时 30% 概率把 mask 随机 relax（让本来的 keep 区域变成 generate），强迫模型用 text 重新生成
4. Inference: Dual-CFG via null-VACE（推理时把 VACE 置零做一个 CFG branch）

---

## 7. 训练数据策略

### 7.1 Text-Motion Pairing 质量

当前 pre-extracted embeddings 来自 `qwen3_augmented/`（GPT-4 改写后的 caption）。确保:
- T2M 训练样本使用 **rewritten** caption（语义更精准）
- Completion 训练样本的 caption 也要覆盖（不能只有 motion 没有 text）

### 7.2 数据清洗

使用 `high_quality.json` (456K)，排除 low_quality (85K)。

### 7.3 扩充 T2M 数据

考虑将 HumanML3D (15K) 和 MotionX (80K) 的 text-motion pairs 引入 Phase 1 的 pure T2M 样本池：
- 这些数据有高质量 text annotation
- 用于补充 pure T2M 训练样本（当前只有 HyMotion 400h 的子集有 caption）

---

## 8. 评测计划

### 8.1 核心指标

| Task | Metrics | Threshold |
|------|---------|-----------|
| E1 (T2M) | R-precision, FID, MM-Dist | 至少达到 T2M 1.0 的 80% |
| E3 (In-Between) | MPJPE, foot_skating, boundary_smoothness | ≤ v2 uncond best |
| E4 (End-Effector) | End-effector precision, MPJPE | ≤ KIMODO |
| E10 (Part Control) | Part MPJPE, overall quality | ≤ v2 uncond |
| E1+condition | 有 condition 时 text 是否仍生效 | Δ(with-text, no-text) > 10% on R-precision |

### 8.2 Ablation 矩阵

| Ablation | 什么被验证 | Config |
|----------|----------|--------|
| No CondEncoder (text only) | 基线 T2M 能力 | `v3_ablation_text_only.py` |
| No Timestep Gate (fixed 0.5) | Gate 的价值 | `v3_ablation_no_gate.py` |
| No Role Tokens | Role embedding 的价值 | `v3_ablation_no_role.py` |
| VACE concat + DSCF | 是否可以两者兼有 | `v3_ablation_vace_plus_dscf.py` |
| CondEncoder depth 2/4/8 | 容量需求 | sweep |

---

## 9. Timeline

| Week | 任务 | 机器 | 预期产出 |
|------|------|------|---------|
| W1 (5/8-5/14) | 代码实现 + Debug smoke test | debug_machine_1 | 架构跑通, loss 下降 |
| W2 (5/15-5/21) | Phase 0 + T2M 保持验证 | debug_machine_1+2 | T2M 不退化 |
| W3 (5/22-5/28) | Phase 1 pilot (500 epochs, 8GPU) | debug_machine_1 | 初步 E1+E3 指标 |
| W4 (5/29-6/4) | Full Phase 1 (Taiji 4×8) + Eval | Taiji | 全面评测 |
| W5 (6/5-6/11) | CFG sweep + Phase 2 + 论文 | Taiji + local | 最终模型 + paper draft |

---

## 10. 论文 Story Line (Draft)

**Title**: "Motion Condition Fusion via Dual-Stream Cross-Attention with Timestep-Adaptive Gating"

**核心论点**: 
> Existing motion completion models that concatenate spatial conditions in the input layer create an optimization shortcut that suppresses text understanding. We propose Dual-Stream Condition Fusion (DSCF) that injects spatial conditions through cross-attention at the same architectural level as text, combined with a timestep-adaptive gate that naturally allocates "semantic planning" to early timesteps and "spatial execution" to late timesteps.

**Novelty claims**:
1. First analysis of the "VACE shortcut" phenomenon in multi-condition motion generation
2. Cross-attention based spatial condition injection for motion (vs concat/add/impute)
3. Timestep-adaptive multi-modal fusion gate
4. Dual-CFG paradigm enabling independent control of semantic and spatial guidance

**Experiments**:
- T2M quality preservation (vs T2M 1.0 baseline)
- Text adherence under spatial constraints (novel eval: swap captions and measure output change)
- Spatial constraint precision (vs KIMODO, MoGenDiT)
- Ablation on each component
- Dual-CFG scale analysis

---

## 11. Quick-Start: 实现第一步

```bash
# 1. 创建新文件目录
mkdir -p configs/hymotion_m2m_v3

# 2. 先实现最简版本 (无 CondEncoder, 只加 Role Token + 去掉 VACE concat)
#    验证: T2M 能力是否保持 (因为去掉 VACE 后模型应该退回 T2M 模式)

# 3. 加入 CondEncoder, 验证 condition 信息能通过 cross-attn 传递
#    测试: 给定 prefix, 模型是否能续写合理 motion

# 4. 加入 Timestep Gate, 验证 gate 分化
#    监控: gate values at t=0.1 vs t=0.9

# 5. 完整 Dual-CFG 推理
```

---

## Appendix A: 为什么不直接增加 T2M 训练比例

"把 pure_gen 从 16% 提到 50% 能否解决？"

**不能**, 原因:
1. 即使 50% pure T2M, 在剩余 50% 有 motion condition 时, text 的梯度仍被 VACE shortcut 稀释
2. 增加 pure_gen 会减少 completion 训练量 → completion 质量下降
3. 根本矛盾: **两个 condition 的 pathway 不对称**（一个在 input 层, 一个在 attention 层）

唯一的根本解法是让两个 condition 在相同的架构层级竞争。

## Appendix B: 为什么不直接用 UMO 的 Adapter 方案

UMO 的 0.207M adapter 太弱, 因为:
1. 它只有 1 层 MLP, 无法编码复杂的 per-joint mask pattern
2. 它不支持 per-joint-group 粒度 (只有 frame-level P/G/E)
3. Element-wise add 破坏了 T2M 的 input 分布 (虽然论文声称影响小, 但我们的 mask 复杂度远超 UMO)

我们的 CondEncoder 可以理解为 "UMO adapter 的超级版" — 多层 attention, per-joint-group 粒度, cross-attention 注入 (非 add)。

## Appendix C: 为什么 Imputation (KIMODO-style) 不够

Imputation 要求:
1. 训练时 x_t[known] = clean (MAN, 已实现)
2. 推理时每步替换 x_t[known] = clean

问题:
1. **text 仍被淹没**: 即使用 imputation, 模型可以从 x_t 的 known 区域直接读信息, 仍然不需要 text
2. **boundary 问题**: imputation 的 known/gen boundary 仍存在不连续 (SOAR 只能缓解)
3. **CFG 不兼容**: imputation 要求每步替换, 和 dual-CFG 的 null-condition branch 冲突

DSCF + cross-attention 从根本上消除了 "x_t 里有 clean 值" → "不需要 text" 的逻辑链。

## Appendix D: 潜在的代码实现 Bug 检查清单

在开始 v3 之前, 必须验证当前仓库没有遗留的 null_embedding bug:

```bash
# 检查 null_vtxt_feat 是否正确从 T2M checkpoint 加载
python3 -c "
import torch
ckpt = torch.load('checkpoints/HY-Motion-1.0/HY-Motion-1.0-Lite/latest.ckpt', map_location='cpu')
keys = [k for k in ckpt.keys() if 'null' in k.lower() or 'vtxt' in k.lower()]
print('Null-related keys in T2M ckpt:', keys)
for k in keys:
    v = ckpt[k]
    print(f'  {k}: shape={v.shape}, norm={v.norm():.4f}, min={v.min():.4f}, max={v.max():.4f}')
"

# 检查当前 M2M checkpoint 的 null_vtxt_feat
python3 -c "
import torch
ckpt = torch.load('work_dirs/hymotion_m2m_v2_caption_local_046b/checkpoint-epoch_498/model.safetensors', map_location='cpu')
null_keys = [k for k in ckpt.keys() if 'null' in k]
print('Null keys in M2M ckpt:', null_keys)
for k in null_keys:
    v = ckpt[k]
    print(f'  {k}: shape={v.shape}, norm={v.norm():.4f}, all_zero={v.abs().max() < 1e-6}')
"
```

**如果 null_vtxt_feat 全零** → 这就是 caption 模型失败的直接原因之一!（参考 Historical Bug Record: 2026-03-27 框架 bug）。即使框架 bug 已修, 如果当前训练的 checkpoint 是在 bug 修复前开始的, null embedding 仍可能是错的。

---

*文档结束。所有临时文档按照项目规范存储在 `docs/temp/`。*
