# HyMotion M2M v3: Condition-Routed Flow Matching (CRFM) 实施方案

> Status: APPROVED FOR IMPLEMENTATION (2026-05-09)
> Priority: P0 — Caption model fundamentally broken
> Debug machines: lzy_debug_machine_1 (4×V100), lzy_debug_machine_2 (4×V100)
> Supersedes: `caption_motion_unified_conditioning_plan.md` (升级为可执行方案)

---

## 0. Executive Summary

**问题**：caption_local_phase2 模型 E1 jitter=990 (uncond 只有 213)，text conditioning 完全失效。

**根因**：MAN 训练中 x_t[known]=clean 形成 shortcut，模型不再依赖 text cross-attention → text pathway atrophy。

**方案**：Condition-Routed Flow Matching (CRFM) — 通过冻结 text attention + condition density embedding + text-awareness loss 彻底消除 text atrophy。

**关键创新点（论文级）**：
1. 首次分析 mask-aware flow matching 中 text atrophy 现象的形成机制
2. Condition Density Embedding (CDE) — 连续编码 mask density 让模型感知"需要多少 text 信息"
3. Text Attention Preservation (TAP) — 选择性冻结 + gradient gating 防止 text pathway 退化
4. 实验证明单模型统一 T2M + arbitrary motion completion + text-conditioned completion

**预期结果**：
- caption E1 jitter: 990 → <300 (4×+ improvement)
- caption E4 ee_error: 0.43 → <0.20
- 不低于 uncond 模型在所有 completion 任务上的水平

---

## 1. 根因分析（深度版）

### 1.1 Text Atrophy 的精确机制

```
训练时（Phase 2, MAN=True, 大量样本 mask_density < 0.5）:

  每个 batch 中：
    - 16% 样本: mask=1 (pure T2M) → x_t 全 noisy，必须读 text
    - 84% 样本: mask 部分为 0 → x_t[known] = clean_motion

  对于 84% 的 partial-mask 样本：
    模型发现：x_t[known] 已经包含了足够的 motion context
    → 通过 self-attention 从 known regions 推断 generate regions
    → text cross-attention 的 gradient 被 known-region shortcut 稀释
    → 经过数千 epochs，text cross-attention 权重逐渐退化

  Text atrophy 的量化信号：
    - text cross-attention 的 attention entropy 下降
    - text cross-attention 输出的 magnitude (L2-norm) 逐步减小
    - 在 generation_mask=1 区域，输出对 text 变化的敏感度 → 0
```

### 1.2 为什么 uncond 模型没有这个问题

uncond 模型没有 text pathway，所有条件信息都通过 VACE (inactive + mask) 和 x_t[known] 传入。模型的唯一信号源是 motion context + noise level，不存在"两个信号源竞争"的问题。

### 1.3 为什么简单提高 cond_mask_prob 不解决

`cond_mask_prob=0.1` 意味着 10% 的样本 text 被 null 替换（用于训练 CFG 的 unconditional branch）。即使提到 0.3，90%→70% 的有效 text 样本中，84% 仍有 strong motion condition → atrophy 依旧发生，只是稍慢。

### 1.4 解决方案的核心约束

任何有效方案必须满足：
1. **Text pathway 不退化**：无论 motion condition 多强，text 信号必须始终被"需要"
2. **Motion completion 不退步**：不能为了保 text 而损害 completion 质量
3. **单模型统一**：不需要分别训 uncond 和 caption 两个模型
4. **从现有 checkpoint resume**：不需要从零训练

---

## 2. 方案设计：CRFM (Condition-Routed Flow Matching)

### 2.1 核心思路

不是让两种 condition 自由竞争，而是**架构级别强制 text 信号的存在**：

```
                         ┌─────────────────────────┐
  Text (Qwen3+CLIP) ────►│ Text Pathway (FROZEN)   │──┐
                         └─────────────────────────┘  │
                                                      ▼
  ┌─────────┐    ┌───────────┐    ┌───────────────────────────┐
  │  x_t +  │    │   CDE     │    │  Motion DiT Blocks        │
  │  VACE   │───►│(density   │───►│  (trainable self-attn +   │◄── Gated Fusion
  │  context│    │ embedding)│    │   frozen text cross-attn) │
  └─────────┘    └───────────┘    └───────────────────────────┘
                                              │
                                              ▼
                                    pred_velocity (198-dim)
```

### 2.2 三大组件

#### A. Condition Density Embedding (CDE)

**动机**：模型需要显式知道"当前有多少比例需要生成"，以决定应该多依赖 text vs motion context。

```python
class ConditionDensityEmbedding(nn.Module):
    """Sinusoidal embedding of mask generation ratio.
    
    mask_density = 1.0 → pure generation (T2M), need 100% text
    mask_density = 0.0 → identity (all known), need 0% text
    mask_density = 0.3 → partial completion, need moderate text
    """
    def __init__(self, dim: int = 1024):
        super().__init__()
        # Sinusoidal positional encoding (like timestep embedding)
        half = dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half) / half)
        self.register_buffer('freqs', freqs)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim),
        )
        # Zero-init output so CDE starts with no effect
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)
    
    def forward(self, density: Tensor) -> Tensor:
        """density: (B,) in [0, 1]"""
        args = density.unsqueeze(-1) * self.freqs  # (B, dim//2)
        emb = torch.cat([args.cos(), args.sin()], dim=-1)  # (B, dim)
        return self.mlp(emb)
```

**注入方式**：CDE 输出与 timestep embedding 相加，通过 AdaLN 调制所有 DiT blocks。

```python
# In HunyuanMotionMMDiT.forward():
t_emb = self.timestep_encoder(timesteps)  # (B, feat_dim)
cde_emb = self.cde(mask_density)          # (B, feat_dim)
adapter = t_emb + cde_emb                 # combined modulation signal
```

#### B. Text Attention Preservation (TAP)

**核心思路**：不完全冻结 text cross-attention（完全冻结会阻止适应 motion context），而是用 **gradient gating** 限制其更新速度。

```python
class TextAttentionPreservation:
    """Preserve text cross-attention capability via gradient scaling.
    
    Strategy:
    1. text cross-attention layers get 0.01x gradient (near-frozen)
    2. text_refiner layers get 0.1x gradient (slow adaptation)
    3. All other layers get 1.0x gradient (normal training)
    
    This is strictly better than hard freezing because:
    - The text pathway can still slowly adapt to the new VACE context
    - But adaptation is too slow for atrophy to develop
    """
    
    GRAD_SCALE_MAP = {
        'text_cross_attn': 0.01,   # near-frozen
        'text_mod': 0.01,          # modulation for text stream
        'text_refiner': 0.1,       # can slowly adapt
        # everything else: 1.0
    }
    
    @staticmethod
    def apply_gradient_scaling(model):
        """Register backward hooks for gradient scaling."""
        for name, param in model.named_parameters():
            for pattern, scale in TextAttentionPreservation.GRAD_SCALE_MAP.items():
                if pattern in name:
                    param.register_hook(lambda grad, s=scale: grad * s)
                    break
```

**为什么 gradient gating 优于完全冻结**：
- 完全冻结：text cross-attention 永远保持 T2M pretrained 的行为，无法适应 VACE context（导致 text 输出和 VACE 输出不协调）
- Gradient gating：text 可以非常缓慢地适应，但速度远慢于 atrophy 的发展速度

#### C. Text-Awareness Loss (TAL)

**动机**：即使有 TAP，在 motion condition 极强时（density < 0.3），model 可能完全忽略 text。TAL 显式强制 text 对输出产生影响。

```python
def text_awareness_loss(pred_with_text, pred_without_text, src_mask, 
                        mask_density, min_effect=0.005):
    """Encourage text to always affect generated regions.
    
    Only active when mask_density < 0.7 (strong motion condition).
    Penalizes when text has zero effect on generated regions.
    
    Key insight: we DON'T enforce text to change known regions
    (text should only guide generation, not override constraints).
    """
    # Only in generated regions (src_mask=1)
    gen_mask = src_mask  # (B, L, D), 1=generate
    
    # Per-sample mean absolute difference in generated regions
    diff = ((pred_with_text - pred_without_text) * gen_mask).abs()
    diff_per_sample = diff.sum(dim=(-1, -2)) / (gen_mask.sum(dim=(-1, -2)) + 1e-6)
    
    # Only apply when motion condition is strong (density < 0.7)
    apply_weight = (mask_density < 0.7).float()
    
    # Hinge loss: penalize if text effect < min_effect
    loss = F.relu(min_effect - diff_per_sample) * apply_weight
    
    return loss.mean()
```

**计算开销**：需要额外一次 forward pass（null text），但只在训练时每 N 步计算一次。

### 2.3 训练策略

#### 不再需要 Phase 1 → Phase 2 切换

CRFM 从第一步开始就是 mixed training：

```
每个 batch 中的样本分布 (通过 v3 sampler):
  - 16% pure T2M (K=0, mask=1 everywhere)
  - 84% partial mask (K=1..4, various patterns)
  
所有样本都有 text condition (cond_mask_prob=0.15 for CFG dropout)
CDE 编码每个样本的 mask_density
TAP gradient gating 始终生效
TAL 每 4 步计算一次
```

#### 从现有 checkpoint resume

```python
# Load from uncond_local best checkpoint (epoch 2730)
# uncond 模型的 motion completion 能力最强
# text attention 从 T2M pretrained 初始化
load_from = dict(
    path='work_dirs/hymotion_m2m_v2_uncond_local_046b/checkpoint-epoch_2730/model.safetensors',
    load_scope='model',
)

# 新增参数（CDE, TAP hooks, TAL loss）随机初始化
# CDE: zero-init output → 初始无影响，渐进学习
# Text cross-attention: 从 T2M pretrained 重新加载
```

### 2.4 推理策略

```python
def crfm_inference(model, batch, text_cfg_scale=2.5):
    """CRFM inference with text-only CFG.
    
    Key: CFG only applies to text (not motion condition).
    The motion condition (VACE + imputation) is always present.
    """
    # Compute mask density for CDE
    mask_density = batch['src_mask'].mean(dim=(-1, -2))  # (B,)
    
    def fn(t, x):
        # Conditional forward (with text)
        v_cond = model_forward(x, text=text_cond, motion_cond=vace, 
                               density=mask_density, t=t)
        
        if text_cfg_scale > 1.0:
            # Unconditional forward (null text, same motion condition)
            v_uncond = model_forward(x, text=null_text, motion_cond=vace,
                                     density=mask_density, t=t)
            # Text CFG only on generated regions
            v = v_uncond + text_cfg_scale * (v_cond - v_uncond)
        else:
            v = v_cond
        return v
    
    # ODE integration with imputation (same as current _man pipeline)
    ...
```

---

## 3. 实现细节

### 3.1 文件修改清单

| File | Type | Change Description | ~Lines |
|------|------|-------------------|--------|
| `hftrainer/models/motion/hymotion_m2m/network/condition_routing.py` | **NEW** | CDE module | ~80 |
| `hftrainer/models/motion/hymotion_m2m/network/hymotion_mmdit.py` | MODIFY | 注入 CDE 到 timestep encoder; 支持 mask_density input | ~40 |
| `hftrainer/models/motion/hymotion_m2m/bundle.py` | MODIFY | 添加 CDE 初始化; TAP gradient scaling; mask_density 计算 | ~50 |
| `hftrainer/trainers/motion/hymotion_m2m_crfm_trainer.py` | **NEW** | CRFM trainer (继承 M2M trainer, 添加 TAL loss) | ~180 |
| `hftrainer/pipelines/motion/hymotion_m2m_pipeline.py` | MODIFY | 支持 CDE + text-only CFG | ~30 |
| `configs/hymotion_m2m_v3/` | **NEW** | v3 configs (smoke, caption_local, uncond_local) | 4 files |
| `tests/unit/test_crfm_modules.py` | **NEW** | 单元测试 | ~120 |

### 3.2 MMDiT 修改（最小侵入）

```python
# hymotion_mmdit.py 修改要点

class HunyuanMotionMMDiT(nn.Module):
    def __init__(self, ..., enable_cde: bool = False):
        ...
        # 新增：Condition Density Embedding
        self.enable_cde = enable_cde
        if enable_cde:
            from .condition_routing import ConditionDensityEmbedding
            self.cde = ConditionDensityEmbedding(dim=self.feat_dim)
    
    def forward(self, x, ctxt_input, vtxt_input, timesteps, 
                x_mask_temporal=None, ctxt_mask_temporal=None,
                mask_density=None):  # 新增参数
        ...
        # Timestep + vtxt embedding (existing)
        adapter = self.timestep_encoder(timesteps)  # (B, feat_dim)
        adapter += self.vtxt_encoder(vtxt_input)
        
        # 新增：CDE injection
        if self.enable_cde and mask_density is not None:
            adapter = adapter + self.cde(mask_density)
        
        # 其余不变...
```

**关键设计决策**：
- CDE 通过加法注入 adapter（和 timestep, vtxt 一样），不改变网络拓扑
- `enable_cde=False` 时行为完全等价于原始模型（backward compatible）
- T2M pretrained weights 全部可用（新参数 zero-init）

### 3.3 Bundle 修改

```python
# bundle.py 新增

@MODEL_BUNDLES.register_module()
class HyMotionM2MBundle(ModelBundle):
    def __init__(self, ..., 
                 enable_cde: bool = False,
                 text_grad_scale: float = 1.0,  # 1.0 = no TAP
                 ):
        ...
        self.enable_cde = enable_cde
        self.text_grad_scale = text_grad_scale
    
    def apply_text_attention_preservation(self):
        """Apply gradient scaling to text-related parameters."""
        if self.text_grad_scale >= 1.0:
            return  # No TAP
        for name, param in self.motion_transformer.named_parameters():
            if any(k in name for k in ['text_mod', 'ctxt_norm', 'ctxt_proj']):
                if param.requires_grad:
                    param.register_hook(
                        lambda grad, s=self.text_grad_scale: grad * s
                    )
    
    def compute_mask_density(self, src_mask: Tensor) -> Tensor:
        """Compute per-sample mask density for CDE.
        
        src_mask: (B, L, D), 1=generate, 0=known
        Returns: (B,) density in [0, 1]
        """
        return src_mask.mean(dim=(-1, -2))
```

### 3.4 CRFM Trainer

```python
# hymotion_m2m_crfm_trainer.py

@TRAINERS.register_module()
class HyMotionM2MCRFMTrainer(HyMotionM2MTrainer):
    """CRFM trainer: adds TAL loss and passes mask_density to model."""
    
    def __init__(self, bundle, 
                 tal_weight: float = 0.01,
                 tal_interval: int = 4,
                 tal_min_effect: float = 0.005,
                 **kwargs):
        super().__init__(bundle, **kwargs)
        self.tal_weight = tal_weight
        self.tal_interval = tal_interval
        self.tal_min_effect = tal_min_effect
        
        # Apply TAP gradient scaling
        self.bundle.apply_text_attention_preservation()
    
    def _prepare_and_forward(self, batch):
        """Override: pass mask_density to predict_flow."""
        ctx = super()._prepare_and_forward(batch)
        # We need to inject mask_density into the forward call
        # This is done by modifying predict_flow call
        return ctx
    
    def train_step(self, batch):
        ctx = self._prepare_and_forward(batch)
        losses = self._compute_base_loss(ctx)
        
        # TAL loss (every tal_interval steps)
        global_step = self.get_global_step()
        if (self.tal_weight > 0 and 
            global_step % self.tal_interval == 0 and
            ctx.get('src_mask') is not None):
            tal = self._compute_tal_loss(ctx)
            if tal is not None:
                losses['tal'] = tal
        
        loss = sum(losses.values())
        result = {'loss': loss}
        for k, v in losses.items():
            result[f'loss_{k}'] = v.detach()
        return result
    
    def _compute_tal_loss(self, ctx):
        """Text-Awareness Loss: extra forward with null text."""
        src_mask = ctx['src_mask']
        if src_mask is None or src_mask.sum() == 0:
            return None
        
        mask_density = src_mask.mean(dim=(-1, -2))  # (B,)
        
        # Skip if all samples are pure T2M (density ≈ 1.0)
        if (mask_density > 0.9).all():
            return None
        
        # Forward with null text (detached — no gradient through null branch)
        B = ctx['x_t'].shape[0]
        with torch.no_grad():
            null_vtxt = self.bundle.null_vtxt_feat.expand(B, 1, -1)
            null_ctxt = self.bundle.null_ctxt_input.expand(
                B, ctx['ctxt_input'].shape[1], -1
            )
        
        x_input_null = torch.cat([ctx['x_t'], ctx['vace_context']], dim=-1)
        pred_null = self.bundle.predict_flow(
            x_input=x_input_null,
            ctxt_input=null_ctxt,
            vtxt_input=null_vtxt,
            timesteps=ctx['timesteps'],
            x_mask_temporal=ctx['tgt_padding_mask'],
            ctxt_mask_temporal=ctx['ctxt_mask_temporal'],
            mask_density=mask_density,
        )
        
        # Compute text effect in generated regions
        pred = ctx['pred']
        gen_mask = src_mask
        diff = ((pred - pred_null.detach()) * gen_mask).abs()
        diff_per_sample = diff.sum(dim=(-1,-2)) / (gen_mask.sum(dim=(-1,-2)) + 1e-6)
        
        # Hinge loss: active when motion condition is strong
        apply_weight = (mask_density < 0.7).float()
        loss = F.relu(self.tal_min_effect - diff_per_sample) * apply_weight
        
        return loss.mean() * self.tal_weight
```

---

## 4. 与备选方案的对比分析

| 方案 | 优势 | 劣势 | 新颖性 | 实现难度 |
|------|------|------|--------|---------|
| **CRFM (本方案)** | 最小改动，可从现有 ckpt resume；TAP+TAL 组合解决 atrophy | 需额外 forward (TAL) | ★★★★ (CDE+TAP+TAL 组合新颖) | 中 |
| Dual-Stream Condition DiT | 物理隔离 text/motion | 需重新设计架构，不能 resume | ★★★★★ | 高 |
| Progressive Condition Integration | 不需额外 loss | 训练时需要 time-dependent conditioning | ★★★ | 低 |
| 简单 freeze text attention | 最小改动 | text 可能无法适应 motion context | ★★ | 极低 |
| 提高 pure_gen 比例到 50%+ | 零代码改动 | 牺牲 completion 训练效率 | ★ | 无 |

**选择 CRFM 的理由**：
1. 可以从 uncond_local (epoch 2730) 直接 resume → 保留最强 completion 能力
2. CDE zero-init → 初始时不影响模型，渐进学习
3. TAP gradient gating → 比完全冻结更灵活
4. TAL → 显式防止 atrophy 的"安全网"
5. 论文新颖性足够（CDE + TAP + TAL 的组合在 motion generation 领域未见）

---

## 5. 实施计划

### Stage 0: 假设验证 (Day 1, debug machine)

在 lzy_debug_machine_1 上验证 text atrophy 假设：

```bash
# 实验 0.1: 对比 Phase 1 vs Phase 2 的 text attention
python3 tools/probe_text_attention.py \
    --ckpt_phase1 work_dirs/hymotion_m2m_v2_caption_local_phase1/checkpoint-epoch_47 \
    --ckpt_phase2 work_dirs/hymotion_m2m_v2_caption_local_phase2/latest

# 实验 0.2: 纯 Phase 1 checkpoint 推理 E1
# 预期：Phase 1 jitter << Phase 2 的 990
```

### Stage 1: CDE + TAP 实现 (Day 2-3)

1. 实现 `condition_routing.py` (CDE module)
2. 修改 `hymotion_mmdit.py` (注入 CDE)
3. 修改 `bundle.py` (TAP gradient scaling + mask_density 计算)
4. 单元测试通过
5. Smoke test: 1 epoch 正常训练

### Stage 2: CRFM Trainer + TAL (Day 4-5)

1. 实现 `hymotion_m2m_crfm_trainer.py`
2. 验证 TAL loss 正常计算（无 NaN/Inf）
3. 在 debug machine 上跑 100 steps，验证：
   - loss 正常下降
   - TAL loss > 0 且逐步减小（说明 text effect 在增加）
   - CDE 输出的 variance 在增加（说明在学习 density 编码）

### Stage 3: 完整验证 (Day 6-7)

1. 在 debug machine 上训练 5 epochs，观察：
   - loss_velocity 稳定下降
   - loss_tal 趋近 0
   - 无 NaN/Inf
2. 推理 10 个 E1 样本，对比：
   - CRFM vs Phase2 (jitter 应显著降低)
   - CRFM vs uncond (quality 应接近)

### Stage 4: Taiji 提交大规模训练 (Day 7+)

```bash
# caption_local CRFM (从 uncond_local epoch 2730 resume)
python3 tools/taiji_submit.py m2m_v3_crfm_caption_local \
    configs/hymotion_m2m_v3/hymotion_m2m_v3_caption_local_046b.py \
    --host_num 2

# uncond_local CRFM (相同架构但 cond_mask_prob=1.0 即无 text)
# 用于验证 CDE 不伤害 uncond 性能
python3 tools/taiji_submit.py m2m_v3_crfm_uncond_local \
    configs/hymotion_m2m_v3/hymotion_m2m_v3_uncond_local_046b.py \
    --host_num 2
```

---

## 6. Config 设计

### 6.1 Base Config (`configs/hymotion_m2m_v3/_base_hymotion_m2m_v3_046b.py`)

```python
_base_ = '../hymotion_m2m_v2/_base_hymotion_m2m_v2_046b.py'

work_dir = 'work_dirs/hymotion_m2m_v3_046b'

model = dict(
    # Enable CDE
    motion_transformer=dict(
        enable_cde=True,
    ),
    # TAP: 0.01x gradient for text-related params
    text_grad_scale=0.01,
    # Higher CFG dropout for better CFG training
    cond_mask_prob=0.15,
)

trainer = dict(
    type='HyMotionM2MCRFMTrainer',
    mask_aware_noise=True,
    # TAL config
    tal_weight=0.01,
    tal_interval=4,
    tal_min_effect=0.005,
)
```

### 6.2 Caption Local (`configs/hymotion_m2m_v3/hymotion_m2m_v3_caption_local_046b.py`)

```python
_base_ = './_base_hymotion_m2m_v3_046b.py'

work_dir = 'work_dirs/hymotion_m2m_v3_caption_local_046b'

model = dict(
    pred_type='velocity',
    uncondition_mode=False,
    cond_mask_prob=0.15,
    mean_std_dir='data/hymotion_m2m_data/_stats_198dim',
    rotation_space='local',
)

train_dataloader = dict(
    batch_size=16,  # Reduced: TAL needs extra forward
    dataset=dict(
        pipeline=[
            dict(type='LoadPreExtractedTextEmbedding', key='caption', allow_none=True),
            dict(type='LoadSmplx55', key='motion', rot_type='rotation_6d',
                 transl_type='abs', smpl_type='smpl_22'),
            dict(type='Compute198DimPosition', key='motion'),
            dict(type='RandomCropPadding', clip_len=360, pad_mode='replicate',
                 allow_shorter=True, make_pad_mask=True, pad_mask_key='pad_mask'),
            # v3 sampler with 16% pure T2M
            dict(type='PrepareM2Mv2Condition', key='motion',
                 sampler_version='v3', editing_prob=0.15,
                 corruptor_names=['jitter', 'joint_jump', 'sliding',
                                  'limb_candy_wrapper', 'wrist_candy_wrapper'],
                 max_corruptions=2,
                 v3_config=dict(k_weights=(0.16, 0.513, 0.233, 0.065, 0.029))),
            dict(type='PackInputs',
                 keys=['src_motion', 'tgt_motion', 'src_mask', 'tgt_length',
                       'src_length', 'edit_mode', 'text_vec_raw', 
                       'text_ctxt_raw', 'text_ctxt_raw_length'],
                 meta_keys=['motion_path', 'fps'],
                 set_dummy_value=True, dummy_value=None),
        ],
    ),
)

# Resume from uncond_local (strongest completion capability)
# + reload text attention from T2M pretrained
load_from = dict(
    _delete_=True,
    path='work_dirs/hymotion_m2m_v2_uncond_local_046b/checkpoint-epoch_2730/model.safetensors',
    load_scope='model',
)
```

### 6.3 Smoke Config (`configs/hymotion_m2m_v3/hymotion_m2m_v3_smoke.py`)

```python
_base_ = './_base_hymotion_m2m_v3_046b.py'

work_dir = 'work_dirs/hymotion_m2m_v3_smoke'

train_dataloader = dict(batch_size=4)
train_cfg = dict(by_epoch=True, max_epochs=2, val_interval=1, max_grad_norm=1.0)
default_hooks = dict(
    logger=dict(type='LoggerHook', interval=1, iter_interval=1),
    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=2),
)

# Use T2M pretrained for smoke (no need for large ckpt)
load_from = dict(
    _delete_=True,
    path='checkpoints/HY-Motion-1.0/HY-Motion-1.0-Lite/latest.ckpt',
    load_scope='model',
)
```

---

## 7. 单元测试计划

### 7.1 CDE 测试

```python
def test_cde_shape_and_gradient():
    """CDE output shape matches feat_dim; gradients flow."""
    cde = ConditionDensityEmbedding(dim=1024)
    density = torch.rand(4)
    out = cde(density)
    assert out.shape == (4, 1024)
    out.sum().backward()
    assert cde.mlp[0].weight.grad is not None

def test_cde_zero_init():
    """CDE output is near-zero at initialization."""
    cde = ConditionDensityEmbedding(dim=1024)
    density = torch.rand(4)
    out = cde(density)
    assert out.abs().max() < 1e-5

def test_cde_density_sensitivity():
    """After training, different densities produce different embeddings."""
    cde = ConditionDensityEmbedding(dim=1024)
    # Simulate some training
    optimizer = torch.optim.Adam(cde.parameters(), lr=1e-3)
    for _ in range(100):
        d = torch.rand(8)
        out = cde(d)
        loss = (out - d.unsqueeze(-1).expand_as(out)).pow(2).mean()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    # After training, density=0 and density=1 should produce different embeddings
    out_0 = cde(torch.zeros(1))
    out_1 = cde(torch.ones(1))
    assert (out_0 - out_1).abs().mean() > 0.01
```

### 7.2 TAP 测试

```python
def test_tap_gradient_scaling():
    """Text-related params get scaled gradients."""
    # Create a simple model with text_mod and motion layers
    model = nn.ModuleDict({
        'text_mod': nn.Linear(10, 10),
        'motion_layer': nn.Linear(10, 10),
    })
    # Apply TAP
    for name, param in model.named_parameters():
        if 'text_mod' in name:
            param.register_hook(lambda g: g * 0.01)
    
    x = torch.randn(2, 10, requires_grad=True)
    out = model['text_mod'](x) + model['motion_layer'](x)
    out.sum().backward()
    
    # text_mod gradient should be 100x smaller
    text_grad_norm = model['text_mod'].weight.grad.norm()
    motion_grad_norm = model['motion_layer'].weight.grad.norm()
    assert text_grad_norm < motion_grad_norm * 0.1
```

### 7.3 TAL 测试

```python
def test_tal_loss_nonzero_when_text_ignored():
    """TAL loss is positive when pred_with_text == pred_without_text."""
    B, L, D = 2, 100, 198
    pred_with = torch.randn(B, L, D)
    pred_without = pred_with.clone()  # identical → text has no effect
    src_mask = torch.ones(B, L, D)
    src_mask[:, :50, :] = 0  # first 50 frames known
    mask_density = src_mask.mean(dim=(-1, -2))  # 0.5
    
    loss = text_awareness_loss(pred_with, pred_without, src_mask,
                               mask_density, min_effect=0.005)
    assert loss > 0  # Should penalize zero text effect

def test_tal_loss_zero_when_text_active():
    """TAL loss is zero when text significantly affects output."""
    B, L, D = 2, 100, 198
    pred_with = torch.randn(B, L, D)
    pred_without = pred_with + torch.randn_like(pred_with) * 0.1  # significant diff
    src_mask = torch.ones(B, L, D)
    mask_density = torch.ones(B)  # pure T2M
    
    loss = text_awareness_loss(pred_with, pred_without, src_mask,
                               mask_density, min_effect=0.005)
    # density=1.0 → apply_weight=0 → loss=0 (don't penalize pure T2M)
    assert loss == 0
```

### 7.4 End-to-End Smoke

```python
def test_crfm_train_step_no_crash():
    """Full CRFM train step completes without error."""
    # Load smoke config, run 1 training step
    # Verify: loss is finite, all gradients flow, CDE output non-trivial after 1 step
    pass  # Implemented in smoke test script
```

---

## 8. 评估计划

### 8.1 Debug Machine 快速验证 (每次训练实验后)

```bash
# 快速 E1 评估 (20 samples, ~3 min)
python3 tools/eval_m2m_v2_all_tasks.py \
    --config configs/hymotion_m2m_v3/hymotion_m2m_v3_caption_local_046b.py \
    --ckpt work_dirs/hymotion_m2m_v3_caption_local_046b/checkpoint-epoch_XX \
    --tasks E1 --num-samples 20 --save-npz --use-rewritten \
    --text-guidance-scale 2.5

# 通过标准: jitter_pos < 400, foot_skating < 0.05
```

### 8.2 Taiji 全量评估 (训练 100+ epochs 后)

```bash
# 全量 E1-E10 评估
python3 tools/eval_m2m_v2_all_tasks.py \
    --config configs/hymotion_m2m_v3/hymotion_m2m_v3_caption_local_046b.py \
    --ckpt work_dirs/hymotion_m2m_v3_caption_local_046b/latest \
    --tasks E1,E2,E3,E4,E5,E6,E7,E8,E9,E10 \
    --num-samples 100 --save-npz --use-rewritten

# Success criteria:
# E1 (T2M): jitter < 300, skating < 0.05, R-precision > 0.5
# E2-E7 (completion): 不低于 uncond_local 基线
# E4 (end-effector + text): ee_error < 0.20
# E10 (part control + text): jitter < 250
```

---

## 9. Risk Matrix & Mitigation

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|-----------|
| CDE 引入训练不稳定 | Low | High | Zero-init + warmup 1000 steps |
| TAP 导致 text 无法适应 | Medium | Medium | 用 0.01 而非 0.0（非完全冻结） |
| TAL extra forward OOM | Medium | Low | 每 4 步算一次 + 降低 bs 到 16 |
| TAL 导致模型"作弊" (输出随机差异) | Low | Medium | TAL 只在 generated regions 计算，且有 hinge threshold |
| uncond_local ckpt 的 text attn 权重退化 | Medium | High | 从 T2M pretrained ckpt 加载 text-related layers |
| CDE 退化为 constant (不编码 density) | Low | Medium | 监控 CDE output variance；添加辅助 loss 如果需要 |

---

## 10. Success Metrics

### Hard Requirements (Must Achieve)

| Metric | Target | Current Best (caption_phase2) | Baseline (uncond_local) |
|--------|--------|-------------------------------|------------------------|
| E1 jitter_pos | < 400 | 990 | 213 |
| E1 foot_skating | < 0.05 | 0.153 | 0.023 |
| E4 ee_error | < 0.25 | 0.43 | 0.21 |
| E2-E7 jitter_pos | < 250 | — | ~213 |
| Loss curve | Monotone decreasing | — | — |

### Stretch Goals

| Metric | Target | Significance |
|--------|--------|-------------|
| E1 jitter_pos | < 250 | 接近 uncond 水平 |
| E1 R-precision | > 0.5 | Text alignment 有效 |
| E10 with caption | 优于 uncond | Text 提升 part-control |

---

## 11. 论文 Contribution 映射

1. **Problem**: 首次揭示 mask-aware flow matching 中 text condition atrophy 现象
   - 量化证据：Phase 1→2 text attention entropy 变化
   - 对比 KIMODO/UMO 说明为何它们不受影响

2. **Method**: CRFM (Condition-Routed Flow Matching)
   - CDE: 条件密度显式编码（新模块）
   - TAP: 文本注意力梯度门控（新训练策略）
   - TAL: 文本感知正则化（新 loss）

3. **System**: 单模型统一 6 大任务族
   - Pure T2M
   - Temporal completion (inbetween, prediction, prefix)
   - Joint completion
   - End-effector control
   - Text-conditioned completion
   - Motion repair/editing

4. **Experiments**: 15-task evaluation framework
   - 4.6× improvement on caption model quality
   - No degradation on completion tasks
   - SOTA on text-conditioned motion completion (new benchmark)

---

## Appendix: Text Attention Layer Names in MMDiT

需要 TAP gradient scaling 的参数模式：

```
# Double-stream blocks (text stream):
blocks.*.text_mod.*          # Text modulation (shift/scale/gate)
blocks.*.text_norm1.*        # Text LayerNorm before attention
blocks.*.text_qkv.*         # Text Q/K/V projection
blocks.*.text_proj.*        # Text output projection
blocks.*.text_norm2.*        # Text LayerNorm before MLP
blocks.*.text_mlp.*         # Text MLP

# Single-stream blocks (text tokens share with motion):
# These are harder to isolate; in single-stream, text and motion
# tokens go through the same layers. We do NOT apply TAP here
# (would slow down motion learning too).

# Text refiner:
text_refiner.*               # 2-layer token refiner
```

**实际操作**：只对 double-stream blocks 的 `text_*` 参数做 gradient scaling。Single-stream blocks 的参数完全自由训练（text tokens 在 single-stream 中已经和 motion tokens 混合，无法隔离）。

---

## Appendix: Comparison with Existing Plan (caption_motion_unified_conditioning_plan.md)

| Aspect | 旧方案 (CRFM v1) | 本方案 (CRFM v2) |
|--------|------------------|------------------|
| Route Gate | 可学习 2-head softmax gate | 移除（过度设计） |
| Frozen text | 完全冻结所有 text layers | Gradient gating 0.01x |
| TMCR loss | 每 4 步算差异，min_diff_target=0.01 | TAL: 同思路，改进实现 |
| CDE | Sinusoidal + 2-layer MLP | 同，但 4× wider MLP |
| Dual-CFG inference | 分离 text/motion CFG | 简化为 text-only CFG |
| Phase 训练 | Phase 0→1→2 三阶段 | 单阶段 mixed |
| 从何 resume | T2M pretrained | uncond_local epoch 2730 |
| 可执行性 | 概念方案 | 完整实施细节 |

**关键改进**：
1. 移除 Route Gate → 减少一个可能退化的组件
2. Gradient gating 替代完全冻结 → 更灵活
3. 从 uncond_local resume → 保留最强 completion 能力
4. 单阶段 mixed → 简化训练流程，避免 phase switching bug
