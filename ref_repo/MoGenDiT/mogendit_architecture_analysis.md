# MoGenDIT Complete Architecture Analysis

**Document Purpose**: Comprehensive technical specification of the MoGenDIT motion generation and refinement framework from `/apdcephfs_cq10/share_1467498/home/chengxuzuo/projects/MoGenDIT/`

**Date**: 2026-03-25
**Version**: 1.0 (Based on old codebase audit)

---

## 1. Motion Representation Layer

### 1.1 Core Motion Representation: OccamMotionRep

**File**: `/apdcephfs_cq10/share_1467498/home/chengxuzuo/projects/MoGenDIT/motion_process/motion_representation.py` (Lines 649-1048)

#### 1.1.1 Initialization Parameters
```python
class OccamMotionRep:
    def __init__(self, keep_hand=False, global_pose=True, fps=30):
        self.n_joint = 22  # SMPL skeleton without hand joints
        self.global_pose = global_pose  # If True, uses global orientation; False = local
        self.fps = fps  # Frame rate (default 30)
        
        # Dimension breakdown:
        d_pose = self.n_joint * 6  # 22 joints × 6D rotation = 132 dims
        d_joint = self.n_joint * 3  # 22 joints × 3D positions = 66 dims
        d_trans = 3                  # Root translation = 3 dims
        
        self.data_dim = 132 + 66 + 3 = 201  # Total: 201-dim motion
```

**Key Feature**: Uses rotation_6d representation (column-major: `[R00,R10,R20,R01,R11,R21]` format)

#### 1.1.2 Encoding Function

**File**: Lines 677-706

```python
def encode(self, pose: torch.Tensor, joint: torch.Tensor, trans: torch.Tensor):
    """
    Input shapes:
    - pose: (T, 22, 3, 3)  # Rotation matrices for each joint
    - joint: (T, 22, 3)    # Global 3D joint positions
    - trans: (T, 3)        # Root translation
    
    Output: (T, 201)  # Flattened motion vector
    
    Process:
    1. If global_pose=True: Apply forward kinematics to convert local→global
    2. Convert rotation matrices → 6D representation
    3. Flatten and concatenate: [pose_6d(132) | joint_pos(66) | trans(3)]
    """
    if self.global_pose:
        pose = self.body_model.forward_kinematics(pose)
    pose = rotation_matrix_to_r6d(pose).reshape(-1, self.n_joint, 6)
    pose = pose.flatten(1)          # (T, 132)
    joint = joint.flatten(1)        # (T, 66)
    trans = trans.flatten(1)        # (T, 3)
    motion = torch.cat([pose, joint, trans], dim=-1)  # (T, 201)
    return motion
```

#### 1.1.3 Decoding Function

**File**: Lines 752-779

```python
def decode(self, motion: torch.Tensor):
    """
    Input: (T, 201)
    
    Output:
    - pose: (T, 22, 3, 3)  # Rotation matrices
    - joint: (T, 22, 3)    # 3D joint positions
    - trans: (T, 3)        # Root translation
    
    Reverse Process:
    1. Extract components by mask: pose(132) | joint(66) | trans(3)
    2. Reshape pose back to (T, 22, 6)
    3. Convert 6D → rotation matrices
    4. If global_pose=True: Apply inverse kinematics
    """
    T = motion.shape[0]
    pose_flat = motion[:, self.pose_mask]           # (T, 132)
    joint_flat = motion[:, self.joint_mask]         # (T, 66)
    trans_flat = motion[:, self.trans_mask]         # (T, 3)
    
    pose = pose_flat.view(T, self.n_joint, 6)
    pose = r6d_to_rotation_matrix(pose).reshape(T, self.n_joint, 3, 3)
    if self.global_pose:
        pose = self.body_model.inverse_kinematics(pose)
    joint = joint_flat.view(T, self.n_joint, 3)
    trans = trans_flat.view(T, 3)
    
    return pose, joint, trans
```

#### 1.1.4 Normalization (Egocentric Alignment)

**File**: Lines 781-831

```python
def normalization(self, motion: torch.Tensor, ref_idx=0, height_reset=False):
    """
    Aligns motion to egocentric frame:
    - ref_idx frame's root orientation → world Z-axis
    - ref_idx frame's root X-Z position → origin
    
    Implementation:
    1. Extract rotation matrix of ref_idx frame's root (Pelvis, joint 0)
    2. Calculate ego transformation: R_ego_gv_inv = get_ego_gv(R_ref).T
       (Converts any direction to face Z-axis)
    3. Apply rotation to all poses: R' = R_ego_gv_inv @ R
    4. Apply translation to all joints: joint' = R_ego_gv_inv @ (joint - joint_ref_xz)
    """
    R_ego_gv_inv = get_ego_gv(pose[ref_idx, 0]).transpose(-2, -1)
    # Apply to poses
    if not self.global_pose:
        pose[:, 0] = R_ego_gv_inv.matmul(pose[:, 0])
    else:
        pose = R_ego_gv_inv.matmul(pose)
    
    # Apply to joints
    global_joint = joint + trans
    global_joint[:, :, [0, 2]] -= global_joint[ref_idx:ref_idx+1, :1, [0, 2]]
    global_joint = R_ego_gv_inv.matmul(global_joint.unsqueeze(-1)).view_as(global_joint)
    
    return motion_with_updated_components
```

#### 1.1.5 Kinematic Loss (Temporal Consistency)

**File**: Lines 1001-1047

```python
def kinematic_loss_batch(self, R6d, joint, length=None, l1_weight=0.0, l2_weight=1.0):
    """
    Enforces skeleton rigidity: all joint offsets stay constant across frames.
    
    Components:
    1. Skeleton offset loss (rigid body constraint):
       loss_rigid = MSE(offset_t, offset_0) 
       where offset_t = joint_positions - root_position
    
    2. L1 penalty (optional):
       loss_rigid += L1(offset_t, offset_0)
    
    Masking: If length provided, only compute loss for valid frames
    """
    R = r6d_to_rotation_matrix(R6d.clone()).reshape(-1, n_j, 3, 3)
    joint = joint.clone().reshape(b, -1, n_j, 3)
    
    # Compute skeleton offsets from motion
    offsets_from_motion = self.body_model.get_skeleton_offsets(
        pose=R, joint=joint, global_pose=self.global_pose, require_grad=True
    ).reshape(b, -1, n_j, 3)
    
    init_skeleton_offsets = offsets_from_motion[:, [0]]
    loss_rigid_body = F.mse_loss(
        offsets_from_motion,
        init_skeleton_offsets.expand_as(offsets_from_motion),
        reduction="none"
    ) * l2_weight
    
    if l1_weight > 0.0:
        loss_rigid_body += l1_weight * F.l1_loss(...)
    
    return loss_rigid_body.mean()
```

### 1.2 Alternative Representations

**File**: Lines 1-648 contain deprecated/alternative representations:
- `HM263XRep`: Padded 263-dim format (pose + joint + stationary)
- `Motion291Rep`: 291-dim with velocity component

---

## 2. Model Architecture Layer

### 2.1 Core Model: MoreDiff (Diffusion Transformer)

**File**: `/apdcephfs_cq10/share_1467498/home/chengxuzuo/projects/MoGenDIT/model/more_diff.py` (Lines 253-504)

#### 2.1.1 Model Configuration by Size

```python
def get_MoreDiff_model(data_dim, version="0.1B"):
    """
    Three model sizes available:
    """
    if version == "0.3B":
        d_model = 1024    # Hidden dimension
        n_head = 16       # Attention heads
        n_stack = 18      # Transformer blocks
    elif version == "0.1B":
        d_model = 768
        n_head = 12
        n_stack = 12
    elif version == "0.03B":
        d_model = 512
        n_head = 8
        n_stack = 8
    
    model = MoreDiff(
        d_motion=data_dim,      # Input: 201 for OccamMotionRep
        d_model=d_model,
        d_cond=22*3,            # Condition: 66-dim (skeleton structure)
        n_head=n_head,
        n_stack=n_stack,
        dropout=0.0,
        window_size=90,         # Sliding window attention (rows 493)
    )
    return model
```

#### 2.1.2 Model Forward Pass

**File**: Lines 312-336

```python
class MoreDiff(BaseModel):
    def forward(self, x_wrapped: dict, t: torch.Tensor) -> torch.Tensor:
        """
        Input:
        - x_wrapped: Dict with keys:
            - 'x_t': (batch, seq_len, 201)  # Noisy motion
            - 'cond': (batch, 1, 66) or None
            - 'mask': (batch, seq_len, 201)  # Observed regions mask
            - 'padding_mask': (batch, seq_len)  # Valid frame mask
        - t: (batch,)  # Diffusion timestep
        
        Process:
        1. Permute to seq-first: [seq_len, batch, dim]
        2. Concatenate motion + mask: x_cat = [x_t | mask]
        3. Embed to d_model: motion_2_token(x_cat)
        4. Pass through n_stack DiT blocks with RoPE
        5. Decode back to motion: token_2_motion(output)
        
        Output: (batch, seq_len, 201)  # Predicted motion x_0
        """
        x = x_wrapped["x_t"]  # (batch, seq_len, 201)
        cond = x_wrapped["cond"]  # (batch, 1, 66) or None
        mask = x_wrapped["mask"]  # (batch, seq_len, 201)
        
        x = x.permute(1, 0, 2)  # [seq_len, batch, 201]
        cond = cond.permute(1, 0, 2)  # [1, batch, 66]
        mask = mask.permute(1, 0, 2)  # [seq_len, batch, 201]
        
        x = torch.cat([x, mask], dim=-1)  # [seq_len, batch, 402]
        x = self.motion_2_token(x)  # [seq_len, batch, d_model]
        
        for encoder_block in self.encoder_blocks:
            x = encoder_block(x, t, cond, padding_mask)
        
        x = self.token_2_motion(x)  # [seq_len, batch, 201]
        x = x.permute(1, 0, 2)  # [batch, seq_len, 201]
        return x
```

### 2.2 Attention Mechanism: RoPE + Sliding Window

**File**: Lines 9-95 (RoPE class and window mask function)

#### 2.2.1 Rotary Position Embedding (RoPE)

**File**: Lines 9-79

```python
class RoPE(nn.Module):
    """
    Rotary Position Embedding - applies rotation to Q/K based on position
    Enables length extrapolation without learnable parameters
    """
    def __init__(self, head_dim: int, max_seq_len: int = 5000, base: int = 10000):
        self.head_dim = head_dim
        # Precompute frequency matrix: theta = base^(-2i/d) for i in [0, d/2)
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
    
    @staticmethod
    def apply_rotary_emb(x: torch.Tensor, angles: torch.Tensor):
        """
        x shape: [seq_len, batch_size, n_head, head_dim]  (seq-first format!)
        angles shape: [seq_len, head_dim//2]
        
        Process:
        1. Split x into even/odd dimensions: x1, x2
        2. Apply rotation: x1' = x1*cos(theta) - x2*sin(theta)
                          x2' = x1*sin(theta) + x2*cos(theta)
        3. Interleave back
        """
        x1, x2 = x[..., ::2], x[..., 1::2]
        cos_angles = torch.cos(angles).unsqueeze(1).unsqueeze(2)
        sin_angles = torch.sin(angles).unsqueeze(1).unsqueeze(2)
        rotated_x1 = x1 * cos_angles - x2 * sin_angles
        rotated_x2 = x1 * sin_angles + x2 * cos_angles
        return torch.cat([rotated_x1, rotated_x2], dim=-1)
```

#### 2.2.2 Sliding Window Attention Mask

**File**: Lines 83-95

```python
def get_window_mask(seq_len: int, window_size: int, device: torch.device):
    """
    Restricts attention to local window around each position
    
    Example: window_size=90 (half_win=45) means each position attends to
             ±45 neighbors, total 91 positions
    
    Computation:
    1. Distance matrix: dist[i,j] = |i - j| for all positions
    2. Within window (dist ≤ half_win): mask = 0.0 (allow attention)
    3. Outside window (dist > half_win): mask = -1e9 (block attention)
    4. Output shape: (1, 1, seq_len, seq_len) for broadcasting
    """
    half_win = window_size // 2  # 90 // 2 = 45
    idx = torch.arange(seq_len, device=device)
    dist = torch.abs(idx.unsqueeze(0) - idx.unsqueeze(1))
    window_mask = torch.where(dist <= half_win, 0.0, -1e9)
    window_mask = window_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]
    return window_mask
```

### 2.3 DiT Block (Transformer Block with AdaLN)

**File**: Lines 98-249

```python
class DiTBlock(nn.Module):
    """
    Diffusion Transformer block with:
    - Adaptive Layer Normalization (AdaLN) for timestep conditioning
    - Multi-head self-attention with RoPE + sliding window
    - MLP feed-forward network
    """
    def __init__(self, d_model: int, d_cond: int, n_head: int, ..., window_size=None, rope=None):
        self.d_model = d_model
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.window_size = window_size if window_size else -1
        self.rope = rope
        
        # Projection layers
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        # Conditioning and normalization
        self.adaLN = AdaLN(d_model=d_model)
        self.timestep_embedding = TimestepEmbedder(d_model=d_model, max_timesteps=1000)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.cond_embed = nn.Linear(d_cond, d_model)
        
        # Feed-forward
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.SiLU(),
            nn.Linear(d_model * 2, d_model),
        )
    
    def forward(self, src: torch.Tensor, t: torch.Tensor, cond: torch.Tensor, padding_mask=None):
        """
        Inputs:
        - src: [seq_len, batch_size, d_model]
        - t: [batch_size]  (timestep)
        - cond: [1, batch_size, d_cond]  (skeleton condition)
        - padding_mask: [batch_size, seq_len]  (1 = valid, 0 = padding)
        
        Output: [seq_len, batch_size, d_model]
        """
        # 1. Fuse condition + timestep
        cond_emb = self.cond_embed(cond)
        time_emb = self.timestep_embedding(t)  # [1, batch, d_model]
        cond_fusion = cond_emb + time_emb
        
        # 2. AdaLN generates per-layer modulation parameters
        gate_msa, shift_msa, scale_msa, gate_ffn, shift_ffn, scale_ffn = self.adaLN(cond_fusion)
        
        # 3. Generate attention masks
        attn_mask = None
        if self.window_size > 0:
            attn_mask = get_window_mask(seq_len, self.window_size, src.device)
            attn_mask = attn_mask.expand(batch_size, self.n_head, seq_len, seq_len)
        
        if padding_mask is not None:
            # Convert padding_mask [batch, seq_len] → [batch, n_head, seq_len, seq_len]
            padding_mask_expanded = padding_mask.unsqueeze(1).unsqueeze(1)
            padding_mask_expanded = padding_mask_expanded.expand(batch_size, self.n_head, seq_len, seq_len)
            padding_mask_expanded = padding_mask_expanded.to(dtype=src.dtype) * -1e9
            attn_mask = (attn_mask + padding_mask_expanded) if attn_mask else padding_mask_expanded
        
        # 4. Self-attention with ada-shift-scale normalization
        x = ada_shift_scale(self.norm1(src), shift_msa, scale_msa)
        q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        x_attn = self._multihead_attention(q, k, v, attn_mask)
        x = src + gate_msa * x_attn
        
        # 5. Feed-forward
        x = ada_shift_scale(self.norm2(x), shift_ffn, scale_ffn)
        x = x + gate_ffn * self.ffn(x)
        
        return x
    
    def _multihead_attention(self, q, k, v, attn_mask=None):
        """
        Seq-first multi-head attention with RoPE
        
        Input shape: [seq_len, batch_size, d_model]
        """
        seq_len, batch_size = q.shape[0], q.shape[1]
        
        # Split heads
        q = q.reshape(seq_len, batch_size, self.n_head, self.head_dim)
        k = k.reshape(seq_len, batch_size, self.n_head, self.head_dim)
        v = v.reshape(seq_len, batch_size, self.n_head, self.head_dim)
        
        # Apply RoPE
        if self.rope:
            q = self.rope(q)
            k = self.rope(k)
        
        # Compute attention scores
        q = q.permute(1, 2, 0, 3)  # [batch, n_head, seq_len, head_dim]
        k = k.permute(1, 2, 3, 0)  # [batch, n_head, head_dim, seq_len]
        v = v.permute(1, 2, 0, 3)  # [batch, n_head, seq_len, head_dim]
        
        attn_weights = (q @ k) / math.sqrt(self.head_dim)
        if attn_mask is not None:
            attn_weights = attn_weights + attn_mask
        
        attn_weights = torch.softmax(attn_weights, dim=-1)
        attn_output = attn_weights @ v
        
        # Merge heads and output projection
        attn_output = attn_output.permute(2, 0, 1, 3).reshape(seq_len, batch_size, self.d_model)
        attn_output = self.out_proj(attn_output)
        
        return attn_output
```

### 2.4 Conditioning Mechanism: AdaLN

**File**: Lines 438-462

```python
class AdaLN(nn.Module):
    """
    Adaptive Layer Normalization - generates per-layer normalization parameters
    from timestep+condition features
    
    Outputs 6 tensors (2 for attn, 2 for FFN):
    - gate_msa, shift_msa, scale_msa for attention block
    - gate_ffn, shift_ffn, scale_ffn for FFN block
    
    Usage: x_norm = x * (scale + 1) + shift, then scaled by gate
    """
    def __init__(self, d_model):
        super().__init__()
        # MLP to project condition to 6*d_model parameters
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_model, 6 * d_model, bias=True)
        )
        nn.init.zeros_(self.adaLN_modulation[1].weight)
        nn.init.zeros_(self.adaLN_modulation[1].bias)
    
    def forward(self, c: torch.Tensor):
        """
        c shape: [1, batch, d_model]
        
        Output: 6 tensors each [1, batch, d_model]
        """
        return self.adaLN_modulation(c).chunk(6, dim=-1)

def ada_shift_scale(x, shift, scale):
    return x * (scale + 1) + shift
```

### 2.5 Timestep Embedding

**File**: Lines 355-409

```python
class TimestepEmbedder(nn.Module):
    """
    Encodes diffusion timestep t ∈ [0, num_timesteps) to d_model-dim embedding
    Uses sinusoidal encoding followed by MLP
    """
    def __init__(self, d_model: int, max_timesteps: int = 1000):
        self.d_model = d_model
        # Frequency matrix for sinusoidal encoding
        self.freqs = nn.Parameter(
            torch.exp(torch.linspace(0, math.log(10000.0), d_model // 2) * -1),
            requires_grad=False
        )
        # MLP to process sinusoidal features
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model, bias=True),
            nn.SiLU(),
            nn.Linear(d_model, d_model, bias=True),
        )
    
    def forward(self, timesteps: torch.Tensor):
        """
        timesteps: [batch_size]
        
        Output: [1, batch_size, d_model]  (unsqueezed for broadcasting)
        """
        x = timesteps.unsqueeze(1).float()  # [batch, 1]
        emb = torch.cat(
            [torch.sin(x * self.freqs), torch.cos(x * self.freqs)],
            dim=1
        )  # [batch, d_model]
        emb = self.mlp(emb)
        return emb.unsqueeze(0)  # [1, batch, d_model]
```

---

## 3. Diffusion Framework Layer

**File**: `/apdcephfs_cq10/share_1467498/home/chengxuzuo/projects/MoGenDIT/EasyDiffusion/base_diffusion.py`

### 3.1 Core Diffusion: GaussianDiffusion

**File**: Lines 38-128

```python
class GaussianDiffusion:
    """
    Gaussian Diffusion Process - encapsulates forward (q) and reverse (p) processes
    """
    def __init__(
        self,
        num_timesteps: int = 1000,
        beta_schedule: BetaSchedule = BetaSchedule.COSINE,
        model_mean_type: ModelMeanType = ModelMeanType.START_X,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        noise_remap_mode: str = "identity",
    ):
        """
        Beta schedules control noise level at each timestep
        
        LINEAR: β_t ∈ [beta_start, beta_end] linearly
        COSINE: β_t based on cosine schedule (smoother, better for small timesteps)
        """
        self.num_timesteps = num_timesteps  # Typically 1000
        self.model_mean_type = model_mean_type  # EPSILON or START_X
        self.noise_remap_mode = noise_remap_mode
        
        # Initialize beta sequence
        if beta_schedule == BetaSchedule.LINEAR:
            self.betas = self._linear_beta_schedule(beta_start, beta_end, num_timesteps)
        elif beta_schedule == BetaSchedule.COSINE:
            self.betas = self._cosine_beta_schedule(num_timesteps)
        
        # Precompute diffusion schedule parameters
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = np.cumprod(self.alphas)  # ᾱ_t = ∏(1-β_i)
        
        # Store as torch tensors for GPU computation
        self.alphas_cumprod_t = torch.tensor(...).unsqueeze(0)
        self.sqrt_alphas_cumprod_t = torch.sqrt(self.alphas_cumprod_t)
        self.sqrt_one_minus_alphas_cumprod_t = torch.sqrt(1 - self.alphas_cumprod_t)
        
        # Posterior distribution parameters for reverse process
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.posterior_mean_coef1 = self.betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.posterior_mean_coef2 = (1.0 - self.alphas_cumprod_prev) * np.sqrt(self.alphas) / (1.0 - self.alphas_cumprod)
    
    def _cosine_beta_schedule(self, num_timesteps: int, s: float = 0.008):
        """
        Cosine schedule: β_t = 1 - (cos((t+s)/(1+s) * π/2))^2
        Advantage: slower noise increase at beginning, faster at end
        """
        def alpha_bar(t):
            return np.cos((t + s) / (1 + s) * np.pi / 2) ** 2
        
        betas = []
        for i in range(num_timesteps):
            t1 = i / num_timesteps
            t2 = (i + 1) / num_timesteps
            betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), 0.999))
        return np.array(betas, dtype=np.float64)
```

### 3.2 Forward Process (q_sample)

**File**: Lines 129-169

```python
def q_sample(
    self,
    x0: torch.Tensor,
    t: torch.Tensor,
    noise: Optional[torch.Tensor] = None,
    obs_mask: Optional[torch.Tensor] = None,
    length_mask: Optional[torch.Tensor] = None,
):
    """
    Forward diffusion: x_t = √ᾱ_t * x_0 + √(1-ᾱ_t) * ε
    
    Inputs:
    - x0: [batch, seq_len, 201]  (clean motion)
    - t: [batch]  (timestep index)
    - obs_mask: [batch, seq_len, 201]  (observed regions, ≥0.99999)
    - length_mask: [batch, seq_len]  (valid frame regions)
    
    Outputs:
    - x_t: [batch, seq_len, 201]  (noisy motion at timestep t)
    - noise: [batch, seq_len, 201]  (noise that was applied)
    
    Key feature: Observed regions (keyframes) are NOT noised
    """
    if noise is None:
        noise = torch.randn_like(x0, device=x0.device)
    noise = noise_remapping(noise, mode=self.noise_remap_mode)
    
    # Extract schedule parameters for timestep t
    sqrt_alphas_cumprod = self._extract(self.sqrt_alphas_cumprod_t, t, x0.shape)
    sqrt_one_minus_alphas_cumprod = self._extract(self.sqrt_one_minus_alphas_cumprod_t, t, x0.shape)
    
    # Apply forward diffusion formula
    x_noise = sqrt_alphas_cumprod * x0 + sqrt_one_minus_alphas_cumprod * noise
    
    # Create noise mask: regions to apply noise
    noise_mask = torch.ones_like(noise, dtype=torch.bool)
    if obs_mask is not None:
        noise_mask[obs_mask >= (1 - 1e-5)] = False  # Don't noise observed regions
    if length_mask is not None:
        noise_mask *= length_mask
    
    # Apply selective noising
    x_t = x0.clone()
    x_t[noise_mask] = x_noise[noise_mask]
    noise[~noise_mask] *= 0
    
    return x_t, noise
```

### 3.3 Reverse Process Components

**File**: Lines 171-238

```python
def p_mean_variance(self, model: Callable, x_wrap: dict, t: torch.Tensor):
    """
    Reverse diffusion: computes mean and variance for p(x_{t-1} | x_t)
    
    Uses model prediction (either EPSILON or START_X) to infer x_0
    Then computes posterior mean: μ(x_t, x_0) and posterior variance: Σ
    
    Equations:
    - If EPSILON mode: x_0 = (x_t - √(1-ᾱ_t)*ε_pred) / √ᾱ_t
    - If START_X mode: x_0 = model_output directly
    
    - Posterior mean: μ = (β̃_t / (1-ᾱ_t)) * x_0 + ((1-ᾱ_{t-1}) / (1-ᾱ_t)) * x_t
      where β̃_t = (1 - ᾱ_{t-1}) / (1 - ᾱ_t) * β_t
    """
    model_output = model(x_wrap, t)
    
    if self.model_mean_type == ModelMeanType.EPSILON:
        pred_x0 = self._predict_x0_from_eps(x_wrap["x_t"], t, model_output)
    elif self.model_mean_type == ModelMeanType.START_X:
        pred_x0 = model_output
    
    # Compute posterior mean
    posterior_mean = (
        self._extract(self.posterior_mean_coef1_t, t, x_wrap["x_t"].shape) * pred_x0
        + self._extract(self.posterior_mean_coef2_t, t, x_wrap["x_t"].shape) * x_wrap["x_t"]
    )
    
    # Compute posterior log-variance
    posterior_variance = self._extract(self.posterior_variance_t, t, x_wrap["x_t"].shape)
    posterior_log_variance = torch.log(posterior_variance)
    
    return {
        "mean": posterior_mean,
        "variance": posterior_variance,
        "log_variance": posterior_log_variance,
        "pred_x0": pred_x0,
    }
```

---

## 4. Training Framework

**File**: `/apdcephfs_cq10/share_1467498/home/chengxuzuo/projects/MoGenDIT/trainer/my_trainer.py`

### 4.1 Distributed Trainer Architecture

**File**: Lines 27-122

```python
class MoGenDitDistributedTrainer(DistributedLMTrainer):
    """
    Multi-GPU distributed training with DDP (DistributedDataParallel)
    Supports gradient synchronization, EMA checkpointing, log management
    """
    def __init__(
        self,
        args,
        train_platform,
        model: nn.Module,
        diffusion,
        data,
        motion_rep,
    ):
        super().__init__(
            model=model,
            data=data,
            optimizer=None,
            ema_decay=getattr(args, 'ema_decay', 0.999),
            ema_start_step=getattr(args, 'ema_start_step', 2000)
        )
        
        self.args = args
        self.batch_size = args.batch_size
        self.log_interval = args.log_interval
        self.save_interval = args.save_interval
        self.mask_scheduler = MotionMaskScheduler(motion_rep=motion_rep)
        self.body_model = AnimoSMPLBody()
        
        # DDP wrapper (only in distributed mode)
        if self.distributed:
            self.model = DDP(
                self.model.to(self.device),
                device_ids=[self.gpu],
                find_unused_parameters=False
            )
            # Reinitialize EMA model for wrapped model
            self.ema_model = copy.deepcopy(self.model.module)
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )
        
        # Data loading with DistributedSampler
        self.data_sampler = DistributedSampler(self.data) if self.distributed else None
        self.data_loader = DataLoader(
            dataset=self.data,
            batch_size=self.batch_size,
            shuffle=(self.data_sampler is None),
            sampler=self.data_sampler,
            drop_last=False,
            num_workers=4,
            pin_memory=True,
        )
        
        self.diffusion = diffusion
        self.schedule_sampler = create_named_schedule_sampler(
            args.schedule_sampler_type, diffusion
        )
        self.motion_rep = motion_rep
```

### 4.2 Training Loop

**File**: Lines 199-272

```python
def train(self, iters, keyframe_modes):
    """
    Main training loop with producer-consumer pattern for data loading
    """
    torch.cuda.empty_cache()
    
    total_steps_per_epoch = len(self.data_loader)
    num_epochs = math.ceil(iters / total_steps_per_epoch)
    
    self.model.train()
    
    for epoch in range(num_epochs):
        if self.distributed:
            self.data_sampler.set_epoch(epoch)  # Reshuffle each epoch
        
        self.data_queue = Queue(maxsize=12)  # Buffer for prefetching
        
        # Start producer thread for data loading
        data_producer_thread = Thread(
            target=self.data_producer,
            kwargs={"keyframe_modes": keyframe_modes}
        )
        data_producer_thread.start()
        
        while self.iter < target_iter:
            train_data_patch = self.data_queue.get()
            if train_data_patch is None:
                data_producer_thread.join(timeout=1)
                break
            
            losses = self.forward_backward(train_data_patch)
            self.iter += 1
            
            # Logging (main process only)
            if self.is_main_process() and self.iter % self.log_interval == 0:
                self.log_writer.add_scalars(...)
            
            # Checkpointing (main process only)
            if self.is_main_process() and self.iter % self.save_interval == 0:
                self.save(folder_path=self.save_dir, model_name=self.args.model_name)
        
        self.epoch += 1
    
    if self.distributed:
        dist.destroy_process_group()
```

### 4.3 Data Preparation with Motion Degradation

**File**: Lines 127-197

```python
def data_producer(self, keyframe_modes):
    """
    Producer thread: loads batch, applies degradation, adds noise
    """
    for batch, length in self.data_loader:
        batch = batch.to(self.device)
        length = length.to(self.device)
        
        # 1. Generate keyframe mask (which regions are observed)
        keyframe_mask = self.mask_scheduler.get_formulated_mask(
            motion=batch,
            length=length,
            mode_formula=keyframe_modes,
        )  # Shape: [batch, seq_len, 201]
        
        # 2. Generate padding mask (valid frames)
        bool_length_mask = self.mask_scheduler.get_length_mask_bool(motion=batch, length=length)
        padding_mask = (bool_length_mask[..., 0] == 0).float()
        
        # 3. Optional motion degradation (50% of batches)
        x0 = batch.clone()
        if self.args.motion_degradation and self.args.degrade_rate > 0:
            degradation_idx = random_index(batch.shape[0], sampling_rate=self.args.degrade_rate)
            keyframe_mask[degradation_idx] *= 0  # Clear all observed regions
            
            # Keep first N frames clean (reference frames)
            ref_frames = random.randint(1, 10)
            keyframe_mask[degradation_idx, :ref_frames] += 1
            
            # Apply degradation to motion (noise, jitter, etc.)
            x0[degradation_idx] = self.motion_rep.motion_degradation_batch(
                motion=batch[degradation_idx],
                keyframe_mask=keyframe_mask[degradation_idx],
                length=length[degradation_idx],
                bool_length_mask=bool_length_mask[degradation_idx],
            )
        
        # 4. Forward diffusion: x_0 → x_t + noise
        t, weight = self.schedule_sampler.sample(batch.shape[0], self.device)
        x_t, noise = self.diffusion.q_sample(
            x0=x0,
            t=t,
            obs_mask=keyframe_mask,
            length_mask=bool_length_mask
        )
        
        # 5. Prepare model input
        x_wrapped = self.model.wrap_inputs(
            x=x_t,
            cond=None,  # Unconditional for now
            mask=keyframe_mask,
            padding_mask=padding_mask
        )
        
        self.data_queue.put((x_wrapped, t, x0, noise, weight, bool_length_mask, length))
    
    self.data_queue.put(None)  # Signal end of data
```

### 4.4 Loss Computation

**File**: Lines 274-336

```python
def forward_backward(self, train_data_patch):
    """
    Forward pass through model, compute losses, backward pass
    
    Loss components:
    1. Denoising loss: L2 between pred_x0 and x0
    2. Geometric loss: skeleton rigidity (rigid body constraint)
    3. Drift loss: consistency between joint positions and velocities
    """
    x_wrapped, t, x_0, noise, weight, bool_length_mask, length = train_data_patch
    
    # Forward pass
    pred_x0 = self.model(x_wrapped, t)  # [batch, seq, 201]
    
    # Extract components
    pred_pose = pred_x0[:, :, self.motion_rep.pose_mask]  # [batch, seq, 132]
    gt_pose = x_0[:, :, self.motion_rep.pose_mask]
    
    pred_joint = pred_x0[:, :, self.motion_rep.joint_mask]  # [batch, seq, 66]
    gt_joint = x_0[:, :, self.motion_rep.joint_mask]
    
    pred_trans = pred_x0[:, :, self.motion_rep.trans_mask]  # [batch, seq, 3]
    gt_trans = x_0[:, :, self.motion_rep.trans_mask]
    
    # Loss 1: L2 denoising loss
    loss_denoise = F.smooth_l1_loss(
        pred_pose, gt_pose, reduction="none"
    ) * self.args.loss_weight_pose
    
    # Loss 2: Geometric loss (skeleton rigidity + drift)
    loss_rigid, loss_drift = geometric_loss_batch(
        R6d=pred_pose,
        joint=pred_joint,
        trans=pred_trans,
        global_pose=True,
        length=length,
        l1_weight=0.0,
        l2_weight=self.args.loss_weight_rigid,
    )
    
    # Total loss
    loss_total = (
        loss_denoise.mean() +
        loss_rigid * self.args.loss_weight_rigid +
        loss_drift * self.args.loss_weight_drift
    ) * weight.unsqueeze(1)
    
    # Backward pass
    self.optimizer.zero_grad()
    loss_total.mean().backward()
    if self.args.clip_grad_norm > 0:
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip_grad_norm)
    self.optimizer.step()
    
    # EMA update
    if self.ema_model is not None:
        self.update_ema()
    
    return {
        "loss": {"denoise": loss_denoise.mean(), "rigid": loss_rigid, "drift": loss_drift},
        "metrics": {"weight_avg": weight.mean()}
    }
```

### 4.5 Loss Functions

**File**: `/apdcephfs_cq10/share_1467498/home/chengxuzuo/projects/MoGenDIT/trainer/geometric_loss.py` (Lines 99-176)

```python
def geometric_loss_batch(R6d, joint, trans, global_pose=False, length=None, l1_weight=0.0, l2_weight=1.0):
    """
    Geometric/physical loss to enforce skeleton consistency
    
    Three components:
    1. Rigid body loss: joint offsets constant over time
    2. Forward kinematics loss: joint positions match FK from rotations
    3. Drift loss: motion derivative consistency
    """
    b = R6d.shape[0]
    R = r6d_to_rotation_matrix(R6d.clone()).reshape(-1, 24, 3, 3)
    joint = joint.clone().reshape(b, -1, 24, 3)
    trans = trans.clone().reshape(b, -1, 3)
    
    # 1. Rigid body constraint: offsets should stay constant
    offsets_from_motion = get_skeleton_offsets(
        pose=R, joint=joint, global_pose=global_pose
    ).reshape(b, -1, 24, 3)
    init_skeleton_offsets = offsets_from_motion[:, [0]]
    
    loss_rigid_body = F.mse_loss(
        offsets_from_motion,
        init_skeleton_offsets.expand_as(offsets_from_motion),
        reduction="none"
    ) * l2_weight
    
    # 2. Velocity drift loss: global motion should match velocity
    global_joint = (joint + trans.unsqueeze(-2)).reshape(b, -1, 24, 3)
    global_joint_delta = global_joint[:, 1:] - global_joint[:, :-1]
    loss_drift = F.mse_loss(
        global_joint_delta,
        vel[:, :-1] / fps,  # fps = 30 typically
        reduction="none"
    ) * l2_weight
    
    # Optional L1 penalty
    if l1_weight > 0.0:
        loss_rigid_body += l1_weight * F.l1_loss(...)
        loss_drift += l1_weight * F.l1_loss(...)
    
    # Mask by valid length
    if length is not None:
        mask = torch.zeros_like(loss_rigid_body[:, :, :1, :1])
        for i in range(b):
            mask[i, :length[i]] = 1.0
        loss_rigid_body = (loss_rigid_body * mask.expand_as(loss_rigid_body)).sum() / mask.expand_as(loss_rigid_body).sum()
        loss_drift = (loss_drift * mask[:, 1:].expand_as(loss_drift)).sum() / mask[:, 1:].expand_as(loss_drift).sum()
    
    return loss_rigid_body, loss_drift
```

---

## 5. Motion Refinement Pipeline

**File**: `/apdcephfs_cq10/share_1467498/home/chengxuzuo/projects/MoGenDIT/motion_process/motion_refiner.py` (Lines 19-330)

### 5.1 MoreDiffRefiner Class

**File**: Lines 19-117

```python
class MoreDiffRefiner:
    """
    Unified interface for motion refinement using the diffusion model
    Supports three refinement modes: denoise, ada_denoise, trans_regen
    """
    def __init__(self, motion_rep, model, diffusion):
        self.motion_rep = motion_rep
        self.model = model
        self.diffusion = diffusion
    
    def refine(
        self,
        motion,                   # [1, seq_len, 201]
        cond,                     # [1, 1, 66] or None
        step=10,                  # Number of diffusion steps
        eta=1.0,                  # Noise scale
        mode="denoise",           # Refinement mode
        window_size=224,          # For windowed processing
        use_windowed=False,       # Whether to use windowing
        fast_sampling=True,       # Use fewer timesteps (trans_regen only)
    ):
        """
        High-level refinement interface
        
        Modes:
        1. denoise: Low-level noise removal, 10 steps, eta=1.0
        2. ada_denoise: Adaptive denoising  
        3. trans_regen: Regenerate translation only, slower (50+ steps), eta=0.0
        """
        if not use_windowed:
            # Single-pass processing
            return self._non_windowed_refine(motion, cond, mask, keep_mask, ...)
        else:
            # Windowed processing for long sequences
            return self._windowed_refine(motion, cond, ...)
```

### 5.2 Refinement Modes

**File**: Lines 36-117

#### Mode 1: Denoise
```python
def _denoise_mode(self, motion, cond, mask, keep_mask, step=10, eta=1.0, imputation_mode="skip_last"):
    """
    Light refinement: remove small artifacts/noise
    - 10-50 diffusion steps
    - eta=1.0 (full noise injection)
    - Only affects regions marked by mask
    """
    _motion = self.motion_rep.normalization(motion.squeeze(0)).unsqueeze(0)
    with torch.no_grad():
        x_wrap = self.model.wrap_inputs(_motion, cond, mask, None)
        _motion = self.diffusion.denoise(
            x_wrap=x_wrap,
            model=self.model,
            num_timesteps=step,
            eta=eta,
            mask=keep_mask,
            imputation_mode=imputation_mode,
        )
    return _motion
```

#### Mode 2: Regenerate (trans_regen)
```python
def _regen_mode(self, motion, cond, mask, keep_mask=None, eta=0.0, ..., fast_sampling=True):
    """
    Full regeneration: recreate problematic regions from scratch
    - 50-999 diffusion steps (or fewer with fast_sampling)
    - eta=0.0 (deterministic, uses posterior variance)
    - Slower but higher quality
    - Can use custom_timesteps for faster convergence
    """
    _motion = self.motion_rep.normalization(motion.squeeze(0)).unsqueeze(0)
    with torch.no_grad():
        x_wrap = self.model.wrap_inputs(_motion, cond, mask, None)
        _motion = self.diffusion.ddim_sample_loop(
            x_wrap=x_wrap,
            model=self.model,
            eta=eta,
            mask=keep_mask,
            custom_timesteps=custom_timesteps if fast_sampling else None,
            imputation_mode=imputation_mode,
        )
    return _motion
```

**Custom timesteps for fast sampling** (File: Lines 5-16):
```python
custom_timesteps = [999, 750, 500, 250, 100, 50, 25, 10, 5, 0]
# Only 10 steps instead of full 1000, ~90% speedup with minor quality loss
```

### 5.3 Windowed Refinement

**File**: Lines 182-263

```python
def refine(..., use_windowed=True, window_size=224, prev_padding=20):
    """
    For long sequences: split into overlapping windows, process each, stitch together
    
    Algorithm:
    1. Window = 224 frames
    2. Overlap = 20 frames (previous padding)
    3. For each window:
       a. Extract window from motion
       b. Mark previous frames as fixed (keep_mask)
       c. Apply refinement to new frames
       d. Calculate translation distance per frame
       e. If distance > 4.0m, truncate window (walking too far)
       f. Pre-stitch output to original motion
    4. Move to next window
    """
    current_idx = 0
    prev_frame_pad = 0
    
    while current_idx < motion.shape[1]:
        begin = current_idx
        end = min(begin + window_size, motion.shape[1])
        _motion = motion[:, begin:end]
        length = end - begin
        
        # Mark previous frames as fixed
        _mask = torch.zeros_like(_motion)
        _mask[:, :prev_frame_pad] += 1  # Previous padding frames fixed
        
        # Mark regions to regenerate based on mode
        if mode in ["denoise", "ada_denoise"]:
            _mask[:, :1] += 1  # Keep first frame
        elif mode == "trans_regen":
            _mask[:, :, pose_mask] += 1  # Keep all pose
            _mask[:, :, joint_mask] += 1
        
        _motion = self._non_windowed_refine(
            _motion, cond, _mask, _keep_mask, ...
        )
        
        # **Automatic cutoff**: Find where character walks > 4m
        _trans = _motion[..., trans_mask].reshape(-1, 3)
        cutoff_in_segment = None
        
        if _trans.shape[0] > 0:
            first_frame_trans = _trans[0:1, :]
            frame_distances = torch.norm(_trans - first_frame_trans, dim=1)
            distance_exceed_mask = frame_distances > 4.0
            cutoff_indices = torch.where(distance_exceed_mask)[0]
            
            if len(cutoff_indices) > 0:
                cutoff_in_segment = cutoff_indices[0].item()
                # Ensure at least 30 frames of new content before cutoff
                end = max(begin + cutoff_in_segment, begin + prev_frame_pad + 30)
                end = min(end, motion.shape[1])
                _motion = _motion[:, :(end - begin)]
        
        # Pre-stitch output to reference motion at beginning
        _motion = self.motion_rep.pre_stitch(
            _motion[0, :],
            ref_motion=motion[0, [begin]],
            reset_height=False,
            stitch_joint_idx=0,
        )
        
        # Update full motion
        motion[0, begin:end] = _motion
        prev_frame_pad = prev_padding
        current_idx += (end - begin) - prev_frame_pad
    
    return motion
```

---

## 6. Physics Simulation

**File**: `/apdcephfs_cq10/share_1467498/home/chengxuzuo/projects/MoGenDIT/animo/simulator.py` (Lines 17-300)

### 6.1 FlatGroundSimulator

**File**: Lines 17-90

```python
class FlatGroundSimulator:
    """
    Physics-based motion simulator enforcing ground contact and friction constraints
    Uses dual-level PD control: angular (joint rotations) + linear (root translation)
    Solves QP (Quadratic Programming) for contact forces
    """
    
    # Contact analysis
    adj_chain = {0: [], 7: [4,1,0], 8: [5,2,0], 10: [7,4,1,0], 11: [8,5,2,0]}
    
    # Physical parameters
    G = 9.8                # Gravity (m/s²)
    mu = 0.8              # Coefficient of friction
    
    def __init__(self, skeleton: AnimoSkeleton, fps=30, eps=1e-1):
        self.q_dof = skeleton.q_dof  # Degrees of freedom (75: 3 translation + 72 rotation)
        
        # PD gains adapt to fps (for numerical stability)
        self.kp_q = 1 * fps**2    # Angular stiffness (fps² scaling)
        self.kd_q = fps           # Angular damping
        self.kp_p = 1 * fps**2    # Linear position stiffness
        self.kd_p = fps           # Linear position damping
        
        self.w_qp = 0.5 * min(1.0, fps / 60)  # QP solver weight
        self.w_vel_ref = 0.2  # Reference velocity weight (vs. physics constraints)
        
        self.dt = 1 / fps
        self.eps = eps
        self.body_model = skeleton
```

### 6.2 Dual-Level Control System

**File**: Lines 115-195

```python
def update_state(self, pose, vel=None, trans=None):
    """
    Two-level PD control:
    1. Angular level: control joint rotations to match target pose
    2. Linear level: control root translation to match target + maintain contact
    
    Process:
    1. Encode target pose to generalized coordinates q_ref
    2. Compute angular PD error: des_qddot = Kp(q_ref - q) - Kd*qdot
    3. Compute linear PD error: des_pddot = Kp(p_ref - p) - Kd*pdot
    4. Solve QP to find qdot that satisfies both constraints
    5. Apply friction/contact constraints
    """
    
    # 1. Get target configurations
    q_ref = self.body_model.q_encode(pose, torch.zeros(3)).flatten()
    q_delta = q_ref - self.q
    q_delta[3:] = normalize_angle(q_delta[3:])  # Wrap angles to [-π, π]
    
    # 2. Desired accelerations (from PD control)
    des_qddot = self.kp_q * q_delta - self.kd_q * self.qdot
    des_pddot = self.kp_p * (p_ref - self.p) - self.kd_p * self.pdot
    
    des_qdot = self.qdot + des_qddot * self.dt
    des_pdot = self.pdot + des_pddot * self.dt
    
    # 3. Jacobian matrix (maps joint velocities to end-effector velocities)
    Js = self.body_model.get_Jacobian(R=self.pose)
    
    # 4. Solve QP: minimize ||Js @ qdot - des_pdot||²
    # Subject to: friction constraints, etc.
    P = Js.T @ Js + regularization
    q = -Js.T @ des_pdot
    x = solve_qp(P, q, solver="quadprog")  # Quadratic programming solver
    
    qdot_qp = torch.tensor(x)
    
    # 5. Blend QP solution with desired velocity (weighted)
    qdot_fusion = qdot_qp * self.w_qp + des_qdot * (1 - self.w_qp)
    
    # 6. Integrate velocity to update configuration
    q_fusion = self.q + qdot_fusion * self.dt
    pose_optim, _ = self.body_model.q_decode(q_fusion)
    
    _, p_optim = self.body_model.forward_kinematics(pose_optim, calc_joint=True)
    p_com = self.body_model.calc_com_position(p_optim)
```

### 6.3 Contact and Friction Handling

**File**: Lines 206-250

```python
def update_state(...):  # Lines 206 onwards
    """
    After pose optimization, handle global translation with friction
    
    Logic:
    1. Check if character is grounded (minimal height = ground contact)
    2. If grounded and insufficient vertical momentum to jump:
       - Apply static friction constraints
       - Limit horizontal acceleration by: |a_xz| ≤ μ * (|a_y + g|)
    3. If in air:
       - Apply drag/air resistance
       - Reduce horizontal control authority
    """
    
    # Contact detection
    if self.t[1] <= self.get_minimal_height(self.p):
        self.float_flag = False
        contact_joint_idx = self.contact_judge(p_com)  # Which joints touch ground?
        
        # Vertical momentum check
        total_y_momentum = self.tdot[1] + self.p_com_dot[1]
        gravity_impulse = self.G * self.dt
        
        if total_y_momentum - gravity_impulse <= 0:  # Can't jump
            # Static friction: horizontal acceleration limited by normal force
            for contact_joint in contact_joint_idx:
                tdot_local = (p_prev[contact_joint] - p_optim[contact_joint]) / self.dt
                tddot = (tdot_local - self.tdot) / self.dt
                
                max_tddot_xz = (tddot[1] + p_com_ddot[1] + self.G).abs() * self.mu
                com_tddot_xz = torch.norm(tddot[[0,2]] + p_com_ddot[[0,2]])
                
                if com_tddot_xz > max_tddot_xz:
                    # Scale down horizontal acceleration to fit friction cone
                    tddot[[0,2]] *= max_tddot_xz / (com_tddot_xz + eps)
        else:  # Can jump
            # Apply reduced friction during flight
            tddot_fusion *= 0.5  # Reduced horizontal control in air
    else:
        self.float_flag = True  # Airborne
        # Reduce velocity damping during flight
```

---

## 7. Key Hyperparameters and Configurations

### Model Sizes
```
0.03B: d_model=512,  n_head=8,  n_stack=8   (minimal)
0.1B:  d_model=768,  n_head=12, n_stack=12  (recommended)
0.3B:  d_model=1024, n_head=16, n_stack=18  (large)
```

### Training
```
- Diffusion steps: 1000
- Beta schedule: COSINE (smooth noise increase)
- Model mean type: START_X (predict x_0 directly)
- Motion representation: OccamMotionRep (201-dim)
- Batch size: 64-256 (distributed training)
- Learning rate: 1e-4 with AdamW
- EMA decay: 0.999
- EMA start step: 2000
```

### Refinement
```
Denoise:      steps=10,   eta=1.0   (light, fast)
Ada_denoise:  steps=10-50, eta=varies (adaptive)
Trans_regen:  steps=50-999, eta=0.0 (full, slow)
Fast_sampling: custom_timesteps=[999,750,500,250,100,50,25,10,5,0] (~90% speedup)
Window_size:  224 frames (~7.5 sec at 30fps)
Prev_padding: 20 frames (~0.67 sec overlap)
```

### Physics Simulation (FlatGroundSimulator)
```
- Gravity: G = 9.8 m/s²
- Friction coeff: μ = 0.8
- PD gains: Kp_q = fps², Kd_q = fps (adaptive)
- QP blend weight: w_qp = 0.5 * min(1.0, fps/60)
- Velocity ref weight: w_vel_ref = 0.2
- Contact feet: indices [7, 8, 10, 11] (ankles + toes)
```

---

## 8. Rotation 6D Convention (CRITICAL)

**File**: `/apdcephfs_cq10/share_1467498/home/chengxuzuo/projects/MoGenDIT/trainer/geometric_loss.py` (Line 179-196)

```python
def r6d_to_rotation_matrix(r6d: torch.Tensor):
    """
    MoGenDIT uses COLUMN-MAJOR convention!
    6D = first two COLUMNS of 3×3 rotation matrix, flattened
    
    6D layout: [R00, R10, R20, R01, R11, R21]  ← Column-major
              (first 3 are column 0, next 3 are column 1)
    
    Reconstruction (Gram-Schmidt orthonormalization):
    1. Normalize column 0: c0 = normalize(r6d[0:3])
    2. Orthogonalize column 1: c1 = normalize(r6d[3:6] - <c0, r6d[3:6]> * c0)
    3. Cross product: c2 = c0 × c1
    4. Output: R = [c0 | c1 | c2]
    """
    r6d = r6d.reshape(-1, 6)
    column0 = normalize_tensor_eps(r6d[:, 0:3])
    column1 = normalize_tensor_eps(
        r6d[:, 3:6] - (column0 * r6d[:, 3:6]).sum(dim=1, keepdim=True) * column0
    )
    column2 = column0.cross(column1, dim=1)
    r = torch.stack((column0, column1, column2), dim=-1)
    return r
```

**From articulate library** (used in MoGenDIT):
```python
# articulate/math/angular.py
def r6d_to_rotation_matrix(r6d: torch.Tensor):
    """Same column-major convention as above"""
    # Implementation details in articulate library
```

**WARNING**: HyMotion M2M uses ROW-MAJOR convention (incompatible!)
Row-major: `[R00, R01, R10, R11, R20, R21]` (first two ROWS)
Must convert with index reordering: `[0, 2, 4, 1, 3, 5]` to switch between them.

---

## Summary Table

| Component | Type | Size | Key Files |
|-----------|------|------|-----------|
| Motion Repr | OccamMotionRep | 201-dim | motion_representation.py:649-1048 |
| Model | MoreDiff DiT | 0.03B/0.1B/0.3B | more_diff.py:253-504 |
| Diffusion | GaussianDiffusion | 1000 steps | base_diffusion.py:38-300 |
| Training | MoGenDitDistributedTrainer | DDP | my_trainer.py:27-400 |
| Refinement | MoreDiffRefiner | 3 modes | motion_refiner.py:19-330 |
| Physics | FlatGroundSimulator | QP-based | simulator.py:17-300 |
| Loss | GeometricLoss | Rigidity+Drift | geometric_loss.py:99-176 |

