# PRISM Jitter Mechanisms: Detailed Visual Analysis

## 1. CFG GUIDANCE AMPLIFICATION FLOW

```
Denoising Step i:
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  Transformer Forward Pass (Conditioned)                    │
│  ────────────────────────────────────────────────────────  │
│  Input: [latents, timestep_i, prompt_embeddings]           │
│  Output: noise_pred_cond                                   │
│                                                             │
│  Transformer Forward Pass (Unconditioned)                  │
│  ────────────────────────────────────────────────────────  │
│  Input: [latents, timestep_i, negative_prompt_embeddings] │
│  Output: noise_pred_uncond                                 │
│                                                             │
│  CFG Scaling (Line 437-438):                               │
│  ────────────────────────────────────────────────────────  │
│  ┌─ noise_pred = noise_uncond + guidance_scale × (...)     │
│  └─ guidance_scale = 5.0                                   │
│                                                             │
│  AMPLIFICATION: 5× larger than base noise predictions!     │
│                                                             │
│  Example:                                                   │
│  ├─ noise_pred_uncond  = [0.1, 0.2, 0.3]  (baseline)      │
│  ├─ noise_pred_cond    = [0.4, 0.5, 0.6]  (conditioned)   │
│  ├─ difference         = [0.3, 0.3, 0.3]  (guidance)      │
│  └─ final noise_pred   = [0.1, 0.2, 0.3] + 5×[0.3, 0.3, 0.3]
│                        = [1.6, 1.7, 1.8]  ← 5-6× LARGER!   │
│                                                             │
│  Scheduler Step:                                            │
│  ────────────────────────────────────────────────────────  │
│  latents_i+1 = f(noise_pred, timestep, latents_i)         │
│  ↑ 5× larger noise → 5× larger latent changes             │
│  ↑ Velocity ∝ Δlatents ∴ velocity 5× larger too!          │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. KAFS KINEMATIC ASYNCHRONY

```
Denoising Timeline (Timesteps 1000 → 0):

Standard Denoising:
═════════════════════
Joint: pelvis | wrist | pelvis | wrist | ...
Time:  t      | t     | t      | t     |
Noise: N(t)   | N(t)  | N(t-1) | N(t-1)|


With KAFS (alpha_pelvis=0.85, alpha_wrist=1.15):
═════════════════════════════════════════════════════
Joint:      pelvis | wrist  | pelvis | wrist   |
Raw time:   1000   | 1000   | 800    | 800     |
Scaled time: 850   | 1150   | 680    | 920     |  ← DIFFERENT!
Noise level: N(850)| N(1150)| N(680) | N(920)  |

Effect at Kinematic Chain:
──────────────────────────
Wrist = shoulder + upper_arm + forearm_pose
      = f(pelvis_pose, ...)  ← depends on ancestor joints

At step 500:
├─ Pelvis denoised with t'=500×0.85=425    (less noisy)
├─ Wrist denoised with t'=500×1.15=575     (more noisy)
└─ IK inconsistency: wrist pos from "future" noise level!
   Pelvis still noisy from step 425
   Wrist fully denoised from step 575
   → Position inconsistency → jitter


Within-chain Velocity:
─────────────────────
v_wrist = (wrist_t+1 - wrist_t) / Δt
        = f(pelvis_noise_t, wrist_noise_t)
        
If pelvis_noise >> wrist_noise or vice versa:
  → velocity 2-3× baseline
```

---

## 3. DENORMALIZATION CASCADE

```
Latent Space (Normalized):
──────────────────────────
noise_pred ∈ [-1, 1] range
latents ∈ [−3σ, +3σ] ≈ [-3, 3]
Δlatents_frame ≈ 0.1-0.2 (small normalized changes)


Denormalization Step 1 (Line 598):
──────────────────────────────────
latents_denorm = latents × latents_std + latents_mean

Example with per-channel statistics:
Channel 0: latents_std = 2.0  → multiply by 2.0
Channel 1: latents_std = 0.5  → multiply by 0.5
Channel 2: latents_std = 1.5  → multiply by 1.5

Δlatents_frame_denorm:
├─ Channel 0: 0.1 × 2.0 = 0.2  (2× amplification)
├─ Channel 1: 0.1 × 0.5 = 0.05 (0.5× dampening)
└─ Channel 2: 0.1 × 1.5 = 0.15 (1.5× amplification)


VAE Decode:
───────────
[16, T_latent, 23] → [T, 23, 6]

High-std channels already amplified 2× in latent space
→ VAE decoder uses amplified features
→ Output motion has 2-4× higher amplitude in those joints


Denormalization Step 2 (Line 628):
──────────────────────────────────
motion_denorm = motion × motion_std + motion_mean

motion_std ranges:
├─ Translation: std ≈ 1.2-1.5 m
├─ Rotation (6D): std ≈ 0.8-1.2
├─ Velocity: std ≈ 0.5-1.5 m/s

motion_frame_denorm ∝ Δlatents_denorm × motion_std
                    ∝ 0.2 × 1.2  (for high-std latent channel)
                    ≈ 0.24


CUMULATIVE AMPLIFICATION:
───────────────────────
      CFG (5×) × Latent_denorm (2×) × Motion_denorm (1.2×) × KAFS (1.3×)
    = 5 × 2 × 1.2 × 1.3
    ≈ 15.6× potential amplification!

But typically observed: 5-10× because not all mechanisms active simultaneously
```

---

## 4. SEGMENT BOUNDARY DISCONTINUITY

```
Autoregressive Generation:

Segment 1 (Prompt A: "Walk forward"):
═════════════════════════════════════
Generate frames [0, 128]
│
├─ Frame 0-50: Starting motion (denoising in progress)
├─ Frame 50-100: Mid-segment (smooth continuation of A)
├─ Frame 100-128: End of segment A
│   └─ Frame 128 = LAST FRAME = Condition for segment 2
│
Full motion Segment 1: natural progression
Velocity pattern: smooth (learned from "walk forward" training)


Segment 2 (Prompt B: "Turn around"):
═══════════════════════════════════════
Frame 128 forced as condition (Frame 129 = 128)
│
├─ Frame 129 (generated, free): Continue B motion, different prompt!
│   └─ NOT conditioned to smoothly follow A
│   └─ Instead trained to START motion B (turn around)
│
├─ Frame 130-178: Motion B continues
│
Problem at boundary:
════════════════════
Frame 128 (end of A): velocity_A = (pos_128 - pos_127) / dt
Frame 129 (start of B): velocity_B = (pos_129 - pos_128) / dt

pos_128 from: "walk forward" distribution (Segment 1 training)
pos_129 from: "turn around" distribution (Segment 2 training)

These are different motion distributions!
→ velocity jump: v_jump = |velocity_B - velocity_A|
→ Typically 2-5× baseline velocity


Velocity profile across boundary:
─────────────────────────────────
  v
  │     Segment 1         Segment 2
  │    (Smooth)          (Smooth)
  │    /‾‾‾‾‾\            /‾‾‾‾‾
  │   /       \   JUMP   /
  │  /         \    ↑    /
  │_/__________\___↓____/___________→ frames
   127  128   129  130  131
        ↑ boundary discontinuity
        
Velocity_129 can be 2-5× higher than Velocity_128


Why not fixed by overlap_frames=1?
──────────────────────────────────
├─ overlap_frames=1: Skip 1 frame during concatenation
├─ But Frame 129 is STILL generated independently
├─ The skip doesn't smooth the discontinuity
├─ It just removes the "duplicate" (which was Frame 128)
└─ Boundary between Segment 1's last and Segment 2's second frame
   is still hard cut!
```

---

## 5. COMBINED AMPLIFICATION STACK

```
Signal Flow with All Jitter Sources:

┌─────────────────────────────────────────────────────┐
│ Baseline Noise (from diffusion model)                │
│ Magnitude: σ_base ≈ 0.05-0.1                         │
└──────────────────┬──────────────────────────────────┘
                   │
                   ↓
        ┌──────────────────────┐
        │ CFG Scaling (5.0×)    │
        │ σ → 5×σ              │
        └──────────┬───────────┘
                   │
                   ↓
        ┌──────────────────────┐
        │ Latent Denormalization│
        │ (2.0× for high-std)  │
        │ σ → 2×σ              │
        └──────────┬───────────┘
                   │
                   ↓
        ┌──────────────────────┐
        │ VAE Decode           │
        │ (preserves magnitude)│
        └──────────┬───────────┘
                   │
                   ↓
        ┌──────────────────────┐
        │ Motion Denormalization│
        │ (1.2× motion_std)    │
        │ σ → 1.2×σ            │
        └──────────┬───────────┘
                   │
                   ↓
        ┌──────────────────────┐
        │ KAFS Kinematic Async │
        │ (1.3× for misaligned)│
        └──────────┬───────────┘
                   │
                   ↓
        ┌──────────────────────┐
        │ Segment Boundary     │
        │ (2-5× discontinuity) │
        └──────────┬───────────┘
                   │
                   ↓
    ┌───────────────────────────────┐
    │ FINAL JITTER                  │
    │ σ_final = 0.075 × 5 × 2 × 1.2 │
    │         × 1.3 × 2.5           │
    │         ≈ 2.8 (within segment)│
    │         ≈ 5-7 (at boundary)   │
    └───────────────────────────────┘
```

---

## 6. Frame-to-Frame Velocity Visualization

```
Observed Motion: "Walk Forward" then "Turn Around"

Position (frontal view):
┌─────────────────────────────────────┐
│  Y ^                                 │
│    │      Turn!                      │
│    │      /‾‾‾‾\   Segment 2        │
│    │     /      \                    │
│    │   Walk→     \                   │
│    │ /‾‾‾‾‾\     \                  │
│    │/       \     \___→              │
│    ├─────────┼─────────── X          │
│                                      │
└─────────────────────────────────────┘

Frame-to-Frame Velocity Magnitude:
┌─────────────────────────────────────┐
│  v (m/s)                             │
│  │                                    │
│  │   Segment 1 (smooth)              │
│  │  ╱‾‾‾‾‾\     /\    /‾\           │
│  │ ╱       \___╱  \__╱   ╲          │
│  │                    ↓ JUMP AT BOUNDARY
│  │                    \╱‾‾╲ Segment 2
│  │                   2-5× │╲╱‾‾╲
│  │─────────┼─────────────┼───────→  │
│ 0│      [Frame]         [128→129]   │
│         ↑ Within-segment ↑ Between  │
│         Low variance     High jump   │
│                                      │
└─────────────────────────────────────┘

Jitter = sqrt(var(within-segment) + var(between-segment))
       ≈ sqrt(1² + 3²)
       ≈ 3.16

With CFG=5 + denorm: ≈ 8-10 observed
```

