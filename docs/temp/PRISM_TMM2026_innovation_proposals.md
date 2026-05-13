# PRISM TMM 2026 Paper Strategy

> Updated: 2026-05-13
> Status: Paper edits complete. Repositioned from "engineering combination" to "insight paper."

---

## 一、Problem Diagnosis

### 1.1 ECCV Reviewer Feedback

Three reviewers converged on the same weakness:
- **Reviewer cL6x**: "incremental novelty" — per-joint tokenization from MoGenTS, noise-free conditioning from Diffusion Forcing
- **Reviewer qAQ1**: "both contributions exist in prior work" — lacking independent technical innovation
- **Reviewer FG3s**: "limited contribution" — engineering combination rather than methodological innovation

### 1.2 Root Cause

The ECCV submission framed PRISM as a "careful engineering combination" of existing techniques. This framing invited the "incremental" verdict even though the experimental results were strong.

### 1.3 Resolution Strategy

Rather than adding new modules (which would require retraining), we repositioned the paper around a **principled insight** that was always present but poorly articulated:

> **Latent-generator alignment principle**: Aligning the latent token structure with the physical structure of the generation target fundamentally simplifies the generative learning problem.

This is not a new module — it is a new way of understanding *why* the architecture works, backed by controlled ablation evidence.

---

## 二、Survey of Related Innovations (Reference)

| Method | Innovation | Relevance to PRISM |
|--------|-----------|-------------------|
| **ANT** (2025) | Frequency-aware adaptive denoising stages, Dynamic CFG | Frequency-domain perspective |
| **Free-T2M** (2025) | DCT low-frequency consistency loss + semantic consistency loss | Frequency-domain loss |
| **LMR** (2025) | Dual-granularity tokenizer (reasoning + execution latent) | Hierarchical generation |
| **POMP** (CVPR 2025) | Kinematics-dynamics dual module + phase manifold bridge | Physics consistency |
| **FlashMo/MotionSiT** (2025) | SO(3) Lie-group joint rotation diffusion | Rotation-space geometry |
| **UniMoGen** (2025) | Kinematic-aware attention masks (joint-ancestor attention) | Structured attention |
| **HY-Motion** (2024) | Window attention + asymmetric masking + DPO/GRPO | Training paradigm |
| **CoDA** (2025) | Part-wise diffusion + gradient-flow coordination | Part-wise denoising |
| **MoGenTS** (NeurIPS 2024) | Per-joint discrete tokenization for VQ | Encoding-side per-joint benefit |

---

## 三、Current Paper Positioning

### 3.1 Core Contribution: Latent-Generator Alignment Principle

The key insight: when each latent token corresponds to a single kinematic unit, the flow-matching velocity field decomposes from **one heterogeneous multi-scale regression** (root trajectory in meters + per-joint rotations in radians, mixed scales and dynamics) into **K=23 coordinated homogeneous predictions**, each operating in a uniform 6D rotation space.

**Controlled ablation evidence**: Switching from 1D monolithic latent to 2D joint-factorized latent (same generator, same data, same training) improves FID by **2.5x** (0.137 → 0.055 on MotionHub). This isolates the structural benefit from any generator or data effect.

### 3.2 Differentiation from MoGenTS

- **MoGenTS** analyzed per-joint tokenization from the **encoding** perspective: better VQ codes per joint, improved codebook utilization and reconstruction quality.
- **PRISM** identifies a deeper **generation-side** mechanism: when each latent token is a kinematic unit, the flow-matching velocity field structurally decomposes into K homogeneous sub-problems. This latent-generator alignment is a quality lever largely orthogonal to generator scaling or data curation.

### 3.3 Per-Token Timestep with Physical Semantics

On the joint-factorized grid, per-token noise levels (from Diffusion Forcing) gain **physical semantics**: clean root tokens anchor the global trajectory while noisy limb tokens are progressively refined. This unifies text-to-motion, pose-conditioned generation, and segment-chained streaming in a single model without task-specific architectures.

### 3.4 Paper Narrative Arc

1. **Challenge**: Monolithic motion latents create a structural mismatch — the generator must learn one velocity field for heterogeneous quantities with different scales and dynamics.
2. **Principle**: Latent-generator alignment — match latent token structure to physical structure of the generation target.
3. **Instantiation**: Joint-factorized causal VAE + flow-matching DiT on 2D (time × joint) grid + per-token timestep conditioning with Self-Forcing.
4. **Evidence**: 2.5× FID improvement from structural change alone; SOTA on HumanML3D, MotionHub, BABEL; 50-scenario user study.
5. **Takeaway**: Latent-generator alignment is a quality lever largely orthogonal to model scaling — a finding relevant beyond motion generation.

---

## 四、Files Modified for TMM Resubmission

| File | Changes |
|------|---------|
| `main.tex` | Abstract rewritten: alignment principle framing, 2.5× FID claim, removed "focused application" language |
| `sec/sec_1_introduction.tex` | Three-paragraph rewrite: "latent-structure problem" challenge, "latent-generator alignment" insight, contribution list without self-deprecation |
| `sec/sec_2_related_work.tex` | MoGenTS differentiation: encoding-side vs generation-side; clarified Diffusion Forcing attribution |
| `sec/sec_3_method.tex` | "Novelty decomposition" → principle-level contribution; VAE motivation → alignment violation; DiT motivation → decomposed velocity field |
| `sec/sec_5_conclusion.tex` | Full rewrite: leads with alignment principle, quantitative 2.5× claim, removed "focused application" framing |

---

## References

- MoGenTS: NeurIPS 2024, per-joint discrete tokenization
- Diffusion Forcing: NeurIPS 2024, per-token noise levels
- Self-Forcing: autoregressive self-conditioning
- ANT: https://arxiv.org/abs/2506.02452
- Free-T2M: https://arxiv.org/abs/2501.18232
- LMR (Think Before You Move): https://arxiv.org/abs/2512.24100
- POMP: CVPR 2025
- UniMoGen: skeleton-agnostic motion generation
- HY-Motion: window attention + DPO/GRPO scaling
