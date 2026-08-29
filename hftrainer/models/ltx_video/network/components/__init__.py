# MODIFIED BY HFTRAINER: relocated into repository-owned layers and adapted for local execution.
# See hftrainer/models/ltx_video/UPSTREAM.md and LICENSE.ltx-2.x.
"""
Diffusion pipeline components.
Submodules:
    diffusion_steps - Diffusion stepping algorithms (EulerDiffusionStep, EulerAncestralDiffusionStep)
    guiders         - Guidance strategies (CFGGuider, STGGuider, APG variants)
    noisers         - Noise samplers (GaussianNoiser)
    patchifiers     - Latent patchification (VideoLatentPatchifier, AudioPatchifier)
    protocols       - Protocol definitions (Patchifier, etc.)
    schedulers      - Sigma schedulers (LTX2Scheduler, LinearQuadraticScheduler)
"""
