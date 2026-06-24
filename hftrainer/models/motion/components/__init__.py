"""Shared neural network components used by multiple motion model-zoo methods.

Sub-packages:
  - wan_blocks/       Wan-style NN building blocks (causal conv, enc/dec, attention, etc.)
  - utils/            Geometry helpers, smoothnet, tensor utilities
  - hunyuan_motion/   Shared HunyuanMotion backbone/text/loss compatibility API

Motion-domain utilities such as body models, motion processors, skeleton math,
and retargeting live under ``hftrainer.motion``. The legacy
``components.body_models``, ``components.motion_processor`` and
``components.retarget`` modules are compatibility wrappers only.
"""
