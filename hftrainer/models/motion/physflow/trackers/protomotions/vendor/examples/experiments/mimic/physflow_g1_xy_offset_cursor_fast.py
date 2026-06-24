# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
"""Faster-adapting PhysFlow G1 tracker variant for online adversarial fine-tuning.

Same position-aware (XY/velocity) observations as ``physflow_g1_xy_offset_stable.py``
but with a less conservative learning rate so the policy can adapt to the KIMODO-G1
generated motion distribution within a practical number of PPO epochs. Used by the
PhysFlow online adversarial loop when warm-starting from the stable tracker.
"""

from pathlib import Path

_BASE_PATH = Path(__file__).with_name("physflow_g1_xy_offset.py")

_ns = {"__file__": str(_BASE_PATH)}
exec(compile(_BASE_PATH.read_text(), str(_BASE_PATH), "exec"), _ns)

terrain_config = _ns["terrain_config"]
scene_lib_config = _ns["scene_lib_config"]
motion_lib_config = _ns["motion_lib_config"]
env_config = _ns["env_config"]
configure_robot_and_simulator = _ns["configure_robot_and_simulator"]
apply_inference_overrides = _ns["apply_inference_overrides"]


def agent_config(robot_cfg, env_cfg, args):
    cfg = _ns["agent_config"](robot_cfg, env_cfg, args)

    # ~4x faster than the stable variant to adapt to the generated distribution
    # while still well below the released all-purpose tracker's update magnitude.
    cfg.model.actor_optimizer.lr = 2e-5
    cfg.model.critic_optimizer.lr = 1e-4
    cfg.model.discriminator_optimizer.lr = 1e-4
    cfg.model.disc_critic_optimizer.lr = 1e-4

    cfg.num_mini_epochs = 1
    cfg.e_clip = 0.1
    cfg.actor_clip_frac_threshold = 0.45
    cfg.gradient_clip_val = 20.0

    cfg.task_reward_w = 1.0
    cfg.amp_parameters.discriminator_reward_w = 0.5

    cfg.l2c2.lambda_l2c2 = 2.0
    return cfg
