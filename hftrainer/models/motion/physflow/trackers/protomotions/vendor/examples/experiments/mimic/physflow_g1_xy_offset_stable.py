# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
"""Stable PhysFlow G1 tracker with global displacement observations.

This variant keeps the position-aware observation changes from
``physflow_g1_xy_offset.py`` but makes the first adaptation stage more
conservative. The released G1 deploy policy was trained without XY-offset
targets, so opening those channels from a partial warm start can otherwise
produce large PPO updates and early falls.
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

    # First make the new XY/velocity channels useful without destabilizing the
    # strong released pose-tracking prior.
    cfg.model.actor_optimizer.lr = 5e-6
    cfg.model.critic_optimizer.lr = 5e-5
    cfg.model.discriminator_optimizer.lr = 5e-5
    cfg.model.disc_critic_optimizer.lr = 5e-5

    cfg.num_mini_epochs = 1
    cfg.e_clip = 0.1
    cfg.actor_clip_frac_threshold = 0.45
    cfg.gradient_clip_val = 20.0

    # Prioritize tracking while bootstrapping translation following. AMP is kept
    # on, but at a lower weight than the released all-purpose tracker.
    cfg.task_reward_w = 1.0
    cfg.amp_parameters.discriminator_reward_w = 0.5

    # Keep noisy/clean consistency strong for the newly exposed target channels.
    cfg.l2c2.lambda_l2c2 = 2.0
    return cfg
