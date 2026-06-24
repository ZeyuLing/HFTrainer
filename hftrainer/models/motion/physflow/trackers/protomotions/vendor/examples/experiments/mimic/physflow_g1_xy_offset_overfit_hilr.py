# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
"""Higher-actor-lr overfit variant of the position-aware (xy_offset) G1 tracker.

Identical to ``physflow_g1_xy_offset_overfit.py`` (same 369-d position-aware
arch, DR + reset-noise disabled) but with a 4x larger actor learning rate so the
newly-exposed xy_offset / anchor-velocity channels (zero-weighted at warm start
by design) learn translation-following faster. Used as a parallel comparison
against the conservative (lr=5e-6) overfit run to see if global-translation
reconstruction (eval/gt_error) can be driven down materially faster without
destabilizing the strong released pose-tracking prior.
"""

from pathlib import Path

_BASE_PATH = Path(__file__).with_name("physflow_g1_xy_offset_overfit.py")

_ns = {"__file__": str(_BASE_PATH)}
exec(compile(_BASE_PATH.read_text(), str(_BASE_PATH), "exec"), _ns)

terrain_config = _ns["terrain_config"]
scene_lib_config = _ns["scene_lib_config"]
motion_lib_config = _ns["motion_lib_config"]
env_config = _ns["env_config"]
configure_robot_and_simulator = _ns["configure_robot_and_simulator"]
apply_inference_overrides = _ns["apply_inference_overrides"]

_base_agent_config = _ns["agent_config"]


def agent_config(robot_cfg, env_cfg, args):
    cfg = _base_agent_config(robot_cfg, env_cfg, args)
    # 4x the conservative actor lr to accelerate learning of the translation
    # (xy_offset/anchor-vel) channels. Critic gets a matching modest bump so the
    # value function keeps up with the faster-moving policy.
    cfg.model.actor_optimizer.lr = 2e-5
    cfg.model.critic_optimizer.lr = 1e-4
    return cfg
