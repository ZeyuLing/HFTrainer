# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
"""No-domain-randomization smoke variant of the PhysFlow G1 rehearsal tracker.

This keeps the same AMP/L2C2/replay objective as
``physflow_g1_released_rehearsal.py`` but disables simulator domain
randomization. It is intended for fast guarded-adversarial smoke runs where we
first need to verify that the training objective improves over the released
pretrained tracker before reintroducing robustness randomization.
"""

from examples.experiments.mimic import physflow_g1_released_rehearsal as base


terrain_config = base.terrain_config
scene_lib_config = base.scene_lib_config
motion_lib_config = base.motion_lib_config
env_config = base.env_config
agent_config = base.agent_config
apply_inference_overrides = base.apply_inference_overrides


def configure_robot_and_simulator(robot_cfg, simulator_cfg, args):
    """Apply the base robot setup, then remove simulator DR for smoke training."""

    base.configure_robot_and_simulator(robot_cfg, simulator_cfg, args)
    simulator_cfg.domain_randomization = None
