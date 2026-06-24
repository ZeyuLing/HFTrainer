# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
"""PhysFlow G1 tracker with position-aware observations and anchor position reward.

The stable XY/velocity tracker can see reference displacement, but the base
BeyondMimic deploy reward is still mostly local pose/velocity tracking. This
variant adds an explicit global anchor-position reward so locomotion references
such as "walk forward" train against translation error directly.
"""

from pathlib import Path

_BASE_PATH = Path(__file__).with_name("physflow_g1_xy_offset_stable.py")

_ns = {"__file__": str(_BASE_PATH)}
exec(compile(_BASE_PATH.read_text(), str(_BASE_PATH), "exec"), _ns)

terrain_config = _ns["terrain_config"]
scene_lib_config = _ns["scene_lib_config"]
motion_lib_config = _ns["motion_lib_config"]
configure_robot_and_simulator = _ns["configure_robot_and_simulator"]
apply_inference_overrides = _ns["apply_inference_overrides"]


def env_config(robot_cfg, args):
    from protomotions.envs.component_factories import (
        anchor_pos_error_term_factory,
        global_anchor_pos_rew_factory,
    )

    cfg = _ns["env_config"](robot_cfg, args)
    cfg.reward_components["global_anchor_pos"] = global_anchor_pos_rew_factory(
        weight=1.0,
        sigma=0.5,
    )
    cfg.termination_components["bad_anchor_pos"] = anchor_pos_error_term_factory(
        threshold=1.0,
    )
    return cfg


def agent_config(robot_cfg, env_cfg, args):
    from protomotions.envs.component_factories import anchor_pos_metric_factory

    cfg = _ns["agent_config"](robot_cfg, env_cfg, args)
    cfg.evaluator.evaluation_components["anchor_pos"] = anchor_pos_metric_factory(
        threshold=0.75,
    )
    return cfg
