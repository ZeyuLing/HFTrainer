# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
"""Small-data jump overfit variant for the released G1 BeyondMimic tracker.

This intentionally stays close to ``physflow_g1_released_rehearsal.py`` so the
released g1-bones checkpoint can warm-start strictly. The default path keeps
actor observation dimensions unchanged; set ``PHYSFLOW_JUMP_ROOT_HEIGHT_OBS=1``
only when training from scratch or using a compatible checkpoint.
"""

import os
from pathlib import Path

_BASE_PATH = Path(__file__).with_name("physflow_g1_released_rehearsal.py")

_ns = {"__file__": str(_BASE_PATH)}
exec(compile(_BASE_PATH.read_text(), str(_BASE_PATH), "exec"), _ns)

terrain_config = _ns["terrain_config"]
scene_lib_config = _ns["scene_lib_config"]
motion_lib_config = _ns["motion_lib_config"]
apply_inference_overrides = _ns["apply_inference_overrides"]

_base_env_config = _ns["env_config"]
_base_agent_config = _ns["agent_config"]
_base_configure = _ns["configure_robot_and_simulator"]


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def env_config(robot_cfg, args):
    from protomotions.envs.component_factories import (
        anchor_pos_error_term_factory,
        contact_match_rew_factory,
        global_anchor_pos_rew_factory,
        mimic_target_poses_reduced_coords_factory,
        reduced_coords_obs_factory,
        relative_body_pos_error_term_factory,
    )

    cfg = _base_env_config(robot_cfg, args)

    # Let RSI sample airborne / takeoff frames instead of always biasing toward
    # the first standing frame.
    cfg.motion_manager.init_start_prob = float(
        os.environ.get("PHYSFLOW_JUMP_INIT_START_PROB", "0.0")
    )

    # Jump references need explicit root translation pressure; the released
    # reward is mostly local pose/velocity.
    cfg.reward_components["global_anchor_pos"] = global_anchor_pos_rew_factory(
        weight=float(os.environ.get("PHYSFLOW_JUMP_ANCHOR_POS_W", "1.0")),
        sigma=float(os.environ.get("PHYSFLOW_JUMP_ANCHOR_POS_SIGMA", "0.5")),
    )
    cfg.reward_components["contact_match"] = contact_match_rew_factory(
        weight=float(os.environ.get("PHYSFLOW_JUMP_CONTACT_W", "-0.1")),
    )

    cfg.termination_components["bad_anchor_pos"] = anchor_pos_error_term_factory(
        threshold=float(os.environ.get("PHYSFLOW_JUMP_ANCHOR_POS_TERM", "1.0")),
    )
    cfg.termination_components["bad_motion_body_pos"] = relative_body_pos_error_term_factory(
        threshold=float(os.environ.get("PHYSFLOW_JUMP_BODY_POS_TERM", "0.5")),
    )

    if _env_flag("PHYSFLOW_JUMP_ROOT_HEIGHT_OBS", default=False):
        cfg.observation_components["noisy_reduced_coords_obs"] = reduced_coords_obs_factory(
            use_noisy=True,
            root_height_obs=True,
            root_vel_obs=False,
        )
        cfg.observation_components["clean_reduced_coords_obs"] = reduced_coords_obs_factory(
            use_noisy=False,
            root_height_obs=True,
            root_vel_obs=False,
        )
        cfg.observation_components[
            "noisy_mimic_reduced_coords_target_poses"
        ] = mimic_target_poses_reduced_coords_factory(
            use_noisy=True,
            include_dof_vel=True,
            include_xy_offset=False,
        )
        cfg.observation_components[
            "clean_mimic_reduced_coords_target_poses"
        ] = mimic_target_poses_reduced_coords_factory(
            use_noisy=False,
            include_dof_vel=True,
            include_xy_offset=False,
        )

    return cfg


def agent_config(robot_cfg, env_cfg, args):
    from protomotions.envs.component_factories import anchor_pos_metric_factory

    cfg = _base_agent_config(robot_cfg, env_cfg, args)
    cfg.evaluator.evaluation_components["anchor_pos"] = anchor_pos_metric_factory(
        threshold=float(os.environ.get("PHYSFLOW_JUMP_ANCHOR_POS_METRIC", "0.75")),
    )
    return cfg


def configure_robot_and_simulator(robot_cfg, simulator_cfg, args):
    _base_configure(robot_cfg, simulator_cfg, args)

    # Overfit signal should not be washed out by domain randomization.
    simulator_cfg.domain_randomization = None
    # The jump-overfit run is a clean tracking fit, not a robustness/push test.
    # Disable projectile actors explicitly; IsaacGym's projectile tensor path is
    # unnecessary here and can trip CUDA illegal-access failures during startup.
    simulator_cfg.projectile.num_projectiles = int(
        os.environ.get("PHYSFLOW_JUMP_NUM_PROJECTILES", "0")
    )

    try:
        from protomotions.simulator.base_simulator.config import RobotNoiseConfig

        robot_cfg.reset_noise = RobotNoiseConfig(
            dof_pos_noise=0.02,
            root_pos_noise=[0.01, 0.01, 0.005],
            root_rot_noise=[0.02, 0.02, 0.05],
            root_vel_noise=[0.02, 0.02, 0.01],
            root_ang_vel_noise=[0.02, 0.02, 0.02],
        )
    except Exception:
        pass
