# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
"""Overfit-debug variant of the position-aware (xy_offset) G1 tracker.

Goal: a clean small-data overfit sanity test. Same 369-d position-aware
architecture as ``physflow_g1_xy_offset_stable.py`` (so it matches the FIXED
369-d warm-start checkpoint), but with domain randomization and reset noise
DISABLED so the in-training evaluator produces a trustworthy reconstruction
curve. If the (correctly warm-started) tracker cannot drive reconstruction
error toward ~0 on a tiny motion set under these clean conditions, the bug is
elsewhere; if it can, the pipeline is sound.
"""

from pathlib import Path

_BASE_PATH = Path(__file__).with_name("physflow_g1_xy_offset_stable.py")

_ns = {"__file__": str(_BASE_PATH)}
exec(compile(_BASE_PATH.read_text(), str(_BASE_PATH), "exec"), _ns)

terrain_config = _ns["terrain_config"]
scene_lib_config = _ns["scene_lib_config"]
motion_lib_config = _ns["motion_lib_config"]
env_config = _ns["env_config"]
agent_config = _ns["agent_config"]
apply_inference_overrides = _ns["apply_inference_overrides"]

_base_configure = _ns["configure_robot_and_simulator"]


def configure_robot_and_simulator(robot_cfg, simulator_cfg, args):
    # Apply the released robot/sim configuration first (sets DR + reset noise).
    _base_configure(robot_cfg, simulator_cfg, args)

    # --- Overfit: remove all domain randomization for a clean signal ---
    simulator_cfg.domain_randomization = None

    # Shrink reset noise so episodes start near the reference pose. A little
    # noise is kept to avoid degenerate identical inits across envs.
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
