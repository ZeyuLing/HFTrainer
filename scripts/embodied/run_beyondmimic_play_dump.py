#!/usr/bin/env python3
"""Run a trained BeyondMimic policy and dump executed G1 qpos frames."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BM_ROOT = PROJECT_ROOT / "ref_repo/BeyondMimic"


pre_parser = argparse.ArgumentParser(add_help=False)
pre_parser.add_argument("--bm-root", type=Path, default=DEFAULT_BM_ROOT)
pre_args, _ = pre_parser.parse_known_args()
BM_ROOT = pre_args.bm_root.resolve()
os.chdir(BM_ROOT)
sys.path.insert(0, str(BM_ROOT / "scripts/rsl_rl"))
sys.path.insert(0, str(BM_ROOT / "source/whole_body_tracking"))

from isaaclab.app import AppLauncher  # noqa: E402

import cli_args  # noqa: E402


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--bm-root", type=Path, default=DEFAULT_BM_ROOT)
parser.add_argument("--dump-npz", type=Path, required=True)
parser.add_argument("--dump-json", type=Path, default=None)
parser.add_argument("--rollout-steps", type=int, default=0)
parser.add_argument("--disable-eval-randomization", action="store_true", default=True)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--motion_file", type=str, default=None, help="Path to the motion file.")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from isaaclab.envs import (  # noqa: E402
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import get_checkpoint_path  # noqa: E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

import whole_body_tracking.tasks  # noqa: F401,E402


def _disable_if_present(obj: object, *names: str) -> None:
    for name in names:
        if hasattr(obj, name):
            setattr(obj, name, None)


def _zero_ranges(command_cfg: object) -> None:
    command_cfg.pose_range = {k: (0.0, 0.0) for k in ("x", "y", "z", "roll", "pitch", "yaw")}
    command_cfg.velocity_range = {k: (0.0, 0.0) for k in ("x", "y", "z", "roll", "pitch", "yaw")}
    command_cfg.joint_position_range = (0.0, 0.0)


def _tensor_to_numpy(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()


def _robot_root_state(robot) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    data = robot.data
    if hasattr(data, "root_pos_w") and hasattr(data, "root_quat_w"):
        root_pos = data.root_pos_w
        root_quat = data.root_quat_w
    else:
        root_state = data.root_state_w
        root_pos = root_state[:, :3]
        root_quat = root_state[:, 3:7]
    if hasattr(data, "root_lin_vel_w") and hasattr(data, "root_ang_vel_w"):
        root_lin = data.root_lin_vel_w
        root_ang = data.root_ang_vel_w
    else:
        root_state = data.root_state_w
        root_lin = root_state[:, 7:10]
        root_ang = root_state[:, 10:13]
    return root_pos, root_quat, root_lin, root_ang


def _current_qpos(robot, env_id: int = 0) -> np.ndarray:
    root_pos, root_quat, _, _ = _robot_root_state(robot)
    qpos = torch.cat([root_pos[env_id], root_quat[env_id], robot.data.joint_pos[env_id]], dim=-1)
    return _tensor_to_numpy(qpos).astype(np.float32)


def _set_reference_state(env_unwrapped, env_id: int = 0) -> None:
    command = env_unwrapped.command_manager.get_term("motion")
    robot = env_unwrapped.scene["robot"]
    command.time_steps[env_id] = 0
    env_ids = torch.tensor([env_id], dtype=torch.long, device=command.device)
    root_pos = command.motion.body_pos_w[0, 0].unsqueeze(0) + env_unwrapped.scene.env_origins[env_ids]
    root_quat = command.motion.body_quat_w[0, 0].unsqueeze(0)
    root_lin = command.motion.body_lin_vel_w[0, 0].unsqueeze(0)
    root_ang = command.motion.body_ang_vel_w[0, 0].unsqueeze(0)
    joint_pos = command.motion.joint_pos[0].unsqueeze(0)
    joint_vel = command.motion.joint_vel[0].unsqueeze(0)
    robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
    robot.write_root_state_to_sim(torch.cat([root_pos, root_quat, root_lin, root_ang], dim=-1), env_ids=env_ids)
    env_unwrapped.scene.write_data_to_sim()


def _bool_done(value) -> bool:
    if isinstance(value, torch.Tensor):
        return bool(value.detach().cpu().reshape(-1)[0].item())
    arr = np.asarray(value).reshape(-1)
    return bool(arr[0]) if arr.size else False


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg):
    agent_cfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else 1
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    if args_cli.motion_file is None:
        raise ValueError("--motion_file is required")
    env_cfg.commands.motion.motion_file = args_cli.motion_file

    if args_cli.disable_eval_randomization:
        _zero_ranges(env_cfg.commands.motion)
        if hasattr(env_cfg.observations, "policy"):
            env_cfg.observations.policy.enable_corruption = False
        _disable_if_present(env_cfg.events, "push_robot", "physics_material", "add_joint_default_pos", "base_com")

    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    print(f"[beyondmimic-dump] loading checkpoint: {resume_path}")
    print(f"[beyondmimic-dump] motion_file: {args_cli.motion_file}")

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = RslRlVecEnvWrapper(env)

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    obs, _ = env.get_observations()
    _set_reference_state(env.unwrapped, 0)
    obs, _ = env.get_observations()

    command = env.unwrapped.command_manager.get_term("motion")
    robot = env.unwrapped.scene["robot"]
    control_dt = float(env.unwrapped.cfg.decimation * env.unwrapped.cfg.sim.dt)
    max_steps = int(args_cli.rollout_steps) if args_cli.rollout_steps > 0 else int(command.motion.time_step_total - 1)
    max_steps = max(1, min(max_steps, int(command.motion.time_step_total - 1)))

    qpos_frames = [_current_qpos(robot)]
    done_step: int | None = None
    with torch.inference_mode():
        for step in range(1, max_steps + 1):
            actions = policy(obs)
            if actions.dim() == 1:
                actions = actions.unsqueeze(0)
            obs, _, dones, extras = env.step(actions)
            qpos_frames.append(_current_qpos(robot))
            terminated = _bool_done(dones)
            if not terminated and hasattr(env.unwrapped, "termination_manager"):
                terminated = _bool_done(env.unwrapped.termination_manager.terminated)
            if terminated:
                done_step = step
                break

    qpos = np.stack(qpos_frames, axis=0).astype(np.float32)
    qvel = np.zeros((qpos.shape[0], qpos.shape[1] - 1), dtype=np.float32)
    if qpos.shape[0] > 1:
        qvel[1:, :3] = (qpos[1:, :3] - qpos[:-1, :3]) / control_dt
        qvel[1:, 6:] = (qpos[1:, 7:] - qpos[:-1, 7:]) / control_dt

    args_cli.dump_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args_cli.dump_npz,
        qpos=qpos,
        qvel=qvel,
        frequency=np.float32(1.0 / control_dt),
        joint_names=np.array(["root", *list(robot.joint_names)], dtype=object),
        source_motion=np.array(str(args_cli.motion_file)),
        checkpoint=np.array(str(resume_path)),
        done_step=np.array(-1 if done_step is None else done_step, dtype=np.int32),
    )
    summary = {
        "dump_npz": str(args_cli.dump_npz),
        "checkpoint": str(resume_path),
        "motion_file": str(args_cli.motion_file),
        "frames": int(qpos.shape[0]),
        "fps": float(1.0 / control_dt),
        "planned_steps": int(max_steps),
        "done_step": done_step,
        "completed": done_step is None and int(qpos.shape[0]) >= max_steps,
    }
    if args_cli.dump_json is not None:
        args_cli.dump_json.parent.mkdir(parents=True, exist_ok=True)
        args_cli.dump_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
