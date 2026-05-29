#!/usr/bin/env python3
"""Minimal IsaacGym create_sim probe for remote environment debugging."""

from __future__ import annotations

import argparse
import faulthandler

faulthandler.enable()

from isaacgym import gymapi  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--compute-device-id", type=int, default=0)
    parser.add_argument("--graphics-device-id", type=int, default=-1)
    parser.add_argument("--gpu-pipeline", type=int, choices=[0, 1], default=1)
    parser.add_argument("--physx-gpu", type=int, choices=[0, 1], default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    gym = gymapi.acquire_gym()
    sim_params = gymapi.SimParams()
    sim_params.physx.use_gpu = bool(args.physx_gpu)
    sim_params.use_gpu_pipeline = bool(args.gpu_pipeline)
    print(
        "create_sim",
        {
            "compute_device_id": args.compute_device_id,
            "graphics_device_id": args.graphics_device_id,
            "gpu_pipeline": bool(args.gpu_pipeline),
            "physx_gpu": bool(args.physx_gpu),
        },
        flush=True,
    )
    sim = gym.create_sim(
        args.compute_device_id,
        args.graphics_device_id,
        gymapi.SIM_PHYSX,
        sim_params,
    )
    print("created", sim, flush=True)
    gym.destroy_sim(sim)
    print("destroyed", flush=True)


if __name__ == "__main__":
    main()
