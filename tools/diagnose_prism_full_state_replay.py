"""Replay PRISM loss after loading a full Accelerator/FSDP checkpoint."""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import hftrainer  # noqa: F401
from mmengine.config import Config

from hftrainer.runner.accelerate_runner import AccelerateRunner
from diagnose_prism_fsdp_update import (
    fixed_timestep_suite,
    raw_first_batch,
    replay_loss,
    seeded_fixed_timestep_loss,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--frozen-module-checkpoint", default=None)
    parser.add_argument("--skip-config-load-from", action="store_true")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def load_checkpoint_mmap(path: str) -> Dict[str, Any]:
    if os.path.isdir(path):
        path = os.path.join(path, "model.pt")
    try:
        return torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    except TypeError:
        return torch.load(path, map_location="cpu", weights_only=False)


def filter_frozen_latent_state(state: Dict[str, Any]) -> Dict[str, Any]:
    return {
        key: state[key]
        for key in ("vae", "smpl_pose_processor", "__bundle_params__")
        if key in state
    }


def load_frozen_modules_one_rank_at_a_time(
    runner: AccelerateRunner,
    checkpoint: Optional[str],
) -> None:
    if not checkpoint:
        return
    is_distributed = torch.distributed.is_initialized()
    num_processes = runner.accelerator.num_processes if is_distributed else 1
    my_rank = runner.accelerator.process_index if is_distributed else 0
    for loading_rank in range(num_processes):
        if my_rank == loading_rank:
            state = filter_frozen_latent_state(load_checkpoint_mmap(checkpoint))
            runner.bundle.load_state_dict_selective(state, strict=False)
            del state
        if is_distributed:
            torch.distributed.barrier()


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    if args.skip_config_load_from:
        cfg.load_from = None
    runner = AccelerateRunner.from_cfg(cfg)

    runner.accelerator.wait_for_everyone()
    runner.accelerator.load_state(args.checkpoint)
    runner.accelerator.wait_for_everyone()
    load_frozen_modules_one_rank_at_a_time(runner, args.frozen_module_checkpoint)
    runner.accelerator.wait_for_everyone()

    data_iter = iter(runner.train_dataloader)
    fixed_batch = next(data_iter)
    raw_eval_batch = raw_first_batch(
        runner.cfg,
        fixed_batch["motion"].shape[0],
        runner.accelerator.device,
    )

    result = {
        "config": args.config,
        "checkpoint": args.checkpoint,
        "frozen_module_checkpoint": args.frozen_module_checkpoint,
        "skip_config_load_from": bool(args.skip_config_load_from),
        "rank": runner.accelerator.process_index,
        "num_processes": runner.accelerator.num_processes,
        "same_seed": replay_loss(runner, fixed_batch, args.seed),
        "fixed_timestep": fixed_timestep_suite(runner, fixed_batch),
        "raw_first_batch_fixed_timestep": fixed_timestep_suite(runner, raw_eval_batch),
        "raw_first_batch_seeded_999": seeded_fixed_timestep_loss(
            runner,
            raw_eval_batch,
            999,
            args.seed,
        ),
    }

    runner.accelerator.wait_for_everyone()
    if runner.accelerator.is_main_process:
        text = json.dumps(result, indent=2, ensure_ascii=False)
        print(text)
        if args.output:
            os.makedirs(os.path.dirname(args.output), exist_ok=True)
            with open(args.output, "w", encoding="utf-8") as handle:
                handle.write(text)


if __name__ == "__main__":
    main()
