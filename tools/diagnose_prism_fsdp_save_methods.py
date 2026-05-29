"""Compare FSDP model-only checkpoint extraction methods for PRISM."""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import hftrainer  # noqa: F401
from mmengine.config import Config

from hftrainer.runner.accelerate_runner import AccelerateRunner
from diagnose_prism_fsdp_update import (
    fixed_timestep_suite,
    grad_norm,
    raw_first_batch,
    reduce_scalar,
    replay_loss,
    scalar,
    seeded_fixed_timestep_loss,
    set_seed,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--steps", type=int, default=7)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def bundle_params(bundle) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for name, param in bundle.named_parameters(recurse=False):
        out[name] = param.detach().cpu().clone()
    for name, buf in bundle.named_buffers(recurse=False):
        out[name] = buf.detach().cpu().clone()
    return out


def nested_state(runner: AccelerateRunner, transformer_state: Dict[str, torch.Tensor]):
    state = {"__hftrainer_meta__": runner.bundle.checkpoint_metadata()}
    state["transformer"] = {
        key: value.detach().cpu().clone()
        for key, value in transformer_state.items()
        if isinstance(value, torch.Tensor)
    }
    params = bundle_params(runner.bundle)
    if params:
        state["__bundle_params__"] = params
    return state


def save_rank0(path: str, state: Dict[str, Any]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)


def collect_fsdp_state_dict(runner: AccelerateRunner):
    module = runner.bundle.transformer
    state = runner.accelerator.get_state_dict(module)
    if not runner.accelerator.is_main_process:
        return None
    return nested_state(runner, state)


def collect_summon_unwrapped_state_dict(runner: AccelerateRunner):
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

    module = runner.bundle.transformer
    unwrapped = runner.accelerator.unwrap_model(module)
    with FSDP.summon_full_params(
        module,
        recurse=True,
        writeback=False,
        rank0_only=True,
        offload_to_cpu=True,
    ):
        if not runner.accelerator.is_main_process:
            return None
        state = unwrapped.state_dict()
        return nested_state(runner, state)


def collect_dcp_model_state_dict(runner: AccelerateRunner):
    from torch.distributed.checkpoint.state_dict import (
        StateDictOptions,
        get_model_state_dict,
    )

    module = runner.bundle.transformer
    options = StateDictOptions(
        full_state_dict=True,
        cpu_offload=True,
        broadcast_from_rank0=False,
    )
    state = get_model_state_dict(module, options=options)
    if not runner.accelerator.is_main_process:
        return None
    return nested_state(runner, state)


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    runner = AccelerateRunner.from_cfg(cfg)

    trainable_modules = [
        getattr(runner.bundle, name)
        for name in runner.bundle._trainable_modules
        if isinstance(getattr(runner.bundle, name), torch.nn.Module)
    ]

    data_iter = iter(runner.train_dataloader)
    fixed_batch = next(data_iter)
    raw_eval_batch = raw_first_batch(
        runner.cfg,
        fixed_batch["motion"].shape[0],
        runner.accelerator.device,
    )

    pre = replay_loss(runner, fixed_batch, args.seed)
    pre_fixed = fixed_timestep_suite(runner, fixed_batch)
    pre_raw_fixed = fixed_timestep_suite(runner, raw_eval_batch)
    pre_raw_seeded_999 = seeded_fixed_timestep_loss(
        runner,
        raw_eval_batch,
        999,
        args.seed,
    )

    rows = []
    for step in range(args.steps):
        batch = fixed_batch if step == 0 else next(data_iter)
        set_seed(args.seed + step)
        runner.bundle.train()
        runner.trainer.train()
        with runner.accelerator.accumulate(*trainable_modules):
            out = runner.trainer.train_step(batch)
            loss = out["loss"]
            runner.accelerator.backward(loss)
            runner._sync_orphan_param_grads()
            gnorm = grad_norm(runner)
            if runner.max_grad_norm is not None:
                runner.accelerator.clip_grad_norm_(
                    list(runner.bundle.trainable_parameters()),
                    runner.max_grad_norm,
                )
            for opt in runner.optimizers.values():
                opt.step()
                opt.zero_grad()
            for sched in runner.lr_schedulers.values():
                sched.step()
        rows.append(
            {
                "step": step,
                "train_loss": reduce_scalar(runner, scalar(out["loss"])),
                "train_loss_transl": reduce_scalar(runner, scalar(out["loss_transl"])),
                "train_loss_rot": reduce_scalar(runner, scalar(out["loss_rot"])),
                "grad_norm": gnorm,
            }
        )

    post_same = replay_loss(runner, fixed_batch, args.seed)
    post_next = replay_loss(runner, fixed_batch, args.seed + 1)
    post_fixed = fixed_timestep_suite(runner, fixed_batch)
    post_raw_fixed = fixed_timestep_suite(runner, raw_eval_batch)
    post_raw_seeded_999 = seeded_fixed_timestep_loss(
        runner,
        raw_eval_batch,
        999,
        args.seed,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    methods = {}

    fsdp_state = collect_fsdp_state_dict(runner)
    if runner.accelerator.is_main_process:
        fsdp_path = os.path.join(args.output_dir, "model_accelerator_get_state_dict.pt")
        save_rank0(fsdp_path, fsdp_state)
        methods["accelerator_get_state_dict"] = fsdp_path
    runner.accelerator.wait_for_everyone()

    summon_state = collect_summon_unwrapped_state_dict(runner)
    if runner.accelerator.is_main_process:
        summon_path = os.path.join(args.output_dir, "model_summon_unwrapped.pt")
        save_rank0(summon_path, summon_state)
        methods["summon_unwrapped"] = summon_path
    runner.accelerator.wait_for_everyone()

    dcp_state = collect_dcp_model_state_dict(runner)
    if runner.accelerator.is_main_process:
        dcp_path = os.path.join(args.output_dir, "model_dcp_get_model_state_dict.pt")
        save_rank0(dcp_path, dcp_state)
        methods["dcp_get_model_state_dict"] = dcp_path
    runner.accelerator.wait_for_everyone()

    runner_state = runner._state_dict_to_save()
    if runner.accelerator.is_main_process:
        runner_path = os.path.join(args.output_dir, "model_runner_state_dict_to_save.pt")
        save_rank0(runner_path, runner_state)
        methods["runner_state_dict_to_save"] = runner_path
    runner.accelerator.wait_for_everyone()

    result = {
        "config": args.config,
        "steps": args.steps,
        "rank": runner.accelerator.process_index,
        "num_processes": runner.accelerator.num_processes,
        "pre_same_seed": pre,
        "pre_fixed_timestep": pre_fixed,
        "pre_raw_first_batch_fixed_timestep": pre_raw_fixed,
        "pre_raw_first_batch_seeded_999": pre_raw_seeded_999,
        "step_rows": rows,
        "post_same_seed": post_same,
        "post_next_seed": post_next,
        "post_fixed_timestep": post_fixed,
        "post_raw_first_batch_fixed_timestep": post_raw_fixed,
        "post_raw_first_batch_seeded_999": post_raw_seeded_999,
        "methods": methods,
    }
    runner.accelerator.wait_for_everyone()
    if runner.accelerator.is_main_process:
        with open(os.path.join(args.output_dir, "summary.json"), "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, ensure_ascii=False)
        print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
