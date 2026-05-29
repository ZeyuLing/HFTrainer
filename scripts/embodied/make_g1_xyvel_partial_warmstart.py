#!/usr/bin/env python3
"""Create a partial warm-start checkpoint for the PhysFlow G1 XY/velocity tracker.

The released G1 deploy checkpoint observes:
  reduced_obs(64) + target(256) + previous_actions(29) = 349 actor inputs.

The PhysFlow position-aware tracker observes:
  reduced_obs(64) + target(276) + previous_actions(29) = 369 actor inputs,
where the extra 20 target dimensions are 4 future steps * (xy_offset[2] +
anchor_vel[3]).

This script copies all compatible weights and expands the actor's first
normalization/linear layer so the new channels start with zero influence.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


OLD_REDUCED = 64
OLD_TARGET = 256
NEW_TARGET = 276
PREV_ACTIONS = 29


def _expand_vector(old: torch.Tensor, fill: float) -> torch.Tensor:
    new = torch.full((OLD_REDUCED + NEW_TARGET + PREV_ACTIONS,), fill, dtype=old.dtype)
    new[:OLD_REDUCED] = old[:OLD_REDUCED]
    old_prev_start = OLD_REDUCED + OLD_TARGET
    new_prev_start = OLD_REDUCED + NEW_TARGET
    new[OLD_REDUCED : OLD_REDUCED + OLD_TARGET] = old[OLD_REDUCED:old_prev_start]
    new[new_prev_start:] = old[old_prev_start:]
    return new


def _expand_actor_first_weight(old: torch.Tensor) -> torch.Tensor:
    new = torch.zeros((old.shape[0], OLD_REDUCED + NEW_TARGET + PREV_ACTIONS), dtype=old.dtype)
    new[:, :OLD_REDUCED] = old[:, :OLD_REDUCED]
    old_prev_start = OLD_REDUCED + OLD_TARGET
    new_prev_start = OLD_REDUCED + NEW_TARGET
    new[:, OLD_REDUCED : OLD_REDUCED + OLD_TARGET] = old[:, OLD_REDUCED:old_prev_start]
    new[:, new_prev_start:] = old[:, old_prev_start:]
    return new


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    source = Path(args.source)
    output = Path(args.output)
    ckpt = torch.load(source, map_location="cpu", weights_only=False)
    model = ckpt["model"]

    model["_actor.mu.norm.running_obs_norm.mean"] = _expand_vector(
        model["_actor.mu.norm.running_obs_norm.mean"],
        fill=0.0,
    )
    model["_actor.mu.norm.running_obs_norm.var"] = _expand_vector(
        model["_actor.mu.norm.running_obs_norm.var"],
        fill=1.0,
    )
    model["_actor.mu.mlp.0.weight"] = _expand_actor_first_weight(
        model["_actor.mu.mlp.0.weight"]
    )

    ckpt["epoch"] = 0
    ckpt["step_count"] = 0
    ckpt["run_start_time"] = 0
    ckpt["best_evaluated_score"] = None
    ckpt["skip_optimizer_load"] = True
    for key in [
        "actor_optimizer",
        "critic_optimizer",
        "discriminator_optimizer",
        "disc_critic_optimizer",
    ]:
        ckpt.pop(key, None)

    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, output)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
