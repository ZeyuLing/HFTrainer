#!/usr/bin/env python3
"""Create a CORRECT partial warm-start checkpoint for the PhysFlow G1 XY/velocity tracker.

Background / bug being fixed
----------------------------
The released G1 deploy actor observes (in this concat order):

    reduced_obs(64) + target(256) + previous_actions(29) = 349 inputs.

The position-aware tracker turns on ``include_xy_offset`` and
``include_anchor_vel`` in ``mimic_target_poses_reduced_coords_factory``.
That grows the target block from 256 -> 276.

CRITICAL: the target block is NOT a single contiguous chunk that can be
extended at its tail. It is built per future step and then flattened
(``obs.view(num_envs, -1)``). With future_steps = [1, 2, 4, 8] there are
4 steps, and each step's feature vector is:

    released : [rot6(6), dof_vel(29), dof_pos(29)]                = 64
    posaware : [rot6(6), dof_vel(29), dof_pos(29), xy(2), vel(3)] = 69

So the 5 new channels are inserted at the END OF EACH STEP block, i.e.
they are interleaved through the 276-dim target at offsets 64, 133, 202,
271 -- not appended at offset 256.

The previous script (make_g1_xyvel_partial_warmstart.py) wrongly assumed
``new_target = [old_256 | new_20]``. That only aligns future step 0; steps
1/2/3 are shifted by 5/10/15 channels and the zero-fill overwrites real
step-3 pose features. This silently destroys the pretrained prior for the
[2,4,8] horizon.

This script remaps the actor first linear layer and the obs-norm mean/var
PER STEP, copying each step's 64 released channels into the correct slot and
zero-initialising the 5 new channels so that, at init, the network behaves
IDENTICALLY to the released policy (new channels have zero weight).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


REDUCED = 64
PREV_ACTIONS = 29
N_STEPS = 4
PER_STEP_OLD = 64          # rot6 + dof_vel(29) + dof_pos(29)
EXTRA_PER_STEP = 5         # xy(2) + anchor_vel(3)
PER_STEP_NEW = PER_STEP_OLD + EXTRA_PER_STEP  # 69

OLD_TARGET = N_STEPS * PER_STEP_OLD   # 256
NEW_TARGET = N_STEPS * PER_STEP_NEW   # 276
OLD_TOTAL = REDUCED + OLD_TARGET + PREV_ACTIONS  # 349
NEW_TOTAL = REDUCED + NEW_TARGET + PREV_ACTIONS  # 369


def _remap_last_dim(old: torch.Tensor, fill: float) -> torch.Tensor:
    """Remap a tensor whose LAST dim is the 349-d actor input to 369-d.

    Works for 1-D (mean/var, shape [349]) and 2-D weight (shape [H, 349]).
    """
    assert old.shape[-1] == OLD_TOTAL, (old.shape, OLD_TOTAL)
    new_shape = list(old.shape)
    new_shape[-1] = NEW_TOTAL
    new = torch.full(new_shape, fill, dtype=old.dtype)

    # 1) reduced_obs block (unchanged)
    new[..., :REDUCED] = old[..., :REDUCED]

    # 2) interleaved target block: copy each step's 64 channels, leave the
    #    trailing 5 channels of each new step block at `fill`.
    for step in range(N_STEPS):
        old_off = REDUCED + step * PER_STEP_OLD
        new_off = REDUCED + step * PER_STEP_NEW
        new[..., new_off:new_off + PER_STEP_OLD] = old[..., old_off:old_off + PER_STEP_OLD]
        # new[..., new_off+PER_STEP_OLD : new_off+PER_STEP_NEW] stays = fill

    # 3) previous_actions block (unchanged, shifted by +20)
    old_prev = REDUCED + OLD_TARGET
    new_prev = REDUCED + NEW_TARGET
    new[..., new_prev:new_prev + PREV_ACTIONS] = old[..., old_prev:old_prev + PREV_ACTIONS]
    return new


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--keep-optimizer",
        action="store_true",
        help="Keep optimizer states (warm optimizer). Default drops them and sets "
             "skip_optimizer_load=True so the first layer can adapt cleanly.",
    )
    args = parser.parse_args()

    source = Path(args.source)
    output = Path(args.output)
    ckpt = torch.load(source, map_location="cpu", weights_only=False)
    model = ckpt["model"]

    w = model["_actor.mu.mlp.0.weight"]
    assert w.shape[1] == OLD_TOTAL, f"unexpected actor first layer width {w.shape}"

    model["_actor.mu.norm.running_obs_norm.mean"] = _remap_last_dim(
        model["_actor.mu.norm.running_obs_norm.mean"], fill=0.0
    )
    model["_actor.mu.norm.running_obs_norm.var"] = _remap_last_dim(
        model["_actor.mu.norm.running_obs_norm.var"], fill=1.0
    )
    model["_actor.mu.mlp.0.weight"] = _remap_last_dim(w, fill=0.0)

    ckpt["epoch"] = 0
    ckpt["step_count"] = 0
    ckpt["run_start_time"] = 0
    ckpt["best_evaluated_score"] = None

    if not args.keep_optimizer:
        ckpt["skip_optimizer_load"] = True
        for key in [
            "actor_optimizer",
            "critic_optimizer",
            "discriminator_optimizer",
            "disc_critic_optimizer",
        ]:
            ckpt.pop(key, None)
    else:
        ckpt["skip_optimizer_load"] = False

    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, output)
    print(f"Wrote {output}")
    print(f"  actor first layer: {tuple(w.shape)} -> "
          f"{tuple(model['_actor.mu.mlp.0.weight'].shape)}")


if __name__ == "__main__":
    main()
