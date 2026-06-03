#!/usr/bin/env python3
"""Create a warm-start checkpoint from the RELEASED BeyondMimic G1 tracker.

The released last.ckpt has epoch=45660. ProtoMotions' training loop runs
`while current_epoch < max_epochs`, and load_parameters() restores
current_epoch from the checkpoint. Warm-starting directly would therefore
immediately exceed any sane budget and train for 0 epochs.

This script copies the released MODEL weights verbatim (same architecture,
include_xy_offset=False -> guaranteed key match, NO structural change that
caused the previous catastrophic-forgetting failure) but resets the training
bookkeeping (epoch / step_count / best_score) to 0/None so a fresh fine-tune
run can proceed. Optimizer states are dropped (load_parameters ignores them;
optimizers are re-created fresh anyway).
"""
from pathlib import Path

import torch

PROTO = Path("/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer/ref_repo/ProtoMotions")
SRC = PROTO / "data/pretrained_models/motion_tracker/g1-bones-deploy/last.ckpt"
OUT_DIR = Path(
    "/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer/output/physflow_kimodo_g1/checkpoints"
)
OUT = OUT_DIR / "g1_released_warmstart_epoch0.ckpt"


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"loading released ckpt: {SRC}")
    ck = torch.load(SRC, map_location="cpu", weights_only=False)
    print(f"  source epoch={ck.get('epoch')} step_count={ck.get('step_count')}")

    # Keep ALL released keys (model + optimizers + adv_ema) so the AMP/PPO
    # load_parameters path doesn't KeyError, but reset training bookkeeping and
    # set skip_optimizer_load=True so optimizers start fresh for the fine-tune.
    new_ck = dict(ck)
    new_ck["epoch"] = 0
    new_ck["step_count"] = 0
    new_ck["run_start_time"] = None
    new_ck["best_evaluated_score"] = None
    new_ck["skip_optimizer_load"] = True
    print(f"saving warm-start ckpt (epoch=0, skip_optimizer_load=True, {len(new_ck['model'])} tensors): {OUT}")
    torch.save(new_ck, OUT)
    print("DONE")


if __name__ == "__main__":
    main()
