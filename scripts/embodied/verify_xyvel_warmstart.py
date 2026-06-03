#!/usr/bin/env python3
"""Verify the xy/vel warmstart remap preserves the released actor at init.

We feed a random released-layout input (349) and the SAME values placed into
the correct interleaved 369-layout (with arbitrary garbage in the 5 new
channels per step). A correct remap must produce an identical first-layer
pre-activation, because the new channels are zero-weighted.

We run both the FIXED remap and the OLD (buggy, append-at-end) remap to show
the difference.
"""
import sys
import torch

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import make_g1_xyvel_partial_warmstart_FIXED as fx

REDUCED, PREV = fx.REDUCED, fx.PREV_ACTIONS
N, PO, PN, EX = fx.N_STEPS, fx.PER_STEP_OLD, fx.PER_STEP_NEW, fx.EXTRA_PER_STEP
OLD_T, NEW_T = fx.OLD_TARGET, fx.NEW_TARGET
OLD_TOT, NEW_TOT = fx.OLD_TOTAL, fx.NEW_TOTAL


def normalize(x, mean, var, clamp=5.0):
    return torch.clamp((x - mean) / torch.sqrt(var + 1e-5), -clamp, clamp)


def old_remap_vec(old, fill):
    new = torch.full((NEW_TOT,), fill, dtype=old.dtype)
    new[:REDUCED] = old[:REDUCED]
    old_prev = REDUCED + OLD_T
    new_prev = REDUCED + NEW_T
    new[REDUCED:REDUCED + OLD_T] = old[REDUCED:old_prev]
    new[new_prev:] = old[old_prev:]
    return new


def old_remap_w(old, fill):
    new = torch.full((old.shape[0], NEW_TOT), fill, dtype=old.dtype)
    new[:, :REDUCED] = old[:, :REDUCED]
    old_prev = REDUCED + OLD_T
    new_prev = REDUCED + NEW_T
    new[:, REDUCED:REDUCED + OLD_T] = old[:, REDUCED:old_prev]
    new[:, new_prev:] = old[:, old_prev:]
    return new


def build_new_input_interleaved(x_old):
    """Place 349-layout values into the correct 369 interleaved slots; put
    random garbage into the 5 new channels of each step."""
    x_new = torch.randn(NEW_TOT)
    x_new[:REDUCED] = x_old[:REDUCED]
    for s in range(N):
        oo = REDUCED + s * PO
        no = REDUCED + s * PN
        x_new[no:no + PO] = x_old[oo:oo + PO]
        # x_new[no+PO:no+PN] stays random garbage (the new xy/vel channels)
    op, npr = REDUCED + OLD_T, REDUCED + NEW_T
    x_new[npr:npr + PREV] = x_old[op:op + PREV]
    return x_new


def main():
    ck = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
    m = ck["model"]
    W = m["_actor.mu.mlp.0.weight"].double()
    mean = m["_actor.mu.norm.running_obs_norm.mean"].double()
    var = m["_actor.mu.norm.running_obs_norm.var"].double()
    assert W.shape[1] == OLD_TOT

    torch.manual_seed(0)
    x_old = torch.randn(OLD_TOT).double()
    x_new = build_new_input_interleaved(x_old).double()

    out_old = W @ normalize(x_old, mean, var)

    # FIXED remap
    Wf = fx._remap_last_dim(W, 0.0)
    mf = fx._remap_last_dim(mean, 0.0)
    vf = fx._remap_last_dim(var, 1.0)
    out_fixed = Wf @ normalize(x_new, mf, vf)

    # OLD buggy remap
    Wb = old_remap_w(W, 0.0)
    mb = old_remap_vec(mean, 0.0)
    vb = old_remap_vec(var, 1.0)
    out_bug = Wb @ normalize(x_new, mb, vb)

    print(f"max|out_fixed - out_old| = {(out_fixed - out_old).abs().max().item():.3e}")
    print(f"max|out_bug   - out_old| = {(out_bug   - out_old).abs().max().item():.3e}")
    print(f"mean|out_bug  - out_old| = {(out_bug   - out_old).abs().mean().item():.3e}")
    ok = (out_fixed - out_old).abs().max().item() < 1e-9
    print("FIXED remap preserves released actor:", "PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
