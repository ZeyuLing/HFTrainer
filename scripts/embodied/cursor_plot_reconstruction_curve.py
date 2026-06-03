#!/usr/bin/env python3
"""Plot the REFERENCE-RECONSTRUCTION curve from a ProtoMotions tracker run.

Per the user's directive, training effectiveness for the position-aware G1
tracker is judged by whether the reference motion is being RECONSTRUCTED, i.e.
the in-sim tracking-error metrics should go DOWN over epochs (not by survival /
episode_length, which previously gave a false sense of progress).

Reads the TensorBoard event file(s) under
  ref_repo/ProtoMotions/results/<experiment>/
and reports / plots the key reconstruction metrics:
  eval/gt_error/mean          global translation (body position) error  [m]
  eval/max_joint_error/mean   worst joint angle error                    [rad]
  eval/relative_body_pos/mean relative body position error               [m]
  eval/gr_error/mean          global rotation error                      [rad]
  eval/success_rate           fraction of motions tracked w/o failure
For contrast we also pull a survival/return signal if present.

Usage:
  python3 scripts/embodied/cursor_plot_reconstruction_curve.py [experiment_name]
"""
import sys
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer")
EXP = sys.argv[1] if len(sys.argv) > 1 else "physflow_g1_released_rehearsal_v1"
RESULTS = ROOT / "ref_repo" / "ProtoMotions" / "results" / EXP
OUT_PNG = ROOT / "output/physflow_kimodo_g1" / f"reconstruction_curve_{EXP}.png"

RECON_TAGS = {
    "eval/gt_error/mean": ("gt_error (body pos) [m]", "lower=better"),
    "eval/max_joint_error/mean": ("max_joint_error [rad]", "lower=better"),
    "eval/relative_body_pos/mean": ("relative_body_pos [m]", "lower=better"),
    "eval/gr_error/mean": ("gr_error (rot) [rad]", "lower=better"),
}
SUCCESS_TAG = "eval/success_rate"


def load_events():
    files = sorted(RESULTS.rglob("events.out.tfevents*"))
    if not files:
        print(f"NO EVENT FILES YET under {RESULTS}")
        return None
    ea = EventAccumulator(str(files[-1].parent), size_guidance={"scalars": 0})
    ea.Reload()
    return ea


def series(ea, tag):
    if tag not in ea.Tags().get("scalars", []):
        return [], []
    ev = ea.Scalars(tag)
    return [e.step for e in ev], [e.value for e in ev]


def main():
    ea = load_events()
    if ea is None:
        return
    avail = ea.Tags().get("scalars", [])

    # Print a compact table of the reconstruction metrics over steps/epochs.
    print(f"\n=== Reference-reconstruction metrics: {EXP} ===")
    cols = list(RECON_TAGS.keys()) + [SUCCESS_TAG]
    cols = [c for c in cols if c in avail]
    if not cols:
        print("(no eval metrics logged yet — initial eval still running)")
        print("available scalar tags sample:", avail[:20])
        return

    # Build a step-indexed table from the first reconstruction tag present
    base_steps, _ = series(ea, cols[0])
    data = {c: dict(zip(*series(ea, c))) for c in cols}
    header = "step      " + "  ".join(f"{c.split('/')[1][:12]:>12s}" for c in cols)
    print(header)
    for s in base_steps:
        row = f"{s:<9d} " + "  ".join(
            (f"{data[c][s]:>12.4f}" if s in data[c] else f"{'-':>12s}") for c in cols
        )
        print(row)

    # Trend summary (first vs last)
    print("\n=== Trend (first -> last eval) ===")
    for c in cols:
        st, vs = series(ea, c)
        if len(vs) >= 1:
            arrow = "" if len(vs) < 2 else (" DOWN(better)" if vs[-1] < vs[0] else " UP")
            tail = f" -> {vs[-1]:.4f}{arrow}" if len(vs) >= 2 else ""
            print(f"  {c:32s}: {vs[0]:.4f}{tail}  (n={len(vs)})")

    # Plot
    plt.figure(figsize=(11, 7))
    for i, (tag, (label, _)) in enumerate(RECON_TAGS.items(), 1):
        st, vs = series(ea, tag)
        if not st:
            continue
        ax = plt.subplot(2, 3, i)
        ax.plot(st, vs, "-o", ms=3)
        ax.set_title(label)
        ax.set_xlabel("step")
        ax.grid(alpha=0.3)
    st, vs = series(ea, SUCCESS_TAG)
    if st:
        ax = plt.subplot(2, 3, 5)
        ax.plot(st, vs, "-o", ms=3, color="green")
        ax.set_title("success_rate (higher=better)")
        ax.set_xlabel("step")
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.3)
    plt.suptitle(f"Reference reconstruction over training: {EXP}")
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=110)
    print(f"\nsaved plot: {OUT_PNG}")


if __name__ == "__main__":
    main()
