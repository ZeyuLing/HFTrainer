#!/usr/bin/env python3
"""Overlay reference-reconstruction curves of multiple tracker fine-tune runs.

Draws gt_error / max_joint_error / relative_body_pos / success_rate vs epoch for
each experiment, with the epoch-0 (warmstarted-released) baseline drawn as a
dashed horizontal line. Lower error = better reconstruction.
"""
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer")
RES = ROOT / "ref_repo" / "ProtoMotions" / "results"
OUT = ROOT / "output/physflow_kimodo_g1/reconstruction_curve_compare.png"

RUNS = [
    ("v1 cold-opt task0.5/amp2.0", "physflow_g1_released_rehearsal_v1", "tab:blue"),
    ("v2 warm-opt task1.0/amp0.25", "physflow_g1_released_rehearsal_v2_taskheavy", "tab:red"),
]
PANELS = [
    ("eval/gt_error/mean", "gt_error (body pos) [m]  lower=better"),
    ("eval/max_joint_error/mean", "max_joint_error [rad]  lower=better"),
    ("eval/relative_body_pos/mean", "relative_body_pos [m]  lower=better"),
    ("eval/success_rate", "success_rate  higher=better"),
]


def series(exp, tag):
    files = sorted((RES / exp).rglob("events.out.tfevents*"))
    if not files:
        return [], []
    ea = EventAccumulator(str(files[-1].parent), size_guidance={"scalars": 0})
    ea.Reload()
    if tag not in ea.Tags().get("scalars", []):
        return [], []
    ev = ea.Scalars(tag)
    return [e.step for e in ev], [e.value for e in ev]


def main():
    plt.figure(figsize=(13, 9))
    for i, (tag, title) in enumerate(PANELS, 1):
        ax = plt.subplot(2, 2, i)
        for label, exp, color in RUNS:
            st, vs = series(exp, tag)
            if not st:
                continue
            ax.plot(st, vs, "-o", ms=4, color=color, label=label)
            # baseline = first eval point (epoch-0 warmstarted released)
            ax.axhline(vs[0], color=color, ls="--", lw=1, alpha=0.5)
        ax.set_title(title)
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
        if i == 1:
            ax.legend(fontsize=8)
        if tag == "eval/success_rate":
            ax.set_ylim(0.88, 1.005)
    plt.suptitle(
        "PhysFlow G1 tracker — reference reconstruction vs training epoch\n"
        "(dashed = each run's epoch-0 warmstarted-released baseline)"
    )
    plt.tight_layout()
    plt.savefig(OUT, dpi=120)
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
