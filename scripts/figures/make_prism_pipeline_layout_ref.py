#!/usr/bin/env python3
"""Draw a deterministic layout reference for the PRISM pipeline figure.

This is not the final raster artwork.  It fixes composition, arrow semantics,
module grouping, and color hierarchy before image generation is used for the
final visual polish.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import patches  # noqa: E402


OUT_DIR = Path("papers/PRISM_TMM2026/figures")
W, H = 18.0, 9.6


COL = {
    "ink": "#2B2F36",
    "muted": "#667085",
    "panel": "#F7F8FA",
    "panel2": "#FBFCFD",
    "blue": "#315C9E",
    "blue_fill": "#EAF1FB",
    "teal": "#158F86",
    "teal_fill": "#DDF3F0",
    "coral": "#DF6D55",
    "coral_fill": "#FBE7E1",
    "root": "#476EA8",
    "orient": "#4D9A6B",
    "joint": "#D8A43A",
    "purple": "#6F4BB3",
    "purple_fill": "#F2ECFA",
    "orange": "#D8891C",
    "orange_fill": "#FFF2DE",
    "gray_fill": "#EEF1F4",
}


def setup_ax():
    fig, ax = plt.subplots(figsize=(W, H), dpi=160)
    ax.set_xlim(0, 1800)
    ax.set_ylim(0, 960)
    ax.axis("off")
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    return fig, ax


def panel(ax, x, y, w, h, label=None, edge=None, face=None, lw=1.4, r=10):
    patch = patches.FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.012,rounding_size={r}",
        linewidth=lw,
        edgecolor=edge or COL["ink"],
        facecolor=face or COL["panel2"],
    )
    ax.add_patch(patch)
    if label:
        ax.text(
            x + w / 2,
            y + h - 24,
            label,
            ha="center",
            va="center",
            fontsize=12,
            fontweight="bold",
            color=edge or COL["ink"],
        )
    return patch


def arrow(ax, x1, y1, x2, y2, color=None, lw=2.0, dashed=False, label=None, ms=12):
    style = dict(
        arrowstyle="-|>",
        color=color or COL["ink"],
        lw=lw,
        mutation_scale=ms,
        shrinkA=0,
        shrinkB=0,
    )
    if dashed:
        style["linestyle"] = (0, (4, 3))
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=style)
    if label:
        ax.text(
            (x1 + x2) / 2,
            (y1 + y2) / 2 + 14,
            label,
            ha="center",
            va="bottom",
            fontsize=8.5,
            color=color or COL["ink"],
        )


def small_arrow(ax, x1, y1, x2, y2, color, dashed=False):
    arrow(ax, x1, y1, x2, y2, color=color, lw=1.35, dashed=dashed, ms=9)


def grid(ax, x, y, cols, rows, cw, ch, colors, edge="white", lw=1.1):
    for r in range(rows):
        for c in range(cols):
            color = colors[r][c] if isinstance(colors[r], list) else colors[r]
            ax.add_patch(
                patches.Rectangle(
                    (x + c * cw, y + (rows - 1 - r) * ch),
                    cw,
                    ch,
                    facecolor=color,
                    edgecolor=edge,
                    linewidth=lw,
                )
            )
    ax.add_patch(
        patches.Rectangle(
            (x, y),
            cols * cw,
            rows * ch,
            facecolor="none",
            edgecolor=COL["ink"],
            linewidth=0.8,
            alpha=0.55,
        )
    )


def smpl_mesh_slot(ax, x, y, w, h, label):
    panel(ax, x, y, w, h, edge="#A8B0BA", face="#FFFFFF", lw=1.0, r=8)
    cx = x + w / 2
    top = y + h - 44
    # A neutral, mesh-like placeholder that marks where a real SMPL render goes.
    ax.add_patch(patches.Circle((cx, top - 18), 15, facecolor="#C9CED6", edgecolor="#8C96A3", lw=0.8))
    torso = patches.Polygon(
        [
            (cx - 28, top - 45),
            (cx + 28, top - 45),
            (cx + 22, top - 112),
            (cx + 10, top - 145),
            (cx - 10, top - 145),
            (cx - 22, top - 112),
        ],
        closed=True,
        facecolor="#D8DCE2",
        edgecolor="#8C96A3",
        lw=0.8,
    )
    ax.add_patch(torso)
    limbs = [
        ((cx - 28, top - 50), (cx - 58, top - 105), (cx - 45, top - 112), (cx - 17, top - 62)),
        ((cx + 28, top - 50), (cx + 57, top - 100), (cx + 45, top - 110), (cx + 17, top - 62)),
        ((cx - 11, top - 145), (cx - 42, top - 213), (cx - 27, top - 218), (cx + 0, top - 152)),
        ((cx + 11, top - 145), (cx + 40, top - 210), (cx + 26, top - 218), (cx + 0, top - 152)),
    ]
    for pts in limbs:
        ax.add_patch(
            patches.Polygon(pts, closed=True, facecolor="#D8DCE2", edgecolor="#8C96A3", lw=0.8)
        )
    for yy in [top - 62, top - 84, top - 108, top - 132]:
        ax.plot([cx - 25, cx + 25], [yy, yy + 4], color="#ADB5C0", lw=0.6)
    for dx in [-18, 0, 18]:
        ax.plot([cx + dx, cx], [top - 45, top - 145], color="#B7BEC8", lw=0.55)
    ax.text(cx, y + 46, label, ha="center", va="center", fontsize=10.5, fontweight="bold", color=COL["ink"])
    ax.text(cx, y + 20, "real SMPL render slot", ha="center", va="center", fontsize=6.5, color=COL["muted"])


def draw_motion_grid(ax, x, y, w, h):
    panel(ax, x, y, w, h, edge=COL["ink"], face="#FFFFFF", lw=1.0, r=8)
    ax.text(x + w / 2, y + h - 28, "Kinematic-unit grid X", ha="center", va="center", fontsize=11, fontweight="bold", color=COL["ink"])
    gx, gy = x + 50, y + 38
    cw, ch = 24, 20
    cols = 8
    rows = 8
    colors = [COL["root"]] * 2 + [COL["orient"]] * 2 + [COL["joint"]] * 4
    grid(ax, gx, gy, cols, rows, cw, ch, colors)
    ax.text(gx - 18, gy + rows * ch / 2, "kinematic unit", rotation=90, ha="center", va="center", fontsize=8.5)
    ax.text(gx + cols * cw / 2, gy + rows * ch + 21, "time", ha="center", va="center", fontsize=8.5)
    arrow(ax, gx + 42, gy + rows * ch + 11, gx + cols * cw - 10, gy + rows * ch + 11, lw=1.0, ms=7)
    ax.text(x + 17, gy + 7.0 * ch, "root", ha="left", va="center", fontsize=8.5, color=COL["root"], fontweight="bold")
    ax.text(x + 17, gy + 5.0 * ch, "orient", ha="left", va="center", fontsize=8.5, color=COL["orient"], fontweight="bold")
    ax.text(x + 17, gy + 2.2 * ch, "joints", ha="left", va="center", fontsize=8.5, color=COL["joint"], fontweight="bold")


def draw_vae(ax, x, y, w, h):
    panel(ax, x, y, w, h, "Causal Motion VAE", edge=COL["blue"], face="#FFFFFF", lw=1.2, r=10)
    ax.text(x + 80, y + h - 64, "Encoder", ha="center", fontsize=9, fontweight="bold", color=COL["blue"])
    ax.text(x + w - 80, y + h - 64, "Decoder", ha="center", fontsize=9, fontweight="bold", color=COL["blue"])
    for bx in [x + 36, x + w - 136]:
        panel(ax, bx, y + 106, 100, 60, edge="#8FB0D8", face=COL["blue_fill"], lw=0.9, r=8)
        ax.text(bx + 50, y + 136, "causal\nconv", ha="center", va="center", fontsize=8)
        panel(ax, bx, y + 38, 100, 56, edge="#8FB0D8", face=COL["blue_fill"], lw=0.9, r=8)
        ax.text(bx + 50, y + 66, "joint\nattention", ha="center", va="center", fontsize=8)
        arrow(ax, bx + 50, y + 106, bx + 50, y + 94, lw=1.2, ms=7)
    panel(ax, x + w / 2 - 22, y + 87, 44, 44, "Z", edge=COL["ink"], face=COL["gray_fill"], lw=1.0, r=8)
    small_arrow(ax, x + 136, y + 66, x + w / 2 - 22, y + 109, COL["ink"])
    small_arrow(ax, x + w / 2 + 22, y + 109, x + w - 136, y + 66, COL["ink"])


def draw_latent_grid(ax, x, y, w, h):
    panel(ax, x, y, w, h, "Latent grid Z", edge=COL["ink"], face="#FFFFFF", lw=1.0, r=8)
    gx, gy = x + 38, y + 42
    rows, cols, cw, ch = 8, 7, 20, 22
    colors = []
    for _ in range(rows):
        colors.append([COL["teal"] if c < 3 else COL["coral"] for c in range(cols)])
    grid(ax, gx, gy, cols, rows, cw, ch, colors)
    ax.text(gx - 16, gy + rows * ch / 2, "joint", rotation=90, ha="center", va="center", fontsize=8)
    ax.text(gx + cols * cw / 2, gy + rows * ch + 18, "time", ha="center", va="center", fontsize=8)
    ax.add_patch(patches.Rectangle((x + 36, y + 28), 16, 16, facecolor=COL["teal"], edgecolor=COL["ink"], lw=0.5))
    ax.text(x + 58, y + 36, "clean context", va="center", fontsize=7.5)
    ax.add_patch(patches.Rectangle((x + 36, y + 8), 16, 16, facecolor=COL["coral"], edgecolor=COL["ink"], lw=0.5))
    ax.text(x + 58, y + 16, "noisy target", va="center", fontsize=7.5)


def block(ax, x, y, w, h, text, edge=COL["blue"], face=COL["blue_fill"], fs=8.5):
    panel(ax, x, y, w, h, edge=edge, face=face, lw=0.9, r=6)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=COL["ink"])


def draw_kuflowt(ax, x, y, w, h):
    panel(ax, x, y, w, h, "KU-FlowT denoising loop", edge=COL["blue"], face="#FFFFFF", lw=1.6, r=10)
    ax.text(x + w - 32, y + h - 38, "x L", ha="center", va="center", fontsize=11, color=COL["ink"])
    sx, sy = x + 155, y + h - 94
    bw, bh, gap = 135, 42, 20
    names = ["AdaLN", "Temporal Attn", "Joint Attn", "MLP", "velocity v_theta", "Euler update"]
    ys = []
    for i, name in enumerate(names):
        by = sy - i * (bh + gap)
        ys.append(by)
        block(ax, sx, by, bw, bh, name, fs=8)
        if i:
            arrow(ax, sx + bw / 2, by + bh + 2, sx + bw / 2, by + bh, lw=1.1, ms=6)
    # Condition boxes.
    cx = x + 18
    block(ax, cx, y + h - 120, 110, 62, "text\ncondition c", edge="#BBA6CB", face="#F2ECFA", fs=8)
    block(ax, cx, y + h - 210, 110, 62, "per-token\ntimestep tau", edge="#D2B79F", face="#FCF0E7", fs=8)
    small_arrow(ax, cx + 110, y + h - 89, sx, ys[0] + bh / 2, COL["blue"])
    small_arrow(ax, cx + 110, y + h - 179, sx, ys[0] + bh / 2 - 4, COL["blue"])
    # RoPE hooks.
    for by, color in [(ys[1], COL["blue"]), (ys[2], COL["purple"])]:
        circ = patches.Circle((sx + bw + 42, by + bh / 2), 27, facecolor="#FFFFFF", edgecolor=color, lw=1.2)
        ax.add_patch(circ)
        ax.text(sx + bw + 42, by + bh / 2, "RoPE\nQ/K", ha="center", va="center", fontsize=7.2, color=COL["ink"])
        small_arrow(ax, sx + bw, by + bh / 2, sx + bw + 15, by + bh / 2, color)
    # Internal loop.
    ax.plot(
        [sx + bw + 78, sx + bw + 78, sx + bw + 42],
        [ys[5] + bh / 2, ys[1] + bh / 2, ys[1] + bh / 2],
        color=COL["ink"],
        lw=1.0,
    )
    small_arrow(ax, sx + bw + 78, ys[5] + bh / 2, sx + bw + 5, ys[5] + bh / 2, COL["ink"])


def draw_kt_inset(ax, x, y, w, h):
    panel(ax, x, y, w, h, "KT-RoPE", edge=COL["purple"], face=COL["purple_fill"], lw=1.2, r=8)
    ax.text(x + w / 2, y + h - 42, "skeleton coordinate p_j", ha="center", fontsize=8.5, color=COL["purple"], fontweight="bold")
    # Tree.
    nodes = [(x + 50, y + 68), (x + 50, y + 95), (x + 28, y + 42), (x + 72, y + 42), (x + 18, y + 18), (x + 82, y + 18)]
    edges = [(0, 1), (0, 2), (0, 3), (2, 4), (3, 5)]
    for a, b in edges:
        ax.plot([nodes[a][0], nodes[b][0]], [nodes[a][1], nodes[b][1]], color=COL["ink"], lw=0.9)
    for p in nodes:
        ax.add_patch(patches.Circle(p, 4.5, facecolor="#D9DCE3", edgecolor=COL["ink"], lw=0.7))
    ax.text(x + 50, y + 4, "SMPL tree", ha="center", fontsize=6.6)
    small_arrow(ax, x + 100, y + 58, x + 128, y + 58, COL["ink"])
    # Coordinate column.
    ax.text(x + 155, y + 68, "[p1\np2\n...\npj]", ha="center", va="center", fontsize=8)
    ax.text(x + 155, y + 4, "spectral\nscalar", ha="center", fontsize=6.6)
    small_arrow(ax, x + 182, y + 58, x + 210, y + 58, COL["ink"])
    # Rotary circle.
    ax.add_patch(patches.Circle((x + 250, y + 58), 31, facecolor="#FFFFFF", edgecolor=COL["ink"], lw=0.8))
    ax.plot([x + 250, x + 250], [y + 28, y + 88], color=COL["muted"], lw=0.8)
    ax.plot([x + 220, x + 280], [y + 58, y + 58], color=COL["muted"], lw=0.8)
    arrow(ax, x + 250, y + 58, x + 273, y + 82, color=COL["purple"], lw=1.4, ms=8)
    ax.text(x + 250, y + 4, "joint-axis\nRoPE phase", ha="center", fontsize=6.6)


def draw_kafs(ax, x, y, w, h):
    panel(ax, x, y, w, h, "KAFS", edge=COL["orange"], face=COL["orange_fill"], lw=1.2, r=8)
    ax.text(x + w / 2, y + h - 42, "per-joint sigma_j schedule", ha="center", fontsize=8.5, color=COL["orange"], fontweight="bold")
    # Depth skeleton.
    pts = [(x + 64, y + 78), (x + 64, y + 104), (x + 42, y + 62), (x + 86, y + 62), (x + 28, y + 31), (x + 100, y + 31)]
    colors = ["#335CBB", "#2F9E74", "#E3A72F", "#E3A72F", "#D9443F", "#D9443F"]
    for a, b in [(0, 1), (0, 2), (0, 3), (2, 4), (3, 5)]:
        ax.plot([pts[a][0], pts[b][0]], [pts[a][1], pts[b][1]], color=COL["ink"], lw=0.9)
    for p, c in zip(pts, colors):
        ax.add_patch(patches.Circle(p, 5.2, facecolor=c, edgecolor=COL["ink"], lw=0.65))
    ax.text(x + 64, y + 9, "kinematic depth", ha="center", fontsize=6.8)
    # Schedule plot.
    px, py, pw, ph = x + 145, y + 28, 138, 86
    ax.plot([px, px], [py, py + ph], color=COL["ink"], lw=0.8)
    ax.plot([px, px + pw], [py, py], color=COL["ink"], lw=0.8)
    ax.text(px - 10, py + ph, "sigma", ha="right", va="center", fontsize=6.6)
    ax.text(px + pw, py - 14, "step k", ha="right", va="center", fontsize=6.6)
    xs = [px + i * pw / 80 for i in range(81)]
    curve_cols = ["#D9443F", "#E3A72F", "#2F9E74", "#335CBB"]
    for j, c in enumerate(curve_cols):
        ys = [py + ph * ((1 - i / 80) ** (0.75 + 0.28 * j)) for i in range(81)]
        ax.plot(xs, ys, color=c, lw=1.4)


def draw_interface(ax, x, y, w, h):
    panel(ax, x, y, w, h, "Clean-context / noisy-target interface", edge="#9AA4B2", face="#FFFFFF", lw=1.0, r=8)
    names = ["T2M", "TP2M", "AR streaming"]
    starts = [x + 44, x + 282, x + 530]
    for name, sx in zip(names, starts):
        ax.text(sx + 78, y + h - 56, name, ha="center", va="center", fontsize=10, fontweight="bold")
    # T2M
    grid(ax, starts[0], y + 75, 6, 3, 24, 20, [[COL["coral"]] * 6 for _ in range(3)])
    ax.text(starts[0] + 72, y + 50, "all noisy target", ha="center", fontsize=7.5, color=COL["coral"])
    # TP2M
    colors = [[COL["teal"]] * 3 + [COL["coral"]] * 4 for _ in range(3)]
    grid(ax, starts[1], y + 75, 7, 3, 24, 20, colors)
    ax.text(starts[1] + 36, y + 50, "clean prefix", ha="center", fontsize=7.5, color=COL["teal"])
    ax.text(starts[1] + 118, y + 50, "noisy target", ha="center", fontsize=7.5, color=COL["coral"])
    # AR
    colors = [[COL["teal"]] * 3 + [COL["coral"]] * 4 for _ in range(3)]
    grid(ax, starts[2], y + 75, 7, 3, 24, 20, colors)
    ax.text(starts[2] + 36, y + 50, "clean tail", ha="center", fontsize=7.5, color=COL["teal"])
    ax.text(starts[2] + 118, y + 50, "next segment", ha="center", fontsize=7.5, color=COL["coral"])
    # Legend
    ax.add_patch(patches.Rectangle((x + 248, y + 18), 17, 17, facecolor=COL["teal"], edgecolor=COL["ink"], lw=0.5))
    ax.text(x + 270, y + 26, "clean context", va="center", fontsize=7.5)
    ax.add_patch(patches.Rectangle((x + 392, y + 18), 17, 17, facecolor=COL["coral"], edgecolor=COL["ink"], lw=0.5))
    ax.text(x + 414, y + 26, "noisy target", va="center", fontsize=7.5)


def draw_self_forcing(ax, x, y, w, h):
    panel(ax, x, y, w, h, "Self-Forcing training", edge="#9AA4B2", face="#FFFFFF", lw=1.0, r=8)
    ax.text(x + 82, y + h - 74, "generated\nsegment", ha="center", va="center", fontsize=8)
    grid(ax, x + 42, y + 93, 4, 2, 19, 20, [["#D8DCE2"] * 4 for _ in range(2)], edge="#FFFFFF")
    arrow(ax, x + 125, y + 113, x + 168, y + 113, lw=1.3, ms=8)
    panel(ax, x + 178, y + 75, 82, 76, edge=COL["blue"], face=COL["blue_fill"], lw=1.0, r=8)
    ax.text(x + 219, y + 113, "VAE\nencode", ha="center", va="center", fontsize=9, color=COL["blue"], fontweight="bold")
    arrow(ax, x + 260, y + 113, x + 304, y + 113, lw=1.3, ms=8)
    grid(ax, x + 315, y + 93, 4, 2, 19, 20, [[COL["teal"]] * 4 for _ in range(2)], edge="#FFFFFF")
    ax.text(x + 352, y + h - 74, "clean tail\ncontext", ha="center", va="center", fontsize=8)
    # Dashed training loop.
    ax.annotate(
        "",
        xy=(x + 331, y + 83),
        xytext=(x + 70, y + 83),
        arrowprops=dict(
            arrowstyle="-|>",
            color=COL["blue"],
            lw=1.3,
            linestyle=(0, (4, 3)),
            connectionstyle="arc3,rad=-0.28",
            mutation_scale=9,
        ),
    )
    ax.text(x + w / 2, y + 34, "generated tail becomes context", ha="center", fontsize=7.5, color=COL["blue"], fontweight="bold")


def draw_line_legend(ax, x, y):
    ax.text(x, y + 72, "Arrow semantics", fontsize=9.5, fontweight="bold", color=COL["ink"])
    arrow(ax, x, y + 50, x + 58, y + 50, lw=1.7, ms=8)
    ax.text(x + 68, y + 50, "data flow", va="center", fontsize=7.5)
    arrow(ax, x, y + 29, x + 58, y + 29, color=COL["orange"], lw=1.3, ms=8)
    ax.text(x + 68, y + 29, "sampling control", va="center", fontsize=7.5)
    arrow(ax, x, y + 8, x + 58, y + 8, color=COL["blue"], lw=1.3, dashed=True, ms=8)
    ax.text(x + 68, y + 8, "training loop", va="center", fontsize=7.5)


def main():
    fig, ax = setup_ax()
    # Top row.
    smpl_mesh_slot(ax, 35, 610, 130, 280, "SMPL\nmotion")
    arrow(ax, 170, 750, 205, 750)
    draw_motion_grid(ax, 210, 610, 250, 280)
    arrow(ax, 465, 750, 500, 750)
    draw_vae(ax, 505, 610, 300, 280)
    arrow(ax, 810, 750, 845, 750)
    draw_latent_grid(ax, 850, 610, 180, 280)
    arrow(ax, 1035, 750, 1070, 750)
    draw_kuflowt(ax, 1075, 510, 430, 420)
    arrow(ax, 1510, 750, 1560, 750, label="decode Z -> SMPL")
    smpl_mesh_slot(ax, 1568, 610, 160, 280, "SMPL\nmotion output")
    # Internal callouts.
    draw_kt_inset(ax, 1172, 338, 314, 146)
    draw_kafs(ax, 1510, 338, 255, 210)
    small_arrow(ax, 1438, 780, 1368, 438, COL["purple"], dashed=True)
    small_arrow(ax, 1510, 454, 1468, 604, COL["orange"])
    small_arrow(ax, 1510, 474, 1422, 576, COL["orange"])
    # Bottom row.
    draw_interface(ax, 35, 305, 690, 235)
    draw_self_forcing(ax, 760, 305, 380, 235)
    # Region labels are subtle and help imagegen preserve compact layout.
    ax.text(250, 932, "Representation and VAE", fontsize=9.5, color=COL["muted"], fontweight="bold")
    ax.text(1135, 944, "Generator core", fontsize=9.5, color=COL["muted"], fontweight="bold")
    ax.text(1520, 565, "Inference control", fontsize=9.5, color=COL["muted"], fontweight="bold")
    ax.set_ylim(255, 960)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    svg = OUT_DIR / "fig_pipeline_layout_ref_0611.svg"
    png = OUT_DIR / "fig_pipeline_layout_ref_0611.png"
    fig.savefig(svg, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(png, bbox_inches="tight", pad_inches=0.04)
    print(svg.resolve())
    print(png.resolve())


if __name__ == "__main__":
    main()
