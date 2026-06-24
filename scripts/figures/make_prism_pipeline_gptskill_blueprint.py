#!/usr/bin/env python3
"""Draw a controlled PRISM pipeline figure blueprint.

This figure follows the GPT-image skill's research-figure guidance, but keeps
arrow routing and scientific relationships deterministic.  It is intended as a
conference-paper-quality SVG/PNG baseline and, if needed, as a reference for
image-model polishing.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.image as mpimg  # noqa: E402
from matplotlib import patches  # noqa: E402


OUT_DIR = Path("papers/PRISM_TMM2026/figures")
SMPL_CROP = OUT_DIR / "fig_pipeline_smpl_crop_0611.png"
W, H = 20.48, 11.52

COL = {
    "ink": "#222B36",
    "muted": "#667085",
    "line": "#354052",
    "panel": "#FFFFFF",
    "soft": "#F6F8FB",
    "soft2": "#F9FAFC",
    "blue": "#1F5CA8",
    "blue2": "#EAF2FF",
    "teal": "#0E8E82",
    "teal2": "#DDF3F0",
    "coral": "#E2644A",
    "coral2": "#FFE7DF",
    "purple": "#7057B8",
    "purple2": "#F1ECFA",
    "amber": "#D98A18",
    "amber2": "#FFF1DA",
    "root": "#3D73C8",
    "orient": "#3FA36B",
    "joint": "#F0A51A",
    "gray": "#E7EBF0",
    "gray2": "#C7CED8",
}


def setup_ax():
    fig, ax = plt.subplots(figsize=(W, H), dpi=160)
    ax.set_xlim(0, 2048)
    ax.set_ylim(0, 1152)
    ax.axis("off")
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    return fig, ax


def card(ax, x, y, w, h, title=None, edge=None, face=None, lw=1.35, r=13):
    p = patches.FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.012,rounding_size={r}",
        linewidth=lw,
        edgecolor=edge or COL["line"],
        facecolor=face or COL["panel"],
    )
    ax.add_patch(p)
    if title:
        ax.text(
            x + w / 2,
            y + h - 28,
            title,
            ha="center",
            va="center",
            fontsize=13,
            fontweight="bold",
            color=edge or COL["ink"],
        )
    return p


def label(ax, x, y, s, size=10, weight="normal", color=None, ha="center", va="center", rot=0):
    ax.text(
        x,
        y,
        s,
        ha=ha,
        va=va,
        fontsize=size,
        fontweight=weight,
        color=color or COL["ink"],
        rotation=rot,
        family="DejaVu Sans",
    )


def rect(ax, x, y, w, h, text="", edge=None, face=None, lw=1.0, r=8, size=9, color=None, weight="normal"):
    card(ax, x, y, w, h, None, edge=edge or "#AAB4C2", face=face or COL["soft"], lw=lw, r=r)
    if text:
        label(ax, x + w / 2, y + h / 2, text, size=size, weight=weight, color=color)


def arrow(ax, x1, y1, x2, y2, color=None, lw=1.7, ms=11, label_text=None, label_offset=13):
    ax.annotate(
        "",
        xy=(x2, y2),
        xytext=(x1, y1),
        arrowprops=dict(
            arrowstyle="-|>",
            lw=lw,
            color=color or COL["line"],
            mutation_scale=ms,
            shrinkA=0,
            shrinkB=0,
        ),
    )
    if label_text:
        label(ax, (x1 + x2) / 2, (y1 + y2) / 2 + label_offset, label_text, size=8.2, color=color or COL["line"])


def elbow_arrow(ax, pts, color=None, lw=1.45, ms=10, dashed=False):
    color = color or COL["line"]
    for (x1, y1), (x2, y2) in zip(pts[:-2], pts[1:-1]):
        ax.plot([x1, x2], [y1, y2], color=color, lw=lw, linestyle=(0, (4, 3)) if dashed else "-")
    (x1, y1), (x2, y2) = pts[-2], pts[-1]
    ax.annotate(
        "",
        xy=(x2, y2),
        xytext=(x1, y1),
        arrowprops=dict(
            arrowstyle="-|>",
            lw=lw,
            color=color,
            mutation_scale=ms,
            shrinkA=0,
            shrinkB=0,
            linestyle=(0, (4, 3)) if dashed else "-",
        ),
    )


def grid(ax, x, y, cols, rows, cw, ch, colors, stroke="white", lw=1.1):
    for r in range(rows):
        for c in range(cols):
            fc = colors[r][c] if isinstance(colors[r], list) else colors[r]
            ax.add_patch(
                patches.Rectangle(
                    (x + c * cw, y + (rows - 1 - r) * ch),
                    cw,
                    ch,
                    facecolor=fc,
                    edgecolor=stroke,
                    linewidth=lw,
                )
            )
    ax.add_patch(patches.Rectangle((x, y), cols * cw, rows * ch, fill=False, edgecolor="#9BA6B5", linewidth=0.8))


def smpl_icon(ax, cx, cy, scale=1.0, alpha=1.0):
    """A neutral mesh-like SMPL proxy, deliberately not a skeleton stick figure."""
    edge = "#8A95A3"
    fill = "#D7DCE3"
    ax.add_patch(patches.Circle((cx, cy + 76 * scale), 13 * scale, facecolor=fill, edgecolor=edge, lw=0.8, alpha=alpha))
    torso = patches.Polygon(
        [
            (cx - 25 * scale, cy + 48 * scale),
            (cx + 25 * scale, cy + 48 * scale),
            (cx + 20 * scale, cy - 28 * scale),
            (cx + 9 * scale, cy - 60 * scale),
            (cx - 9 * scale, cy - 60 * scale),
            (cx - 20 * scale, cy - 28 * scale),
        ],
        closed=True,
        facecolor=fill,
        edgecolor=edge,
        lw=0.8,
        alpha=alpha,
    )
    ax.add_patch(torso)
    limbs = [
        [(cx - 24 * scale, cy + 42 * scale), (cx - 52 * scale, cy - 24 * scale), (cx - 42 * scale, cy - 34 * scale), (cx - 12 * scale, cy + 30 * scale)],
        [(cx + 24 * scale, cy + 42 * scale), (cx + 52 * scale, cy - 24 * scale), (cx + 42 * scale, cy - 34 * scale), (cx + 12 * scale, cy + 30 * scale)],
        [(cx - 9 * scale, cy - 58 * scale), (cx - 32 * scale, cy - 135 * scale), (cx - 18 * scale, cy - 140 * scale), (cx + 1 * scale, cy - 64 * scale)],
        [(cx + 9 * scale, cy - 58 * scale), (cx + 32 * scale, cy - 135 * scale), (cx + 18 * scale, cy - 140 * scale), (cx - 1 * scale, cy - 64 * scale)],
    ]
    for pts in limbs:
        ax.add_patch(patches.Polygon(pts, closed=True, facecolor=fill, edgecolor=edge, lw=0.75, alpha=alpha))
    for dy in [32, 12, -8, -28, -48]:
        ax.plot([cx - 20 * scale, cx + 20 * scale], [cy + dy * scale, cy + (dy + 3) * scale], color="#AEB7C2", lw=0.55 * scale, alpha=alpha)
    for dx in [-15, 0, 15]:
        ax.plot([cx + dx * scale, cx], [cy + 45 * scale, cy - 58 * scale], color="#B6BEC9", lw=0.48 * scale, alpha=alpha)


def draw_smpl_card(ax, x, y, w, h, title):
    card(ax, x, y, w, h, edge="#8D9AAF", face="#FFFFFF", lw=1.1, r=10)
    if SMPL_CROP.exists():
        img = mpimg.imread(SMPL_CROP)
        mesh_h = 224
        mesh_w = min(w - 46, mesh_h * img.shape[1] / img.shape[0])
        cx = x + w / 2
        ax.imshow(img, extent=(cx - mesh_w / 2, cx + mesh_w / 2, y + 82, y + 82 + mesh_h), zorder=3)
    else:
        scale = min(0.98, max(0.72, w / 165))
        smpl_icon(ax, x + w / 2, y + 163, scale)
    label(ax, x + w / 2, y + 44, title, size=11, weight="bold")


def draw_motion_grid(ax, x, y, w, h):
    card(ax, x, y, w, h, "Kinematic-unit grid X", edge=COL["line"], face="#FFFFFF", lw=1.25, r=10)
    gx, gy = x + 68, y + 35
    rows, cols, cw, ch = 9, 9, 22, 21
    colors = [COL["root"]] * 2 + [COL["orient"]] * 3 + [COL["joint"]] * 4
    grid(ax, gx, gy, cols, rows, cw, ch, colors)
    label(ax, gx + cols * cw / 2, gy + rows * ch + 24, "time", size=8.5)
    arrow(ax, gx + 48, gy + rows * ch + 13, gx + cols * cw - 6, gy + rows * ch + 13, lw=1.0, ms=7)
    label(ax, gx - 44, gy + rows * ch / 2, "kinematic unit", size=8.2, rot=90)
    arrow(ax, gx - 29, gy + rows * ch - 6, gx - 29, gy + 8, lw=1.0, ms=7)
    label(ax, gx - 6, gy + 7.9 * ch, "root", size=8.2, color=COL["root"], weight="bold", ha="right")
    label(ax, gx - 6, gy + 5.3 * ch, "orient", size=8.2, color=COL["orient"], weight="bold", ha="right")
    label(ax, gx - 6, gy + 2.0 * ch, "joints", size=8.2, color=COL["joint"], weight="bold", ha="right")


def draw_vae(ax, x, y, w, h):
    card(ax, x, y, w, h, "Causal Motion VAE", edge=COL["blue"], face="#FFFFFF", lw=1.35, r=10)
    label(ax, x + 82, y + h - 67, "Encoder", size=9, weight="bold", color=COL["blue"])
    label(ax, x + w - 82, y + h - 67, "Decoder", size=9, weight="bold", color=COL["blue"])
    for bx in (x + 38, x + w - 138):
        rect(ax, bx, y + 130, 100, 54, "causal\nconv", edge="#9CB6D7", face=COL["blue2"], size=8.2)
        rect(ax, bx, y + 58, 100, 58, "joint\nattention", edge="#9CB6D7", face=COL["blue2"], size=8.2)
        arrow(ax, bx + 50, y + 130, bx + 50, y + 116, lw=1.1, ms=7)
    rect(ax, x + w / 2 - 24, y + 101, 48, 48, "Z", edge=COL["line"], face=COL["gray"], size=12, weight="bold")
    elbow_arrow(ax, [(x + 138, y + 87), (x + w / 2 - 24, y + 125)], lw=1.15, ms=8)
    elbow_arrow(ax, [(x + w / 2 + 24, y + 125), (x + w - 138, y + 87)], lw=1.15, ms=8)


def draw_latent_grid(ax, x, y, w, h):
    card(ax, x, y, w, h, "Latent grid Z", edge=COL["line"], face="#FFFFFF", lw=1.25, r=10)
    gx, gy = x + 48, y + 44
    rows, cols, cw, ch = 8, 7, 20, 22
    colors = [[COL["teal"] if c < 3 else COL["coral"] for c in range(cols)] for _ in range(rows)]
    grid(ax, gx, gy, cols, rows, cw, ch, colors)
    label(ax, gx + cols * cw / 2, gy + rows * ch + 23, "time", size=8.5)
    arrow(ax, gx + 44, gy + rows * ch + 12, gx + cols * cw - 8, gy + rows * ch + 12, lw=1.0, ms=7)
    label(ax, gx - 24, gy + rows * ch / 2, "joint", size=8.5, rot=90)
    arrow(ax, gx - 13, gy + rows * ch - 7, gx - 13, gy + 7, lw=1.0, ms=7)
    ax.add_patch(patches.Rectangle((x + 45, y + 33), 15, 15, facecolor=COL["teal"], edgecolor=COL["line"], lw=0.5))
    label(ax, x + 68, y + 40.5, "clean context", size=7.5, ha="left")
    ax.add_patch(patches.Rectangle((x + 45, y + 12), 15, 15, facecolor=COL["coral"], edgecolor=COL["line"], lw=0.5))
    label(ax, x + 68, y + 19.5, "noisy target", size=7.5, ha="left")


def draw_tree(ax, x, y, scale=1.0, color=COL["line"]):
    pts = {
        "root": (x, y + 55 * scale),
        "spine": (x, y + 32 * scale),
        "lh": (x - 24 * scale, y + 20 * scale),
        "rh": (x + 24 * scale, y + 20 * scale),
        "lk": (x - 38 * scale, y - 8 * scale),
        "rk": (x + 38 * scale, y - 8 * scale),
        "la": (x - 18 * scale, y - 4 * scale),
        "ra": (x + 18 * scale, y - 4 * scale),
    }
    edges = [("root", "spine"), ("spine", "lh"), ("spine", "rh"), ("lh", "lk"), ("rh", "rk"), ("spine", "la"), ("spine", "ra")]
    for a, b in edges:
        ax.plot([pts[a][0], pts[b][0]], [pts[a][1], pts[b][1]], color=color, lw=1.0)
    for px, py in pts.values():
        ax.add_patch(patches.Circle((px, py), 4.2 * scale, facecolor="white", edgecolor=color, lw=1.0))


def draw_kuflowt(ax, x, y, w, h):
    card(ax, x, y, w, h, "KU-FlowT", edge=COL["blue"], face="#FBFDFF", lw=1.8, r=13)
    # Column containers.
    cx1, cw1 = x + 18, 145
    cx2, cw2 = x + 184, 265
    cx3, cw3 = x + 472, 194
    for cx, cw, title in ((cx1, cw1, "Conditioning"), (cx2, cw2, "Factorized DiT × L"), (cx3, cw3, "Flow sampler")):
        card(ax, cx, y + 20, cw, h - 62, title, edge="#B8C9E6", face="#FFFFFF", lw=1.0, r=9)

    # Conditioning column.
    rect(ax, cx1 + 23, y + 300, 104, 44, "text c", edge="#C8B8E8", face=COL["purple2"], size=8.5)
    rect(ax, cx1 + 23, y + 220, 104, 48, "mixed Zτ", edge="#93D0C9", face=COL["teal2"], size=8.5)
    rect(ax, cx1 + 23, y + 138, 104, 48, "τj\n(per-token)", edge="#F0BE75", face=COL["amber2"], size=8.0)
    rect(ax, cx1 + 23, y + 58, 104, 48, "AdaLN", edge="#9CB6D7", face=COL["blue2"], size=9)
    arrow(ax, cx1 + 75, y + 138, cx1 + 75, y + 106, lw=1.1, ms=8)

    # DiT column.
    rect(ax, cx2 + 24, y + 304, 160, 50, "Temporal Attn", edge="#8FB0D8", face=COL["blue2"], size=9)
    rect(ax, cx2 + 24, y + 224, 160, 50, "Joint Attn", edge="#8FB0D8", face=COL["blue2"], size=9)
    rect(ax, cx2 + 24, y + 144, 160, 50, "MLP", edge="#8FB0D8", face=COL["blue2"], size=9)
    arrow(ax, cx2 + 104, y + 304, cx2 + 104, y + 274, lw=1.15, ms=8)
    arrow(ax, cx2 + 104, y + 224, cx2 + 104, y + 194, lw=1.15, ms=8)
    rect(ax, cx2 + 190, y + 312, 52, 30, "time\nRoPE", edge=COL["blue"], face="#F4F8FF", size=6.5, color=COL["blue"], weight="bold")

    # KT-RoPE is a local badge attached to Joint Attn, not a separate layer.
    kt_x, kt_y, kt_w, kt_h = cx2 + 190, y + 207, 62, 84
    card(ax, kt_x, kt_y, kt_w, kt_h, None, edge=COL["purple"], face=COL["purple2"], lw=1.0, r=8)
    label(ax, kt_x + kt_w / 2, kt_y + kt_h - 13, "KT-RoPE", size=6.8, weight="bold", color=COL["purple"])
    draw_tree(ax, kt_x + 17, kt_y + 21, 0.33, COL["purple"])
    label(ax, kt_x + 37, kt_y + 47, "pj", size=7.0, weight="bold", color=COL["purple"])
    ax.add_patch(patches.Circle((kt_x + 45, kt_y + 24), 11, fill=False, edgecolor=COL["purple"], lw=0.9))
    arrow(ax, kt_x, kt_y + 43, cx2 + 184, y + 249, color=COL["purple"], lw=0.9, ms=6)

    # Local conditioning arrows into DiT; all short and orthogonal.
    elbow_arrow(ax, [(cx1 + cw1, y + 322), (cx2 + 24, y + 329)], lw=1.15, ms=8)
    elbow_arrow(ax, [(cx1 + cw1, y + 244), (cx2 + 24, y + 249)], lw=1.15, ms=8)
    elbow_arrow(ax, [(cx1 + 75, y + 58), (cx1 + 75, y + 43), (cx2 + 24, y + 43), (cx2 + 24, y + 169)], lw=1.0, ms=7)

    # Flow sampler column.
    rect(ax, cx3 + 26, y + 144, 95, 48, "velocity vθ", edge="#9CB6D7", face=COL["blue2"], size=8.5)
    rect(ax, cx3 + 26, y + 72, 95, 48, "Euler step", edge="#9CB6D7", face=COL["blue2"], size=8.5)
    arrow(ax, cx2 + 184, y + 169, cx3 + 26, y + 169, lw=1.15, ms=8)
    arrow(ax, cx3 + 73, y + 144, cx3 + 73, y + 120, lw=1.15, ms=8)
    rect(ax, cx3 + 114, y + 60, 67, 116, "", edge=COL["amber"], face=COL["amber2"], lw=1.0)
    label(ax, cx3 + 147, y + 158, "KAFS", size=8.8, weight="bold", color=COL["amber"])
    label(ax, cx3 + 147, y + 135, "depth dj", size=6.5, color=COL["amber"])
    label(ax, cx3 + 147, y + 113, "σj(k)", size=6.8, color=COL["amber"])
    # Tiny schedule curves.
    for i, c in enumerate(["#377BD1", "#25A06A", "#E45A4D", "#7A55C7"]):
        xs = [cx3 + 127 + t * 7 for t in range(6)]
        ys = [y + 76 + (30 - i * 5) * (1 - (t / 5) ** (1.15 + i * 0.2)) for t in range(6)]
        ax.plot(xs, ys, color=c, lw=1.0)
    # Local sidecar semantics, not data-flow.
    elbow_arrow(ax, [(cx3 + 114, y + 145), (cx3 + 105, y + 145), (cx3 + 105, y + 120)], color=COL["amber"], lw=1.0, ms=7)
    elbow_arrow(ax, [(cx3 + 114, y + 93), (cx3 + 121, y + 93)], color=COL["amber"], lw=1.0, ms=7)
    label(ax, cx3 + 102, y + 137, "τj", size=7.2, color=COL["amber"], ha="right")
    label(ax, cx3 + 123, y + 93, "Δσj", size=7.2, color=COL["amber"], ha="left")
    # Denoised latent output inside sampler.
    colors = [[COL["teal"] for _ in range(4)] for _ in range(2)]
    grid(ax, cx3 + 38, y + 27, 4, 2, 17, 17, colors, lw=0.9)
    label(ax, cx3 + 73, y + 18, "denoised Z", size=7.2, color=COL["muted"])
    arrow(ax, cx3 + 73, y + 72, cx3 + 73, y + 61, lw=1.0, ms=7)


def draw_interface(ax, x, y, w, h):
    card(ax, x, y, w, h, "Clean / noisy token interface", edge="#8D9AAF", face="#FFFFFF", lw=1.1, r=11)
    labels = [("T2M", 0), ("TP2M", 1), ("AR", 2)]
    start_x = x + 82
    for title, i in labels:
        bx = start_x + i * 225
        label(ax, bx + 78, y + h - 72, title, size=10, weight="bold")
        rows, cols, cw, ch = 3, 7, 20, 20
        if title == "T2M":
            colors = [[COL["coral"] for _ in range(cols)] for _ in range(rows)]
            sub = [("all noisy target", COL["coral"])]
        elif title == "TP2M":
            colors = [[COL["teal"] if c < 3 else COL["coral"] for c in range(cols)] for _ in range(rows)]
            sub = [("clean prefix", COL["teal"]), ("noisy target", COL["coral"])]
        else:
            colors = [[COL["teal"] if c < 5 else COL["coral"] for c in range(cols)] for _ in range(rows)]
            sub = [("clean tail", COL["teal"]), ("next segment", COL["coral"])]
        grid(ax, bx, y + 73, cols, rows, cw, ch, colors)
        label(ax, bx + cols * cw / 2, y + 151, "time", size=7.5)
        arrow(ax, bx + 18, y + 142, bx + cols * cw, y + 142, lw=0.9, ms=6)
        label(ax, bx - 24, y + 104, "joint", size=7.5, rot=90)
        arrow(ax, bx - 12, y + 133, bx - 12, y + 76, lw=0.9, ms=6)
        if len(sub) == 1:
            label(ax, bx + 70, y + 45, sub[0][0], size=7.7, color=sub[0][1])
        else:
            label(ax, bx + 34, y + 45, sub[0][0], size=7.0, color=sub[0][1])
            label(ax, bx + 118, y + 45, sub[1][0], size=7.0, color=sub[1][1])
    ax.add_patch(patches.Rectangle((x + w / 2 - 68, y + 18), 16, 16, facecolor=COL["teal"], edgecolor=COL["line"], lw=0.5))
    label(ax, x + w / 2 - 45, y + 26, "clean context", size=7.5, ha="left")
    ax.add_patch(patches.Rectangle((x + w / 2 + 82, y + 18), 16, 16, facecolor=COL["coral"], edgecolor=COL["line"], lw=0.5))
    label(ax, x + w / 2 + 105, y + 26, "noisy target", size=7.5, ha="left")


def draw_self_forcing(ax, x, y, w, h):
    card(ax, x, y, w, h, "Self-Forcing", edge="#8D9AAF", face="#FFFFFF", lw=1.1, r=11)
    label(ax, x + 105, y + h - 72, "generated\nsegment", size=8.5)
    for i in range(4):
        ax.add_patch(patches.Rectangle((x + 54 + i * 24, y + 112), 20, 34, facecolor=COL["gray"], edgecolor="#9AA4B1", lw=0.8))
    label(ax, x + 187, y + 128, "...", size=13, color=COL["muted"])
    arrow(ax, x + 204, y + 129, x + 263, y + 129, lw=1.2, ms=8)
    rect(ax, x + 275, y + 88, 90, 82, "VAE\nencode", edge=COL["blue"], face=COL["blue2"], size=10, color=COL["blue"], weight="bold")
    arrow(ax, x + 365, y + 129, x + 438, y + 129, lw=1.2, ms=8)
    label(ax, x + 493, y + h - 72, "clean tail\ncontext", size=8.5)
    for i in range(3):
        ax.add_patch(patches.Rectangle((x + 438 + i * 26, y + 112), 21, 34, facecolor=COL["teal"], edgecolor="white", lw=1.0))
    label(ax, x + 528, y + 129, "...", size=13, color=COL["muted"])
    ax.annotate(
        "",
        xy=(x + 92, y + 99),
        xytext=(x + 478, y + 99),
        arrowprops=dict(
            arrowstyle="-|>",
            lw=1.3,
            color=COL["blue"],
            linestyle=(0, (3, 3)),
            connectionstyle="arc3,rad=-0.28",
            mutation_scale=9,
        ),
    )
    label(ax, x + w / 2, y + 34, "generated tail as context", size=7.9, color=COL["blue"], weight="bold")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = setup_ax()

    # Top rail.
    draw_smpl_card(ax, 32, 758, 174, 300, "SMPL\nmotion")
    draw_motion_grid(ax, 258, 758, 282, 300)
    draw_vae(ax, 585, 758, 308, 300)
    draw_latent_grid(ax, 935, 758, 218, 300)
    draw_kuflowt(ax, 1170, 595, 700, 463)
    draw_smpl_card(ax, 1888, 758, 128, 300, "SMPL\noutput")

    arrow(ax, 206, 908, 258, 908, lw=1.8, ms=11)
    arrow(ax, 540, 908, 585, 908, lw=1.8, ms=11)
    arrow(ax, 893, 908, 935, 908, lw=1.8, ms=11)
    arrow(ax, 1153, 908, 1170, 908, lw=1.8, ms=11)
    arrow(ax, 1870, 908, 1888, 908, lw=1.8, ms=11, label_text="decode", label_offset=14)

    # Subtle band labels.
    ax.plot([30, 2018], [565, 565], color="#E4E8EF", lw=1.0)
    label(ax, 53, 548, "unified conditional generation", size=8.5, color=COL["muted"], ha="left")

    # Bottom rail.
    draw_interface(ax, 42, 142, 835, 330)
    draw_self_forcing(ax, 938, 142, 608, 330)

    card(ax, 1596, 142, 384, 330, "Narrative composition", edge="#BCC7D8", face="#FFFFFF", lw=1.0, r=11)
    for i, txt in enumerate(["c1: walk", "c2: turn", "c3: jump"]):
        rect(ax, 1636 + i * 102, 320, 78, 44, txt, edge="#C9D2E3", face=COL["soft"], size=8.0)
        if i < 2:
            arrow(ax, 1714 + i * 102, 342, 1730 + i * 102, 342, lw=1.0, ms=6)
    y0 = 230
    for i, col in enumerate([COL["teal"], COL["teal"], COL["coral"], COL["coral"]]):
        ax.add_patch(patches.Rectangle((1668 + i * 34, y0), 28, 48, facecolor=col, edgecolor="white", lw=1.0))
    arrow(ax, 1808, y0 + 24, 1858, y0 + 24, lw=1.0, ms=7)
    for i, col in enumerate([COL["teal"], COL["coral"], COL["coral"], COL["coral"]]):
        ax.add_patch(patches.Rectangle((1864 + i * 26, y0), 21, 48, facecolor=col, edgecolor="white", lw=1.0))
    label(ax, 1788, 189, "LLM plan → segment-wise AR generation", size=8.2, color=COL["muted"])

    png = OUT_DIR / "fig_pipeline_gptskill_blueprint_0611.png"
    svg = OUT_DIR / "fig_pipeline_gptskill_blueprint_0611.svg"
    fig.savefig(png, dpi=180, bbox_inches="tight", pad_inches=0.06)
    fig.savefig(svg, bbox_inches="tight", pad_inches=0.06)
    print(png)
    print(svg)


if __name__ == "__main__":
    main()
