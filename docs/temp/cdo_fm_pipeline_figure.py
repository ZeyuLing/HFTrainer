#!/usr/bin/env python3
"""
CDO-FM (Condition-Decoupled Orchestration Flow Matching) Method Pipeline Overview Figure.

Generates a publication-quality pipeline diagram for the CDO-FM method paper.
Run: python docs/temp/cdo_fm_pipeline_figure.py

Output: docs/temp/cdo_fm_pipeline_overview.pdf / .png

Layout:
  - Left side: Two input encoders (Text path on top, Motion condition path on bottom)
  - Center: MMDiT Backbone (vertical column, top-to-bottom processing)
  - Right side: Two core innovation panels (PDCT, CPOS) with mini plots
  - Bottom: Output Motion
Note: Dual Root is an engineering setting (SMPL vs KIMODO), not a core innovation.
      It appears only as a small annotation on the representation in the left column.
Note: No title in figure — title goes in the paper caption.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, ArrowStyle
import matplotlib.patheffects as pe
import numpy as np

# ============================================================
# Color Palette (academic pastel, print-friendly)
# ============================================================
C = dict(
    # Inputs
    text_input      = '#D6EAF8',  # light blue
    motion_input    = '#D4E6F1',  # slightly different blue
    # Encoders
    text_enc        = '#FDEBD0',  # light orange
    vace_enc        = '#FAD7A0',  # medium orange
    v3_sampler      = '#D2B4DE',  # medium purple
    # Backbone
    backbone_bg     = '#D5F5E3',  # light green
    dual_stream     = '#A9DFBF',  # medium green
    single_stream   = '#7DCEA0',  # darker green
    input_proj      = '#ABEBC6',  # soft green
    flow_pred       = '#F5CBA7',  # peach
    timestep        = '#D5DBDB',  # light grey
    # Innovation panels
    pdct_bg         = '#F5EEF8',  # very light purple
    pdct_accent     = '#8E44AD',  # purple accent
    cpos_bg         = '#FDEDEC',  # very light coral
    cpos_accent     = '#C0392B',  # red accent
    # Output
    output          = '#FCF3CF',  # light yellow
    output_border   = '#B7950B',  # gold border
    # Misc
    border          = '#2C3E50',  # dark blue-grey
    text_dark       = '#1B2631',  # near black
    arrow_main      = '#566573',  # medium grey
    arrow_text      = '#E67E22',  # orange (text path)
    arrow_motion    = '#2980B9',  # blue (motion path)
    arrow_innov     = '#8E44AD',  # purple (innovation link)
    # MAN diagram colors
    man_xt          = '#AED6F1',  # light blue for x_t channel
    man_reactive    = '#A9DFBF',  # light green for reactive channel
    man_mask        = '#F9E79F',  # light yellow for mask channel
)


def draw_box(ax, x, y, w, h, label, sublabel=None, color='#D6EAF8',
             fontsize=10, sublabel_fontsize=7, text_color=None,
             bold=True, border_color=None, linewidth=1.2, alpha=0.92,
             zorder=2, pad=0.08):
    """Draw a rounded rectangle with centered label."""
    if text_color is None:
        text_color = C['text_dark']
    if border_color is None:
        border_color = C['border']

    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad={pad}",
        facecolor=color, edgecolor=border_color,
        linewidth=linewidth, alpha=alpha, zorder=zorder,
    )
    ax.add_patch(box)

    weight = 'bold' if bold else 'normal'
    if sublabel:
        ax.text(x + w / 2, y + h * 0.62, label,
                ha='center', va='center', fontsize=fontsize,
                fontweight=weight, color=text_color, zorder=zorder + 1)
        ax.text(x + w / 2, y + h * 0.28, sublabel,
                ha='center', va='center', fontsize=sublabel_fontsize,
                fontstyle='italic', color='#566573', zorder=zorder + 1)
    else:
        ax.text(x + w / 2, y + h / 2, label,
                ha='center', va='center', fontsize=fontsize,
                fontweight=weight, color=text_color, zorder=zorder + 1)

    return (x, y, w, h)  # return bounds for arrow targeting


def arrow(ax, x1, y1, x2, y2, color=None, lw=1.5, ls='-', zorder=1,
          connectionstyle='arc3,rad=0.0', shrinkA=3, shrinkB=3):
    """Draw an arrow."""
    if color is None:
        color = C['arrow_main']
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(
                    arrowstyle='->', color=color, lw=lw,
                    connectionstyle=connectionstyle,
                    shrinkA=shrinkA, shrinkB=shrinkB,
                    linestyle=ls,
                ), zorder=zorder)


def arrow_curved(ax, x1, y1, x2, y2, rad=0.15, **kwargs):
    """Draw a curved arrow."""
    arrow(ax, x1, y1, x2, y2, connectionstyle=f'arc3,rad={rad}', **kwargs)


# ============================================================
# Schematic illustration helpers
# ============================================================

def draw_text_input_icon(ax, cx, cy, scale=1.0):
    """Draw a speech-bubble + text-lines schematic for text input."""
    s = scale
    # Speech bubble outline
    bubble_w, bubble_h = 1.1 * s, 0.55 * s
    bx = cx - bubble_w / 2
    by = cy - bubble_h / 2
    bubble = FancyBboxPatch(
        (bx, by), bubble_w, bubble_h,
        boxstyle="round,pad=0.05",
        facecolor='white', edgecolor='#5DADE2',
        linewidth=1.0, alpha=0.9, zorder=5,
    )
    ax.add_patch(bubble)

    # Text lines inside the bubble (3 short horizontal lines)
    line_y_offsets = [0.15 * s, 0 * s, -0.15 * s]
    line_widths = [0.65 * s, 0.55 * s, 0.35 * s]
    for dy, lw_line in zip(line_y_offsets, line_widths):
        ax.plot([cx - lw_line / 2, cx + lw_line / 2],
                [cy + dy, cy + dy],
                color='#5DADE2', linewidth=1.8, alpha=0.7, zorder=6,
                solid_capstyle='round')

    # Small tail/pointer at bottom-left of bubble
    tail_x = bx + 0.15 * s
    tail_y = by
    ax.plot([tail_x, tail_x - 0.1 * s, tail_x + 0.1 * s],
            [tail_y, tail_y - 0.12 * s, tail_y],
            color='#5DADE2', linewidth=1.0, alpha=0.9, zorder=5)


def draw_stick_figure(ax, cx, cy, scale=1.0, color='#2980B9', alpha=0.8):
    """Draw a minimal stick figure."""
    s = scale * 0.35
    # Head
    head = plt.Circle((cx, cy + 0.5 * s), 0.12 * s, fill=False,
                       edgecolor=color, linewidth=1.2, alpha=alpha, zorder=6)
    ax.add_patch(head)
    # Body
    ax.plot([cx, cx], [cy + 0.38 * s, cy - 0.05 * s],
            color=color, linewidth=1.2, alpha=alpha, zorder=6)
    # Arms
    ax.plot([cx - 0.2 * s, cx + 0.2 * s], [cy + 0.28 * s, cy + 0.28 * s],
            color=color, linewidth=1.0, alpha=alpha, zorder=6)
    # Legs
    ax.plot([cx, cx - 0.15 * s], [cy - 0.05 * s, cy - 0.35 * s],
            color=color, linewidth=1.0, alpha=alpha, zorder=6)
    ax.plot([cx, cx + 0.15 * s], [cy - 0.05 * s, cy - 0.35 * s],
            color=color, linewidth=1.0, alpha=alpha, zorder=6)


def draw_stick_keyframes(ax, cx, cy, n=3, spacing=0.55, scale=1.0):
    """Draw n stick figures in a row as motion keyframes with dashed lines between."""
    start_x = cx - (n - 1) * spacing * scale / 2
    poses = []
    # Different arm/leg angles for each keyframe
    arm_angles = [(-0.2, 0.2), (0.15, -0.15), (-0.05, 0.25)]
    leg_angles = [(-0.15, 0.15), (0.1, -0.1), (-0.12, 0.18)]

    for i in range(n):
        x = start_x + i * spacing * scale
        s = scale * 0.3
        color = '#2980B9'
        al = 0.5 + 0.2 * (i / max(n - 1, 1))  # increasing alpha

        # Head
        head = plt.Circle((x, cy + 0.5 * s), 0.1 * s, fill=False,
                           edgecolor=color, linewidth=1.0, alpha=al, zorder=6)
        ax.add_patch(head)
        # Body
        ax.plot([x, x], [cy + 0.4 * s, cy - 0.05 * s],
                color=color, linewidth=1.0, alpha=al, zorder=6)
        # Arms (varied poses)
        la, ra = arm_angles[i % len(arm_angles)]
        ax.plot([x + la * s, x + ra * s],
                [cy + 0.25 * s + abs(la) * 0.3 * s, cy + 0.25 * s + abs(ra) * 0.3 * s],
                color=color, linewidth=0.8, alpha=al, zorder=6)
        # Legs (varied poses)
        ll, rl = leg_angles[i % len(leg_angles)]
        ax.plot([x, x + ll * s], [cy - 0.05 * s, cy - 0.3 * s],
                color=color, linewidth=0.8, alpha=al, zorder=6)
        ax.plot([x, x + rl * s], [cy - 0.05 * s, cy - 0.3 * s],
                color=color, linewidth=0.8, alpha=al, zorder=6)

        # Dashed lines between keyframes
        if i < n - 1:
            x_next = start_x + (i + 1) * spacing * scale
            ax.plot([x + 0.12 * s, x_next - 0.12 * s],
                    [cy + 0.1 * s, cy + 0.1 * s],
                    color='#999', linewidth=0.6, linestyle=':', alpha=0.5, zorder=5)


def draw_motion_sequence(ax, cx, cy, n=5, spacing=0.35, scale=1.0):
    """Draw a motion sequence as a series of stick figures with a motion trail."""
    start_x = cx - (n - 1) * spacing * scale / 2

    # Motion trail (gradient line behind the figures)
    for i in range(n - 1):
        x1 = start_x + i * spacing * scale
        x2 = start_x + (i + 1) * spacing * scale
        al = 0.15 + 0.15 * (i / max(n - 2, 1))
        ax.plot([x1, x2], [cy - 0.08 * scale, cy - 0.08 * scale],
                color=C['output_border'], linewidth=2.5, alpha=al, zorder=4,
                solid_capstyle='round')

    for i in range(n):
        x = start_x + i * spacing * scale
        s = scale * 0.22
        al = 0.3 + 0.6 * (i / max(n - 1, 1))  # fade in
        color = C['output_border']

        # Head
        head = plt.Circle((x, cy + 0.5 * s), 0.08 * s, fill=False,
                           edgecolor=color, linewidth=0.8, alpha=al, zorder=6)
        ax.add_patch(head)
        # Body
        ax.plot([x, x], [cy + 0.42 * s, cy - 0.05 * s],
                color=color, linewidth=0.8, alpha=al, zorder=6)
        # Arms — vary across sequence
        arm_spread = 0.15 + 0.1 * np.sin(i * 1.2)
        ax.plot([x - arm_spread * s, x + arm_spread * s],
                [cy + 0.28 * s, cy + 0.28 * s],
                color=color, linewidth=0.6, alpha=al, zorder=6)
        # Legs
        leg_angle = 0.12 + 0.06 * np.sin(i * 1.5)
        ax.plot([x, x - leg_angle * s], [cy - 0.05 * s, cy - 0.28 * s],
                color=color, linewidth=0.6, alpha=al, zorder=6)
        ax.plot([x, x + leg_angle * s], [cy - 0.05 * s, cy - 0.28 * s],
                color=color, linewidth=0.6, alpha=al, zorder=6)

    # Arrow at end indicating continuation
    last_x = start_x + (n - 1) * spacing * scale + 0.08 * scale
    ax.annotate('', xy=(last_x + 0.2 * scale, cy),
                xytext=(last_x + 0.02 * scale, cy),
                arrowprops=dict(arrowstyle='->', color=C['output_border'],
                                lw=1.5, alpha=0.6),
                zorder=6)


def draw_man_diagram(ax, cx, cy, w, h):
    """Draw MAN (Mask-Aware Noise) 3-channel diagram.

    Shows 3 horizontal channels stacked vertically:
    - Channel 1 (top):    x_t(MAN) — known=clean, unknown=noise
    - Channel 2 (middle): reactive — src_motion x src_mask
    - Channel 3 (bottom): mask — binary 1/0
    Each rendered as a colored strip with mini schematic content.
    """
    # Strip dimensions
    strip_h = h * 0.22
    gap = h * 0.06
    total_h = 3 * strip_h + 2 * gap
    y_start = cy + total_h / 2 - strip_h

    channels = [
        ('$x_t$(MAN)', C['man_xt'], '#2471A3'),
        ('reactive', C['man_reactive'], '#1B7A3D'),
        ('mask', C['man_mask'], '#B7950B'),
    ]

    label_w = w * 0.32  # label region width
    strip_w = w - label_w - 0.1  # content strip width
    strip_x = cx - w / 2 + label_w + 0.05

    for i, (label, bg_color, text_color) in enumerate(channels):
        sy = y_start - i * (strip_h + gap)

        # Label on the left
        ax.text(cx - w / 2 + label_w / 2, sy + strip_h / 2,
                label, ha='center', va='center', fontsize=5.5,
                color=text_color, fontweight='bold', zorder=8)

        # Content strip
        strip = FancyBboxPatch(
            (strip_x, sy), strip_w, strip_h,
            boxstyle="round,pad=0.02",
            facecolor=bg_color, edgecolor=text_color,
            linewidth=0.6, alpha=0.7, zorder=6
        )
        ax.add_patch(strip)

        # Mini content inside each strip
        n_cells = 10
        cell_w = strip_w / n_cells
        if i == 0:  # x_t(MAN): show known=solid, unknown=hatched
            for j in range(n_cells):
                cx_j = strip_x + (j + 0.5) * cell_w
                if j in [2, 3, 7]:  # "known" positions — solid clean color
                    rect = FancyBboxPatch(
                        (strip_x + j * cell_w + 0.01, sy + 0.01),
                        cell_w - 0.02, strip_h - 0.02,
                        boxstyle="square,pad=0", facecolor='#85C1E9',
                        edgecolor='none', alpha=0.5, zorder=7)
                    ax.add_patch(rect)
                else:  # "unknown" — show noise texture (small random dots)
                    np.random.seed(42 + j)
                    for _ in range(3):
                        dx = np.random.uniform(0.02, cell_w - 0.02)
                        dy = np.random.uniform(0.02, strip_h - 0.02)
                        ax.plot(strip_x + j * cell_w + dx, sy + dy,
                                '.', color='#5DADE2', markersize=1.5,
                                alpha=0.4, zorder=7)

        elif i == 1:  # reactive: show partial signal (some cells filled)
            for j in range(n_cells):
                if j in [2, 3, 7]:  # same "known" positions
                    rect = FancyBboxPatch(
                        (strip_x + j * cell_w + 0.01, sy + 0.01),
                        cell_w - 0.02, strip_h - 0.02,
                        boxstyle="square,pad=0", facecolor='#82E0AA',
                        edgecolor='none', alpha=0.5, zorder=7)
                    ax.add_patch(rect)

        elif i == 2:  # mask: binary 1/0
            for j in range(n_cells):
                cx_j = strip_x + (j + 0.5) * cell_w
                val = '0' if j in [2, 3, 7] else '1'
                ax.text(cx_j, sy + strip_h / 2, val,
                        ha='center', va='center', fontsize=4.5,
                        color=text_color, fontweight='bold', alpha=0.7, zorder=8)

    # Concatenation brace or arrow at bottom
    brace_y = y_start - 2 * (strip_h + gap) - 0.05
    ax.text(cx, brace_y, 'concat $\\to$ (B, L, 594)',
            ha='center', va='top', fontsize=5, color='#7F8C8D',
            fontstyle='italic', zorder=8)


# ============================================================
# Inset plot helpers
# ============================================================

def draw_pdct_curve(ax, x, y, w, h):
    """PDCT density schedule curve."""
    inset = ax.inset_axes([x, y, w, h], transform=ax.transData)
    steps = np.linspace(0, 1, 300)
    sa, sb = 0.2, 0.65
    density = np.piecewise(steps,
        [steps < sa, (steps >= sa) & (steps < sb), steps >= sb],
        [lambda t: 0.15,
         lambda t: 0.15 + (0.55 - 0.15) * (t - sa) / (sb - sa),
         lambda t: 0.55])

    inset.fill_between(steps, density, alpha=0.25, color=C['pdct_accent'])
    inset.plot(steps, density, color=C['pdct_accent'], linewidth=2.0)
    inset.axvline(x=sa, color='#999', linewidth=0.5, linestyle='--')
    inset.axvline(x=sb, color='#999', linewidth=0.5, linestyle='--')
    inset.text(sa / 2, 0.50, 'A', ha='center', fontsize=7, fontweight='bold', color=C['pdct_accent'])
    inset.text((sa + sb) / 2, 0.50, 'B', ha='center', fontsize=7, fontweight='bold', color=C['pdct_accent'])
    inset.text((sb + 1) / 2, 0.50, 'C', ha='center', fontsize=7, fontweight='bold', color=C['pdct_accent'])

    inset.set_xlim(0, 1)
    inset.set_ylim(0, 0.62)
    inset.set_xlabel('Training Step', fontsize=6, labelpad=1)
    inset.set_ylabel('E[$\\rho$]', fontsize=6, labelpad=1)
    inset.tick_params(labelsize=5, length=2, pad=1)
    inset.spines['top'].set_visible(False)
    inset.spines['right'].set_visible(False)
    inset.set_facecolor(C['pdct_bg'])
    return inset


def draw_cpos_curve(ax, x, y, w, h):
    """CPOS schedule curves."""
    inset = ax.inset_axes([x, y, w, h], transform=ax.transData)
    t = np.linspace(0, 1, 300)
    w_text = np.exp(-((t - 0.3) / 0.25)**2)  # normalized bell
    alpha = 1 / (1 + np.exp(-10 * (t - 0.4)))  # sigmoid

    inset.plot(t, w_text, color='#E74C3C', linewidth=2.0, label='$w_{text}(t)$ (CFG)')
    inset.plot(t, alpha, color='#2980B9', linewidth=2.0, label='$\\alpha(t)$ (replace)')
    inset.fill_between(t, w_text, alpha=0.1, color='#E74C3C')
    inset.fill_between(t, alpha, alpha=0.1, color='#2980B9')

    inset.set_xlim(0, 1)
    inset.set_ylim(-0.05, 1.15)
    inset.set_xlabel('ODE time $t$', fontsize=6, labelpad=1)
    inset.legend(fontsize=5.5, loc='center right', framealpha=0.7,
                 handlelength=1.2, borderpad=0.3)
    inset.tick_params(labelsize=5, length=2, pad=1)
    inset.spines['top'].set_visible(False)
    inset.spines['right'].set_visible(False)
    inset.set_facecolor(C['cpos_bg'])

    # Phase annotations
    inset.text(0.12, 0.95, 'text CFG\ndominant', fontsize=5.5, color='#E74C3C',
               ha='center', fontweight='bold', va='top')
    inset.text(0.85, 0.95, 'condition\nreplacement', fontsize=5.5, color='#2980B9',
               ha='center', fontweight='bold', va='top')
    return inset


def main():
    # ============================================================
    # Figure Setup (no title — title goes in paper caption)
    # ============================================================
    fig, ax = plt.subplots(1, 1, figsize=(17, 10.5))
    ax.set_xlim(0, 17)
    ax.set_ylim(0, 10.5)
    ax.set_aspect('equal')
    ax.axis('off')
    fig.patch.set_facecolor('white')

    # ============================================================
    # LAYOUT CONSTANTS
    # ============================================================
    # Left column (encoders): x = 0.4 ~ 3.6
    # Center backbone:        x = 4.2 ~ 9.2
    # Right panels:           x = 10.0 ~ 16.6
    # Vertical: 10.0 (top) → 0.3 (bottom)
    # Without title, we can use more vertical space

    LX = 0.4   # left column x
    LW = 3.0   # left column width
    BX = 4.3   # backbone x
    BW = 4.8   # backbone width
    RX = 10.0  # right column x
    RW = 6.6   # right column width

    # ============================================================
    # LEFT: Text Input with speech-bubble illustration
    # ============================================================
    text_input_y = 9.0
    text_input_h = 0.85
    draw_box(ax, LX, text_input_y, LW, text_input_h,
             '', None,  # empty label — we draw the icon instead
             color=C['text_input'], fontsize=9, sublabel_fontsize=6.5)
    # Speech bubble icon
    draw_text_input_icon(ax, LX + LW * 0.32, text_input_y + text_input_h * 0.5, scale=0.52)
    # Label text to the right of icon
    ax.text(LX + LW * 0.68, text_input_y + text_input_h * 0.6,
            'Text Input', ha='center', va='center',
            fontsize=9, fontweight='bold', color=C['text_dark'], zorder=5)
    ax.text(LX + LW * 0.68, text_input_y + text_input_h * 0.28,
            '"a person walks..."',
            ha='center', va='center', fontsize=6, fontstyle='italic',
            color='#566573', zorder=5)

    # ============================================================
    # LEFT: Text Encoder
    # ============================================================
    text_enc_y = 7.65
    text_enc_h = 0.85
    draw_box(ax, LX, text_enc_y, LW, text_enc_h,
             'Text Encoder', 'Frozen Qwen3 + CLIP-L',
             color=C['text_enc'], fontsize=9.5, sublabel_fontsize=7)

    # dim labels
    ax.text(LX + LW / 2, text_enc_y - 0.3,
            '$c_{txt}$(B,S,4096) + $v_{txt}$(B,1,768)',
            ha='center', fontsize=6, color='#7F8C8D', fontstyle='italic')

    # Text Input → Text Encoder
    arrow(ax, LX + LW / 2, text_input_y, LX + LW / 2, text_enc_y + text_enc_h,
          color=C['arrow_text'], lw=1.5)

    # ============================================================
    # LEFT: Motion Condition with stick-figure keyframes
    # ============================================================
    motion_cond_y = 5.7
    motion_cond_h = 0.95
    draw_box(ax, LX, motion_cond_y, LW, motion_cond_h,
             '', None,
             color=C['motion_input'], fontsize=9, sublabel_fontsize=6.5)
    # Stick figure keyframes illustration
    draw_stick_keyframes(ax, LX + LW * 0.5, motion_cond_y + motion_cond_h * 0.55,
                         n=3, spacing=0.65, scale=1.0)
    # Label below figures
    ax.text(LX + LW / 2, motion_cond_y + 0.12,
            'Motion Condition  (keyframes / trajectory / dense)',
            ha='center', va='center', fontsize=6.5, fontweight='bold',
            color=C['text_dark'], zorder=5)

    # ============================================================
    # LEFT: Structured Condition Sampler (renamed from V3 Condition Sampler)
    # ============================================================
    v3_y = 4.3
    v3_h = 0.85
    draw_box(ax, LX, v3_y, LW, v3_h,
             'Structured Condition Sampler', 'Rank-K Boolean Tensor Prior',
             color=C['v3_sampler'], fontsize=7.5, sublabel_fontsize=6.5)
    ax.text(LX + LW / 2, v3_y - 0.18,
            '$M = \\bigoplus_{k=1}^{K}(t_k \\otimes d_k)$',
            ha='center', fontsize=7, color='#6C3483', fontstyle='italic')

    # ============================================================
    # LEFT: VACE Cond. Encoder + MAN diagram
    # ============================================================
    vace_y = 1.9
    vace_h = 1.85  # taller to accommodate MAN diagram
    draw_box(ax, LX, vace_y, LW, vace_h,
             '', None,  # we place label and MAN diagram manually
             color=C['vace_enc'], fontsize=9, sublabel_fontsize=7)
    # Box label at top
    ax.text(LX + LW / 2, vace_y + vace_h - 0.2,
            'VACE Cond. Encoder', ha='center', va='center',
            fontsize=9, fontweight='bold', color=C['text_dark'], zorder=5)
    ax.text(LX + LW / 2, vace_y + vace_h - 0.45,
            'Mask-Aware Noise (MAN)',
            ha='center', va='center', fontsize=7, fontstyle='italic',
            color='#566573', zorder=5)

    # MAN 3-channel visual diagram inside the VACE box
    draw_man_diagram(ax, LX + LW / 2, vace_y + vace_h * 0.35,
                     w=LW - 0.3, h=vace_h * 0.55)

    # Motion Condition → V3 → VACE arrows
    mc = LX + LW / 2
    arrow(ax, mc, motion_cond_y, mc, v3_y + v3_h, color=C['arrow_motion'], lw=1.5)
    arrow(ax, mc, v3_y, mc, vace_y + vace_h, color=C['arrow_motion'], lw=1.5)

    # ============================================================
    # LEFT: Motion Representation (bottom-left, small annotation)
    # ============================================================
    draw_box(ax, LX, 0.8, LW, 0.7,
             'Motion Representation',
             '198-dim: trans(3)+rot6d(132)+pos(63)',
             color='#D5DBDB', fontsize=7.5, sublabel_fontsize=5.5,
             border_color='#7F8C8D', linewidth=0.8, alpha=0.6)

    # Small dashed arrow: representation feeds into VACE
    arrow(ax, mc, 1.5, mc, vace_y, color='#7F8C8D', lw=0.8, ls='--')

    # ============================================================
    # CENTER: MMDiT Backbone
    # ============================================================
    # Background container — extend higher since no title
    bb_pad = 0.15
    backbone_bg = FancyBboxPatch(
        (BX, 0.8), BW, 9.2,
        boxstyle=f"round,pad={bb_pad}",
        facecolor=C['backbone_bg'], edgecolor=C['border'],
        linewidth=2.0, alpha=0.3, zorder=0
    )
    ax.add_patch(backbone_bg)
    ax.text(BX + BW / 2, 9.75,
            'MMDiT Backbone (reused, no new params)',
            ha='center', va='center', fontsize=11, fontweight='bold',
            color='#1B4F2A', zorder=3)

    # Center x for backbone elements
    cx = BX + BW / 2

    # Input Projection
    draw_box(ax, BX + 0.4, 8.65, BW - 0.8, 0.6,
             'Input Projection',
             'text_emb + motion_emb $\\to$ hidden',
             color=C['input_proj'], fontsize=8.5, sublabel_fontsize=6.5,
             bold=False, linewidth=0.8)

    # Dual-Stream Blocks
    ds_y = 6.5
    ds_h = 1.6
    draw_box(ax, BX + 0.3, ds_y, BW - 0.6, ds_h,
             'Dual-Stream Blocks ($\\times N_{double}$)', None,
             color=C['dual_stream'], fontsize=10)
    # Sub-annotations in dual-stream
    ax.text(cx - 0.65, ds_y + 0.6, 'Motion\nStream',
            ha='center', va='center', fontsize=7.5, color='#1B5E20',
            fontweight='bold')
    ax.text(cx + 0.65, ds_y + 0.6, 'Text\nStream',
            ha='center', va='center', fontsize=7.5, color='#1B5E20',
            fontweight='bold')
    # Bidirectional arrow between streams
    ax.annotate('', xy=(cx + 0.15, ds_y + 0.57), xytext=(cx - 0.15, ds_y + 0.57),
                arrowprops=dict(arrowstyle='<->', color='#1B5E20', lw=1.0),
                zorder=4)
    ax.text(cx, ds_y + 0.28, 'Joint Attention',
            ha='center', va='center', fontsize=6.5, color='#1B5E20',
            fontstyle='italic')

    # Arrow: Input Proj → Dual-Stream
    arrow(ax, cx, 8.65, cx, ds_y + ds_h, color=C['arrow_main'], lw=1.8)

    # Single-Stream Blocks
    ss_y = 4.5
    ss_h = 1.45
    draw_box(ax, BX + 0.3, ss_y, BW - 0.6, ss_h,
             'Single-Stream Blocks ($\\times N_{single}$)', None,
             color=C['single_stream'], fontsize=10)
    ax.text(cx, ss_y + 0.45, '[motion; text] $\\to$ Self-Attn $\\to$ FFN',
            ha='center', va='center', fontsize=7.5, color='#145A28',
            fontstyle='italic')

    # Arrow: Dual → Single
    arrow(ax, cx, ds_y, cx, ss_y + ss_h, color=C['arrow_main'], lw=1.8)

    # Flow Velocity Prediction
    fv_y = 3.0
    fv_h = 0.8
    draw_box(ax, BX + 0.5, fv_y, BW - 1.0, fv_h,
             'Flow Velocity Prediction',
             '$v_\\theta$ (B, L, 198)',
             color=C['flow_pred'], fontsize=9, sublabel_fontsize=7.5)

    # Arrow: Single → Flow Vel
    arrow(ax, cx, ss_y, cx, fv_y + fv_h, color=C['arrow_main'], lw=1.8)

    # Timestep + AdaLN (side box)
    draw_box(ax, BX + 0.5, 2.0, BW - 1.0, 0.55,
             'Timestep t + AdaLN',
             '$t \\to$ sinusoidal $\\to$ adaptive LN',
             color=C['timestep'], fontsize=7.5, sublabel_fontsize=5.5,
             bold=False, linewidth=0.6, alpha=0.7)
    # Side arrow from timestep to backbone
    arrow(ax, cx, 2.55, cx + 0.01, fv_y, color='#999', lw=0.8, ls='--')

    # ============================================================
    # CENTER: Output Motion with motion-sequence illustration
    # ============================================================
    out_y = 0.9
    out_h = 0.8
    draw_box(ax, BX + 0.6, out_y, BW - 1.2, out_h,
             '', None,  # we draw the illustration instead
             color=C['output'], fontsize=10, sublabel_fontsize=8,
             border_color=C['output_border'], linewidth=2.0)
    # Motion sequence illustration
    draw_motion_sequence(ax, cx, out_y + out_h * 0.55,
                         n=5, spacing=0.4, scale=1.1)
    # Label
    ax.text(cx, out_y + 0.07, 'Output Motion (B, L, 198)',
            ha='center', va='center', fontsize=7, fontweight='bold',
            color=C['output_border'], zorder=5)

    # Arrow: Flow Vel → Output
    arrow(ax, cx, fv_y, cx, out_y + out_h, color=C['output_border'], lw=2.0)

    # ============================================================
    # ARROWS: Encoders → Backbone
    # ============================================================
    # Text Encoder → Input Projection (horizontal right)
    arrow(ax, LX + LW, text_enc_y + text_enc_h * 0.5, BX + 0.4, 8.9,
          color=C['arrow_text'], lw=2.0)
    ax.text(3.85, 8.6, '$c_{txt}$', fontsize=7, color=C['arrow_text'],
            fontweight='bold', ha='center')

    # VACE Encoder → Input Projection (horizontal right, then up via curve)
    arrow_curved(ax, LX + LW, vace_y + vace_h * 0.6, BX + 0.4, 8.7,
                 rad=-0.25, color=C['arrow_motion'], lw=2.0)
    ax.text(3.85, 6.0, 'cond\n(B,L,594)', fontsize=6, color=C['arrow_motion'],
            fontweight='bold', ha='center')

    # ============================================================
    # RIGHT PANEL 1: PDCT (Training Innovation) — Top Half
    # ============================================================
    pdct_y = 5.6
    pdct_h = 4.3
    pdct_box = FancyBboxPatch(
        (RX, pdct_y), RW, pdct_h,
        boxstyle="round,pad=0.1",
        facecolor=C['pdct_bg'], edgecolor=C['pdct_accent'],
        linewidth=1.5, alpha=0.5, zorder=0, linestyle='--'
    )
    ax.add_patch(pdct_box)

    ax.text(RX + RW / 2, pdct_y + pdct_h - 0.25,
            '[Training] Progressive Density Curriculum Training (PDCT)',
            ha='center', va='center', fontsize=9.5, fontweight='bold',
            color=C['pdct_accent'])

    # Bullet points
    ax.text(RX + 0.3, pdct_y + pdct_h - 0.65,
            'Zero extra parameters — schedule modification only\n\n'
            'Phase A: Low density (E[$\\rho$]$\\approx$0.15)\n'
            '  $\\to$ Text pathway must be established first\n'
            'Phase B: Linear density ramp $\\to$ gradual introduction\n'
            '  $\\to$ Model learns to fuse text + motion conditions\n'
            'Phase C: Full distribution (E[$\\rho$]$\\approx$0.55)\n'
            '  $\\to$ Full conditional generation capability',
            ha='left', va='top', fontsize=7, color=C['text_dark'],
            linespacing=1.35)

    # PDCT curve inset
    draw_pdct_curve(ax, RX + 3.2, pdct_y + 0.2, 3.2, 1.8)

    # Dashed arrow: PDCT → Structured Condition Sampler
    arrow_curved(ax, RX, pdct_y + 1.0, LX + LW, v3_y + v3_h * 0.5,
                 rad=0.3, color=C['pdct_accent'], lw=1.2, ls='--')
    ax.text(5.0, 5.5, 'controls $\\rho$\ndistribution',
            fontsize=6, color=C['pdct_accent'], ha='center',
            fontstyle='italic', rotation=0)

    # ============================================================
    # RIGHT PANEL 2: CPOS (Inference Innovation) — Bottom Half
    # ============================================================
    cp_y = 0.8
    cp_h = 4.4
    cp_box = FancyBboxPatch(
        (RX, cp_y), RW, cp_h,
        boxstyle="round,pad=0.1",
        facecolor=C['cpos_bg'], edgecolor=C['cpos_accent'],
        linewidth=1.5, alpha=0.5, zorder=0, linestyle='--'
    )
    ax.add_patch(cp_box)

    ax.text(RX + RW / 2, cp_y + cp_h - 0.25,
            '[Inference] Condition-Progressive ODE Sampling (CPOS)',
            ha='center', va='center', fontsize=9.5, fontweight='bold',
            color=C['cpos_accent'])

    ax.text(RX + 0.3, cp_y + cp_h - 0.65,
            'Zero extra parameters — inference schedule only\n\n'
            '$w_{text}(t)$: Time-varying text CFG (bell-shaped)\n'
            '  $\\to$ ODE early: strong text guidance → global semantics\n'
            '$\\alpha(t)$: Condition replacement schedule (sigmoid)\n'
            '  $\\to$ ODE late: replace predictions → clean conditions\n'
            'Only 2 forward passes (same cost as standard CFG)\n'
            '$\\alpha(t)$ is post-processing, not CFG — zero extra cost',
            ha='left', va='top', fontsize=7, color=C['text_dark'],
            linespacing=1.35)

    # CPOS curve inset
    draw_cpos_curve(ax, RX + 3.2, cp_y + 0.2, 3.2, 1.75)

    # Dashed arrow: CPOS → Flow Velocity Prediction
    arrow_curved(ax, RX, cp_y + 2.0, BX + BW - 0.5, fv_y + 0.1,
                 rad=-0.15, color=C['cpos_accent'], lw=1.2, ls='--')
    ax.text(9.5, 2.8, 'schedule\n$w_{text}(t)$, $\\alpha(t)$',
            fontsize=6, color=C['cpos_accent'], ha='center',
            fontstyle='italic')

    # ============================================================
    # COLOR LEGEND (bottom strip)
    # ============================================================
    legend_items = [
        ('Input', C['text_input'], '-'),
        ('Encoder', C['text_enc'], '-'),
        ('Backbone', C['backbone_bg'], '-'),
        ('Training (PDCT)', C['pdct_bg'], '--'),
        ('Inference (CPOS)', C['cpos_bg'], '--'),
        ('Output', C['output'], '-'),
    ]
    lx_start = 5.0
    lx_step = 2.0
    ly = 0.35
    for i, (name, color, ls) in enumerate(legend_items):
        lxi = lx_start + i * lx_step
        box = FancyBboxPatch(
            (lxi, ly), 0.35, 0.2,
            boxstyle="round,pad=0.02",
            facecolor=color, edgecolor=C['border'],
            linewidth=0.5, alpha=0.8, linestyle=ls
        )
        ax.add_patch(box)
        ax.text(lxi + 0.55, ly + 0.1, name,
                fontsize=5.5, va='center', ha='left', color=C['text_dark'])

    # ============================================================
    # Save
    # ============================================================
    plt.tight_layout(pad=0.3)
    out_base = 'docs/temp/cdo_fm_pipeline_overview'
    fig.savefig(f'{out_base}.png', dpi=250, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    fig.savefig(f'{out_base}.pdf', bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"Saved: {out_base}.png ({250} dpi) and {out_base}.pdf")
    plt.close()


if __name__ == '__main__':
    main()
