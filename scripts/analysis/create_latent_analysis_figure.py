"""
Create a compact 2-panel figure for the PRISM paper's latent analysis section.
Panel (a): Per-token/channel latent std — showing CV normalization effect
Panel (b): Per-token/channel velocity magnitude — showing balanced velocity targets
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

# Paths
analysis_dir = Path('/apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer/papers/PRISM_TMM2026/analysis_results')
output_dir = Path('/apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer/papers/PRISM_TMM2026/figures')
output_dir.mkdir(parents=True, exist_ok=True)

# Load data
d = np.load(str(analysis_dir / 'analysis_data.npz'), allow_pickle=True)

token_names = d['token_names']
z_std_2d = d['z_std_per_joint_2d']
z_std_1d = d['z_std_per_channel_1d']
v_mag_2d = d['v_mag_2d_per_joint']
v_mag_1d = d['v_mag_1d_per_channel']

# Compute CVs
cv_2d_std = np.std(z_std_2d) / np.mean(z_std_2d)
cv_1d_std = np.std(z_std_1d) / np.mean(z_std_1d)
cv_2d_vel = np.std(v_mag_2d) / np.mean(v_mag_2d)
cv_1d_vel = np.std(v_mag_1d) / np.mean(v_mag_1d)

# Figure style
plt.rcParams.update({
    'font.size': 9,
    'axes.labelsize': 10,
    'axes.titlesize': 10,
    'xtick.labelsize': 7,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.dpi': 300,
    'font.family': 'sans-serif',
})

# Colors
color_2d = '#2196F3'  # Blue
color_1d = '#FF5722'  # Red/Orange

fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.5))

# --- Panel (a): Latent std per token/channel ---
ax = axes[0]
x_2d = np.arange(len(z_std_2d))
x_1d = np.arange(len(z_std_1d))

ax.bar(x_2d, z_std_2d, width=0.7, color=color_2d, alpha=0.8, label=f'2D (CV={cv_2d_std:.3f})')
# For 1D, offset and use different width since different number of channels
# We plot 1D on same axis but as a horizontal band showing range
ax.axhline(y=np.mean(z_std_2d), color=color_2d, linestyle='--', linewidth=1.0, alpha=0.7)
ax.axhline(y=1.0, color='gray', linestyle=':', linewidth=0.8, alpha=0.5, label='$\\mathcal{N}(0,1)$ target')

# Add 1D as shaded range
ax.axhspan(np.min(z_std_1d), np.max(z_std_1d), alpha=0.15, color=color_1d, label=f'1D range (CV={cv_1d_std:.3f})')
ax.axhline(y=np.mean(z_std_1d), color=color_1d, linestyle='--', linewidth=1.0, alpha=0.7)

ax.set_xlabel('Joint token index')
ax.set_ylabel('Latent std $\\sigma$')
ax.set_title('(a) Latent std per token')
ax.set_ylim(0.8, 1.15)
ax.set_xticks([0, 5, 10, 15, 22])
ax.legend(loc='upper right', framealpha=0.9, edgecolor='none')

# Annotate
ax.annotate(f'2D: all joints $\\approx$ 0.92',
           xy=(11, np.mean(z_std_2d)), xytext=(11, 0.88),
           fontsize=7, color=color_2d, ha='center',
           arrowprops=dict(arrowstyle='->', color=color_2d, lw=0.8))

# --- Panel (b): Velocity magnitude per token/channel ---
ax = axes[1]
ax.bar(x_2d, v_mag_2d, width=0.7, color=color_2d, alpha=0.8, label=f'2D (CV={cv_2d_vel:.4f})')
ax.axhline(y=np.mean(v_mag_2d), color=color_2d, linestyle='--', linewidth=1.0, alpha=0.7)

# 1D as shaded range (different scale, so we need a twin axis)
ax2 = ax.twinx()
ax2.bar(x_1d + 0.0, v_mag_1d, width=0.5, color=color_1d, alpha=0.3, label=f'1D (CV={cv_1d_vel:.4f})')
ax2.axhline(y=np.mean(v_mag_1d), color=color_1d, linestyle='--', linewidth=1.0, alpha=0.7)
ax2.set_ylabel('1D velocity mag.', color=color_1d, fontsize=8)
ax2.tick_params(axis='y', colors=color_1d)
ax2.set_ylim(1.0, 1.25)

ax.set_xlabel('Joint token / channel index')
ax.set_ylabel('2D velocity magnitude', color=color_2d)
ax.tick_params(axis='y', colors=color_2d)
ax.set_title(f'(b) Velocity target magnitude')
ax.set_ylim(5.2, 5.55)
ax.set_xticks([0, 5, 10, 15, 22])

# Combined legend
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right', framealpha=0.9, edgecolor='none')

plt.tight_layout()

# Save
for ext in ['pdf', 'png']:
    fig.savefig(str(output_dir / f'fig_latent_analysis.{ext}'), bbox_inches='tight', dpi=300)
    print(f"Saved: {output_dir / f'fig_latent_analysis.{ext}'}")

plt.close()

# Also create a simpler, cleaner version with just bar plots side by side
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8))

# --- Panel (a): Latent std comparison ---
ax = axes[0]

# Group: 2D bars for 23 joints, then gap, then 1D bars for 16 channels
n_2d = len(z_std_2d)
n_1d = len(z_std_1d)
gap = 2
x_2d_pos = np.arange(n_2d)
x_1d_pos = np.arange(n_2d + gap, n_2d + gap + n_1d)

bars_2d = ax.bar(x_2d_pos, z_std_2d, width=0.8, color=color_2d, alpha=0.85, label=f'2D per-joint (CV = {cv_2d_std:.3f})')
bars_1d = ax.bar(x_1d_pos, z_std_1d, width=0.8, color=color_1d, alpha=0.85, label=f'1D per-channel (CV = {cv_1d_std:.3f})')

ax.axhline(y=1.0, color='gray', linestyle=':', linewidth=0.8, alpha=0.6, label='$\\mathcal{N}(0,1)$ target')
ax.axhline(y=np.mean(z_std_2d), color=color_2d, linestyle='--', linewidth=0.8, alpha=0.5)
ax.axhline(y=np.mean(z_std_1d), color=color_1d, linestyle='--', linewidth=0.8, alpha=0.5)

# Add bracket labels
mid_2d = np.mean(x_2d_pos)
mid_1d = np.mean(x_1d_pos)
ax.text(mid_2d, 0.855, '23 joints', ha='center', fontsize=7, color=color_2d)
ax.text(mid_1d, 0.855, '16 channels', ha='center', fontsize=7, color=color_1d)

ax.set_ylim(0.85, 1.15)
ax.set_ylabel('Latent std $\\sigma$')
ax.set_title('(a) Latent standard deviation')
ax.set_xticks([])
ax.legend(loc='upper right', fontsize=7, framealpha=0.9, edgecolor='none')

# Add CV ratio annotation
ax.annotate(f'CV ratio: {cv_1d_std/cv_2d_std:.1f}$\\times$',
           xy=(n_2d + gap//2, 1.08), fontsize=8, ha='center', fontweight='bold',
           bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', edgecolor='gray', alpha=0.8))

# --- Panel (b): Velocity magnitude comparison (normalized for visual comparison) ---
ax = axes[1]

# Normalize both to their means for visual comparison of CV
v_2d_norm = v_mag_2d / np.mean(v_mag_2d)
v_1d_norm = v_mag_1d / np.mean(v_mag_1d)

bars_2d = ax.bar(x_2d_pos, v_2d_norm, width=0.8, color=color_2d, alpha=0.85, label=f'2D per-joint (CV = {cv_2d_vel:.4f})')
bars_1d = ax.bar(x_1d_pos, v_1d_norm, width=0.8, color=color_1d, alpha=0.85, label=f'1D per-channel (CV = {cv_1d_vel:.4f})')

ax.axhline(y=1.0, color='gray', linestyle=':', linewidth=0.8, alpha=0.6)

ax.text(mid_2d, 0.955, '23 joints', ha='center', fontsize=7, color=color_2d)
ax.text(mid_1d, 0.955, '16 channels', ha='center', fontsize=7, color=color_1d)

ax.set_ylim(0.95, 1.06)
ax.set_ylabel('Normalized velocity magnitude')
ax.set_title('(b) Flow-matching velocity targets')
ax.set_xticks([])
ax.legend(loc='upper right', fontsize=7, framealpha=0.9, edgecolor='none')

# Add CV ratio annotation
ax.annotate(f'CV ratio: {cv_1d_vel/cv_2d_vel:.1f}$\\times$',
           xy=(n_2d + gap//2, 1.045), fontsize=8, ha='center', fontweight='bold',
           bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', edgecolor='gray', alpha=0.8))

plt.tight_layout()

for ext in ['pdf', 'png']:
    fig.savefig(str(output_dir / f'fig_latent_analysis.{ext}'), bbox_inches='tight', dpi=300)
    print(f"Saved (v2): {output_dir / f'fig_latent_analysis.{ext}'}")

plt.close()
print("Done!")
