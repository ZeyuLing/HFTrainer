#!/usr/bin/env python3
"""Render FBX orig vs cleaned side-by-side comparison video (headless, no GPU needed).

Uses matplotlib to draw stick figure from FBX bone positions, outputs MP4.
"""
import fbx
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path
import subprocess, tempfile, os, sys

FBX_DIR = Path('data/lightai_data/CJGame_MB/raw')

# Standard skeleton hierarchy for stick figure
SKELETON_BONES = {
    'hips','spine','spine1','spine2','spine3','neck','neck1','head',
    'leftshoulder','leftarm','leftforearm','lefthand',
    'rightshoulder','rightarm','rightforearm','righthand',
    'leftupleg','leftleg','leftfoot','lefttoebase',
    'rightupleg','rightleg','rightfoot','righttoebase',
}

SKELETON_LINKS = [
    ('Hips','Spine'),('Spine','Spine1'),('Spine1','Spine2'),('Spine2','Spine3'),
    ('Spine3','Neck'),('Neck','Neck1'),('Neck1','Head'),
    ('Spine3','LeftShoulder'),('LeftShoulder','LeftArm'),('LeftArm','LeftForeArm'),('LeftForeArm','LeftHand'),
    ('Spine3','RightShoulder'),('RightShoulder','RightArm'),('RightArm','RightForeArm'),('RightForeArm','RightHand'),
    ('Hips','LeftUpLeg'),('LeftUpLeg','LeftLeg'),('LeftLeg','LeftFoot'),('LeftFoot','LeftToeBase'),
    ('Hips','RightUpLeg'),('RightUpLeg','RightLeg'),('RightLeg','RightFoot'),('RightFoot','RightToeBase'),
]

def load_fbx_positions(path, fps=30):
    """Load bone world positions from FBX."""
    manager = fbx.FbxManager.Create()
    ios = fbx.FbxIOSettings.Create(manager, "IOSRoot")
    manager.SetIOSettings(ios)
    importer = fbx.FbxImporter.Create(manager, "")
    if not importer.Initialize(str(path), -1, manager.GetIOSettings()):
        manager.Destroy(); return None, None
    scene = fbx.FbxScene.Create(manager, "s")
    importer.Import(scene)
    stack = scene.GetCurrentAnimationStack()
    if not stack:
        scene.Destroy(); manager.Destroy(); return None, None
    ts = stack.GetLocalTimeSpan()
    start = ts.GetStart().GetSecondDouble()
    stop = ts.GetStop().GetSecondDouble()
    n_frames = int(round((stop - start) * fps))

    bones = {}
    def collect(node):
        name = node.GetName()
        short = name.split(':')[-1] if ':' in name else name
        if short.lower() in SKELETON_BONES:
            bones[short] = node
        for i in range(node.GetChildCount()):
            collect(node.GetChild(i))
    collect(scene.GetRootNode())

    positions = {}  # bone_name -> (n_frames, 3)
    for name, node in bones.items():
        pos = np.zeros((n_frames, 3))
        for fi in range(n_frames):
            t = fbx.FbxTime()
            t.SetSecondDouble(start + fi / fps)
            gt = node.EvaluateGlobalTransform(t)
            tr = gt.GetT()
            pos[fi] = [tr[0], tr[1], tr[2]]
        positions[name] = pos

    scene.Destroy(); manager.Destroy()
    return positions, n_frames


def render_comparison_video(base_name, output_path, fps=30, max_frames=None):
    """Render orig vs cleaned side-by-side video."""
    print(f'Loading {base_name}...')
    pos_o, nf_o = load_fbx_positions(FBX_DIR / f'{base_name}.fbx', fps)
    pos_c, nf_c = load_fbx_positions(FBX_DIR / f'{base_name}_cleaned.fbx', fps)
    if pos_o is None or pos_c is None:
        print(f'  Failed to load'); return False

    nf = min(nf_o, nf_c)
    if max_frames:
        nf = min(nf, max_frames)

    # Find bounding box for consistent view
    all_pts = []
    for pos in [pos_o, pos_c]:
        for name, p in pos.items():
            all_pts.append(p[:nf])
    all_pts = np.concatenate(all_pts, axis=0)
    center = np.mean(all_pts, axis=0)
    span = np.max(np.abs(all_pts - center)) * 1.3

    tmpdir = tempfile.mkdtemp()
    print(f'  Rendering {nf} frames...')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), facecolor='#1a1a2e')

    for fi in range(nf):
        for ax, pos, title, color in [
            (ax1, pos_o, 'Original', '#60a5fa'),
            (ax2, pos_c, 'Cleaned', '#34d399'),
        ]:
            ax.clear()
            ax.set_facecolor('#1a1a2e')
            ax.set_xlim(center[0] - span, center[0] + span)
            ax.set_ylim(center[1] - span * 0.3, center[1] + span * 1.2)
            ax.set_aspect('equal')
            ax.set_title(title, color=color, fontsize=14, fontweight='bold')
            ax.tick_params(colors='#444')
            ax.spines[:].set_color('#333')

            # Draw links (front view: X=horizontal, Y=vertical)
            for a, b in SKELETON_LINKS:
                if a in pos and b in pos:
                    pa = pos[a][fi]
                    pb = pos[b][fi]
                    ax.plot([pa[0], pb[0]], [pa[1], pb[1]], color=color, linewidth=2, alpha=0.8)

            # Draw joints
            for name, p in pos.items():
                ax.plot(p[fi, 0], p[fi, 1], 'o', color=color, markersize=4, alpha=0.9)

        fig.suptitle(f'{base_name}  |  Frame {fi}/{nf}', color='#999', fontsize=10)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        fig.savefig(f'{tmpdir}/frame_{fi:05d}.png', dpi=100, facecolor=fig.get_facecolor())

        if fi % 100 == 0:
            print(f'    {fi}/{nf}')

    plt.close(fig)

    # ffmpeg
    print(f'  Encoding MP4...')
    # Try encoders in order of preference
    for encoder, extra_args in [
        ('h264_nvenc', ['-preset', 'fast', '-cq', '22']),
        ('libx264', ['-preset', 'fast', '-crf', '22']),
        ('mpeg4', ['-q:v', '5']),
    ]:
        result = subprocess.run([
            'ffmpeg', '-y', '-framerate', str(fps),
            '-i', f'{tmpdir}/frame_%05d.png',
            '-c:v', encoder, '-pix_fmt', 'yuv420p',
            *extra_args, '-movflags', '+faststart',
            str(output_path)
        ], capture_output=True, text=True)
        if result.returncode == 0:
            break
    if result.returncode != 0:
        print(f'  ffmpeg error: {result.stderr[-300:]}')
        for f in Path(tmpdir).glob('*.png'):
            f.unlink()
        os.rmdir(tmpdir)
        return False

    # Cleanup
    for f in Path(tmpdir).glob('*.png'):
        f.unlink()
    os.rmdir(tmpdir)
    print(f'  Saved: {output_path}')
    return True


if __name__ == '__main__':
    cases = [
        # Significant
        ('SLDF_DualBlades_Stand_Run_FR_001', 'significant'),
        ('SLDF_DualBlades_Stand_Run_F_001', 'significant'),
        # Minor
        ('DAD_Spear_Crouch_Turn_L_001', 'minor'),
        ('Player_Mojiang_Lv1_Namaste_Throw_Start_002', 'minor'),
        # Finger only
        ('Boss_Mojiang_Lv1_Buddha_Attack_004', 'finger_only'),
        # Length changed
        ('DAD_KuangWarrior_Skill04_005', 'length_changed'),
    ]

    out_dir = Path('docs/temp/demo_videos')
    out_dir.mkdir(exist_ok=True)

    for name, cat in cases:
        out_path = out_dir / f'{cat}_{name}.mp4'
        if out_path.exists():
            print(f'Skip existing: {out_path}')
            continue
        render_comparison_video(name, out_path, fps=30, max_frames=300)
