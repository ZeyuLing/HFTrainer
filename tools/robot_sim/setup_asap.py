#!/usr/bin/env python3
"""Setup helper for ASAP/HumanoidVerse environment.

This script helps set up the external dependencies needed for the
Isaac Gym → Sim2Real pipeline:
  1. Clone and install ASAP
  2. Download Isaac Gym Preview 4
  3. Download G1 URDF/MJCF models
  4. Verify the installation

Usage:
    python tools/robot_sim/setup_asap.py --install-dir ~/ASAP
    python tools/robot_sim/setup_asap.py --check-only
"""

import argparse
import os
import subprocess
import sys


def check_python_packages():
    """Check required Python packages."""
    packages = {
        'torch': 'PyTorch',
        'numpy': 'NumPy',
        'scipy': 'SciPy',
    }
    optional = {
        'isaacgym': 'Isaac Gym Preview 4',
        'mujoco': 'MuJoCo',
        'hydra': 'Hydra (ASAP config)',
        'imageio': 'ImageIO (video export)',
    }

    print('=== Required Packages ===')
    all_ok = True
    for pkg, name in packages.items():
        try:
            __import__(pkg)
            print(f'  ✓ {name} ({pkg})')
        except ImportError:
            print(f'  ✗ {name} ({pkg}) — MISSING')
            all_ok = False

    print('\n=== Optional Packages ===')
    for pkg, name in optional.items():
        try:
            __import__(pkg)
            print(f'  ✓ {name} ({pkg})')
        except ImportError:
            print(f'  · {name} ({pkg}) — not installed')

    return all_ok


def install_asap(install_dir):
    """Clone and install ASAP."""
    if os.path.exists(install_dir):
        print(f'ASAP directory already exists: {install_dir}')
        return

    print(f'Cloning ASAP to {install_dir}...')
    subprocess.run([
        'git', 'clone', 'https://github.com/LeCAR-Lab/ASAP.git', install_dir,
    ], check=True)

    print('Installing ASAP...')
    subprocess.run([
        sys.executable, '-m', 'pip', 'install', '-e', install_dir,
    ], check=True)

    isaac_utils_dir = os.path.join(install_dir, 'isaac_utils')
    if os.path.exists(isaac_utils_dir):
        subprocess.run([
            sys.executable, '-m', 'pip', 'install', '-e', isaac_utils_dir,
        ], check=True)

    print('ASAP installed successfully.')


def print_isaac_gym_instructions():
    """Print Isaac Gym installation instructions."""
    print("""
=== Isaac Gym Installation ===

Isaac Gym Preview 4 must be installed manually:

1. Download from: https://developer.nvidia.com/isaac-gym-preview-4
   (requires NVIDIA developer account)

2. Extract and install:
   tar -xvzf IsaacGym_Preview_4_Package.tar.gz
   cd isaacgym/python
   pip install -e .

3. If you get libpython errors:
   export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

4. Verify:
   python -c "import isaacgym; print('Isaac Gym OK')"

=== Alternative: Isaac Lab (newer) ===

Isaac Lab is the successor to Isaac Gym with modern APIs:
   git clone https://github.com/isaac-sim/IsaacLab.git
   cd IsaacLab && ./isaaclab.sh --install

Note: Isaac Gym is still required by ASAP. Isaac Lab can be used
for newer projects but is not yet compatible with ASAP's codebase.
""")


def print_pipeline_overview():
    """Print the full pipeline overview."""
    print("""
═══════════════════════════════════════════════════════════
  HyMotion T2M → G1 Robot Pipeline Overview
═══════════════════════════════════════════════════════════

  ┌─────────────────────┐
  │  Text Prompt        │  "a person walks forward"
  └──────────┬──────────┘
             │
  ┌──────────▼──────────┐
  │  HyMotion T2M-Lite  │  Flow matching, 0.46B MMDiT
  │  (this repo)        │  → SMPL 22-joint rot6d motion
  └──────────┬──────────┘
             │  135-dim or 201-dim per frame
  ┌──────────▼──────────┐
  │  SMPL → G1 Retarget │  Joint mapping + Euler decomp
  │  (this repo)        │  → G1 29-DOF joint angles
  └──────────┬──────────┘
             │  .pkl reference motion file
  ┌──────────▼──────────┐
  │  Isaac Gym / ASAP   │  PPO-based motion tracking
  │  (external)         │  4096 parallel environments
  └──────────┬──────────┘
             │  trained .pt / .onnx policy
  ┌──────────▼──────────┐
  │  Sim2Sim (MuJoCo)   │  Cross-simulator validation
  │  (external)         │
  └──────────┬──────────┘
             │  validated policy
  ┌──────────▼──────────┐
  │  Sim2Real (Unitree) │  Real G1 hardware deployment
  │  (external)         │  via Unitree SDK + ROS2
  └─────────────────────┘

Steps 1-2 are fully implemented in this repo.
Steps 3-5 use ASAP (open-source, CMU LeCAR Lab).

Quick Start:
  # Generate + retarget
  python tools/robot_sim/text_to_g1.py \\
    --prompt "a person walks forward slowly" \\
    --config configs/robot_sim/g1_motion_tracking.py \\
    --checkpoint checkpoints/HY-Motion-1.0/HY-Motion-1.0-Lite/latest.ckpt \\
    --output output/g1_walk/ \\
    --generate-train-cmd --asap-root ~/ASAP

═══════════════════════════════════════════════════════════
""")


def main():
    parser = argparse.ArgumentParser(description='Setup ASAP for G1 motion imitation')
    parser.add_argument('--install-dir', type=str, default=os.path.expanduser('~/ASAP'),
                       help='Directory to install ASAP')
    parser.add_argument('--check-only', action='store_true',
                       help='Only check installation, do not install')
    parser.add_argument('--overview', action='store_true',
                       help='Print pipeline overview')
    args = parser.parse_args()

    if args.overview:
        print_pipeline_overview()
        return

    print('Checking Python packages...\n')
    all_ok = check_python_packages()

    if args.check_only:
        print('\n' + ('All required packages OK.' if all_ok else 'Some packages missing.'))
        print_isaac_gym_instructions()
        return

    if not all_ok:
        print('\nSome required packages are missing. Install them first.')
        return

    install_asap(args.install_dir)
    print_isaac_gym_instructions()
    print_pipeline_overview()


if __name__ == '__main__':
    main()
