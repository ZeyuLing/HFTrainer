#!/usr/bin/env python3
"""Launch KAFS ablation: single mode, single GPU, foreground.

Usage:
    CUDA_VISIBLE_DEVICES=0 python3 scripts/eval/launch_kafs_single.py none 200
    CUDA_VISIBLE_DEVICES=1 python3 scripts/eval/launch_kafs_single.py depth_driven 200
"""
import subprocess, sys, os

mode = sys.argv[1] if len(sys.argv) > 1 else "none"
max_samples = sys.argv[2] if len(sys.argv) > 2 else "200"

root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(root)

cmd = [
    sys.executable, "scripts/eval/eval_prism_kafs_ablation.py",
    "--config", "configs/prism/prism_1b_tp2m_multiframe.py",
    "--checkpoint", "work_dirs/prism_1b_tp2m_multiframe/checkpoint-iter_15000",
    "--kafs-mode", mode,
    "--anno-file", "data/annotation/test_motionhub_t2m.json",
    "--data-dir", "data/motionhub",
    "--output-dir", "work_dirs/prism_kafs_ablation",
    "--max-samples", max_samples,
    "--num-inference-steps", "50",
    "--seed", "42",
]
print(f"[launch_kafs_single] mode={mode}, max_samples={max_samples}")
print(f"[launch_kafs_single] cmd: {' '.join(cmd)}")
sys.exit(subprocess.call(cmd))
