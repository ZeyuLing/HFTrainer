#!/usr/bin/env python3
"""Quick GT velocity check - compute body_pose velocity from training data."""
import os
import sys
import json
import numpy as np

sys.path.insert(0, '/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer')


def compute_vel(arr):
    """Mean frame-to-frame L2 velocity."""
    if arr.ndim == 3:
        arr = arr[0]
    return float(np.linalg.norm(np.diff(arr, axis=0), axis=1).mean())


def main():
    # Load a few GT motions from motionhub
    motionhub_root = '/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer/data/motionhub'

    # Load train.json to find motion paths
    train_json = os.path.join(motionhub_root, 'train.json')
    with open(train_json, 'r') as f:
        train_data = json.load(f)

    data_list = train_data.get('data_list', {})

    print(f"Total training samples: {len(data_list)}")

    # Find some humanml3d samples
    hml3d_samples = [(k, v) for k, v in data_list.items() if 'humanml3d' in k.lower()]
    print(f"HumanML3D samples: {len(hml3d_samples)}")

    if not hml3d_samples:
        # Try any sample
        hml3d_samples = list(data_list.items())[:20]

    vels_bp = []
    vels_tr = []

    for name, info in hml3d_samples[:20]:
        path = info.get('motion_path', '')
        if not path:
            continue
        full_path = os.path.join(motionhub_root, path) if not os.path.isabs(path) else path
        if not os.path.isfile(full_path):
            continue

        data = np.load(full_path, allow_pickle=True)

        if 'body_pose' in data:
            bp = data['body_pose']
            if bp.shape[0] > 2:
                vel = compute_vel(bp)
                vels_bp.append(vel)

        if 'transl' in data:
            tr = data['transl']
            if tr.shape[0] > 2:
                vels_tr.append(compute_vel(tr))

        if len(vels_bp) >= 10:
            break

    if vels_bp:
        print(f"\nGT body_pose velocity (axis-angle, L2 per frame):")
        print(f"  samples: {len(vels_bp)}")
        print(f"  mean: {np.mean(vels_bp):.5f}")
        print(f"  std: {np.std(vels_bp):.5f}")
        print(f"  min: {np.min(vels_bp):.5f}")
        print(f"  max: {np.max(vels_bp):.5f}")

    if vels_tr:
        print(f"\nGT transl velocity:")
        print(f"  mean: {np.mean(vels_tr):.5f}")
        print(f"  min: {np.min(vels_tr):.5f}")
        print(f"  max: {np.max(vels_tr):.5f}")

    # Also check the data format
    if hml3d_samples:
        name, info = hml3d_samples[0]
        path = info.get('motion_path', '')
        full_path = os.path.join(motionhub_root, path) if not os.path.isabs(path) else path
        if os.path.isfile(full_path):
            data = np.load(full_path, allow_pickle=True)
            print(f"\nSample '{name}' keys: {list(data.files)}")
            for k in data.files:
                arr = data[k]
                if hasattr(arr, 'shape'):
                    print(f"  {k}: shape={arr.shape}, dtype={arr.dtype}")


if __name__ == '__main__':
    main()
