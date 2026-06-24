"""Build a test anno whose num_frames = GT humanml3d_272 length per canonical id.

The model generates exactly num_frames (gen/req=1.0), and ours' anno num_frames is
~0.84x the GT-272 length on the eval set. Overriding num_frames with the true GT
length makes ours natively generate at the GT time-base (no post-hoc resample).
Canonical id = basename(smplx_path) stem (same mapping as repack_pred_to_272ids).
Entries whose canonical id has no GT-272 file keep their original num_frames.
"""
import json, os, numpy as np

REPO = "/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer"
SRC = os.path.join(REPO, "data/annotation/test_hml3d.json")
DST = os.path.join(REPO, "data/annotation/test_hml3d_gtlen.json")
GT = "/dev/shm/ms272_data/motion_data" if os.path.isdir("/dev/shm/ms272_data/motion_data") \
     else os.path.join(REPO, "ref_repo/MotionStreamer/MotionStreamer/humanml3d_272/motion_data")

LMAP = os.path.join(REPO, "data/annotation/test_hml3d_gtlen_lenmap.json")

raw = json.load(open(SRC))
dl = raw["data_list"]
n_over = n_keep = 0
ratios = []
len_map = {}
for pid, e in dl.items():
    can = os.path.splitext(os.path.basename(e.get("smplx_path", "")))[0]
    g = os.path.join(GT, can + ".npy")
    if can and os.path.exists(g):
        gl = int(np.load(g, mmap_mode="r").shape[0])
        old = e.get("num_frames")
        if old:
            ratios.append(old / gl)
        e["num_frames"] = gl
        len_map[pid] = gl
        n_over += 1
    else:
        n_keep += 1
json.dump(raw, open(DST, "w"))
json.dump(len_map, open(LMAP, "w"))
print(f"len_map -> {LMAP}")
r = np.array(ratios)
print(f"overridden={n_over} kept={n_keep} -> {DST}")
print(f"old/GT length ratio: mean={r.mean():.3f} med={np.median(r):.3f} "
      f"(<1 means old anno was shorter than GT)")
