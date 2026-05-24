"""Analyze G1 robot STL meshes to understand their coordinate systems."""
import trimesh
import numpy as np
from pathlib import Path

MESH_DIR = Path("/apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer/ref_repo/ProtoMotions/protomotions/data/assets/mesh/G1/")

# Analyze a selection of meshes covering different body parts
meshes_to_analyze = [
    "pelvis.stl",
    "left_hip_pitch_link.stl",
    "left_knee_link.stl",
    "left_ankle_pitch_link.stl",
    "torso_link.stl",
    "left_shoulder_pitch_link.stl",
    "left_elbow_link.stl",
    "head_link.stl",
    "right_hip_pitch_link.stl",
    "waist_yaw_link.stl",
]

print("=" * 80)
print("G1 Robot STL Mesh Coordinate System Analysis")
print("=" * 80)

for fname in meshes_to_analyze:
    fpath = MESH_DIR / fname
    if not fpath.exists():
        print(f"\n[SKIP] {fname} not found")
        continue

    mesh = trimesh.load(fpath)

    vertices = np.array(mesh.vertices)
    bbox_min = vertices.min(axis=0)
    bbox_max = vertices.max(axis=0)
    bbox_size = bbox_max - bbox_min
    centroid = mesh.centroid
    center_of_mass = mesh.center_mass if hasattr(mesh, 'center_mass') else centroid

    # Check if centered near origin
    max_offset = np.abs(centroid).max()
    is_near_origin = max_offset < 0.01  # within 1cm of origin

    print(f"\n{'─' * 80}")
    print(f"  {fname}")
    print(f"{'─' * 80}")
    print(f"  Vertices:       {len(vertices)}")
    print(f"  BBox min:       [{bbox_min[0]:+.6f}, {bbox_min[1]:+.6f}, {bbox_min[2]:+.6f}]")
    print(f"  BBox max:       [{bbox_max[0]:+.6f}, {bbox_max[1]:+.6f}, {bbox_max[2]:+.6f}]")
    print(f"  BBox size:      [{bbox_size[0]:.6f}, {bbox_size[1]:.6f}, {bbox_size[2]:.6f}]")
    print(f"  Centroid:       [{centroid[0]:+.6f}, {centroid[1]:+.6f}, {centroid[2]:+.6f}]")
    print(f"  Center of mass: [{center_of_mass[0]:+.6f}, {center_of_mass[1]:+.6f}, {center_of_mass[2]:+.6f}]")
    print(f"  Near origin?    {'YES (body-local frame)' if is_near_origin else 'NO (offset from origin)'}")
    print(f"  Max offset:     {max_offset:.6f} m")

print(f"\n{'=' * 80}")
print("SUMMARY")
print("=" * 80)

# Now do a quick summary of ALL meshes
print("\nAll STL files - centroid distance from origin:")
print(f"{'File':<45} {'|centroid| (m)':>15} {'Frame'}")
print("-" * 75)

all_stls = sorted(MESH_DIR.glob("*.stl"))
offsets = []
for fpath in all_stls:
    mesh = trimesh.load(fpath)
    centroid = mesh.centroid
    dist = np.linalg.norm(centroid)
    offsets.append(dist)
    frame = "LOCAL" if dist < 0.01 else f"OFFSET ({dist:.4f}m)"
    print(f"  {fpath.name:<43} {dist:>12.6f}   {frame}")

offsets = np.array(offsets)
print(f"\n  Mean centroid distance: {offsets.mean():.6f} m")
print(f"  Max centroid distance:  {offsets.max():.6f} m")
print(f"  Min centroid distance:  {offsets.min():.6f} m")
print(f"  Meshes near origin (<1cm): {(offsets < 0.01).sum()} / {len(offsets)}")
print(f"  Meshes offset (>=1cm):     {(offsets >= 0.01).sum()} / {len(offsets)}")
