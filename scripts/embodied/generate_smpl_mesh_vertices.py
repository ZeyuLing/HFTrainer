"""Generate binary vertex files from SMPL mesh JSON files for fast web loading.

Output format per motion:
  Header: num_frames (uint32) | num_vertices (uint32) | fps (float32)
  Per-frame: 6890 * 3 * float16 vertex data

Also saves faces.bin (topology): 13776 * 3 * uint32

Usage:
    python scripts/embodied/generate_smpl_mesh_vertices.py \
        --input-dir output/embodied_t2m_v4/data/smpl_mesh \
        --output-dir output/embodied_t2m_v4/data/smpl_vertices \
        --smpl-model-path checkpoints/smpl_models
"""

import argparse
import json
import struct
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

try:
    import smplx
except ImportError:
    raise ImportError("smplx is required. Install with: pip install smplx")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate binary SMPL mesh vertices from JSON param files"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="output/embodied_t2m_v4/data/smpl_mesh",
        help="Directory containing SMPL mesh JSON files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output/embodied_t2m_v4/data/smpl_vertices",
        help="Output directory for binary vertex files",
    )
    parser.add_argument(
        "--smpl-model-path",
        type=str,
        default="checkpoints/smpl_models",
        help="Path to SMPL model directory",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Max frames per batch for FK (to control GPU memory)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for SMPL forward kinematics",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip processing if output file already exists",
    )
    return parser.parse_args()


def load_smpl_model(model_path: str, gender: str, device: str):
    """Load SMPL-H model with given gender."""
    model = smplx.create(
        model_path,
        model_type="smplh",
        gender=gender,
        use_pca=False,
        batch_size=1,
    )
    model = model.to(device)
    model.eval()
    return model


def process_motion(
    json_path: Path,
    output_path: Path,
    smpl_models: dict,
    model_path: str,
    device: str,
    batch_size: int,
) -> bool:
    """Process a single motion JSON file and save binary vertices.

    Returns True if processed successfully, False otherwise.
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    fps = data.get("fps", 30)
    frames = data["frames"]
    num_frames = len(frames)

    if num_frames == 0:
        print(f"  [WARN] No frames in {json_path.name}, skipping.")
        return False

    # Collect SMPL params from all frames
    # Each frame is a list of agents; we take the first agent
    global_orients = []
    body_poses = []
    left_hand_poses = []
    right_hand_poses = []
    transls = []
    betas_list = []
    gender = None

    for frame_agents in frames:
        if len(frame_agents) == 0:
            print(f"  [WARN] Empty frame in {json_path.name}, skipping file.")
            return False
        agent = frame_agents[0]

        # Get gender (assume consistent across frames)
        if gender is None:
            gender = agent.get("gender", "neutral")

        # Parse params - each is [1, N] shaped
        rh = agent["Rh"][0]  # [3]
        th = agent["Th"][0]  # [3]
        poses = agent["poses"][0]  # [156]
        shapes = agent["shapes"][0]  # [16]

        # global_orient from Rh (axis-angle, 3 values)
        global_orients.append(rh)
        # body_pose: poses[3:66] (21 joints x 3 = 63 values)
        body_poses.append(poses[3:66])
        # left_hand_pose: poses[66:111] (15 joints x 3 = 45 values)
        left_hand_poses.append(poses[66:111])
        # right_hand_pose: poses[111:156] (15 joints x 3 = 45 values)
        right_hand_poses.append(poses[111:156])
        # transl
        transls.append(th)
        # betas: take first 10
        betas_list.append(shapes[:10])

    # Load or get cached SMPL model
    if gender not in smpl_models:
        smpl_models[gender] = load_smpl_model(model_path, gender, device)
    body_model = smpl_models[gender]

    # Convert to tensors
    global_orient_t = torch.tensor(global_orients, dtype=torch.float32, device=device)
    body_pose_t = torch.tensor(body_poses, dtype=torch.float32, device=device)
    left_hand_pose_t = torch.tensor(left_hand_poses, dtype=torch.float32, device=device)
    right_hand_pose_t = torch.tensor(right_hand_poses, dtype=torch.float32, device=device)
    transl_t = torch.tensor(transls, dtype=torch.float32, device=device)
    betas_t = torch.tensor(betas_list, dtype=torch.float32, device=device)

    # Process in batches
    all_vertices = []
    for start in range(0, num_frames, batch_size):
        end = min(start + batch_size, num_frames)

        with torch.no_grad():
            output = body_model(
                global_orient=global_orient_t[start:end],
                body_pose=body_pose_t[start:end],
                left_hand_pose=left_hand_pose_t[start:end],
                right_hand_pose=right_hand_pose_t[start:end],
                transl=transl_t[start:end],
                betas=betas_t[start:end],
            )

        # vertices: [batch, 6890, 3]
        vertices = output.vertices.cpu().numpy()
        all_vertices.append(vertices)

    all_vertices = np.concatenate(all_vertices, axis=0)  # [num_frames, 6890, 3]
    num_vertices = all_vertices.shape[1]

    # Save binary file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        # Header: num_frames (uint32), num_vertices (uint32), fps (float32)
        f.write(struct.pack("<I", num_frames))
        f.write(struct.pack("<I", num_vertices))
        f.write(struct.pack("<f", float(fps)))
        # Per-frame vertex data as float16
        vertices_f16 = all_vertices.astype(np.float16)
        f.write(vertices_f16.tobytes())

    return True


def save_faces(body_model, output_dir: Path):
    """Save the SMPL face topology as a binary file."""
    faces = body_model.faces  # numpy array [13776, 3] int
    faces_uint32 = faces.astype(np.uint32)

    faces_path = output_dir / "faces.bin"
    with open(faces_path, "wb") as f:
        f.write(faces_uint32.tobytes())

    print(f"Saved faces topology: {faces_path} "
          f"({faces_uint32.shape[0]} triangles, "
          f"{faces_path.stat().st_size} bytes)")


def main():
    args = parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all JSON files
    json_files = sorted(input_dir.glob("*.json"))
    if not json_files:
        print(f"No JSON files found in {input_dir}")
        return

    print(f"Found {len(json_files)} JSON files in {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Device: {args.device}")
    print(f"Batch size: {args.batch_size}")
    print(f"Skip existing: {args.skip_existing}")

    # Cache for SMPL models (keyed by gender)
    smpl_models = {}

    # Save faces topology (using neutral model)
    faces_path = output_dir / "faces.bin"
    if not faces_path.exists():
        print("Loading SMPL-H model for faces topology...")
        if "neutral" not in smpl_models:
            smpl_models["neutral"] = load_smpl_model(
                args.smpl_model_path, "neutral", args.device
            )
        save_faces(smpl_models["neutral"], output_dir)
    else:
        print(f"Faces topology already exists: {faces_path}")

    # Process each motion file
    processed = 0
    skipped = 0
    failed = 0

    for json_path in tqdm(json_files, desc="Processing motions"):
        output_path = output_dir / (json_path.stem + ".bin")

        if args.skip_existing and output_path.exists():
            skipped += 1
            continue

        try:
            success = process_motion(
                json_path=json_path,
                output_path=output_path,
                smpl_models=smpl_models,
                model_path=args.smpl_model_path,
                device=args.device,
                batch_size=args.batch_size,
            )
            if success:
                processed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"  [ERROR] Failed to process {json_path.name}: {e}")
            failed += 1

    print(f"\nDone! Processed: {processed}, Skipped: {skipped}, Failed: {failed}")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
