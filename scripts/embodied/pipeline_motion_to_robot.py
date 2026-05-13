#!/usr/bin/env python3
"""End-to-end pipeline: HyMotion eval output -> Robot motion cache.

Chains three conversion steps:
  1. motion_135 NPZ -> SMPL-X NPZ  (motion135_to_smplx.py)
  2. SMPL-X NPZ -> GMR Robot PKL   (gmr_retarget_headless.py)
  3. GMR PKL -> ProtoMotions cache  (gmr_to_protomotions.py)

Usage:
    python scripts/embodied/pipeline_motion_to_robot.py \
        --input work_dirs/.../npz/00000.npz \
        --output data/embodied_debug/robot_cache.pt \
        [--actual-human-height 1.8] \
        [--robot unitree_g1] \
        [--keep-intermediates]

Optionally run ONNX tracker validation:
    python scripts/embodied/pipeline_motion_to_robot.py \
        --input work_dirs/.../npz/00000.npz \
        --output data/embodied_debug/robot_cache.pt \
        --validate
"""
import argparse
import os
import sys
import pathlib
import subprocess
import tempfile
import shutil

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
GMR_ROOT = PROJECT_ROOT / "ref_repo" / "GMR"
PROTOMOTIONS_ROOT = PROJECT_ROOT / "ref_repo" / "ProtoMotions"

# Default paths
DEFAULT_MJCF = PROTOMOTIONS_ROOT / "protomotions" / "data" / "assets" / "mjcf" / "g1_holo_compat.xml"
DEFAULT_ONNX = PROTOMOTIONS_ROOT / "data" / "pretrained_models" / "motion_tracker" / "g1-bones-deploy" / "compiled_models" / "unified_pipeline.onnx"


def run_step(cmd, step_name):
    """Run a pipeline step and check for errors."""
    print(f"\n{'='*60}")
    print(f"  Step: {step_name}")
    print(f"{'='*60}")
    print(f"  CMD: {' '.join(str(c) for c in cmd)}\n")

    result = subprocess.run(
        [str(c) for c in cmd],
        cwd=str(PROJECT_ROOT),
        capture_output=False,  # let output stream to console
    )
    if result.returncode != 0:
        print(f"\nERROR: {step_name} failed with return code {result.returncode}")
        sys.exit(1)
    print(f"\n  {step_name} completed successfully.")


def main():
    parser = argparse.ArgumentParser(
        description="End-to-end pipeline: HyMotion eval output -> Robot motion cache"
    )
    parser.add_argument("--input", required=True,
                        help="Input motion_135 NPZ from HyMotion eval")
    parser.add_argument("--output", required=True,
                        help="Output ProtoMotions .pt cache file")
    parser.add_argument("--robot", default="unitree_g1",
                        help="Target robot (default: unitree_g1)")
    parser.add_argument("--actual-human-height", type=float, default=None,
                        help="Override human height for GMR (default: auto from betas)")
    parser.add_argument("--tgt-fps", type=int, default=30,
                        help="Target FPS for GMR retargeting (default: 30)")
    parser.add_argument("--control-dt", type=float, default=0.02,
                        help="ProtoMotions control dt (default: 0.02 = 50Hz)")
    parser.add_argument("--mjcf", default=str(DEFAULT_MJCF),
                        help="Path to ProtoMotions G1 MJCF XML")
    parser.add_argument("--keep-intermediates", action="store_true",
                        help="Keep intermediate files (SMPL-X NPZ, GMR PKL)")
    parser.add_argument("--validate", action="store_true",
                        help="Run ONNX tracker validation after conversion")
    parser.add_argument("--onnx", default=str(DEFAULT_ONNX),
                        help="Path to ONNX model for validation")
    parser.add_argument("--no-fk-ground-correction", action="store_true",
                        help="Disable FK-based ground correction")
    parser.add_argument("--fk-ground-mode", default="global",
                        choices=["global", "smooth", "perframe"],
                        help="FK ground correction mode (default: global)")
    parser.add_argument("--ground-clearance", type=float, default=0.0,
                        help="Target min foot Z after FK correction (default: 0.0)")
    parser.add_argument("--no-smooth", action="store_true",
                        help="Disable temporal smoothing in gmr_to_protomotions")
    args = parser.parse_args()

    input_path = pathlib.Path(args.input).resolve()
    output_path = pathlib.Path(args.output).resolve()

    if not input_path.exists():
        print(f"ERROR: Input file not found: {input_path}")
        sys.exit(1)

    # Create output directory
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Determine intermediate file paths
    if args.keep_intermediates:
        stem = output_path.stem
        inter_dir = output_path.parent
        smplx_path = inter_dir / f"{stem}_smplx.npz"
        gmr_pkl_path = inter_dir / f"{stem}_gmr.pkl"
    else:
        tmpdir = tempfile.mkdtemp(prefix="embodied_pipeline_")
        smplx_path = pathlib.Path(tmpdir) / "smplx.npz"
        gmr_pkl_path = pathlib.Path(tmpdir) / "gmr_retarget.pkl"

    try:
        # Step 1: motion_135 -> SMPL-X
        run_step([
            sys.executable, SCRIPT_DIR / "motion135_to_smplx.py",
            str(input_path), str(smplx_path),
            "--fps", str(args.tgt_fps),
        ], "motion_135 -> SMPL-X NPZ")

        # Step 2: SMPL-X -> GMR Robot PKL
        gmr_cmd = [
            sys.executable, SCRIPT_DIR / "gmr_retarget_headless.py",
            "--smplx_file", str(smplx_path),
            "--robot", args.robot,
            "--save_path", str(gmr_pkl_path),
            "--tgt_fps", str(args.tgt_fps),
            "--no-offset-to-ground",  # FK correction handles grounding
        ]
        if args.actual_human_height is not None:
            gmr_cmd += ["--actual-human-height", str(args.actual_human_height)]
        run_step(gmr_cmd, "SMPL-X -> GMR Robot PKL")

        # Step 3: GMR PKL -> ProtoMotions cache
        proto_cmd = [
            sys.executable, SCRIPT_DIR / "gmr_to_protomotions.py",
            "--input", str(gmr_pkl_path),
            "--output", str(output_path),
            "--mjcf", str(args.mjcf),
            "--control-dt", str(args.control_dt),
        ]
        if args.no_fk_ground_correction:
            proto_cmd.append("--no-fk-ground-correction")
        proto_cmd += ["--fk-ground-mode", args.fk_ground_mode]
        if args.ground_clearance != 0.0:
            proto_cmd += ["--ground-clearance", str(args.ground_clearance)]
        if args.no_smooth:
            proto_cmd.append("--no-smooth")
        run_step(proto_cmd, "GMR PKL -> ProtoMotions cache")

        print(f"\n{'='*60}")
        print(f"  Pipeline complete!")
        print(f"{'='*60}")
        print(f"  Input:  {input_path}")
        print(f"  Output: {output_path}")
        if args.keep_intermediates:
            print(f"  SMPL-X: {smplx_path}")
            print(f"  GMR:    {gmr_pkl_path}")

        # Step 4 (optional): ONNX tracker validation
        if args.validate:
            if not pathlib.Path(args.onnx).exists():
                print(f"\nWARNING: ONNX model not found: {args.onnx}")
                print("  Skipping validation.")
            else:
                tracker_script = PROTOMOTIONS_ROOT / "deployment" / "test_tracker_mujoco.py"
                run_step([
                    sys.executable, str(tracker_script),
                    "--onnx", str(args.onnx),
                    "--motion", str(output_path),
                    "--loops", "1",
                    "--no-realtime",
                ], "ONNX Tracker Validation")

    finally:
        if not args.keep_intermediates:
            # Clean up temp files
            if 'tmpdir' in locals() and os.path.isdir(tmpdir):
                shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    main()
