#!/usr/bin/env python3
"""Incremental V4 pipeline: process newly completed PyRoki retargets.

Designed to run on Taiji (needs dm_control, mujoco, ONNX tracker).

Steps for each newly completed retarget:
1. Check if NPZ exists but .motion doesn't → run convert_pyroki_retargeted
2. Check if .motion exists but not in pyroki_ids.txt → rebuild cache/JSON
3. Run physics tracker on new caches
4. Convert tracked caches to JSON
5. Update both manifests

Usage:
    # Run once (process all new completions)
    python3 scripts/embodied/incremental_v4_pipeline.py

    # Run in loop mode (check every N seconds)
    python3 scripts/embodied/incremental_v4_pipeline.py --loop --interval 300

    # Dry run
    python3 scripts/embodied/incremental_v4_pipeline.py --dry-run
"""

import argparse
import glob
import json
import os
import pathlib
import shutil
import subprocess
import sys
import time

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
V4_DIR = PROJECT_ROOT / "output" / "embodied_t2m_v4"
RETARGET_DIR = V4_DIR / "data" / "retarget"
CACHE_DIR = V4_DIR / "data" / "caches"
MOTIONS_DIR = V4_DIR / "data" / "motions"
TRACKED_CACHES_DIR = V4_DIR / "data" / "tracked_caches"
TRACKED_MOTIONS_DIR = V4_DIR / "data" / "tracked_motions"
META_DIR = V4_DIR / "data" / "meta"
PYROKI_IDS_FILE = V4_DIR / "data" / "pyroki_ids.txt"

PROTOMOTIONS_ROOT = PROJECT_ROOT / "ref_repo" / "ProtoMotions"
CONVERT_SCRIPT = PROTOMOTIONS_ROOT / "data" / "scripts" / "convert_pyroki_retargeted_robot_motions_to_proto.py"
TRACKER_SCRIPT = PROJECT_ROOT / "scripts" / "embodied" / "run_tracker_export.py"

# Add project paths
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "embodied"))
sys.path.insert(0, str(PROTOMOTIONS_ROOT))


def get_pyroki_ids():
    """Read current pyroki_ids.txt."""
    if not PYROKI_IDS_FILE.exists():
        return set()
    with open(PYROKI_IDS_FILE) as f:
        return set(line.strip() for line in f if line.strip())


def save_pyroki_ids(ids):
    """Write updated pyroki_ids.txt."""
    with open(PYROKI_IDS_FILE, "w") as f:
        for id_ in sorted(ids):
            f.write(id_ + "\n")


def find_npz_without_motion():
    """Find retarget dirs that have NPZ but no .motion file."""
    results = []
    for d in sorted(RETARGET_DIR.iterdir()):
        if not d.is_dir() or not d.name.startswith("motion_v4_"):
            continue
        # Check for .motion
        motion_files = list(d.glob("*.motion"))
        if motion_files:
            continue
        # Check for NPZ
        npz_files = list(d.glob("intermediates/retargeted/*retargeted*.npz"))
        if npz_files:
            oid = d.name.replace("motion_", "")
            results.append({"id": oid, "dir": str(d), "npz": str(npz_files[0])})
    return results


def find_new_motions():
    """Find .motion files not yet in pyroki_ids.txt."""
    known_ids = get_pyroki_ids()
    results = []
    for d in sorted(RETARGET_DIR.iterdir()):
        if not d.is_dir() or not d.name.startswith("motion_v4_"):
            continue
        oid = d.name.replace("motion_", "")
        if oid in known_ids:
            continue
        motion_files = list(d.glob("*.motion"))
        if motion_files:
            results.append({"id": oid, "dir": str(d), "motion": str(motion_files[0])})
    return results


def step1_convert_npz_to_motion(items, dry_run=False):
    """Convert retarget NPZ → .motion using convert_pyroki_retargeted."""
    if not items:
        return []

    print(f"\n{'='*60}")
    print(f"  Step 1: Convert {len(items)} NPZ → .motion")
    print(f"{'='*60}")

    successes = []
    for item in items:
        oid = item["id"]
        retarget_dir = pathlib.Path(item["dir"]) / "intermediates" / "retargeted"
        output_dir = pathlib.Path(item["dir"])

        print(f"\n  [{oid}] Converting NPZ → .motion")
        if dry_run:
            print(f"    DRY RUN: would convert {item['npz']}")
            successes.append(item)
            continue

        env = os.environ.copy()
        env["PYTHONPATH"] = str(PROTOMOTIONS_ROOT) + ":" + env.get("PYTHONPATH", "")
        # Auto-detect rendering backend: prefer EGL (GPU), fall back to OSMesa (CPU)
        # NOTE: ctypes.CDLL("libEGL.so") and nvidia-smi are both unreliable —
        # EGL can fail without proper NVIDIA EGL ICD even with GPUs present.
        # The ONLY reliable test: actually try dm_control import with each backend.
        mujoco_gl = os.environ.get("MUJOCO_GL", "")
        if not mujoco_gl:
            # Try EGL first (fastest, needs GPU + NVIDIA EGL ICD)
            egl_ok = False
            try:
                _test_env = os.environ.copy()
                _test_env["MUJOCO_GL"] = "egl"
                _test_result = subprocess.run(
                    [sys.executable, "-c",
                     "import os; os.environ['MUJOCO_GL']='egl'; from dm_control import mujoco; print('ok')"],
                    capture_output=True, text=True, timeout=15, env=_test_env,
                )
                egl_ok = _test_result.returncode == 0 and "ok" in _test_result.stdout
            except Exception:
                pass

            if egl_ok:
                mujoco_gl = "egl"
            else:
                # Try OSMesa (software rendering, no GPU needed)
                osmesa_ok = False
                try:
                    _test_env = os.environ.copy()
                    _test_env["MUJOCO_GL"] = "osmesa"
                    _test_result = subprocess.run(
                        [sys.executable, "-c",
                         "import os; os.environ['MUJOCO_GL']='osmesa'; from dm_control import mujoco; print('ok')"],
                        capture_output=True, text=True, timeout=15, env=_test_env,
                    )
                    osmesa_ok = _test_result.returncode == 0 and "ok" in _test_result.stdout
                except Exception:
                    pass

                if osmesa_ok:
                    mujoco_gl = "osmesa"
                else:
                    mujoco_gl = "egl"  # last resort
        env["MUJOCO_GL"] = mujoco_gl

        cmd = [
            sys.executable, str(CONVERT_SCRIPT),
            "--retargeted-motion-dir", str(retarget_dir),
            "--output-dir", str(output_dir),
            "--input-fps", "30",
            "--output-fps", "30",
            "--robot-type", "g1",
            "--force-remake",
        ]

        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=120,
                cwd=str(PROTOMOTIONS_ROOT), env=env,
            )
            if result.returncode == 0:
                # Verify .motion was created
                motion_files = list(output_dir.glob("*.motion"))
                if motion_files:
                    print(f"    OK: {motion_files[0].name}")
                    item["motion"] = str(motion_files[0])
                    successes.append(item)
                else:
                    print(f"    ERROR: No .motion file created")
                    print(f"    stdout: {result.stdout[-500:]}")
            else:
                print(f"    ERROR: rc={result.returncode}")
                print(f"    stderr: {result.stderr[-500:]}")
        except subprocess.TimeoutExpired:
            print(f"    ERROR: timeout (120s)")
        except Exception as e:
            print(f"    ERROR: {e}")

    return successes


def step2_rebuild_caches(items, dry_run=False):
    """Rebuild .pt caches and JSON from .motion files."""
    if not items:
        return []

    print(f"\n{'='*60}")
    print(f"  Step 2: Rebuild caches/JSON for {len(items)} motions")
    print(f"{'='*60}")

    from rebuild_v4_from_motion import convert_motion_to_old_cache, extract_metrics
    from convert_cache_to_json import convert_cache_to_json

    successes = []
    for item in items:
        oid = item["id"]
        motion_path = item.get("motion")
        if not motion_path:
            # Find .motion in retarget dir
            retarget_dir = RETARGET_DIR / f"motion_{oid}"
            motion_files = list(retarget_dir.glob("*.motion"))
            if not motion_files:
                print(f"  [{oid}] ERROR: No .motion file found")
                continue
            motion_path = str(motion_files[0])

        print(f"\n  [{oid}] Rebuilding from {os.path.basename(motion_path)}")
        if dry_run:
            print(f"    DRY RUN")
            successes.append(item)
            continue

        try:
            # 2a. Convert to old-format .pt cache
            cache_path = CACHE_DIR / f"{oid}.pt"
            info = convert_motion_to_old_cache(motion_path, str(cache_path))
            print(f"    .pt cache: {info['num_frames']}f @ {info['fps']}fps, "
                  f"root_z={info['root_z_mean']:.3f}m")

            # 2b. Convert to JSON for web viewer
            json_path = MOTIONS_DIR / f"{oid}.json"
            json_info = convert_cache_to_json(motion_path, str(json_path))

            # Also copy to tracked_motions as placeholder
            tracked_json = TRACKED_MOTIONS_DIR / f"{oid}.json"
            if not tracked_json.exists():
                shutil.copy2(str(json_path), str(tracked_json))

            # 2c. Extract metrics
            metrics = extract_metrics(motion_path)
            meta = {
                "id": oid,
                "num_frames": metrics["num_frames"],
                "fps": metrics["fps"],
                "duration": metrics["duration_s"],
                "metrics": metrics,
                "motion_file": motion_path,
                "source": "pyroki",
            }
            with open(META_DIR / f"{oid}.json", "w") as f:
                json.dump(meta, f, indent=2)

            print(f"    Metrics: root_z={metrics['root_height_mean']:.3f}m, "
                  f"fell={metrics['fell']}, max_jvel={metrics['max_joint_velocity']:.1f}")

            successes.append(item)

        except Exception as e:
            print(f"    ERROR: {e}")
            import traceback
            traceback.print_exc()

    return successes


def step3_run_tracker(items, dry_run=False):
    """Run physics tracker on new caches."""
    if not items:
        return []

    print(f"\n{'='*60}")
    print(f"  Step 3: Run tracker on {len(items)} motions")
    print(f"{'='*60}")

    successes = []
    for item in items:
        oid = item["id"]
        cache_path = CACHE_DIR / f"{oid}.pt"
        tracked_path = TRACKED_CACHES_DIR / f"{oid}.pt"

        if not cache_path.exists():
            print(f"  [{oid}] ERROR: No cache at {cache_path}")
            continue

        print(f"\n  [{oid}] Running tracker...")
        if dry_run:
            print(f"    DRY RUN")
            successes.append(item)
            continue

        cmd = [
            sys.executable, str(TRACKER_SCRIPT),
            "--motion", str(cache_path),
            "--output", str(tracked_path),
        ]

        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=300,
                cwd=str(PROJECT_ROOT),
            )
            if result.returncode == 0 and tracked_path.exists():
                print(f"    OK: tracked cache saved")

                # Extract tracker result from output
                for line in result.stdout.split("\n"):
                    if "SUCCESS" in line or "FELL" in line or "fell" in line.lower():
                        print(f"    {line.strip()}")

                item["tracked"] = True
                successes.append(item)
            else:
                print(f"    ERROR: rc={result.returncode}")
                if result.stderr:
                    print(f"    stderr: {result.stderr[-300:]}")
        except subprocess.TimeoutExpired:
            print(f"    ERROR: timeout (300s)")
        except Exception as e:
            print(f"    ERROR: {e}")

    return successes


def step4_convert_tracked(items, dry_run=False):
    """Convert tracked caches to JSON."""
    if not items:
        return []

    print(f"\n{'='*60}")
    print(f"  Step 4: Convert {len(items)} tracked caches to JSON")
    print(f"{'='*60}")

    from convert_cache_to_json import convert_cache_to_json

    successes = []
    for item in items:
        oid = item["id"]
        tracked_cache = TRACKED_CACHES_DIR / f"{oid}.pt"
        tracked_json = TRACKED_MOTIONS_DIR / f"{oid}.json"

        if not tracked_cache.exists():
            print(f"  [{oid}] No tracked cache, skipping")
            continue

        print(f"  [{oid}] Converting tracked cache → JSON")
        if dry_run:
            successes.append(item)
            continue

        try:
            convert_cache_to_json(str(tracked_cache), str(tracked_json))
            successes.append(item)
        except Exception as e:
            print(f"    ERROR: {e}")

    return successes


def step5_update_manifests(new_ids, dry_run=False):
    """Update both manifests with new PyRoki motions."""
    if not new_ids:
        return

    print(f"\n{'='*60}")
    print(f"  Step 5: Update manifests for {len(new_ids)} new motions")
    print(f"{'='*60}")

    if dry_run:
        print("  DRY RUN")
        return

    import torch

    # Update pyroki_ids.txt
    all_ids = get_pyroki_ids()
    all_ids.update(new_ids)
    save_pyroki_ids(all_ids)
    print(f"  pyroki_ids.txt: {len(all_ids)} total")

    # Rebuild full manifests from all data
    all_motion_ids = set()

    # Collect all motion IDs from caches dir
    for pt_file in sorted(CACHE_DIR.glob("v4_*.pt")):
        all_motion_ids.add(pt_file.stem)

    motions_manifest = []
    tracked_manifest = []

    for mid in sorted(all_motion_ids):
        source = "pyroki" if mid in all_ids else "gmr"

        # Load cache to get basic info
        cache_path = CACHE_DIR / f"{mid}.pt"
        try:
            cache = torch.load(str(cache_path), map_location="cpu", weights_only=False)
            if "num_frames" in cache:
                num_frames = int(cache["num_frames"])
            elif "body_pos" in cache:
                num_frames = cache["body_pos"].shape[0]
            else:
                num_frames = 0
            control_dt = float(cache.get("control_dt", 1.0/30.0))
            fps = round(1.0 / control_dt)
        except Exception:
            num_frames = 0
            fps = 30

        entry = {
            "id": mid,
            "source": source,
            "num_frames": num_frames,
            "fps": fps,
        }

        # Load meta if available
        meta_path = META_DIR / f"{mid}.json"
        if meta_path.exists():
            try:
                with open(meta_path) as f:
                    meta = json.load(f)
                metrics = meta.get("metrics", {})
                entry["root_height_mean"] = metrics.get("root_height_mean", 0)
                entry["fell"] = metrics.get("fell", False)
                entry["fall_frame"] = metrics.get("fall_frame", None)
            except Exception:
                entry["fell"] = False
                entry["fall_frame"] = None
        else:
            entry["fell"] = False
            entry["fall_frame"] = None

        # Tracker info
        tracked_cache = TRACKED_CACHES_DIR / f"{mid}.pt"
        if tracked_cache.exists():
            try:
                tc = torch.load(str(tracked_cache), map_location="cpu", weights_only=False)
                tc_body_pos = tc["body_pos"] if isinstance(tc["body_pos"], type(cache_path)) else tc["body_pos"]
                import numpy as np
                if hasattr(tc_body_pos, 'numpy'):
                    bp = tc_body_pos.numpy()
                else:
                    bp = np.array(tc_body_pos)
                root_z = bp[:, 0, 2]
                fell = bool(np.any(root_z < 0.3))
                fall_frame = int(np.argmax(root_z < 0.3)) if fell else None
                entry["tracker_status"] = "fell" if fell else "success"
                entry["tracker_fall_frame"] = fall_frame
                entry["tracker_root_height_min"] = round(float(np.min(root_z)), 4)
                entry["tracker_num_frames"] = int(tc.get("num_frames", bp.shape[0]))
            except Exception as e:
                entry["tracker_status"] = "error"
                entry["tracker_fall_frame"] = None
        else:
            entry["tracker_status"] = "pending"
            entry["tracker_fall_frame"] = None

        motions_manifest.append(entry)
        tracked_manifest.append(entry.copy())

    # Compute stats
    pyroki_count = sum(1 for m in motions_manifest if m.get("source") == "pyroki")
    gmr_count = sum(1 for m in motions_manifest if m.get("source") == "gmr")
    pyroki_fell = sum(1 for m in motions_manifest if m.get("source") == "pyroki" and m.get("fell"))
    gmr_fell = sum(1 for m in motions_manifest if m.get("source") == "gmr" and m.get("fell"))
    tracker_success = sum(1 for m in motions_manifest if m.get("tracker_status") == "success")
    tracker_fell = sum(1 for m in motions_manifest if m.get("tracker_status") == "fell")

    manifest = {
        "total": len(motions_manifest),
        "pyroki_count": pyroki_count,
        "old_gmr_count": gmr_count,
        "pyroki_fell": pyroki_fell,
        "gmr_fell": gmr_fell,
        "tracker_success": tracker_success,
        "tracker_fell": tracker_fell,
        "motions": motions_manifest,
    }

    with open(MOTIONS_DIR / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    tracked_manifest_data = manifest.copy()
    tracked_manifest_data["motions"] = tracked_manifest
    with open(TRACKED_MOTIONS_DIR / "manifest.json", "w") as f:
        json.dump(tracked_manifest_data, f, indent=2)

    print(f"  Manifests updated: {len(motions_manifest)} total "
          f"({pyroki_count} pyroki, {gmr_count} gmr)")
    print(f"  Tracker: {tracker_success} success, {tracker_fell} fell")


def run_once(dry_run=False):
    """Run one pass of the incremental pipeline."""
    t0 = time.time()

    # Step 1: Find NPZ without .motion and convert
    npz_items = find_npz_without_motion()
    if npz_items:
        print(f"\nFound {len(npz_items)} retargets with NPZ but no .motion:")
        for item in npz_items:
            print(f"  - {item['id']}")
        converted = step1_convert_npz_to_motion(npz_items, dry_run)
    else:
        converted = []
        print("\nNo NPZ awaiting .motion conversion.")

    # Step 2: Find new .motion files and rebuild caches
    new_motions = find_new_motions()
    # Also include just-converted items
    converted_ids = {item["id"] for item in converted}
    new_motion_ids_from_converted = []
    for item in converted:
        if item["id"] not in {m["id"] for m in new_motions}:
            new_motion_ids_from_converted.append(item)
    all_new = new_motions + new_motion_ids_from_converted

    if all_new:
        print(f"\nFound {len(all_new)} new .motion files to process:")
        for item in all_new:
            print(f"  - {item['id']}")
        rebuilt = step2_rebuild_caches(all_new, dry_run)
    else:
        rebuilt = []
        print("\nNo new .motion files to rebuild.")

    # Step 3: Run tracker on new caches
    if rebuilt:
        tracked = step3_run_tracker(rebuilt, dry_run)
    else:
        tracked = []

    # Step 4: Convert tracked caches to JSON
    if tracked:
        step4_convert_tracked(tracked, dry_run)

    # Step 5: Update manifests
    new_ids = {item["id"] for item in rebuilt}
    if new_ids:
        step5_update_manifests(new_ids, dry_run)

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"  Incremental pass complete in {elapsed:.0f}s")
    print(f"  Converted NPZ→.motion: {len(converted)}")
    print(f"  Rebuilt caches/JSON: {len(rebuilt)}")
    print(f"  Tracked: {len(tracked)}")
    print(f"{'='*60}\n")

    return len(converted) + len(rebuilt) + len(tracked)


def main():
    parser = argparse.ArgumentParser(description="Incremental V4 pipeline")
    parser.add_argument("--dry-run", action="store_true", help="Print what would be done")
    parser.add_argument("--loop", action="store_true", help="Run in loop mode")
    parser.add_argument("--interval", type=int, default=300, help="Loop interval in seconds")
    parser.add_argument("--max-iterations", type=int, default=100, help="Max loop iterations")
    args = parser.parse_args()

    # Ensure output dirs exist
    for d in [CACHE_DIR, MOTIONS_DIR, TRACKED_CACHES_DIR, TRACKED_MOTIONS_DIR, META_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    if args.loop:
        print(f"Running in loop mode, interval={args.interval}s, max={args.max_iterations}")
        for i in range(args.max_iterations):
            print(f"\n{'#'*60}")
            print(f"  Iteration {i+1}/{args.max_iterations}")
            print(f"  Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"{'#'*60}")

            processed = run_once(args.dry_run)

            if processed == 0:
                # Check if all motions are done
                total_npz = len(list(pathlib.Path(V4_DIR / "data" / "npz").glob("v4_*.npz")))
                total_pyroki = len(get_pyroki_ids())
                total_gmr = total_npz - total_pyroki  # Not all have been attempted
                total_motion = len(list(RETARGET_DIR.glob("motion_v4_*/*.motion")))

                print(f"\n  Status: {total_motion}/{total_npz} retargets complete "
                      f"({total_pyroki} in pyroki_ids.txt)")

                if total_motion >= total_npz:
                    print(f"\n  All {total_npz} retargets complete! Exiting loop.")
                    break

            print(f"\n  Sleeping {args.interval}s...")
            time.sleep(args.interval)
    else:
        run_once(args.dry_run)


if __name__ == "__main__":
    main()
