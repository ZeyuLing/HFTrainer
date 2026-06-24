#!/usr/bin/env python3
"""Verify bundled PhysFlow tracker baselines.

This is the quick guardrail for the ref_repo migration. It verifies that the
three tracker baselines used by PhysFlow resolve to in-repository files and, on
request, runs a tiny Any2Track MuJoCo+ONNX rollout smoke test.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from physflow_tracker_bundle_paths import (  # noqa: E402
    ANY2TRACK_CONFIG,
    ANY2TRACK_G1_MJCF,
    ANY2TRACK_ONNX,
    ANY2TRACK_ROOT,
    HUMANOID_GPT_ONNX,
    HUMANOID_GPT_ROOT,
    HUMANOID_GPT_VENV_PYTHON,
    PROTOMOTIONS_G1_CKPT,
    PROTOMOTIONS_G1_MESH_DIR,
    PROTOMOTIONS_G1_MJCF,
    PROTOMOTIONS_G1_ONNX,
    PROTOMOTIONS_G1_TRACKER_ROOT,
    PROTOMOTIONS_G1_URDF,
    PROTOMOTIONS_ROOT,
)


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _check(label: str, path: Path, want_dir: bool = False) -> dict:
    exists = path.is_dir() if want_dir else path.is_file()
    in_repo = PROJECT_ROOT in path.resolve().parents or path.resolve() == PROJECT_ROOT
    no_ref_repo = "ref_repo" not in path.parts
    ok = bool(exists and in_repo and no_ref_repo)
    return {
        "label": label,
        "path": _rel(path),
        "exists": exists,
        "in_repo": in_repo,
        "no_ref_repo": no_ref_repo,
        "ok": ok,
    }


def _print_table(rows: list[dict]) -> None:
    width = max(len(r["label"]) for r in rows)
    for row in rows:
        status = "OK" if row["ok"] else "FAIL"
        print(f"{status:4} {row['label']:<{width}}  {_rel(PROJECT_ROOT / row['path']) if not Path(row['path']).is_absolute() else row['path']}")


def _import_smoke() -> list[dict]:
    rows = []
    sys.path.insert(0, str(PROTOMOTIONS_ROOT))
    checks = [
        ("tracker", "protomotions.motion_lib", "protomotions.components.motion_lib"),
        ("tracker", "protomotions.deployment", "deployment.motion_utils"),
        ("tracker", "any2track.rollout", "scripts.embodied.eval_opentrack_onnx_mujoco"),
        ("hftrainer", "physflow.rewards", "hftrainer.models.motion.physflow.reward"),
        ("hftrainer", "physflow.any2track_reward", "hftrainer.models.motion.physflow.any2track_reward"),
        ("hftrainer", "physflow.hgpt_reward", "hftrainer.models.motion.physflow.hgpt_reward"),
    ]
    for kind, label, module in checks:
        try:
            __import__(module)
            rows.append({"kind": kind, "label": label, "ok": True, "error": ""})
        except Exception as exc:  # noqa: BLE001
            rows.append({"kind": kind, "label": label, "ok": False, "error": f"{type(exc).__name__}: {exc}"})
    return rows


def _check_hgpt_venv() -> dict:
    if not HUMANOID_GPT_VENV_PYTHON.exists():
        return {
            "label": "humanoid_gpt.venv_imports",
            "ok": False,
            "error": f"missing {_rel(HUMANOID_GPT_VENV_PYTHON)}",
        }
    code = "import jax,mujoco,onnxruntime,flax,scipy,tyro,tree; print('ok')"
    proc = subprocess.run(
        [str(HUMANOID_GPT_VENV_PYTHON), "-c", code],
        cwd=str(HUMANOID_GPT_ROOT),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=120,
    )
    return {
        "label": "humanoid_gpt.venv_imports",
        "ok": proc.returncode == 0,
        "error": proc.stdout.strip()[-500:] if proc.returncode else "",
    }


def _any2track_smoke(max_steps: int) -> dict:
    from scripts.embodied.eval_opentrack_onnx_mujoco import OpenTrackRollout

    lafan = PROJECT_ROOT / "data" / "LAFAN1_Retargeted_for_G1" / "UnitreeG1"
    motions = sorted(lafan.glob("*.npz"))
    if not motions:
        return {
            "label": "any2track.rollout_smoke",
            "ok": False,
            "error": f"no LAFAN1-G1 npz files in {_rel(lafan)}",
        }
    runner = OpenTrackRollout(
        ANY2TRACK_G1_MJCF,
        json.loads(ANY2TRACK_CONFIG.read_text()),
        ANY2TRACK_ONNX,
    )
    row = runner.evaluate_motion(motions[0], max_steps=max_steps)
    return {
        "label": "any2track.rollout_smoke",
        "ok": "mpjpe_mm" in row and "success" in row,
        "error": "",
        "motion": motions[0].name,
        "metrics": {
            "success": row.get("success"),
            "mpjpe_mm": row.get("mpjpe_mm"),
            "mpjve_mps": row.get("mpjve_mps"),
            "root_err_mean": row.get("root_err_mean"),
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--any2track-smoke", action="store_true")
    ap.add_argument("--any2track-max-steps", type=int, default=8)
    ap.add_argument("--check-hgpt-venv", action="store_true")
    ap.add_argument(
        "--skip-hftrainer-imports",
        action="store_true",
        help="Use in tracker-only envs such as the IsaacGym py3.8 venv that do not carry mmengine.",
    )
    args = ap.parse_args()

    path_rows = [
        _check("protomotions.root", PROTOMOTIONS_ROOT, want_dir=True),
        _check("protomotions.g1_tracker_root", PROTOMOTIONS_G1_TRACKER_ROOT, want_dir=True),
        _check("protomotions.g1_onnx", PROTOMOTIONS_G1_ONNX),
        _check("protomotions.g1_ckpt", PROTOMOTIONS_G1_CKPT),
        _check("protomotions.g1_mjcf", PROTOMOTIONS_G1_MJCF),
        _check("protomotions.g1_urdf", PROTOMOTIONS_G1_URDF),
        _check("protomotions.g1_mesh_dir", PROTOMOTIONS_G1_MESH_DIR, want_dir=True),
        _check(
            "protomotions.train_exp",
            PROTOMOTIONS_G1_TRACKER_ROOT / "experiment_config.py",
        ),
        _check("any2track.root", ANY2TRACK_ROOT, want_dir=True),
        _check("any2track.onnx", ANY2TRACK_ONNX),
        _check("any2track.config", ANY2TRACK_CONFIG),
        _check("any2track.g1_mjcf", ANY2TRACK_G1_MJCF),
        _check("humanoid_gpt.root", HUMANOID_GPT_ROOT, want_dir=True),
        _check("humanoid_gpt.onnx", HUMANOID_GPT_ONNX),
        _check("humanoid_gpt.worker", HUMANOID_GPT_ROOT / "physflow_hgpt_judge_server.py"),
        _check("humanoid_gpt.pyproject", HUMANOID_GPT_ROOT / "pyproject.toml"),
        _check("humanoid_gpt.deploy", HUMANOID_GPT_ROOT / "deploy", want_dir=True),
        _check("humanoid_gpt.projects", HUMANOID_GPT_ROOT / "projects", want_dir=True),
        _check(
            "lafan1_g1.data",
            PROJECT_ROOT / "data" / "LAFAN1_Retargeted_for_G1" / "UnitreeG1",
            want_dir=True,
        ),
    ]
    print("== path checks ==")
    _print_table(path_rows)

    import_rows = _import_smoke()
    print("\n== import checks ==")
    for row in import_rows:
        skipped = args.skip_hftrainer_imports and row.get("kind") == "hftrainer" and not row["ok"]
        status = "OK" if row["ok"] else ("SKIP" if skipped else "FAIL")
        suffix = "" if row["ok"] else f"  {row['error']}"
        print(f"{status:4} {row['label']}{suffix}")

    extra_rows = []
    if args.check_hgpt_venv:
        extra_rows.append(_check_hgpt_venv())
    if args.any2track_smoke:
        extra_rows.append(_any2track_smoke(args.any2track_max_steps))
    if extra_rows:
        print("\n== runtime checks ==")
        for row in extra_rows:
            status = "OK" if row["ok"] else "FAIL"
            print(f"{status:4} {row['label']}")
            if row.get("motion"):
                print(f"     motion={row['motion']} metrics={row.get('metrics')}")
            if row.get("error"):
                print(f"     {row['error']}")

    failed = [r for r in path_rows if not r["ok"]]
    failed += [
        r for r in import_rows
        if not r["ok"] and not (args.skip_hftrainer_imports and r.get("kind") == "hftrainer")
    ]
    failed += [r for r in extra_rows if not r["ok"]]
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
