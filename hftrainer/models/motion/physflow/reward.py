"""Frozen-judge physics reward for the PhysFlow online-adversarial loop.

The generator (KIMODO-G1) is scored by how well a *frozen* G1 tracker can
physically execute the generated motion in MuJoCo. This reward is deliberately
decoupled from any tracker that we co-train, so the generator cannot game its
own evaluator (the "judge tracker" of the adversarial design).

It reuses the already-validated scoring stack:
  qpos CSV  --convert-->  ProtoMotions .motion  --MuJoCo+ONNX-->  stats
  stats  -->  compute_g1_adversarial_score   (lower == more trackable == better)

All of this runs in the KIMODO py3.10 env (MuJoCo + onnxruntime), so the whole
online loop lives in a single process -- no IsaacGym required for the reward.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[4]
PROTOMOTIONS_ROOT = PROJECT_ROOT / "ref_repo" / "ProtoMotions"
_SCRIPTS = PROJECT_ROOT / "scripts" / "embodied"
for _p in (str(PROJECT_ROOT), str(PROTOMOTIONS_ROOT), str(_SCRIPTS)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


class PhysicsJudgeReward:
    """Score generated G1 motions with a frozen MuJoCo + ONNX tracker.

    Args:
        onnx_path: Frozen judge tracker (defaults to released ``g1-bones-deploy``).
        mjcf_path: G1 MuJoCo model.
        input_fps / output_fps: CSV / motion fps (KIMODO-G1 is 30).
        error_penalty: score assigned when a motion fails to convert/simulate.
    """

    def __init__(
        self,
        onnx_path: Optional[str] = None,
        mjcf_path: Optional[str] = None,
        input_fps: int = 30,
        output_fps: int = 30,
        robot_json_subsample: int = 4,
        error_penalty: float = 5.0,
        convert_python: Optional[str] = None,
        judges: Optional[List[Dict[str, object]]] = None,
    ) -> None:
        from scripts.embodied.run_g1_rl_tracker_export import DEFAULT_MJCF, DEFAULT_ONNX

        # ---- judge ensemble -------------------------------------------------
        # ``judges`` is an optional list of {"onnx", "mjcf"?, "weight"?, "name"?}.
        # When given, every generated motion is rolled out under EACH judge and
        # the per-judge scores are combined (weighted mean); acceptance gating
        # (no-fall + completion) is taken CONSERVATIVELY across the ensemble so a
        # motion only counts as trackable if *all* judges can execute it. This is
        # what lets the online-adversarial loop swap / blend the frozen released
        # tracker with the co-trained trainee (see physflow_coevolve_orchestrator).
        # With no ``judges`` (the default) behaviour is identical to a single
        # frozen judge at ``onnx_path``.
        if judges:
            self._judges = []
            for j in judges:
                self._judges.append({
                    "onnx": str(j["onnx"]),
                    "mjcf": str(j.get("mjcf") or mjcf_path or DEFAULT_MJCF),
                    "weight": float(j.get("weight", 1.0)),
                    "name": str(j.get("name", os.path.basename(os.path.dirname(str(j["onnx"]))) or "judge")),
                })
        else:
            self._judges = [{
                "onnx": str(onnx_path or DEFAULT_ONNX),
                "mjcf": str(mjcf_path or DEFAULT_MJCF),
                "weight": 1.0,
                "name": "frozen",
            }]
        # primary judge (used for frame-count / control_dt lookup)
        self.onnx_path = self._judges[0]["onnx"]
        self.mjcf_path = self._judges[0]["mjcf"]
        self.input_fps = input_fps
        self.output_fps = output_fps
        self.robot_json_subsample = robot_json_subsample
        self.error_penalty = float(error_penalty)
        # The CSV->.motion converter imports dm_control + protomotions, which
        # live in the IsaacGym py38 venv (not the KIMODO py310 training env).
        self.convert_python = convert_python or os.environ.get(
            "PHYSFLOW_CONVERT_PYTHON", "/root/physflow_isaacgym_py38_cu118/bin/python"
        )

        from scripts.embodied.physflow_g1_scoring import (
            DEFAULT_G1_SCORE_CONFIG,
            compute_g1_adversarial_score,
        )
        from scripts.embodied.run_g1_rl_tracker_export import (
            parse_body_mesh_mapping,
            simulate_and_export,
        )

        self._score_config = DEFAULT_G1_SCORE_CONFIG
        self._compute_score = compute_g1_adversarial_score
        self._simulate = simulate_and_export
        self._parse_body_mesh_mapping = parse_body_mesh_mapping
        # cache body<->mesh mapping per (unique) mjcf so multi-judge ensembles
        # that share the G1 model don't re-parse the XML for every judge.
        self._mesh_cache: Dict[str, object] = {}
        self._body_mesh_mapping = self._mesh_mapping_for(self.mjcf_path)

    def _mesh_mapping_for(self, mjcf_path: str):
        if mjcf_path not in self._mesh_cache:
            self._mesh_cache[mjcf_path] = self._parse_body_mesh_mapping(Path(mjcf_path))
        return self._mesh_cache[mjcf_path]

    @classmethod
    def from_spec_file(cls, spec_path: str, **kwargs) -> "PhysicsJudgeReward":
        """Build a reward from a JSON judge spec written by the co-evolution
        orchestrator: ``{"judges": [{"onnx": ..., "weight": ...}, ...]}``."""
        import json

        with open(spec_path) as f:
            spec = json.load(f)
        return cls(judges=spec.get("judges"), **kwargs)

    # --- CSV dir -> .motion dir (one subprocess for the whole batch) ---
    def _convert_csv_dir(self, csv_dir: Path, proto_dir: Path) -> None:
        cmd = [
            self.convert_python,
            "data/scripts/convert_g1_csv_to_proto.py",
            "--input-dir", str(csv_dir.resolve()),
            "--output-dir", str(proto_dir.resolve()),
            "--input-fps", str(self.input_fps),
            "--output-fps", str(self.output_fps),
            "--pos-units", "m",
            "--rot-format", "quat_wxyz",
            "--joint-units", "rad",
            "--no-has-header",
            "--no-has-frame-column",
            "--force-remake",
        ]
        env = _os_environ()
        env["MUJOCO_GL"] = "disable"
        # The convert script imports the ``protomotions`` package, so the
        # subprocess needs PROTOMOTIONS_ROOT on PYTHONPATH (we only have it on
        # this process's sys.path, which children don't inherit).
        extra_paths = [str(PROTOMOTIONS_ROOT), str(PROJECT_ROOT)]
        env["PYTHONPATH"] = os.pathsep.join(extra_paths + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
        log_path = proto_dir.parent / "convert.log"
        with open(log_path, "w") as lf:
            subprocess.run(cmd, cwd=str(PROTOMOTIONS_ROOT), env=env, check=True,
                           stdout=lf, stderr=subprocess.STDOUT)

    def _expected_frames(self, motion_path: Path) -> int:
        import yaml
        from deployment.motion_utils import MotionPlayer

        with open(self.onnx_path.replace(".onnx", ".yaml")) as f:
            meta = yaml.safe_load(f)
        control_dt = meta["timing"]["control_dt"]
        return int(MotionPlayer(str(motion_path), control_dt=control_dt).total_frames)

    def score_motion_file(self, motion_path: Path, out_json: Path) -> Dict[str, float]:
        """Roll out the motion under EVERY judge and combine.

        - ``score``: weighted mean of per-judge adversarial scores (lower=better).
        - ``completion``: min across judges (conservative).
        - ``fall_detected``: any across judges (conservative -> a motion is only
          'trackable' if no judge in the ensemble falls).
        Per-judge breakdown is returned under ``per_judge`` for logging.
        """
        total_frames = self._expected_frames(motion_path)
        out_json = Path(out_json)
        per_judge: Dict[str, Dict[str, float]] = {}
        wsum = 0.0
        score_acc = 0.0
        completions: List[float] = []
        falls: List[bool] = []
        joint_errs: List[float] = []
        traj_errs: List[float] = []
        for ji, j in enumerate(self._judges):
            jout = out_json if len(self._judges) == 1 else out_json.with_name(
                f"{out_json.stem}__{j['name']}{out_json.suffix}"
            )
            stats = self._simulate(
                onnx_path=j["onnx"],
                motion_file=str(motion_path),
                output_json_path=str(jout),
                mjcf_path=j["mjcf"],
                body_mesh_mapping=self._mesh_mapping_for(j["mjcf"]),
                subsample_factor=self.robot_json_subsample,
            )
            completion = float(stats["total_steps"] / max(total_frames, 1))
            s = float(self._compute_score(
                completion=completion,
                max_joint_error_rad=float(stats.get("max_joint_error_rad", 0.0)),
                root_trajectory_error_mean_m=float(stats.get("root_trajectory_error_mean_m", 0.0)),
                root_displacement_error_m=float(stats.get("root_displacement_error_m", 0.0)),
                fall_detected=bool(stats.get("fall_detected", False)),
                config=self._score_config,
            ))
            w = float(j["weight"])
            score_acc += w * s
            wsum += w
            completions.append(completion)
            falls.append(bool(stats.get("fall_detected", False)))
            joint_errs.append(float(stats.get("max_joint_error_rad", 0.0)))
            traj_errs.append(float(stats.get("root_trajectory_error_mean_m", 0.0)))
            per_judge[j["name"]] = {"score": s, "completion": completion,
                                    "fall_detected": falls[-1]}
        result = {
            "score": float(score_acc / max(wsum, 1e-9)),
            "completion": float(min(completions)),
            "max_joint_error_rad": float(max(joint_errs)),
            "fall_detected": bool(any(falls)),
            "root_trajectory_error_mean_m": float(max(traj_errs)),
        }
        if len(self._judges) > 1:
            result["per_judge"] = per_judge
        return result

    def score_csv_dir(self, csv_dir: Path, work_dir: Path) -> Dict[str, Dict[str, float]]:
        """Convert all CSVs in ``csv_dir`` and score each. Returns {stem: metrics}.

        Adversarial score is LOWER == better (more trackable). Failures get
        ``error_penalty`` so they are never selected as best-of-N.
        """
        csv_dir = Path(csv_dir)
        work_dir = Path(work_dir)
        proto_dir = work_dir / "proto"
        json_dir = work_dir / "json"
        proto_dir.mkdir(parents=True, exist_ok=True)
        json_dir.mkdir(parents=True, exist_ok=True)

        results: Dict[str, Dict[str, float]] = {}
        stems = sorted(p.stem for p in csv_dir.glob("*.csv"))
        try:
            self._convert_csv_dir(csv_dir, proto_dir)
        except Exception as exc:  # convert failed for the whole batch
            for stem in stems:
                results[stem] = {"score": self.error_penalty, "error": f"convert: {exc}"}
            return results

        for stem in stems:
            motions = sorted(proto_dir.glob(f"{stem}*.motion"))
            if not motions:
                results[stem] = {"score": self.error_penalty, "error": "no .motion"}
                continue
            try:
                results[stem] = self.score_motion_file(motions[0], json_dir / f"{stem}.json")
            except Exception as exc:
                results[stem] = {"score": self.error_penalty, "error": str(exc)}
        return results


def _os_environ() -> Dict[str, str]:
    import os

    return os.environ.copy()
