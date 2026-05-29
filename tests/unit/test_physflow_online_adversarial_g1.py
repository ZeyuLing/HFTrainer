import json
from argparse import Namespace

from scripts.embodied.physflow_online_adversarial_g1 import (
    CandidateRecord,
    maybe_update_tracker,
    score_candidate,
    stage_tracker_pool,
    summarize_round,
    tracker_artifact_metadata,
    update_generator_from_hard_cases,
)


def _round_args(**overrides):
    values = dict(
        hard_cases_for_generator=2,
        hard_min_score_for_generator=1.0,
        good_cases_for_tracker=2,
        good_min_completion=0.95,
        good_max_joint_error=0.7,
        good_max_root_trajectory_error=0.25,
        good_max_root_displacement_error=0.35,
        allow_tracker_pool_without_root_metrics=False,
        g1_onnx="old_tracker.onnx",
    )
    values.update(overrides)
    return Namespace(**values)


def _tracker_args(**overrides):
    values = dict(
        run_tracker_update=False,
        min_tracker_motions_for_update=2,
        tracker_python="/usr/bin/python3",
        tracker_experiment="tracker_exp.py",
        g1_tracker_ckpt="tracker.ckpt",
        tracker_simulator="isaacgym",
        tracker_num_envs=16,
        tracker_batch_size=32,
        tracker_steps=5,
        tracker_save_every=0,
        export_tracker_after_update=False,
        activate_exported_tracker=False,
        tracker_export_python=None,
        tracker_export_script="ref_repo/ProtoMotions/deployment/export_bm_tracker_onnx.py",
        validate_tracker_export=False,
        g1_onnx="old_tracker.onnx",
    )
    values.update(overrides)
    return Namespace(**values)


def _score_args(**overrides):
    values = dict(
        g1_onnx="tracker.onnx",
        g1_mjcf="robot.xml",
        g1_urdf="robot.urdf",
        smpl_model_path="smpl",
        pyroki_max_iterations=1,
        retarget_subsample_factor=1,
        retarget_target_raw_frames=120,
        robot_json_subsample=1,
        joint_error_scale=1.0,
        root_trajectory_error_weight=1.0,
        root_trajectory_error_scale=0.5,
        root_displacement_error_weight=0.5,
        root_displacement_error_scale=0.5,
        score_component_cap=2.0,
        fall_penalty=2.0,
        retarget_fail_penalty=5.0,
    )
    values.update(overrides)
    return Namespace(**values)


def test_generator_update_skips_when_no_hard_cases(tmp_path):
    args = Namespace(
        dry_run_updates=False,
        gen_update_steps=2,
        hard_min_score_for_generator=1.0,
    )

    updates = update_generator_from_hard_cases(
        trainer=None,
        hard_cases=[],
        gt_loader=None,
        args=args,
        round_dir=tmp_path,
    )

    skip = json.loads((tmp_path / "generator_update_skipped.json").read_text())
    log_record = json.loads((tmp_path / "generator_updates.jsonl").read_text())

    assert updates == []
    assert skip["status"] == "skipped"
    assert skip["reason"] == "no_hard_cases_above_threshold"
    assert skip["requested_steps"] == 2
    assert log_record == skip
    assert not (tmp_path / "t2m_after_round.pt").exists()


def test_summarize_round_excludes_error_penalties_from_generator_hard_cases(tmp_path):
    onnx = tmp_path / "tracker.onnx"
    yaml = tmp_path / "tracker.yaml"
    onnx.write_text("onnx")
    yaml.write_text("yaml")
    scored_error = CandidateRecord(
        round_idx=0,
        candidate_idx=0,
        prompt="a broken candidate",
        category="locomotion",
        num_frames=0,
        npz_path="broken.npz",
        status="error",
        adversarial_score=5.0,
        error="RuntimeError: simulator missing",
    )
    scored_hard = CandidateRecord(
        round_idx=0,
        candidate_idx=1,
        prompt="a humanoid robot walks forward.",
        category="locomotion",
        num_frames=120,
        npz_path="walk.npz",
        status="scored",
        completion_ratio=1.0,
        max_joint_error_rad=0.5,
        root_trajectory_error_mean_m=1.5,
        root_displacement_error_m=2.8,
        root_metrics_available=True,
        adversarial_score=3.5,
    )

    hard_cases, good_cases = summarize_round(
        [scored_error, scored_hard],
        _round_args(g1_onnx=str(onnx)),
        tmp_path,
        round_idx=0,
    )
    summary = json.loads((tmp_path / "round_summary.json").read_text())

    assert summary["active_g1_tracker"]["g1_onnx_path"] == str(onnx)
    assert summary["active_g1_tracker"]["g1_onnx_md5"] == tracker_artifact_metadata(onnx)["g1_onnx_md5"]
    assert [case.prompt for case in hard_cases] == ["a humanoid robot walks forward."]
    assert good_cases == []
    assert summary["num_hard_eligible"] == 1
    assert summary["num_error_hard_excluded"] == 1
    assert summary["error_hard_excluded"][0]["prompt"] == "a broken candidate"


def test_summarize_round_has_no_hard_cases_when_only_errors_are_high_score(tmp_path):
    scored_error = CandidateRecord(
        round_idx=0,
        candidate_idx=0,
        prompt="a broken candidate",
        category="locomotion",
        num_frames=0,
        npz_path="broken.npz",
        status="error",
        adversarial_score=5.0,
        error="RuntimeError: simulator missing",
    )

    hard_cases, _good_cases = summarize_round(
        [scored_error],
        _round_args(),
        tmp_path,
        round_idx=0,
    )
    summary = json.loads((tmp_path / "round_summary.json").read_text())

    assert hard_cases == []
    assert summary["num_hard_eligible"] == 0
    assert summary["num_error_hard_excluded"] == 1


def test_score_candidate_records_active_tracker_artifacts(tmp_path, monkeypatch):
    onnx = tmp_path / "tracker.onnx"
    yaml = tmp_path / "tracker.yaml"
    motion = tmp_path / "motion.motion"
    onnx.write_text("onnx")
    yaml.write_text("yaml")
    motion.write_text("motion")

    def fake_retarget_npz_to_motion(**_kwargs):
        return motion

    def fake_simulate_and_export(**_kwargs):
        return {
            "total_steps": 10,
            "max_joint_error_rad": 0.2,
            "fall_detected": False,
            "root_height_final": 0.8,
            "root_displacement_ref_m": 0.2,
            "root_displacement_track_m": 0.19,
            "root_displacement_error_m": 0.01,
            "root_trajectory_error_mean_m": 0.02,
            "root_trajectory_error_final_m": 0.02,
        }

    monkeypatch.setattr("scripts.embodied.physflow_online_adversarial_g1.parse_body_mesh_mapping", lambda _path: {})
    monkeypatch.setattr("scripts.embodied.physflow_online_adversarial_g1.retarget_npz_to_motion", fake_retarget_npz_to_motion)
    monkeypatch.setattr("scripts.embodied.physflow_online_adversarial_g1.expected_g1_frames", lambda _motion, _onnx: 10)
    monkeypatch.setattr("scripts.embodied.physflow_online_adversarial_g1.simulate_and_export", fake_simulate_and_export)

    record = CandidateRecord(
        round_idx=0,
        candidate_idx=0,
        prompt="a humanoid robot stands.",
        category="standing",
        num_frames=120,
        npz_path=str(tmp_path / "candidate.npz"),
    )

    scored = score_candidate(record, _score_args(g1_onnx=str(onnx)), tmp_path / "round")

    assert scored.status == "scored"
    assert scored.g1_onnx_path == str(onnx)
    assert scored.g1_onnx_md5 == tracker_artifact_metadata(onnx)["g1_onnx_md5"]
    assert scored.g1_yaml_path == str(yaml)
    assert scored.g1_yaml_md5 == tracker_artifact_metadata(onnx)["g1_yaml_md5"]


def test_stage_tracker_pool_deduplicates_identical_motions(tmp_path):
    motion_a = tmp_path / "a.motion"
    motion_b = tmp_path / "b.motion"
    motion_a.write_text("same motion")
    motion_b.write_text("same motion")
    cases = [
        CandidateRecord(
            round_idx=0,
            candidate_idx=0,
            prompt="a humanoid robot stands.",
            category="standing",
            num_frames=120,
            npz_path="a.npz",
            motion_path=str(motion_a),
            status="scored",
            completion_ratio=1.0,
            max_joint_error_rad=0.2,
            root_trajectory_error_mean_m=0.01,
            root_displacement_error_m=0.01,
            adversarial_score=0.3,
        ),
        CandidateRecord(
            round_idx=0,
            candidate_idx=1,
            prompt="a humanoid robot stands calmly.",
            category="standing",
            num_frames=120,
            npz_path="b.npz",
            motion_path=str(motion_b),
            status="scored",
            completion_ratio=1.0,
            max_joint_error_rad=0.2,
            root_trajectory_error_mean_m=0.01,
            root_displacement_error_m=0.01,
            adversarial_score=0.3,
        ),
    ]

    pool_dir = stage_tracker_pool(cases, tmp_path / "round")
    manifest = json.loads((pool_dir / "manifest.json").read_text())

    assert len(list(pool_dir.glob("*.motion"))) == 1
    assert len(manifest) == 1
    assert manifest[0]["prompt"] == "a humanoid robot stands."
    assert manifest[0]["source_motion_md5"]


def test_tracker_update_skips_when_unique_pool_below_threshold(tmp_path):
    motion = tmp_path / "stand.motion"
    motion.write_text("stand")
    cases = [
        CandidateRecord(
            round_idx=0,
            candidate_idx=0,
            prompt="a humanoid robot stands.",
            category="standing",
            num_frames=120,
            npz_path="stand.npz",
            motion_path=str(motion),
            status="scored",
            completion_ratio=1.0,
            max_joint_error_rad=0.2,
            root_trajectory_error_mean_m=0.01,
            root_displacement_error_m=0.01,
            adversarial_score=0.3,
        ),
    ]

    info = maybe_update_tracker(
        cases,
        _tracker_args(run_tracker_update=True, min_tracker_motions_for_update=2),
        tmp_path / "round",
        round_idx=0,
    )
    saved = json.loads((tmp_path / "round" / "tracker_train_command.json").read_text())

    assert info["num_motions"] == 1
    assert info["min_tracker_motions_for_update"] == 2
    assert info["should_run"] is False
    assert info["skip_reason"] == "below_min_tracker_motions_for_update"
    assert saved["skip_reason"] == "below_min_tracker_motions_for_update"


def test_tracker_update_runs_when_requested_pool_meets_threshold(tmp_path, monkeypatch):
    motion_a = tmp_path / "stand.motion"
    motion_b = tmp_path / "walk.motion"
    motion_a.write_text("stand")
    motion_b.write_text("walk")
    cases = [
        CandidateRecord(
            round_idx=0,
            candidate_idx=0,
            prompt="a humanoid robot stands.",
            category="standing",
            num_frames=120,
            npz_path="stand.npz",
            motion_path=str(motion_a),
            status="scored",
            completion_ratio=1.0,
            max_joint_error_rad=0.2,
            root_trajectory_error_mean_m=0.01,
            root_displacement_error_m=0.01,
            adversarial_score=0.3,
        ),
        CandidateRecord(
            round_idx=0,
            candidate_idx=1,
            prompt="a humanoid robot walks slowly.",
            category="walking",
            num_frames=120,
            npz_path="walk.npz",
            motion_path=str(motion_b),
            status="scored",
            completion_ratio=1.0,
            max_joint_error_rad=0.2,
            root_trajectory_error_mean_m=0.02,
            root_displacement_error_m=0.02,
            adversarial_score=0.4,
        ),
    ]

    calls = []

    class DummyResult:
        returncode = 0

    def fake_run(cmd, cwd, env):
        calls.append({"cmd": cmd, "cwd": cwd, "env": env})
        return DummyResult()

    monkeypatch.setattr("scripts.embodied.physflow_online_adversarial_g1.subprocess.run", fake_run)

    info = maybe_update_tracker(
        cases,
        _tracker_args(run_tracker_update=True, min_tracker_motions_for_update=2),
        tmp_path / "round",
        round_idx=0,
    )

    assert info["num_motions"] == 2
    assert info["should_run"] is True
    assert info["skip_reason"] is None
    assert info["returncode"] == 0
    assert "--motion-file" in info["cmd"]
    assert len(calls) == 1


def test_tracker_update_exports_and_activates_new_onnx(tmp_path, monkeypatch):
    proto_root = tmp_path / "ProtoMotions"
    proto_root.mkdir()
    monkeypatch.setattr("scripts.embodied.physflow_online_adversarial_g1.PROTOMOTIONS_ROOT", proto_root)

    motion_a = tmp_path / "stand.motion"
    motion_b = tmp_path / "walk.motion"
    motion_a.write_text("stand")
    motion_b.write_text("walk")
    cases = [
        CandidateRecord(
            round_idx=0,
            candidate_idx=0,
            prompt="a humanoid robot stands.",
            category="standing",
            num_frames=120,
            npz_path="stand.npz",
            motion_path=str(motion_a),
            status="scored",
            completion_ratio=1.0,
            max_joint_error_rad=0.2,
            root_trajectory_error_mean_m=0.01,
            root_displacement_error_m=0.01,
            adversarial_score=0.3,
        ),
        CandidateRecord(
            round_idx=0,
            candidate_idx=1,
            prompt="a humanoid robot walks slowly.",
            category="walking",
            num_frames=120,
            npz_path="walk.npz",
            motion_path=str(motion_b),
            status="scored",
            completion_ratio=1.0,
            max_joint_error_rad=0.2,
            root_trajectory_error_mean_m=0.02,
            root_displacement_error_m=0.02,
            adversarial_score=0.4,
        ),
    ]
    args = _tracker_args(
        run_tracker_update=True,
        min_tracker_motions_for_update=2,
        export_tracker_after_update=True,
        activate_exported_tracker=True,
        tracker_export_script=str(proto_root / "deployment" / "export_bm_tracker_onnx.py"),
    )
    calls = []

    class DummyResult:
        returncode = 0

    def fake_run(cmd, cwd, env):
        calls.append({"cmd": cmd, "cwd": cwd, "env": env})
        run_dir = proto_root / "results" / "physflow_online_g1_r00"
        if len(calls) == 1:
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "last.ckpt").write_text("checkpoint")
        else:
            export_dir = run_dir / "compiled_models"
            export_dir.mkdir(parents=True, exist_ok=True)
            (export_dir / "unified_pipeline.onnx").write_text("onnx")
            (export_dir / "unified_pipeline.yaml").write_text("yaml")
        return DummyResult()

    monkeypatch.setattr("scripts.embodied.physflow_online_adversarial_g1.subprocess.run", fake_run)

    info = maybe_update_tracker(cases, args, tmp_path / "round", round_idx=0)

    assert len(calls) == 2
    assert info["tracker_export_status"] == "completed"
    assert info["exported_tracker_exists"] is True
    assert info["exported_tracker_yaml_exists"] is True
    assert info["activated_g1_onnx"] == str(proto_root / "results" / "physflow_online_g1_r00" / "compiled_models" / "unified_pipeline.onnx")
    assert args.g1_onnx == info["activated_g1_onnx"]
