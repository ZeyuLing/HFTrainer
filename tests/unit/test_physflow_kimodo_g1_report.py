import json
import os
from pathlib import Path

from scripts.embodied.physflow_kimodo_g1_report import (
    build_next_iteration_plan,
    build_report,
    has_root_metrics,
    root_aware_score,
    write_global_hard_prompt_bank,
    write_next_iteration_script,
    write_global_tracker_pool,
)


def test_report_tracks_root_metric_coverage(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "summary.json").write_text(
        json.dumps(
            {
                "records": [
                    {
                        "prompt_id": "old",
                        "prompt": "old score only",
                        "status": "scored",
                        "completion_ratio": 1.0,
                        "max_joint_error_rad": 0.4,
                    },
                    {
                        "prompt_id": "walk",
                        "prompt": "walk forward",
                        "category": "locomotion",
                        "difficulty": 3,
                        "duration_sec": 5.0,
                        "split": "eval",
                        "seed": 7,
                        "sample_idx": 2,
                        "status": "scored",
                        "completion_ratio": 1.0,
                        "max_joint_error_rad": 0.5,
                        "root_trajectory_error_mean_m": 1.5,
                        "root_displacement_error_m": 2.8,
                        "g1_onnx_path": "/tmp/tracker_a.onnx",
                        "g1_onnx_md5": "abc123",
                        "g1_yaml_path": "/tmp/tracker_a.yaml",
                        "g1_yaml_md5": "def456",
                    },
                ]
            }
        )
    )

    report = build_report([run])
    run_report = report["runs"][0]

    assert run_report["num_root_metric_records"] == 1
    assert run_report["root_metric_coverage"] == 0.5
    assert report["global_hardest"][0]["prompt_id"] == "walk"
    assert report["global_hardest"][0]["root_aware_score"] == 3.5
    assert report["global_hardest"][0]["category"] == "locomotion"
    assert report["global_hardest"][0]["duration_sec"] == 5.0
    assert run_report["num_tracker_artifacts"] == 1
    assert run_report["tracker_artifacts"][0]["g1_onnx_md5"] == "abc123"
    assert report["global_hardest"][0]["g1_onnx_md5"] == "abc123"
    assert report["global_tracker_artifacts"][0]["g1_yaml_md5"] == "def456"


def test_has_root_metrics_and_score_are_independent():
    old_record = {"completion_ratio": 1.0, "max_joint_error_rad": 0.4}
    root_record = {
        "completion_ratio": 1.0,
        "max_joint_error_rad": 0.4,
        "root_trajectory_error_mean_m": 0.5,
        "root_displacement_error_m": 0.0,
    }

    assert not has_root_metrics(old_record)
    assert has_root_metrics(root_record)
    assert root_aware_score(root_record) > root_aware_score(old_record)


def test_write_global_hard_prompt_bank_deduplicates_prompts(tmp_path):
    report = {
        "global_hardest": [
            {
                "prompt_id": "walk_a",
                "prompt": "a humanoid robot walks forward.",
                "category": "locomotion",
                "difficulty": 2,
                "duration_sec": 4.0,
                "root_aware_score": 3.5,
            },
            {
                "prompt_id": "walk_b",
                "prompt": "a humanoid robot walks forward.",
                "category": "locomotion",
                "difficulty": 2,
                "duration_sec": 4.0,
                "root_aware_score": 3.4,
            },
            {
                "prompt_id": "wave",
                "prompt": "a humanoid robot waves.",
                "root_aware_score": 1.2,
            },
        ]
    }

    path = write_global_hard_prompt_bank(report, tmp_path / "hard.jsonl", limit=10)
    records = [json.loads(line) for line in Path(path).read_text().splitlines()]

    assert [record["prompt"] for record in records] == [
        "a humanoid robot walks forward.",
        "a humanoid robot waves.",
    ]
    assert all(record["split"] == "adversarial_hard" for record in records)
    assert records[0]["category"] == "locomotion"
    assert records[0]["difficulty"] == 2
    assert records[0]["tags"][-1] == "score_3.500"


def test_hard_prompt_bank_uses_uncapped_candidate_pool(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    records = []
    for idx in range(12):
        records.append(
            {
                "prompt_id": f"walk_{idx}",
                "prompt": "a humanoid robot walks forward.",
                "category": "locomotion",
                "difficulty": 2,
                "duration_sec": 4.0,
                "status": "scored",
                "completion_ratio": 1.0,
                "max_joint_error_rad": 0.4,
                "root_trajectory_error_mean_m": 1.5 - idx * 0.01,
                "root_displacement_error_m": 2.0,
                "root_metrics_available": True,
            }
        )
    records.append(
        {
            "prompt_id": "turn",
            "prompt": "a humanoid robot turns left.",
            "category": "locomotion",
            "difficulty": 3,
            "duration_sec": 4.0,
            "status": "scored",
            "completion_ratio": 1.0,
            "max_joint_error_rad": 0.4,
            "root_trajectory_error_mean_m": 0.8,
            "root_displacement_error_m": 1.0,
            "root_metrics_available": True,
        }
    )
    (run / "summary.json").write_text(json.dumps({"records": records}))

    report = build_report([run], hard_candidate_limit=20)
    hard_path = write_global_hard_prompt_bank(report, tmp_path / "hard.jsonl", limit=2)
    hard_records = [json.loads(line) for line in hard_path.read_text().splitlines()]

    assert len(report["global_hardest"]) == 10
    assert {record["prompt"] for record in report["global_hardest"]} == {
        "a humanoid robot walks forward."
    }
    assert [record["prompt"] for record in hard_records] == [
        "a humanoid robot walks forward.",
        "a humanoid robot turns left.",
    ]


def test_hard_prompt_bank_excludes_low_score_fillers(tmp_path):
    report = {
        "global_hard_prompt_candidates": [
            {
                "prompt_id": "walk",
                "prompt": "a humanoid robot walks forward.",
                "root_aware_score": 3.5,
            },
            {
                "prompt_id": "stand",
                "prompt": "a humanoid robot stands calmly.",
                "root_aware_score": 0.4,
            },
        ],
        "global_hardest": [],
    }

    path = write_global_hard_prompt_bank(report, tmp_path / "hard.jsonl", limit=5, min_score=1.0)
    records = [json.loads(line) for line in path.read_text().splitlines()]

    assert [record["prompt"] for record in records] == ["a humanoid robot walks forward."]


def test_report_can_write_global_tracker_pool(tmp_path):
    motion = tmp_path / "stand.motion"
    duplicate_motion = tmp_path / "stand_copy.motion"
    motion.write_text("motion")
    duplicate_motion.write_text("motion")
    run = tmp_path / "run"
    run.mkdir()
    (run / "summary.json").write_text(
        json.dumps(
            {
                "records": [
                    {
                        "prompt_id": "stand",
                        "prompt": "a humanoid robot stands.",
                        "status": "scored",
                        "motion_path": str(motion),
                        "completion_ratio": 1.0,
                        "max_joint_error_rad": 0.2,
                        "root_trajectory_error_mean_m": 0.02,
                        "root_displacement_error_m": 0.01,
                        "root_metrics_available": True,
                    },
                    {
                        "prompt_id": "stand_copy",
                        "prompt": "a humanoid robot stands calmly.",
                        "status": "scored",
                        "motion_path": str(duplicate_motion),
                        "completion_ratio": 1.0,
                        "max_joint_error_rad": 0.2,
                        "root_trajectory_error_mean_m": 0.02,
                        "root_displacement_error_m": 0.01,
                        "root_metrics_available": True,
                    },
                ]
            }
        )
    )

    report = build_report([run])
    pool_dir = write_global_tracker_pool(report, tmp_path / "tracker_pool", limit=10)
    manifest = json.loads((pool_dir / "manifest.json").read_text())

    assert report["runs"][0]["num_tracker_pool_eligible"] == 2
    assert len(list(pool_dir.glob("*.motion"))) == 1
    assert manifest[0]["prompt_id"] == "stand"
    assert manifest[0]["source_motion_md5"]


def test_prompt_scoreboard_tracks_latest_and_best_by_prompt(tmp_path):
    old_run = tmp_path / "old"
    new_run = tmp_path / "new"
    old_run.mkdir()
    new_run.mkdir()
    for run, score in [(old_run, 1.5), (new_run, 0.5)]:
        (run / "summary.json").write_text(
            json.dumps(
                {
                    "records": [
                        {
                            "prompt_id": "walk",
                            "prompt": "a humanoid robot walks forward.",
                            "status": "scored",
                            "completion_ratio": 1.0,
                            "max_joint_error_rad": score,
                            "root_trajectory_error_mean_m": 0.0,
                            "root_displacement_error_m": 0.0,
                            "root_metrics_available": True,
                        }
                    ]
                }
            )
        )
    os.utime(old_run / "summary.json", (1000, 1000))
    os.utime(new_run / "summary.json", (2000, 2000))

    report = build_report([old_run, new_run])
    row = report["global_prompt_scoreboard"][0]

    assert row["prompt"] == "a humanoid robot walks forward."
    assert row["num_runs"] == 2
    assert row["first_score"] == 1.5
    assert row["latest_score"] == 0.5
    assert row["best_score"] == 0.5
    assert row["improvement_from_first"] == 1.0
    assert row["num_comparable_root_metric_runs"] == 2
    assert row["comparable_improvement_from_first"] == 1.0
    assert row["latest_run"] == str(new_run)


def test_prompt_scoreboard_separates_legacy_non_root_metrics(tmp_path):
    legacy_run = tmp_path / "legacy"
    root_run = tmp_path / "root"
    legacy_run.mkdir()
    root_run.mkdir()
    (legacy_run / "summary.json").write_text(
        json.dumps(
            {
                "records": [
                    {
                        "prompt_id": "walk",
                        "prompt": "a humanoid robot walks forward.",
                        "status": "scored",
                        "completion_ratio": 1.0,
                        "max_joint_error_rad": 0.2,
                    }
                ]
            }
        )
    )
    (root_run / "summary.json").write_text(
        json.dumps(
            {
                "records": [
                    {
                        "prompt_id": "walk",
                        "prompt": "a humanoid robot walks forward.",
                        "status": "scored",
                        "completion_ratio": 1.0,
                        "max_joint_error_rad": 0.5,
                        "root_trajectory_error_mean_m": 1.0,
                        "root_displacement_error_m": 1.0,
                        "root_metrics_available": True,
                    }
                ]
            }
        )
    )
    os.utime(legacy_run / "summary.json", (1000, 1000))
    os.utime(root_run / "summary.json", (2000, 2000))

    report = build_report([legacy_run, root_run])
    row = report["global_prompt_scoreboard"][0]

    assert row["improvement_from_first"] < 0.0
    assert row["num_comparable_root_metric_runs"] == 1
    assert row["first_comparable_score"] == row["latest_comparable_score"]
    assert row["comparable_improvement_from_first"] == 0.0


def test_next_iteration_plan_uses_materialized_artifacts(tmp_path):
    hard_bank = tmp_path / "hard.jsonl"
    hard_bank.write_text(
        json.dumps(
            {
                "id": "walk",
                "prompt": "a humanoid robot walks forward.",
                "split": "adversarial_hard",
            }
        )
        + "\n"
        + json.dumps(
            {
                "id": "wave",
                "prompt": "a humanoid robot waves.",
                "split": "adversarial_hard",
            }
        )
        + "\n"
    )
    tracker_pool = tmp_path / "tracker_pool"
    tracker_pool.mkdir()
    (tracker_pool / "manifest.json").write_text(json.dumps([{"motion_path": "a.motion"}]))
    args = type(
        "Args",
        (),
        {
            "out": tmp_path / "report.json",
            "next_max_prompts": 8,
            "next_samples_per_prompt": 4,
            "hard_bank_limit": 5,
            "hard_min_score": 1.0,
            "tracker_pool_limit": 7,
            "min_tracker_motions_for_update": 1,
            "next_tracker_experiment_name": "tracker_exp",
            "next_tracker_steps": 123,
            "submit_script": Path("scripts/embodied/submit_physflow_kimodo_adv_sweep_taiji.sh"),
            "tracker_submit_script": Path("scripts/embodied/submit_physflow_g1_tracker_train_taiji.sh"),
        },
    )()
    report = {
        "global_hardest": [{"prompt": "walk"}, {"prompt": "walk"}, {"prompt": "wave"}],
        "score_formula": {"completion": "1 - completion_ratio"},
        "tracker_pool_thresholds": {"require_root_metrics": True},
    }

    plan = build_next_iteration_plan(report, hard_bank, tracker_pool, args)
    script = write_next_iteration_script(plan, tmp_path / "next.sh")

    assert plan["hard_prompt_count"] == 2
    assert plan["tracker_motion_count"] == 1
    assert "PHYSFLOW_MAX_PROMPTS=2" in plan["commands"]["submit_next_adversarial_sweep"]
    assert "PHYSFLOW_SAMPLES_PER_PROMPT=4" in plan["commands"]["submit_next_adversarial_sweep"]
    assert "PHYSFLOW_HARD_MIN_SCORE=1" in plan["commands"]["submit_next_adversarial_sweep"]
    assert "PHYSFLOW_TRAINING_MAX_STEPS=123" in plan["commands"]["train_position_aware_tracker"]
    assert "submit_physflow_g1_tracker_train_taiji.sh" in plan["commands"]["train_position_aware_tracker"]
    assert script.read_text().startswith("#!/usr/bin/env bash")
    assert "cd \"${PROJECT_ROOT}\"" in script.read_text()


def test_next_iteration_plan_suppresses_tracker_command_below_motion_threshold(tmp_path):
    hard_bank = tmp_path / "hard.jsonl"
    hard_bank.write_text(
        json.dumps({"id": "walk", "prompt": "a humanoid robot walks forward.", "split": "adversarial_hard"})
        + "\n"
    )
    tracker_pool = tmp_path / "tracker_pool"
    tracker_pool.mkdir()
    (tracker_pool / "manifest.json").write_text(json.dumps([{"motion_path": "a.motion"}]))
    args = type(
        "Args",
        (),
        {
            "out": tmp_path / "report.json",
            "next_max_prompts": 8,
            "next_samples_per_prompt": 4,
            "hard_bank_limit": 5,
            "hard_min_score": 1.0,
            "tracker_pool_limit": 7,
            "min_tracker_motions_for_update": 2,
            "next_tracker_experiment_name": "tracker_exp",
            "next_tracker_steps": 123,
            "submit_script": Path("scripts/embodied/submit_physflow_kimodo_adv_sweep_taiji.sh"),
            "tracker_submit_script": Path("scripts/embodied/submit_physflow_g1_tracker_train_taiji.sh"),
        },
    )()
    report = {
        "global_hardest": [{"prompt": "walk"}],
        "score_formula": {"completion": "1 - completion_ratio"},
        "tracker_pool_thresholds": {"require_root_metrics": True},
    }

    plan = build_next_iteration_plan(report, hard_bank, tracker_pool, args)
    script = write_next_iteration_script(plan, tmp_path / "next.sh")

    assert plan["tracker_motion_count"] == 1
    assert plan["min_tracker_motions_for_update"] == 2
    assert plan["commands"]["train_position_aware_tracker"] is None
    assert "below threshold" in script.read_text()


def test_next_iteration_plan_propagates_g1_onnx_to_adversarial_sweep(tmp_path):
    hard_bank = tmp_path / "hard.jsonl"
    hard_bank.write_text(
        json.dumps({"id": "walk", "prompt": "a humanoid robot walks forward.", "split": "adversarial_hard"})
        + "\n"
    )
    tracker_onnx = tmp_path / "updated_tracker.onnx"
    tracker_onnx.write_text("onnx")
    args = type(
        "Args",
        (),
        {
            "out": tmp_path / "report.json",
            "next_max_prompts": 8,
            "next_samples_per_prompt": 4,
            "next_g1_onnx": str(tracker_onnx),
            "hard_bank_limit": 5,
            "hard_min_score": 1.0,
            "tracker_pool_limit": 7,
            "min_tracker_motions_for_update": 2,
            "next_tracker_experiment_name": "tracker_exp",
            "next_tracker_steps": 123,
            "submit_script": Path("scripts/embodied/submit_physflow_kimodo_adv_sweep_taiji.sh"),
            "tracker_submit_script": Path("scripts/embodied/submit_physflow_g1_tracker_train_taiji.sh"),
        },
    )()
    report = {
        "global_hardest": [{"prompt": "walk"}],
        "score_formula": {"completion": "1 - completion_ratio"},
        "tracker_pool_thresholds": {"require_root_metrics": True},
        "global_tracker_artifacts": [
            {
                "g1_onnx_path": "/old/tracker.onnx",
                "g1_onnx_md5": "old",
            }
        ],
    }

    plan = build_next_iteration_plan(report, hard_bank, None, args)

    assert plan["next_g1_onnx"] == str(tracker_onnx)
    assert f"PHYSFLOW_G1_ONNX={tracker_onnx}" in plan["commands"]["submit_next_adversarial_sweep"]


def test_next_iteration_plan_auto_uses_single_report_tracker_artifact(tmp_path):
    hard_bank = tmp_path / "hard.jsonl"
    hard_bank.write_text(
        json.dumps({"id": "walk", "prompt": "a humanoid robot walks forward.", "split": "adversarial_hard"})
        + "\n"
    )
    args = type(
        "Args",
        (),
        {
            "out": tmp_path / "report.json",
            "next_max_prompts": 8,
            "next_samples_per_prompt": 4,
            "next_g1_onnx": None,
            "hard_bank_limit": 5,
            "hard_min_score": 1.0,
            "tracker_pool_limit": 7,
            "min_tracker_motions_for_update": 2,
            "next_tracker_experiment_name": "tracker_exp",
            "next_tracker_steps": 123,
            "submit_script": Path("scripts/embodied/submit_physflow_kimodo_adv_sweep_taiji.sh"),
            "tracker_submit_script": Path("scripts/embodied/submit_physflow_g1_tracker_train_taiji.sh"),
        },
    )()
    report = {
        "global_hardest": [{"prompt": "walk"}],
        "score_formula": {"completion": "1 - completion_ratio"},
        "tracker_pool_thresholds": {"require_root_metrics": True},
        "global_tracker_artifacts": [
            {
                "g1_onnx_path": "/single/tracker.onnx",
                "g1_onnx_md5": "single",
            }
        ],
    }

    plan = build_next_iteration_plan(report, hard_bank, None, args)

    assert plan["next_g1_onnx"] == "/single/tracker.onnx"
    assert "PHYSFLOW_G1_ONNX=/single/tracker.onnx" in plan["commands"]["submit_next_adversarial_sweep"]
