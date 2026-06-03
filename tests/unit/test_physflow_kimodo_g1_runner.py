import json
from argparse import Namespace
from pathlib import Path

import pytest


pytest.importorskip("mujoco")
pytest.importorskip("onnxruntime")

from scripts.embodied.physflow_kimodo_g1_runner import (  # noqa: E402
    KimodoRecord,
    compute_adversarial_score,
    load_prompt_bank,
    tracker_artifact_metadata,
    write_adversarial_outputs,
)


def _score_args(**overrides):
    values = dict(
        joint_error_scale=1.0,
        root_trajectory_error_weight=1.0,
        root_trajectory_error_scale=0.5,
        root_displacement_error_weight=0.5,
        root_displacement_error_scale=0.5,
        score_component_cap=2.0,
        fall_penalty=2.0,
    )
    values.update(overrides)
    return Namespace(**values)


def test_load_prompt_bank_accepts_adversarial_metadata(tmp_path):
    prompt_bank = tmp_path / "hard_prompt_bank.jsonl"
    prompt_bank.write_text(
        json.dumps(
            {
                "id": "hard_walk",
                "prompt": "a humanoid robot walks forward.",
                "category": "locomotion",
                "difficulty": 3,
                "duration_sec": 4.0,
                "split": "adversarial_hard",
                "adversarial_score": 3.5,
                "root_trajectory_error_mean_m": 1.2,
            }
        )
        + "\n"
    )

    prompts = load_prompt_bank(prompt_bank)

    assert len(prompts) == 1
    assert prompts[0].id == "hard_walk"
    assert prompts[0].split == "adversarial_hard"
    assert prompts[0].source == "hard_prompt_bank"
    assert prompts[0].tags == []


def test_root_aware_score_promotes_tracking_failures():
    args = _score_args()

    stable_score = compute_adversarial_score(
        completion=1.0,
        max_joint_error_rad=0.5,
        root_trajectory_error_mean_m=0.02,
        root_displacement_error_m=0.01,
        fall_detected=False,
        args=args,
    )
    failed_walk_score = compute_adversarial_score(
        completion=1.0,
        max_joint_error_rad=0.5,
        root_trajectory_error_mean_m=1.5,
        root_displacement_error_m=2.8,
        fall_detected=False,
        args=args,
    )

    assert stable_score < failed_walk_score
    assert failed_walk_score == pytest.approx(3.5)


def test_adversarial_outputs_are_reusable_prompt_bank(tmp_path):
    motion_dir = tmp_path / "motions"
    motion_dir.mkdir()
    stand_motion = motion_dir / "stand.motion"
    walk_motion = motion_dir / "walk.motion"
    stand_motion.write_text("stand")
    walk_motion.write_text("walk")

    records = [
        KimodoRecord(
            prompt_id="stand",
            prompt="a humanoid robot stands.",
            category="standing",
            difficulty=1,
            duration_sec=4.0,
            split="smoke",
            seed=1,
            sample_idx=0,
            output_stem="stand_s00",
            motion_path=str(stand_motion),
            status="scored",
            completion_ratio=1.0,
            max_joint_error_rad=0.2,
            root_trajectory_error_mean_m=0.01,
            root_displacement_error_m=0.01,
            root_metrics_available=True,
        ),
        KimodoRecord(
            prompt_id="walk",
            prompt="a humanoid robot walks forward.",
            category="locomotion",
            difficulty=2,
            duration_sec=4.0,
            split="smoke",
            seed=2,
            sample_idx=0,
            output_stem="walk_s00",
            motion_path=str(walk_motion),
            status="scored",
            completion_ratio=1.0,
            max_joint_error_rad=0.5,
            root_trajectory_error_mean_m=1.5,
            root_displacement_error_m=2.8,
            root_metrics_available=True,
        ),
    ]
    args = Namespace(
        **vars(_score_args()),
        hard_cases=1,
        hard_min_score=1.0,
        good_cases=1,
        good_min_completion=0.95,
        good_max_joint_error=0.7,
        good_max_root_trajectory_error=0.25,
        good_max_root_displacement_error=0.35,
        allow_tracker_pool_without_root_metrics=False,
        samples_per_prompt=2,
        seed=10,
    )

    selection = write_adversarial_outputs(records, args, tmp_path / "selection")
    hard_prompts = load_prompt_bank(Path(selection["hard_prompt_bank"]))

    assert [item.id for item in hard_prompts] == ["walk_hard_s00"]
    assert hard_prompts[0].split == "adversarial_hard"
    assert "score_3.500" in hard_prompts[0].tags
    pooled = list(Path(selection["tracker_motion_pool"]).glob("*.motion"))
    assert [path.name for path in pooled] == ["stand.motion"]
    assert selection["num_tracker_motion_pool_unique"] == 1
    assert selection["num_tracker_motion_pool_duplicates"] == 0
    manifest = json.loads(Path(selection["tracker_motion_pool_manifest"]).read_text())
    assert manifest[0]["prompt_id"] == "stand"
    assert selection["tracker_pool_thresholds"]["max_root_trajectory_error_mean_m"] == 0.25
    assert Path(selection["next_round_commands_script"]).is_file()
    assert "PHYSFLOW_PROMPT_SPLIT=adversarial_hard" in Path(
        selection["next_round_commands_script"]
    ).read_text()


def test_adversarial_outputs_record_active_tracker_artifact(tmp_path):
    motion_dir = tmp_path / "motions"
    motion_dir.mkdir()
    tracker_onnx = tmp_path / "tracker.onnx"
    tracker_yaml = tmp_path / "tracker.yaml"
    walk_motion = motion_dir / "walk.motion"
    tracker_onnx.write_text("onnx")
    tracker_yaml.write_text("yaml")
    walk_motion.write_text("walk")

    records = [
        KimodoRecord(
            prompt_id="walk",
            prompt="a humanoid robot walks forward.",
            category="locomotion",
            difficulty=2,
            duration_sec=4.0,
            split="smoke",
            seed=2,
            sample_idx=0,
            output_stem="walk_s00",
            motion_path=str(walk_motion),
            status="scored",
            completion_ratio=1.0,
            max_joint_error_rad=0.5,
            root_trajectory_error_mean_m=1.5,
            root_displacement_error_m=2.8,
            root_metrics_available=True,
        ),
    ]
    args = Namespace(
        **vars(_score_args()),
        g1_onnx=str(tracker_onnx),
        hard_cases=1,
        hard_min_score=1.0,
        good_cases=0,
        good_min_completion=0.95,
        good_max_joint_error=0.7,
        good_max_root_trajectory_error=0.25,
        good_max_root_displacement_error=0.35,
        allow_tracker_pool_without_root_metrics=False,
        samples_per_prompt=1,
        seed=10,
    )

    selection = write_adversarial_outputs(records, args, tmp_path / "selection")
    expected = tracker_artifact_metadata(tracker_onnx)

    assert selection["active_g1_tracker"]["g1_onnx_md5"] == expected["g1_onnx_md5"]
    assert selection["hard_cases"][0]["g1_onnx_path"] == str(tracker_onnx)
    assert selection["hard_cases"][0]["g1_yaml_md5"] == expected["g1_yaml_md5"]
    assert f"PHYSFLOW_G1_ONNX={tracker_onnx}" in selection["next_commands"]["continue_t2m_hard_prompt_adv_sweep"]


def test_hard_prompt_bank_deduplicates_multi_seed_prompts(tmp_path):
    motion_dir = tmp_path / "motions"
    motion_dir.mkdir()
    walk_a = motion_dir / "walk_a.motion"
    walk_b = motion_dir / "walk_b.motion"
    wave = motion_dir / "wave.motion"
    for path in [walk_a, walk_b, wave]:
        path.write_text(path.stem)

    records = [
        KimodoRecord(
            prompt_id="walk",
            prompt="a humanoid robot walks forward.",
            category="locomotion",
            difficulty=2,
            duration_sec=4.0,
            split="smoke",
            seed=1,
            sample_idx=0,
            output_stem="walk_s00",
            motion_path=str(walk_a),
            status="scored",
            completion_ratio=1.0,
            max_joint_error_rad=0.4,
            root_trajectory_error_mean_m=1.4,
            root_displacement_error_m=2.0,
            root_metrics_available=True,
        ),
        KimodoRecord(
            prompt_id="walk",
            prompt="a humanoid robot walks forward.",
            category="locomotion",
            difficulty=2,
            duration_sec=4.0,
            split="smoke",
            seed=2,
            sample_idx=1,
            output_stem="walk_s01",
            motion_path=str(walk_b),
            status="scored",
            completion_ratio=1.0,
            max_joint_error_rad=0.3,
            root_trajectory_error_mean_m=1.2,
            root_displacement_error_m=1.5,
            root_metrics_available=True,
        ),
        KimodoRecord(
            prompt_id="wave",
            prompt="a humanoid robot waves.",
            category="upper_body",
            difficulty=2,
            duration_sec=4.0,
            split="smoke",
            seed=3,
            sample_idx=0,
            output_stem="wave_s00",
            motion_path=str(wave),
            status="scored",
            completion_ratio=1.0,
            max_joint_error_rad=0.8,
            root_trajectory_error_mean_m=0.2,
            root_displacement_error_m=0.02,
            root_metrics_available=True,
        ),
    ]
    args = Namespace(
        **vars(_score_args()),
        hard_cases=3,
        hard_min_score=1.0,
        good_cases=0,
        good_min_completion=0.95,
        good_max_joint_error=0.7,
        good_max_root_trajectory_error=0.25,
        good_max_root_displacement_error=0.35,
        allow_tracker_pool_without_root_metrics=False,
        samples_per_prompt=2,
        seed=10,
    )

    selection = write_adversarial_outputs(records, args, tmp_path / "selection")
    hard_prompts = load_prompt_bank(Path(selection["hard_prompt_bank"]))

    assert [item.prompt for item in hard_prompts] == [
        "a humanoid robot walks forward.",
        "a humanoid robot waves.",
    ]
    assert [item.id for item in hard_prompts] == ["walk_hard_s00", "wave_hard_s00"]
    assert "PHYSFLOW_MAX_PROMPTS=2" in Path(selection["next_round_commands_script"]).read_text()
    assert "PHYSFLOW_HARD_MIN_SCORE=1" in Path(selection["next_round_commands_script"]).read_text()


def test_hard_prompt_bank_filters_easy_unique_prompts(tmp_path):
    motion_dir = tmp_path / "motions"
    motion_dir.mkdir()
    hard_motion = motion_dir / "hard.motion"
    easy_motion = motion_dir / "easy.motion"
    hard_motion.write_text("hard")
    easy_motion.write_text("easy")

    records = [
        KimodoRecord(
            prompt_id="hard_walk",
            prompt="a humanoid robot walks forward.",
            category="locomotion",
            difficulty=2,
            duration_sec=4.0,
            split="smoke",
            seed=1,
            sample_idx=0,
            output_stem="hard_walk_s00",
            motion_path=str(hard_motion),
            status="scored",
            completion_ratio=1.0,
            max_joint_error_rad=0.4,
            root_trajectory_error_mean_m=1.4,
            root_displacement_error_m=2.0,
            root_metrics_available=True,
        ),
        KimodoRecord(
            prompt_id="easy_stand",
            prompt="a humanoid robot stands calmly.",
            category="standing",
            difficulty=1,
            duration_sec=4.0,
            split="smoke",
            seed=2,
            sample_idx=0,
            output_stem="easy_stand_s00",
            motion_path=str(easy_motion),
            status="scored",
            completion_ratio=1.0,
            max_joint_error_rad=0.2,
            root_trajectory_error_mean_m=0.01,
            root_displacement_error_m=0.01,
            root_metrics_available=True,
        ),
    ]
    args = Namespace(
        **vars(_score_args()),
        hard_cases=3,
        hard_min_score=1.0,
        good_cases=0,
        good_min_completion=0.95,
        good_max_joint_error=0.7,
        good_max_root_trajectory_error=0.25,
        good_max_root_displacement_error=0.35,
        allow_tracker_pool_without_root_metrics=False,
        samples_per_prompt=2,
        seed=10,
    )

    selection = write_adversarial_outputs(records, args, tmp_path / "selection")
    hard_prompts = load_prompt_bank(Path(selection["hard_prompt_bank"]))

    assert [item.prompt for item in hard_prompts] == ["a humanoid robot walks forward."]
    assert [item["prompt_id"] for item in selection["hard_cases"]] == ["hard_walk"]
    assert selection["num_hard_candidates"] == 1
    assert selection["num_below_hard_threshold"] == 1
    assert [item["prompt_id"] for item in selection["top_scored_cases"]] == ["hard_walk", "easy_stand"]
    assert selection["hard_prompt_min_score"] == 1.0


def test_adversarial_outputs_skip_next_hard_command_when_no_hard_prompts(tmp_path):
    motion_dir = tmp_path / "motions"
    motion_dir.mkdir()
    stand_motion = motion_dir / "stand.motion"
    stand_motion.write_text("stand")

    records = [
        KimodoRecord(
            prompt_id="stand",
            prompt="a humanoid robot stands calmly.",
            category="standing",
            difficulty=1,
            duration_sec=4.0,
            split="smoke",
            seed=1,
            sample_idx=0,
            output_stem="stand_s00",
            motion_path=str(stand_motion),
            status="scored",
            completion_ratio=1.0,
            max_joint_error_rad=0.2,
            root_trajectory_error_mean_m=0.01,
            root_displacement_error_m=0.01,
            root_metrics_available=True,
        ),
    ]
    args = Namespace(
        **vars(_score_args()),
        hard_cases=3,
        hard_min_score=1.0,
        good_cases=0,
        good_min_completion=0.95,
        good_max_joint_error=0.7,
        good_max_root_trajectory_error=0.25,
        good_max_root_displacement_error=0.35,
        allow_tracker_pool_without_root_metrics=False,
        samples_per_prompt=2,
        seed=10,
    )

    selection = write_adversarial_outputs(records, args, tmp_path / "selection")
    hard_prompt_bank = Path(selection["hard_prompt_bank"])

    assert load_prompt_bank(hard_prompt_bank) == []
    assert selection["hard_cases"] == []
    assert selection["hard_prompt_records"] == []
    assert selection["num_hard_candidates"] == 0
    assert selection["num_below_hard_threshold"] == 1
    assert selection["next_commands"]["continue_t2m_hard_prompt_adv_sweep"] is None
    assert "skipped: no hard prompts" in Path(selection["next_round_commands_script"]).read_text()


def test_tracker_motion_pool_deduplicates_identical_motion_files(tmp_path):
    motion_dir = tmp_path / "motions"
    motion_dir.mkdir()
    stand_a = motion_dir / "stand_a.motion"
    stand_b = motion_dir / "stand_b.motion"
    stand_a.write_text("same motion")
    stand_b.write_text("same motion")

    records = [
        KimodoRecord(
            prompt_id="stand_a",
            prompt="a humanoid robot stands.",
            category="standing",
            difficulty=1,
            duration_sec=4.0,
            split="smoke",
            seed=1,
            sample_idx=0,
            output_stem="stand_a_s00",
            motion_path=str(stand_a),
            status="scored",
            completion_ratio=1.0,
            max_joint_error_rad=0.2,
            root_trajectory_error_mean_m=0.01,
            root_displacement_error_m=0.01,
            root_metrics_available=True,
        ),
        KimodoRecord(
            prompt_id="stand_b",
            prompt="a humanoid robot stands calmly.",
            category="standing",
            difficulty=1,
            duration_sec=4.0,
            split="smoke",
            seed=2,
            sample_idx=0,
            output_stem="stand_b_s00",
            motion_path=str(stand_b),
            status="scored",
            completion_ratio=1.0,
            max_joint_error_rad=0.2,
            root_trajectory_error_mean_m=0.01,
            root_displacement_error_m=0.01,
            root_metrics_available=True,
        ),
    ]
    args = Namespace(
        **vars(_score_args()),
        hard_cases=0,
        hard_min_score=1.0,
        good_cases=2,
        good_min_completion=0.95,
        good_max_joint_error=0.7,
        good_max_root_trajectory_error=0.25,
        good_max_root_displacement_error=0.35,
        allow_tracker_pool_without_root_metrics=False,
        samples_per_prompt=1,
        seed=10,
    )

    selection = write_adversarial_outputs(records, args, tmp_path / "selection")
    staged_files = list(Path(selection["tracker_motion_pool"]).glob("*.motion"))
    manifest = json.loads(Path(selection["tracker_motion_pool_manifest"]).read_text())

    assert len(staged_files) == 1
    assert len(manifest) == 1
    assert selection["num_tracker_motion_pool_unique"] == 1
    assert selection["num_tracker_motion_pool_duplicates"] == 1
    assert sorted(record["tracker_pool_status"] for record in selection["good_tracker_pool"]) == [
        "duplicate_skipped",
        "staged",
    ]
