from scripts.embodied.physflow_g1_scoring import (
    is_hard_adversarial_case,
    is_good_tracker_motion,
)


def test_hard_adversarial_case_uses_stored_scores():
    assert is_hard_adversarial_case({"adversarial_score": 1.2}, min_score=1.0)
    assert is_hard_adversarial_case({"root_aware_score": 1.2}, min_score=1.0)
    assert not is_hard_adversarial_case({"adversarial_score": 0.8}, min_score=1.0)


def test_hard_adversarial_case_can_compute_score_from_metrics():
    record = {
        "completion_ratio": 1.0,
        "max_joint_error_rad": 0.5,
        "root_trajectory_error_mean_m": 1.5,
        "root_displacement_error_m": 2.8,
    }

    assert is_hard_adversarial_case(record, min_score=1.0)
    assert not is_hard_adversarial_case(record, min_score=4.0)


def test_good_tracker_motion_requires_root_metrics_by_default():
    record = {
        "status": "scored",
        "completion_ratio": 1.0,
        "max_joint_error_rad": 0.2,
        "root_trajectory_error_mean_m": 0.02,
        "root_displacement_error_m": 0.01,
    }

    assert is_good_tracker_motion(record)
    assert not is_good_tracker_motion({**record, "root_metrics_available": False})


def test_good_tracker_motion_rejects_root_drift():
    record = {
        "status": "scored",
        "completion_ratio": 1.0,
        "max_joint_error_rad": 0.2,
        "root_trajectory_error_mean_m": 0.8,
        "root_displacement_error_m": 0.01,
        "root_metrics_available": True,
    }

    assert not is_good_tracker_motion(record)
