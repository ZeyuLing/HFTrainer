"""Shared root-aware adversarial scoring for PhysFlow G1 tracking.

The score is used to rank T2M candidates for hard-prompt mining and to select
good tracked motions for the G1 tracker pool. Keeping it in one module prevents
runner/report/best-of tools from silently drifting apart.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class G1ScoreConfig:
    joint_error_scale: float = 1.0
    root_trajectory_error_weight: float = 1.0
    root_trajectory_error_scale: float = 0.5
    root_displacement_error_weight: float = 0.5
    root_displacement_error_scale: float = 0.5
    score_component_cap: float = 2.0
    fall_penalty: float = 2.0

    def to_dict(self) -> dict[str, float | str]:
        data: dict[str, float | str] = asdict(self)
        data["completion"] = "1 - completion_ratio"
        return data


DEFAULT_G1_SCORE_CONFIG = G1ScoreConfig()
DEFAULT_G1_HARD_PROMPT_MIN_SCORE = 1.0


@dataclass(frozen=True)
class G1TrackerPoolConfig:
    min_completion: float = 0.95
    max_joint_error_rad: float = 0.7
    max_root_trajectory_error_mean_m: float = 0.25
    max_root_displacement_error_m: float = 0.35
    require_root_metrics: bool = True

    def to_dict(self) -> dict[str, float | bool]:
        return asdict(self)


DEFAULT_G1_TRACKER_POOL_CONFIG = G1TrackerPoolConfig()


def config_from_args(args: Any) -> G1ScoreConfig:
    return G1ScoreConfig(
        joint_error_scale=float(getattr(args, "joint_error_scale", DEFAULT_G1_SCORE_CONFIG.joint_error_scale)),
        root_trajectory_error_weight=float(
            getattr(args, "root_trajectory_error_weight", DEFAULT_G1_SCORE_CONFIG.root_trajectory_error_weight)
        ),
        root_trajectory_error_scale=float(
            getattr(args, "root_trajectory_error_scale", DEFAULT_G1_SCORE_CONFIG.root_trajectory_error_scale)
        ),
        root_displacement_error_weight=float(
            getattr(args, "root_displacement_error_weight", DEFAULT_G1_SCORE_CONFIG.root_displacement_error_weight)
        ),
        root_displacement_error_scale=float(
            getattr(args, "root_displacement_error_scale", DEFAULT_G1_SCORE_CONFIG.root_displacement_error_scale)
        ),
        score_component_cap=float(getattr(args, "score_component_cap", DEFAULT_G1_SCORE_CONFIG.score_component_cap)),
        fall_penalty=float(getattr(args, "fall_penalty", DEFAULT_G1_SCORE_CONFIG.fall_penalty)),
    )


def tracker_pool_config_from_args(args: Any) -> G1TrackerPoolConfig:
    return G1TrackerPoolConfig(
        min_completion=float(getattr(args, "good_min_completion", DEFAULT_G1_TRACKER_POOL_CONFIG.min_completion)),
        max_joint_error_rad=float(
            getattr(args, "good_max_joint_error", DEFAULT_G1_TRACKER_POOL_CONFIG.max_joint_error_rad)
        ),
        max_root_trajectory_error_mean_m=float(
            getattr(
                args,
                "good_max_root_trajectory_error",
                DEFAULT_G1_TRACKER_POOL_CONFIG.max_root_trajectory_error_mean_m,
            )
        ),
        max_root_displacement_error_m=float(
            getattr(
                args,
                "good_max_root_displacement_error",
                DEFAULT_G1_TRACKER_POOL_CONFIG.max_root_displacement_error_m,
            )
        ),
        require_root_metrics=not bool(getattr(args, "allow_tracker_pool_without_root_metrics", False)),
    )


def compute_g1_adversarial_score(
    completion: float,
    max_joint_error_rad: float,
    root_trajectory_error_mean_m: float,
    root_displacement_error_m: float,
    fall_detected: bool,
    config: G1ScoreConfig = DEFAULT_G1_SCORE_CONFIG,
) -> float:
    score = (
        (1.0 - float(completion))
        + min(float(max_joint_error_rad) / config.joint_error_scale, config.score_component_cap)
        + config.root_trajectory_error_weight
        * min(float(root_trajectory_error_mean_m) / config.root_trajectory_error_scale, config.score_component_cap)
        + config.root_displacement_error_weight
        * min(float(root_displacement_error_m) / config.root_displacement_error_scale, config.score_component_cap)
    )
    if fall_detected:
        score += config.fall_penalty
    return float(score)


def score_record(record: dict[str, Any], config: G1ScoreConfig = DEFAULT_G1_SCORE_CONFIG) -> float:
    return compute_g1_adversarial_score(
        completion=float(record.get("completion_ratio", 0.0)),
        max_joint_error_rad=float(record.get("max_joint_error_rad", 999.0)),
        root_trajectory_error_mean_m=float(record.get("root_trajectory_error_mean_m", 0.0)),
        root_displacement_error_m=float(record.get("root_displacement_error_m", 0.0)),
        fall_detected=bool(record.get("fall_detected", False)),
        config=config,
    )


def has_root_metrics(record: dict[str, Any]) -> bool:
    if "root_metrics_available" in record:
        return bool(record["root_metrics_available"])
    return (
        "root_trajectory_error_mean_m" in record
        or "root_displacement_error_m" in record
        or "root_displacement_track_m" in record
    )


def is_good_tracker_motion(
    record: dict[str, Any],
    config: G1TrackerPoolConfig = DEFAULT_G1_TRACKER_POOL_CONFIG,
) -> bool:
    if record.get("status") != "scored":
        return False
    if bool(record.get("fall_detected", False)):
        return False
    if config.require_root_metrics and not has_root_metrics(record):
        return False
    return (
        float(record.get("completion_ratio", 0.0)) >= config.min_completion
        and float(record.get("max_joint_error_rad", 999.0)) <= config.max_joint_error_rad
        and float(record.get("root_trajectory_error_mean_m", 999.0))
        <= config.max_root_trajectory_error_mean_m
        and float(record.get("root_displacement_error_m", 999.0)) <= config.max_root_displacement_error_m
    )


def is_hard_adversarial_case(record: dict[str, Any], min_score: float = DEFAULT_G1_HARD_PROMPT_MIN_SCORE) -> bool:
    score = record.get("adversarial_score", record.get("root_aware_score"))
    if score is None:
        score = score_record(record)
    return float(score) >= float(min_score)
