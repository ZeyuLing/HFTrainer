"""
Joint jump corruptor: 局部突变/跳变（Local Discontinuity）。

参考 scripts/m2m/synth_data/lq_local_discon.py，在随机选择的帧与关节上施加：
- sustained_offset: 关节突然偏移并持续若干帧（硬块或梯形渐变）
- burst_noise: 一段内每帧加随机噪声（高频乱跳）
- noisy_freeze: 若干帧内冻结姿态并带漂移+小抖动
- micro_stutter: 随机单帧用前后帧平均替代（微卡顿）

设计原则：
- 仅施加于四肢关节（arm/leg），不对脊椎/颈部做 corrupt，且越靠近末端（手腕、脚踝等）概率越大。
- 帧 pattern 更丰富：单帧多次、稀疏、短/中/长连续段等，更符合真实 mocap 瑕疵。
- 跳变方向偏向单轴（真实传感器常沿某一轴突变），而非完全随机球面。
仅修改 poses，不修改 trans。
"""

from __future__ import annotations

import numpy as np
from typing import Any, Dict, List, Optional, Tuple

from .base_corruptor import BaseCorruptor

# -----------------------------------------------------------------------------
# 四肢关节（仅 arm/leg，不含脊椎）；末端关节权重更高，模拟真实局部跳变分布
# SMPL 24: arm 从 collar/shoulder 到 wrist；leg 从 hip 到 ankle/foot
# -----------------------------------------------------------------------------
# 关节索引 -> 相对权重（末端 > 近端）
LIMB_JOINT_WEIGHTS: Dict[int, float] = {
    # arm: 13,14 collar; 16,17 shoulder; 18,19 elbow; 20,21 wrist
    13: 0.5,
    14: 0.5,
    16: 0.8,
    17: 0.8,
    18: 1.2,
    19: 1.2,
    20: 1.8,
    21: 1.8,
    # leg: 1,2 hip; 4,5 knee; 7,8,10,11 ankle/foot
    1: 0.5,
    2: 0.5,
    4: 0.9,
    5: 0.9,
    7: 1.5,
    8: 1.5,
    10: 1.5,
    11: 1.5,
}
ARM_JOINTS = [13, 14, 16, 17, 18, 19, 20, 21]
LEG_JOINTS = [1, 2, 4, 5, 7, 8, 10, 11]
LIMB_BODY_PARTS: Dict[str, List[int]] = {
    "arm": ARM_JOINTS,
    "leg": LEG_JOINTS,
}
DEFAULT_BODY_PART_NAMES = ("arm", "leg")

# -----------------------------------------------------------------------------
# 跳变策略名称与默认概率
# -----------------------------------------------------------------------------
STRATEGY_SUSTAINED_OFFSET = "sustained_offset"
STRATEGY_BURST_NOISE = "burst_noise"
STRATEGY_NOISY_FREEZE = "noisy_freeze"
STRATEGY_MICRO_STUTTER = "micro_stutter"
DEFAULT_JUMP_STRATEGIES = (
    STRATEGY_SUSTAINED_OFFSET,
    STRATEGY_BURST_NOISE,
    STRATEGY_NOISY_FREEZE,
    STRATEGY_MICRO_STUTTER,
)
DEFAULT_JUMP_STRATEGY_PROBS = (0.25, 0.25, 0.25, 0.25)

# -----------------------------------------------------------------------------
# 强度 -> 事件数范围、持续时长范围、幅度等（可被构造参数覆盖）
# -----------------------------------------------------------------------------
INTENSITY_JOINT_JUMP = {
    "low": {
        "event_count_range": (1, 3),
        "offset_duration_range": (2, 5),
        "burst_duration_range": (3, 6),
        "offset_magnitude": 0.2,
        "burst_noise_scale": 0.1,
        "freeze_drift_scale": 0.003,
        "micro_stutter_count_range": (2, 5),
        "single_repeated_count_range": (2, 5),
        "sparse_frame_prob": 0.12,
    },
    "medium": {
        "event_count_range": (3, 6),
        "offset_duration_range": (4, 12),
        "burst_duration_range": (5, 10),
        "offset_magnitude": 0.5,
        "burst_noise_scale": 0.3,
        "freeze_drift_scale": 0.005,
        "micro_stutter_count_range": (4, 9),
        "single_repeated_count_range": (3, 8),
        "sparse_frame_prob": 0.18,
    },
    "high": {
        "event_count_range": (5, 10),
        "offset_duration_range": (8, 22),
        "burst_duration_range": (10, 25),
        "offset_magnitude": 0.8,
        "burst_noise_scale": 0.6,
        "freeze_drift_scale": 0.008,
        "micro_stutter_count_range": (6, 14),
        "single_repeated_count_range": (5, 12),
        "sparse_frame_prob": 0.25,
    },
}


def _random_limb_joints_with_distal_bias(
    body_part: str,
    num_joints: Optional[int] = None,
    max_joints: Optional[int] = None,
) -> List[int]:
    """从四肢部位（arm/leg）中按权重抽样关节，末端关节概率更大；不选脊椎。"""
    pool = list(LIMB_BODY_PARTS.get(body_part, []))
    if not pool:
        return []
    weights = np.array([LIMB_JOINT_WEIGHTS.get(j, 1.0) for j in pool], dtype=np.float64)
    weights = weights / (weights.sum() + 1e-9)
    n = num_joints if num_joints is not None else np.random.randint(1, min(len(pool), max_joints or 5) + 1)
    n = max(1, min(n, len(pool)))
    chosen = np.random.choice(pool, size=n, replace=False, p=weights)
    return list(chosen)


# -----------------------------------------------------------------------------
# 帧 pattern：更符合真实 mocap 瑕疵（单帧多次、稀疏、短/中/长连续段）
# -----------------------------------------------------------------------------
FRAME_PATTERN_SINGLE_REPEATED = "single_repeated"  # 多个孤立单帧（持续单帧、出现多次）
FRAME_PATTERN_SPARSE = "sparse"  # 稀疏：每帧独立概率
FRAME_PATTERN_SHORT_BURST = "short_burst"  # 2~4 帧连续
FRAME_PATTERN_MEDIUM_SEGMENT = "medium_segment"  # 5~15 帧连续
FRAME_PATTERN_LONG_SEGMENT = "long_segment"  # 较长连续段
DEFAULT_FRAME_PATTERN_PROBS = (0.30, 0.20, 0.20, 0.20, 0.10)  # 提高单帧多次与稀疏占比


def _random_frame_segment(
    F: int,
    duration_range: Tuple[int, int],
) -> Tuple[int, int]:
    """随机一段连续帧 [start, end)，长度在 duration_range 内。"""
    dur = np.random.randint(duration_range[0], min(F, duration_range[1]) + 1)
    start = np.random.randint(0, max(1, F - dur))
    end = min(F, start + dur)
    return start, end


def _sample_jump_direction(magnitude: float, axis_bias_prob: float = 0.65) -> np.ndarray:
    """生成跳变方向：以 axis_bias_prob 概率偏向单轴（模拟真实传感器沿一轴突变），否则随机方向。"""
    if np.random.random() < axis_bias_prob:
        axis = np.random.randint(0, 3)
        err = np.zeros(3, dtype=np.float64)
        err[axis] = 1.0
        err = err + np.random.randn(3).astype(np.float64) * 0.25
    else:
        err = np.random.randn(3).astype(np.float64)
    n = np.linalg.norm(err)
    if n > 1e-9:
        err = err / n * magnitude
    return err


def _get_frames_for_event(
    F: int,
    pattern_type: str,
    params: Dict[str, Any],
    strategy: str,
) -> List[Tuple[int, int]]:
    """
    根据 pattern 类型返回要腐蚀的帧区间列表 [(start, end), ...]，每个区间 [start, end) 左闭右开。
    - single_repeated: 多段，每段长度 1（多帧独立单帧）
    - sparse: 多段，每段长度 1，帧索引按稀疏概率采样
    - short_burst: 一段，长度 2~4
    - medium_segment: 一段，长度 5~15
    - long_segment: 一段，长度取 params 的 offset_duration_range
    """
    segments: List[Tuple[int, int]] = []
    if pattern_type == FRAME_PATTERN_SINGLE_REPEATED:
        num_hits = params.get("single_repeated_count_range", (3, 8))
        count = np.random.randint(num_hits[0], min(F, num_hits[1]) + 1)
        count = min(count, max(1, F // 3))
        indices = np.random.choice(range(F), size=count, replace=False)
        for t in indices:
            segments.append((int(t), int(t) + 1))
    elif pattern_type == FRAME_PATTERN_SPARSE:
        p = params.get("sparse_frame_prob", 0.15)
        p = np.clip(p, 0.05, 0.4)
        mask = np.random.random(F) < p
        indices = np.where(mask)[0]
        if len(indices) > 50:
            indices = np.random.choice(indices, size=50, replace=False)
        for t in indices:
            segments.append((int(t), int(t) + 1))
    elif pattern_type == FRAME_PATTERN_SHORT_BURST:
        dur = np.random.randint(2, min(5, F) + 1)
        start = np.random.randint(0, max(1, F - dur))
        segments.append((start, min(F, start + dur)))
    elif pattern_type == FRAME_PATTERN_MEDIUM_SEGMENT:
        dur = np.random.randint(5, min(16, F) + 1)
        start = np.random.randint(0, max(1, F - dur))
        segments.append((start, min(F, start + dur)))
    elif pattern_type == FRAME_PATTERN_LONG_SEGMENT:
        dur_range = params.get("offset_duration_range", (8, 25))
        dur = np.random.randint(dur_range[0], min(F, dur_range[1]) + 1)
        start = np.random.randint(0, max(1, F - dur))
        segments.append((start, min(F, start + dur)))
    else:
        dur_range = params.get("offset_duration_range", (3, 10))
        start, end = _random_frame_segment(F, dur_range)
        segments.append((start, end))
    return segments


def _apply_sustained_offset(
    poses: np.ndarray,
    segments: List[Tuple[int, int]],
    joint_ids: List[int],
    magnitude: float,
    trapezoid_prob: float = 0.4,
    axis_bias_prob: float = 0.65,
) -> Tuple[np.ndarray, Dict]:
    """对多段 [start, end) 施加同一方向的恒定偏移（方向偏单轴）；单帧段用 hard_block。"""
    poses_mod = poses.copy()
    F, J, D = poses_mod.shape
    max_dur = max((end - start) for start, end in segments) if segments else 0
    use_trapezoid = max_dur > 3 and np.random.random() < trapezoid_prob
    mode = "trapezoid" if use_trapezoid else "hard_block"
    error_vec = _sample_jump_direction(magnitude, axis_bias_prob=axis_bias_prob)

    for start, end in segments:
        duration = end - start
        if duration <= 0:
            continue
        for j in joint_ids:
            if j >= J:
                continue
            if mode == "hard_block":
                poses_mod[start:end, j, :] += error_vec
            else:
                ramp_len = max(1, duration // 4)
                for i in range(ramp_len):
                    alpha = (i + 1) / ramp_len
                    poses_mod[start + i, j, :] += error_vec * alpha
                hold_start = start + ramp_len
                hold_end = end - ramp_len
                if hold_end > hold_start:
                    poses_mod[hold_start:hold_end, j, :] += error_vec
                for i in range(ramp_len):
                    alpha = 1.0 - (i + 1) / ramp_len
                    idx = hold_end + i
                    if idx < F:
                        poses_mod[idx, j, :] += error_vec * alpha

    return poses_mod, {
        "type": STRATEGY_SUSTAINED_OFFSET,
        "segments": segments,
        "joints": joint_ids,
        "mode": mode,
    }


def _apply_burst_noise(
    poses: np.ndarray,
    segments: List[Tuple[int, int]],
    joint_ids: List[int],
    noise_scale: float,
    axis_bias_prob: float = 0.55,
    decay_factor: float = 0.85,
) -> Tuple[np.ndarray, Dict]:
    """多段内每帧加噪声；方向偏单轴，且首帧突变后逐帧衰减（更符合真实突发抖动）。"""
    poses_mod = poses.copy()
    F, J, D = poses_mod.shape
    # 主方向（偏单轴）+ 每帧沿该方向的衰减
    base_dir = _sample_jump_direction(1.0, axis_bias_prob=axis_bias_prob)
    for start, end in segments:
        duration = end - start
        if duration <= 0:
            continue
        # 首帧突变较强，后续帧沿 base_dir 衰减 + 少量各向同性噪声
        for ki, j in enumerate(joint_ids):
            if j >= J:
                continue
            for t in range(duration):
                frame_idx = start + t
                decay = decay_factor**t
                driven = base_dir * (noise_scale * decay) + np.random.randn(3).astype(np.float64) * (noise_scale * 0.3)
                poses_mod[frame_idx, j, :] += driven
    return poses_mod, {"type": STRATEGY_BURST_NOISE, "segments": segments, "joints": joint_ids}


def _apply_noisy_freeze(
    poses: np.ndarray,
    segments: List[Tuple[int, int]],
    joint_ids: List[int],
    drift_scale: float,
    jitter_scale: float = 0.002,
) -> Tuple[np.ndarray, Dict]:
    """多段内将指定关节冻结为各段首帧姿态，并加线性漂移和小抖动。"""
    poses_mod = poses.copy()
    F, J, D = poses_mod.shape
    drift = np.random.randn(J, 3).astype(np.float64) * drift_scale
    for start, end in segments:
        duration = end - start
        if duration <= 0:
            continue
        base_pose = poses_mod[start].copy()
        for t in range(duration):
            frame_idx = start + t
            for j in joint_ids:
                if j >= J:
                    continue
                jitter = np.random.randn(3).astype(np.float64) * jitter_scale
                poses_mod[frame_idx, j, :] = base_pose[j, :] + drift[j, :] * t + jitter
    return poses_mod, {"type": STRATEGY_NOISY_FREEZE, "segments": segments, "joints": joint_ids}


def _apply_micro_stutter(
    poses: np.ndarray,
    frame_indices: List[int],
) -> Tuple[np.ndarray, Dict]:
    """在给定帧用前后帧平均替代（需保证 1 <= idx <= F-2）。"""
    poses_mod = poses.copy()
    F, J, D = poses_mod.shape
    valid = [i for i in frame_indices if 1 <= i < F - 1]
    for idx in valid:
        poses_mod[idx] = (poses_mod[idx - 1] + poses_mod[idx + 1]) * 0.5
    return poses_mod, {"type": STRATEGY_MICRO_STUTTER, "frames": valid}


class JointJumpCorruptor(BaseCorruptor):
    """
    关节跳变 corruptor：在随机帧、随机部位与关节数上施加局部突变（持续偏移、爆发噪声、冻结漂移、微卡顿）。
    仅修改 poses，不修改 trans。
    """

    def __init__(
        self,
        body_model: Optional[Any] = None,
        device: str = "cuda",
    ) -> None:
        """初始化不固定任何腐蚀行为，每次 corrupt() 时随机决定 intensity、部位、策略等。"""
        super().__init__(body_model=body_model, device=device)

    def _apply_corruption(
        self,
        data_mod: Dict,
        poses: np.ndarray,
        trans: np.ndarray,
        **kwargs: Any,
    ) -> Tuple[np.ndarray, np.ndarray, Dict]:
        # 每次 apply 时随机决定 intensity，保证训练时多样性
        intensity = kwargs.get("intensity") or str(np.random.choice(list(INTENSITY_JOINT_JUMP.keys())))
        params = INTENSITY_JOINT_JUMP.get(intensity, INTENSITY_JOINT_JUMP["medium"]).copy()
        event_count_range = params["event_count_range"]
        offset_dur = params["offset_duration_range"]
        burst_dur = params["burst_duration_range"]
        offset_mag = params["offset_magnitude"]
        burst_scale = params["burst_noise_scale"]
        freeze_drift = params["freeze_drift_scale"]
        stutter_count_range = params["micro_stutter_count_range"]
        body_part_names = DEFAULT_BODY_PART_NAMES
        strategy_probs = np.asarray(DEFAULT_JUMP_STRATEGY_PROBS, dtype=np.float64)
        strategy_probs = strategy_probs / strategy_probs.sum()
        frame_patterns = (
            FRAME_PATTERN_SINGLE_REPEATED,
            FRAME_PATTERN_SPARSE,
            FRAME_PATTERN_SHORT_BURST,
            FRAME_PATTERN_MEDIUM_SEGMENT,
            FRAME_PATTERN_LONG_SEGMENT,
        )
        frame_pattern_probs = np.asarray(DEFAULT_FRAME_PATTERN_PROBS, dtype=np.float64)
        frame_pattern_probs = frame_pattern_probs / frame_pattern_probs.sum()
        max_attempts = 30

        F, J, _ = poses.shape
        target_count = np.random.randint(*event_count_range)
        strategies = list(DEFAULT_JUMP_STRATEGIES)
        meta_logs: List[Dict] = []
        affected_components: List[str] = []
        poses_curr = poses.copy()
        attempts = 0

        while len(meta_logs) < target_count and attempts < max_attempts:
            attempts += 1
            strat = str(np.random.choice(strategies, p=strategy_probs))
            body_part = str(np.random.choice(body_part_names))
            joint_ids = _random_limb_joints_with_distal_bias(body_part, max_joints=6)
            if not joint_ids:
                continue

            pattern_type = str(np.random.choice(frame_patterns, p=frame_pattern_probs))
            if strat == STRATEGY_MICRO_STUTTER and pattern_type not in (
                FRAME_PATTERN_SINGLE_REPEATED,
                FRAME_PATTERN_SPARSE,
            ):
                pattern_type = np.random.choice(
                    (FRAME_PATTERN_SINGLE_REPEATED, FRAME_PATTERN_SPARSE),
                    p=np.array([0.6, 0.4]),
                )
            segments = _get_frames_for_event(F, pattern_type, params, strat)
            if not segments:
                continue

            if strat == STRATEGY_SUSTAINED_OFFSET:
                if F < 2:
                    continue
                poses_curr, info = _apply_sustained_offset(poses_curr, segments, joint_ids, offset_mag)
                meta_logs.append(info)
                affected_components.append("sustained_offset")

            elif strat == STRATEGY_BURST_NOISE:
                if F < 2:
                    continue
                poses_curr, info = _apply_burst_noise(poses_curr, segments, joint_ids, burst_scale)
                meta_logs.append(info)
                affected_components.append("burst_noise")

            elif strat == STRATEGY_NOISY_FREEZE:
                if F < 2:
                    continue
                poses_curr, info = _apply_noisy_freeze(poses_curr, segments, joint_ids, freeze_drift)
                meta_logs.append(info)
                affected_components.append("noisy_freeze")

            elif strat == STRATEGY_MICRO_STUTTER:
                if F < 10:
                    continue
                # micro_stutter 需要 1 <= idx < F-1；从 segments 取单帧起点并过滤
                frame_indices = [start for start, end in segments if end == start + 1 and 1 <= start < F - 1]
                if not frame_indices:
                    frame_indices = [start for start, end in segments if 1 <= start < F - 1]
                if not frame_indices:
                    continue
                count = min(len(frame_indices), np.random.randint(*stutter_count_range), max(1, F // 5))
                count = max(1, count)
                if count > len(frame_indices):
                    count = len(frame_indices)
                chosen = list(np.random.choice(len(frame_indices), size=count, replace=False))
                stutter_frames = [frame_indices[i] for i in chosen]
                poses_curr, info = _apply_micro_stutter(poses_curr, stutter_frames)
                if info["frames"]:
                    meta_logs.append(info)
                    affected_components.append("micro_stutter")

        # Build _mask_info: aggregate all events to find corrupted (frame, joint) pairs
        all_segments: List[Tuple[int, int]] = []
        all_joint_ids: set = set()
        for log in meta_logs:
            joints = log.get("joints")
            segments_from_log = log.get("segments")
            frames_from_log = log.get("frames")
            if joints:
                all_joint_ids.update(joints)
            if log.get("joint_index") is not None:
                all_joint_ids.add(log["joint_index"])
            if segments_from_log:
                all_segments.extend(segments_from_log)
            if frames_from_log:
                all_segments.extend([(f, f + 1) for f in frames_from_log])

        _mask_info: Dict = {}
        if all_segments and all_joint_ids:
            _mask_info = {
                "corrupted_segments": all_segments,
                "corrupted_joints": list(all_joint_ids),
                "trans_corrupted": False,
            }

        meta = {
            "synthesis_type": "joint_jump",
            "description": f"Joint Jump / Local Discontinuity ({intensity}): {len(meta_logs)} events",
            "synthesis_method": {
                "pattern_type": "local_discontinuity",
                "intensity_level": intensity,
                "event_count": len(meta_logs),
                "events_log": meta_logs,
            },
            "degradation_details": {"affected_components": list(set(affected_components))},
            "_mask_info": _mask_info,
        }
        return poses_curr, trans, meta


if __name__ == "__main__":
    import argparse
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Test JointJumpCorruptor on real data from refine_hq.json")
    parser.add_argument("--anno", type=str, default="data/hymotion_m2m/anno/refine_hq.json")
    parser.add_argument(
        "--data_root",
        type=str,
        default="data/hymotion_data",
        help="Data root for motion npz paths (tgt_motion_path is relative to this)",
    )
    parser.add_argument("--output_dir", type=str, default="output/test/corruptor/joint_jump_corruptor")
    parser.add_argument("--intensity", type=str, default="medium", choices=["low", "medium", "high"])
    parser.add_argument("--num_samples", type=int, default=3)
    args = parser.parse_args()

    with open(args.anno, "r", encoding="utf-8") as f:
        anno = json.load(f)
    paths = [v["tgt_motion_path"] for v in anno.values() if isinstance(v, dict) and "tgt_motion_path" in v]
    if not paths:
        paths = list(anno.keys())
    chosen = list(np.random.choice(paths, size=min(args.num_samples, len(paths)), replace=False))

    data_root = Path(args.data_root)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    corruptor = JointJumpCorruptor(device="cpu")
    print(f"JointJumpCorruptor intensity={args.intensity}, data_root={data_root}, samples={len(chosen)}")

    for i, rel_path in enumerate(chosen):
        npz_path = data_root / rel_path
        if not npz_path.exists():
            print(f"  skip {i+1}: not found {npz_path}")
            continue
        motion = dict(np.load(str(npz_path), allow_pickle=True))
        if "transl" in motion and "trans" not in motion:
            motion["trans"] = motion["transl"]
        out = corruptor.corrupt(motion, intensity=args.intensity)
        meta = out.get("meta") or {}
        events = (meta.get("synthesis_method") or {}).get("events_log") or []
        corrupted = out.get("corrupted_motion") or {}
        save_name = Path(rel_path).name
        save_path = out_dir / save_name
        to_save = {k: v for k, v in corrupted.items() if isinstance(v, np.ndarray)}
        if to_save:
            np.savez_compressed(str(save_path), **to_save)
        print(f"  {i+1}: {rel_path} -> {save_path} events={len(events)}")
    print("JointJumpCorruptor test OK.")
