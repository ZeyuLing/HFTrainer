"""
Jitter corruptor: algorithmic tremor (Perlin-like noise) and temporal quantization.

随机性（每次 corrupt 调用独立采样，与 RefineOnlineOfflineM2MDataset getitem 一致）：
- 时间随机：以 burst_prob 决定「全帧连续」或「若干 burst 段」；或以 SPARSE_TEMPORAL_PROB 概率用「稀疏帧」（每帧独立概率 p 施加，p 在 SPARSE_FRAME_PROB_RANGE）。
- 部位随机：随机决定只施加 poses、只施加 trans、或两者；对 poses 可随机零化 0~45% 关节，其余用躯干/四肢权重。
- 强度随机：未传 intensity 时在 low/medium/high 中随机。
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d
from typing import Any, Dict, List, Optional, Tuple

from .base_corruptor import BaseCorruptor

# -----------------------------------------------------------------------------
# SMPL 24 关节分组（用于噪声权重：躯干弱、四肢强，模拟真实抖动分布）
# -----------------------------------------------------------------------------
TORSO_JOINT_IDS = [0, 1, 2, 3, 6, 9, 12]
EXTREMITY_JOINT_IDS = [7, 8, 10, 11, 20, 21, 15, 22, 23]

# -----------------------------------------------------------------------------
# Perlin-like 噪声默认参数
# -----------------------------------------------------------------------------
DEFAULT_PERLIN_SIGMAS = (1.0, 5.0, 15.0)
DEFAULT_PERLIN_WEIGHTS = (0.7, 0.2, 0.1)

# -----------------------------------------------------------------------------
# Burst mask 默认参数（控制“施加在哪些帧”）
# -----------------------------------------------------------------------------
DEFAULT_BURST_NUM_RANGE = (1, 4)
DEFAULT_BURST_DURATION_RANGE = (30, 90)
DEFAULT_BURST_FADE_LEN = 10

# -----------------------------------------------------------------------------
# 关节噪声权重（相对 1.0 的倍数）
# -----------------------------------------------------------------------------
DEFAULT_TORSO_WEIGHT = 0.5
DEFAULT_EXTREMITY_WEIGHT = 2.0

# -----------------------------------------------------------------------------
# 策略与强度默认配置
# -----------------------------------------------------------------------------
DEFAULT_STRATEGY_NAMES = ("algo_jitter", "temp_quant")
DEFAULT_STRATEGY_PROBS = (0.7, 0.3)
DEFAULT_MAX_ATTEMPTS = 10
DEFAULT_TEMP_QUANT_JITTER_RATIO = 0.5

# 强度 -> (algo_scale, fps_reduction, burst_prob)；fps_reduction=0 表示不使用时间量化
INTENSITY_PARAMS = {
    "low": (0.001, 0, 0.3),
    "medium": (0.003, 2, 0.6),
    "high": (0.008, 3, 0.8),
}
SPARSE_TEMPORAL_PROB = 0.35
SPARSE_FRAME_PROB_RANGE = (0.2, 0.6)


def _perlin_noise_like(
    shape: Tuple[int, ...],
    scale: float = 1.0,
    sigmas: Tuple[float, ...] = DEFAULT_PERLIN_SIGMAS,
    weights: Tuple[float, ...] = DEFAULT_PERLIN_WEIGHTS,
) -> np.ndarray:
    """Generate Perlin-like smooth noise by blending multi-scale Gaussian-smoothed white noise."""
    n = len(sigmas)
    assert n == len(weights), "sigmas and weights length must match"
    flat_len = int(np.prod(shape))
    bands = [gaussian_filter1d(np.random.randn(flat_len), sigma=s) * w for s, w in zip(sigmas, weights)]
    combined: np.ndarray = sum(bands)
    std_val = float(np.std(combined))
    if std_val < 1e-9:
        std_val = 1e-9
    combined = (combined / std_val * scale).reshape(shape).astype(np.float64)
    return combined


def _burst_mask(
    frames: int,
    burst_prob: float,
    num_burst_range: Tuple[int, int] = DEFAULT_BURST_NUM_RANGE,
    duration_range: Tuple[int, int] = DEFAULT_BURST_DURATION_RANGE,
    fade_len: int = DEFAULT_BURST_FADE_LEN,
) -> Tuple[np.ndarray, str]:
    """
    Randomly decide which frames to corrupt.
    - With probability (1 - burst_prob): apply to all frames (continuous).
    - With probability burst_prob: apply only in num_burst segments, each with random duration and fade.
    Returns (mask of shape (F,), mode string 'continuous' or 'burst_N').
    """
    if np.random.random() >= burst_prob:
        return np.ones(frames, dtype=np.float64), "continuous"
    mask = np.zeros(frames, dtype=np.float64)
    num_bursts = np.random.randint(*num_burst_range)
    for _ in range(num_bursts):
        dur = np.random.randint(*duration_range)
        if frames <= dur:
            mask[:] = 1.0
            return mask, f"burst_{num_bursts}"
        start = np.random.randint(0, frames - dur)
        seg = np.ones(dur)
        if dur > fade_len * 2:
            seg[:fade_len] = np.linspace(0, 1, fade_len)
            seg[-fade_len:] = np.linspace(1, 0, fade_len)
        mask[start : start + dur] = np.maximum(mask[start : start + dur], seg)
    return mask, f"burst_{num_bursts}"


def _sparse_mask(frames: int, frame_prob: float) -> np.ndarray:
    """每帧以概率 frame_prob 施加，时间维度随机。返回 (F,) mask。"""
    return (np.random.random(frames) < frame_prob).astype(np.float64)


def _build_joint_weight_map(
    num_joints: int,
    torso_ids: List[int] = TORSO_JOINT_IDS,
    extremity_ids: List[int] = EXTREMITY_JOINT_IDS,
    torso_weight: float = DEFAULT_TORSO_WEIGHT,
    extremity_weight: float = DEFAULT_EXTREMITY_WEIGHT,
) -> np.ndarray:
    """(1, J, 1) weight per joint: torso < 1 < extremity."""
    weight_map = np.ones((1, num_joints, 1), dtype=np.float64)
    for j in torso_ids:
        if j < num_joints:
            weight_map[0, j, 0] = torso_weight
    for j in extremity_ids:
        if j < num_joints:
            weight_map[0, j, 0] = extremity_weight
    return weight_map


def _random_joint_weight_map(
    num_joints: int,
    *,
    zero_fraction_range: Tuple[float, float] = (0.0, 0.5),
) -> np.ndarray:
    """
    部位随机：在基础躯干/四肢权重上，随机将一部分关节权重置 0（只对部分关节施加）。
    zero_fraction_range: 随机零化的关节比例范围，如 (0.2, 0.5) 表示 20%~50% 的关节不施加。
    """
    base = _build_joint_weight_map(num_joints)
    frac = np.random.uniform(*zero_fraction_range)
    n_zero = max(0, min(num_joints - 1, int(num_joints * frac)))
    if n_zero == 0:
        return base
    indices = np.random.choice(num_joints, size=n_zero, replace=False)
    out = base.copy()
    for j in indices:
        out[0, j, 0] = 0.0
    return out


def _apply_jitter_poses(
    poses: np.ndarray,
    base_scale: float,
    mask: Optional[np.ndarray] = None,
    joint_weight_map: Optional[np.ndarray] = None,
    perlin_scale: float = 1.0,
) -> np.ndarray:
    """Add Perlin-like jitter to poses (F, J, 3). Optional (F,) or (F,1) mask for temporal weighting."""
    poses_mod = poses.copy()
    F, J, D = poses_mod.shape
    raw_noise = _perlin_noise_like((F, J, D), scale=perlin_scale)
    if joint_weight_map is None:
        joint_weight_map = _build_joint_weight_map(J)
    final_noise = raw_noise * joint_weight_map * base_scale
    if mask is not None:
        m = np.asarray(mask)
        if m.ndim == 1:
            m = m[:, np.newaxis]
        final_noise *= m[:, :, np.newaxis]
    poses_mod += final_noise
    return poses_mod


def _apply_jitter_trans(
    trans: np.ndarray,
    base_scale: float,
    mask: Optional[np.ndarray] = None,
    perlin_scale: float = 1.0,
) -> np.ndarray:
    """Add Perlin-like jitter to root translation (F, 3). Optional (F,) or (F,1) mask."""
    trans_mod = trans.copy()
    raw_noise = _perlin_noise_like(trans_mod.shape, scale=perlin_scale)
    final_noise = raw_noise * base_scale
    if mask is not None:
        m = np.asarray(mask)
        if m.ndim == 1:
            m = m[:, np.newaxis]
        final_noise *= m
    trans_mod += final_noise
    return trans_mod


def _apply_temporal_quantization(
    arr: np.ndarray,
    fps_reduction: int,
    jitter_amp: float,
) -> np.ndarray:
    """Downsample along time, linear interpolate back to original length, then add small jitter."""
    F = arr.shape[0]
    if F < fps_reduction * 2:
        return arr.copy()
    indices = np.arange(0, F, fps_reduction)
    if indices[-1] != F - 1:
        indices = np.append(indices, F - 1)
    sampled = arr[indices]
    f = interp1d(indices, sampled, axis=0, kind="linear")
    stepped = f(np.arange(F)).astype(arr.dtype)
    noise = _perlin_noise_like(arr.shape, scale=jitter_amp)
    return stepped + noise


class JitterCorruptor(BaseCorruptor):
    """
    Jitter corruptor: algorithmic tremor (Perlin-like) and/or temporal quantization.

    - 施加策略：每次 corrupt 时随机选「algo_jitter」或「temp_quant」之一（概率可配置）。
    - 施加的帧：由 burst_mask 随机决定全帧连续或若干 burst 段（可配置 burst_prob 等）。
    - 施加部位：默认对「所有关节 + 根位移」施加；关节间按躯干/四肢权重区分。
      可通过 apply_to_poses / apply_to_trans 关闭 poses 或 trans。
    """

    def __init__(
        self,
        body_model: Optional[Any] = None,
        device: str = "cuda",
    ) -> None:
        """初始化不固定任何腐蚀行为，每次 corrupt() 时随机决定 intensity、施加部位等。"""
        super().__init__(body_model=body_model, device=device)

    def _apply_corruption(
        self,
        data_mod: Dict,
        poses: np.ndarray,
        trans: np.ndarray,
        **kwargs: Any,
    ) -> Tuple[np.ndarray, np.ndarray, Dict]:
        # 强度随机：未传 intensity 时在 low/medium/high 中随机（与 refine dataset getitem 一致）
        intensity = kwargs.get("intensity") or str(np.random.choice(list(INTENSITY_PARAMS.keys())))
        algo_scale, temp_quant, burst_prob = INTENSITY_PARAMS.get(intensity, INTENSITY_PARAMS["medium"])
        duration_range = DEFAULT_BURST_DURATION_RANGE

        # 部位随机：随机决定只施加 poses、只施加 trans、或两者
        apply_to_poses = kwargs.get("apply_to_poses") if "apply_to_poses" in kwargs else (np.random.random() < 0.85)
        apply_to_trans = kwargs.get("apply_to_trans") if "apply_to_trans" in kwargs else (np.random.random() < 0.85)
        if not apply_to_poses and not apply_to_trans:
            apply_to_poses = True

        # 部位随机：关节维度随机零化一部分（只对部分关节施加）
        J = poses.shape[1]
        use_random_joints = np.random.random() < 0.6
        joint_weight_map = (
            _random_joint_weight_map(J, zero_fraction_range=(0.0, 0.45))
            if use_random_joints
            else _build_joint_weight_map(J)
        )

        strategies = list(DEFAULT_STRATEGY_NAMES)
        probs = np.asarray(DEFAULT_STRATEGY_PROBS, dtype=np.float64)
        probs = probs / probs.sum()
        max_attempts = DEFAULT_MAX_ATTEMPTS

        F = poses.shape[0]
        meta_logs: List[Dict] = []
        affected_components: List[str] = []
        poses_curr = poses.copy()
        trans_curr = trans.copy()
        attempts = 0

        while not meta_logs and attempts < max_attempts:
            attempts += 1
            poses_curr = poses.copy()
            trans_curr = trans.copy()
            strat = str(np.random.choice(strategies, p=probs))

            if strat == "algo_jitter":
                # 时间随机：以一定概率用「稀疏帧」而非「全帧/burst」
                if np.random.random() < SPARSE_TEMPORAL_PROB:
                    p_frame = float(np.random.uniform(*SPARSE_FRAME_PROB_RANGE))
                    mask = _sparse_mask(F, p_frame)
                    mode_desc = f"sparse_p{p_frame:.2f}"
                else:
                    mask, mode_desc = _burst_mask(
                        F,
                        burst_prob,
                        num_burst_range=DEFAULT_BURST_NUM_RANGE,
                        duration_range=duration_range,
                    )
                if apply_to_poses:
                    poses_curr = _apply_jitter_poses(
                        poses_curr, base_scale=algo_scale, mask=mask, joint_weight_map=joint_weight_map
                    )
                if apply_to_trans:
                    trans_curr = _apply_jitter_trans(trans_curr, base_scale=algo_scale, mask=mask)
                affected_components.append("algorithmic_tremor_rot_trans")
                meta_logs.append({"type": "algo_jitter", "scale": algo_scale, "mode": mode_desc})

            elif strat == "temp_quant" and temp_quant > 1:
                jitter_amp = algo_scale * DEFAULT_TEMP_QUANT_JITTER_RATIO
                if apply_to_poses:
                    poses_curr = _apply_temporal_quantization(
                        poses_curr, fps_reduction=temp_quant, jitter_amp=jitter_amp
                    )
                if apply_to_trans:
                    trans_curr = _apply_temporal_quantization(
                        trans_curr, fps_reduction=temp_quant, jitter_amp=jitter_amp
                    )
                affected_components.append("temporal_aliasing_rot_trans")
                meta_logs.append({"type": "temp_quant", "fps_reduction": temp_quant})

        # Build _mask_info for joint_corrupted_mask generation
        # For jitter, affected joints come from joint_weight_map (> 0 means affected)
        # and frame mask comes from the temporal mask used during corruption.
        # For temp_quant, all frames and joints are affected.
        jm = (joint_weight_map.squeeze() > 0).astype(bool) if apply_to_poses else np.zeros(J, dtype=bool)
        if not meta_logs:
            # No corruption was applied
            _mask_info = {}
        elif meta_logs[-1].get("type") == "algo_jitter":
            _mask_info = {
                "frame_mask": mask,
                "joint_mask": jm,
                "trans_corrupted": apply_to_trans,
            }
        else:
            # temp_quant: all frames, all joints affected
            _mask_info = {
                "frame_mask": np.ones(F, dtype=np.float64),
                "joint_mask": np.ones(J, dtype=bool) if apply_to_poses else np.zeros(J, dtype=bool),
                "trans_corrupted": apply_to_trans,
            }

        meta = {
            "synthesis_type": "jittering",
            "description": f"Realistic Jitter ({intensity})",
            "synthesis_method": {
                "pattern_type": "algorithmic_noise",
                "intensity_level": intensity,
                "events_log": meta_logs,
            },
            "degradation_details": {"affected_components": list(set(affected_components))},
            "_mask_info": _mask_info,
        }
        return poses_curr, trans_curr, meta


if __name__ == "__main__":
    import argparse
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Test JitterCorruptor on real data from refine_hq.json")
    parser.add_argument("--anno", type=str, default="data/hymotion_m2m/anno/refine_hq.json")
    parser.add_argument("--data_dir", type=str, default="data/hymotion_m2m")
    parser.add_argument("--output_dir", type=str, default="output/test/corruptor/jitter_corruptor")
    parser.add_argument("--intensity", type=str, default="medium", choices=["low", "medium", "high"])
    parser.add_argument("--num_samples", type=int, default=3)
    args = parser.parse_args()

    with open(args.anno, "r", encoding="utf-8") as f:
        anno = json.load(f)
    paths = [v["tgt_motion_path"] for v in anno.values() if isinstance(v, dict) and "tgt_motion_path" in v]
    if not paths:
        paths = list(anno.keys())
    chosen = list(np.random.choice(paths, size=min(args.num_samples, len(paths)), replace=False))

    data_dir = Path(args.data_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    corruptor = JitterCorruptor(device="cpu")
    print(f"JitterCorruptor intensity={args.intensity}, samples={len(chosen)}")

    for i, rel_path in enumerate(chosen):
        npz_path = data_dir / rel_path
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
    print("JitterCorruptor test OK.")
