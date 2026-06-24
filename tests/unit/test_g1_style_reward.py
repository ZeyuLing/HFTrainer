import numpy as np
from types import SimpleNamespace

from hftrainer.models.motion.physflow.g1_style_reward import (
    G1StyleBank,
    categorize_style_text,
    qpos_style_feature,
)


def _qpos(speed=0.02, amp=0.2, frames=40):
    qpos = np.zeros((frames, 36), dtype=np.float32)
    qpos[:, 3] = 1.0  # identity wxyz quaternion
    qpos[:, 0] = np.arange(frames, dtype=np.float32) * speed
    phase = np.linspace(0.0, np.pi * 2.0, frames, dtype=np.float32)
    qpos[:, 7:] = amp * np.sin(phase)[:, None]
    return qpos


def test_qpos_style_feature_is_fixed_length_and_finite():
    feat = qpos_style_feature(_qpos(), length=30)
    assert feat.ndim == 1
    assert feat.shape[0] > 100
    assert np.isfinite(feat).all()


def test_style_bank_scores_near_motion_lower(tmp_path):
    walk = _qpos(speed=0.03, amp=0.25)
    stand = _qpos(speed=0.0, amp=0.01)
    bank = G1StyleBank.from_features(
        np.stack([qpos_style_feature(walk), qpos_style_feature(stand)], axis=0),
        labels=["locomotion", "standing"],
        paths=["walk", "stand"],
    )
    path = tmp_path / "style_bank.npz"
    bank.save(path)
    loaded = G1StyleBank.load(path)

    assert loaded.style_cost(_qpos(speed=0.031, amp=0.24), category="locomotion") < 0.1
    assert loaded.style_cost(_qpos(speed=0.0, amp=0.01), category="locomotion") > 0.1


def test_categorize_style_text_prefers_coarse_actions():
    assert categorize_style_text("Neutral_Run_A04_002") == "locomotion"
    assert categorize_style_text("Crowded_KickSth_A03_002") == "dynamic"
    assert categorize_style_text("Ballerina_Transition_A01_002") == "dance"


def test_trainer_style_cost_reweights_candidate_scores():
    from hftrainer.trainers.motion.physflow_trainer import PhysFlowTrainer

    walk = _qpos(speed=0.03, amp=0.25)
    stand = _qpos(speed=0.0, amp=0.01)
    bank = G1StyleBank.from_features(
        np.stack([qpos_style_feature(walk), qpos_style_feature(stand)], axis=0),
        labels=["locomotion", "standing"],
        paths=["walk", "stand"],
    )
    trainer = SimpleNamespace(_style_bank=bank, style_reward_weight=2.0)
    metrics = [{"score": 1.0}, {"score": 1.0}]

    PhysFlowTrainer._add_style_costs(
        trainer,
        metrics,
        np.stack([_qpos(speed=0.031, amp=0.24), _qpos(speed=0.0, amp=0.01)], axis=0),
        num_frames=[40],
        group_size=2,
        captions=["walk forward"],
    )

    assert metrics[0]["physical_score"] == 1.0
    assert metrics[0]["style_category"] == "locomotion"
    assert metrics[0]["score"] < metrics[1]["score"]
