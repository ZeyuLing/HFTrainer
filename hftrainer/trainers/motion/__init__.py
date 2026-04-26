"""Compatibility package for motion trainer imports."""

from hftrainer.trainers.motion.prism_trainer import PrismTrainer
from hftrainer.trainers.motion.vermo_trainer import VermoTrainer
from hftrainer.trainers.motion.hymotion_m2m_trainer import HyMotionM2MTrainer
from hftrainer.trainers.motion.hymotion_m2m_soar_trainer import HyMotionM2MSoarTrainer
from hftrainer.trainers.motion.hymotion_t2m_trainer import HyMotionT2MTrainer
from hftrainer.trainers.motion.hymotion_umo_trainer import HyMotionUMOTrainer

__all__ = [
    'PrismTrainer', 'VermoTrainer',
    'HyMotionM2MTrainer', 'HyMotionM2MSoarTrainer',
    'HyMotionT2MTrainer', 'HyMotionUMOTrainer',
]
