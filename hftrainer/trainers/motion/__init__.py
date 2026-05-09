"""Compatibility package for motion trainer imports."""

from hftrainer.trainers.motion.prism_trainer import PrismTrainer
from hftrainer.trainers.motion.vermo_trainer import VermoTrainer
from hftrainer.trainers.motion.hymotion_m2m_trainer import HyMotionM2MTrainer
from hftrainer.trainers.motion.hymotion_m2m_soar_trainer import HyMotionM2MSoarTrainer
from hftrainer.trainers.motion.hymotion_m2m_crfm_trainer import HyMotionM2MCRFMTrainer
from hftrainer.trainers.motion.hymotion_t2m_trainer import HyMotionT2MTrainer
from hftrainer.trainers.motion.hymotion_umo_trainer import HyMotionUMOTrainer
from hftrainer.trainers.motion.motion_clip_trainer import MotionCLIPTrainer

__all__ = [
    'PrismTrainer', 'VermoTrainer',
    'HyMotionM2MTrainer', 'HyMotionM2MSoarTrainer', 'HyMotionM2MCRFMTrainer',
    'HyMotionT2MTrainer', 'HyMotionUMOTrainer',
    'MotionCLIPTrainer',
]
