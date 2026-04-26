"""VerMo inference pipeline wrapper.

TASK_PROMPTS must use templates from training task definitions to avoid OOD
prompts.  Each entry is taken verbatim from the corresponding task class in
``hftrainer/models/motion/vermo/task_utils/task_lib/``.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from hftrainer.pipelines.base_pipeline import BasePipeline
from hftrainer.registry import PIPELINES


# Each prompt is copied verbatim from the training task's ``templates`` list so
# the model sees an in-distribution task description at inference time.
TASK_PROMPTS = {
    # Caption2Motion.templates[0]
    't2m_1p': 'Create motion from the given description',
    't2m_2p': 'Create motion from the given description',
    # Motion2Caption.templates[0]
    'm2t_1p': 'Caption the given motion.',
    'm2t_2p': 'Caption the given motion.',
    # Music2Dance.templates[42]  (short, no optional mention)
    'm2d': 'Dance to the given music.',
    # Dance2Music.templates[0]
    'd2m': 'Create music that matches the dance motion.',
    # Speech2Gesture.templates[0]
    's2g': 'Given the speech, generate the corresponding gesture motion.',
    # MotionPrediction.templates[0]
    'pred': 'Given the motion of past frames, predict the future motion',
    # MotionInbetween.templates[0]
    'inbetween': 'Given the motion of past frames and future frames, generate the middle frame',
}


@PIPELINES.register_module()
class VermoPipeline(BasePipeline):
    """HFTrainer wrapper around the vendored VerMo backend."""

    def __init__(self, bundle, **kwargs):
        super().__init__(bundle)
        from hftrainer.pipelines.motion.vermo_backend import VermoPipeline as VermoBackendPipeline

        self.backend = VermoBackendPipeline(
            vqvae=bundle.processor.motion_tokenizer,
            audio_tokenizer=bundle.processor.audio_tokenizer,
            text_tokenizer=bundle.processor.text_tokenizer,
            smpl_processor=bundle.processor.smpl_pose_processor,
            lm=bundle.lm,
        )

    def __call__(
        self,
        task: str,
        caption: Optional[str] = None,
        num_person: Optional[int] = None,
        duration: Optional[float] = None,
        music: Optional[str] = None,
        genre: Optional[str] = None,
        audio: Optional[str] = None,
        speech_script: Optional[str] = None,
        motion: Optional[str] = None,
        past_motion: Optional[str] = None,
        future_motion: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        if task not in TASK_PROMPTS:
            raise ValueError(f'Unsupported VerMo task: {task}')
        if task == 't2m_1p' and num_person is None:
            num_person = 1
        if task == 't2m_2p' and num_person is None:
            num_person = 2
        return self.backend(
            task_prompt=TASK_PROMPTS[task],
            num_person=num_person,
            caption=caption,
            duration=duration,
            music=music,
            genre=genre,
            audio=audio,
            speech_script=speech_script,
            past_motion=past_motion,
            future_motion=future_motion,
            motion=motion,
            **kwargs,
        )
