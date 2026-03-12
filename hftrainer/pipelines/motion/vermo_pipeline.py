"""VerMo inference pipeline wrapper."""

from __future__ import annotations

from typing import Any, Dict, Optional

from hftrainer.pipelines.base_pipeline import BasePipeline
from hftrainer.registry import PIPELINES


TASK_PROMPTS = {
    't2m_1p': 'Generate motion sequence from the given caption.',
    't2m_2p': 'Generate multi-person motion sequence from the given caption.',
    'm2t_1p': 'Describe the given motion.',
    'm2t_2p': 'Describe the given multi-person motion.',
    'm2d': 'Dance to the given music.',
    'd2m': 'Add music to this dance.',
    's2g': 'Add body movements to speech.',
    'pred': 'Predict future motion from past motion.',
    'inbetween': 'Interpolate between two motion segments.',
}


@PIPELINES.register_module()
class VermoPipeline(BasePipeline):
    """HFTrainer wrapper around the vendored VerMo backend."""

    def __init__(self, bundle, **kwargs):
        super().__init__(bundle)
        self.backend = bundle.build_backend_pipeline()

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

