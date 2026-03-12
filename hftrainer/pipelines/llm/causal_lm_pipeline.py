"""CausalLM inference pipeline."""

import torch
from typing import Any, Dict, List, Optional, Union

from hftrainer.pipelines.base_pipeline import BasePipeline
from hftrainer.registry import PIPELINES


@PIPELINES.register_module()
class CausalLMPipeline(BasePipeline):
    """
    Inference pipeline for causal language models.

    Usage:
        pipeline = CausalLMPipeline(bundle=causal_lm_bundle)
        texts = pipeline(["What is the capital of France?"])
    """

    def __init__(
        self,
        bundle,
        max_new_tokens: int = 256,
        temperature: float = 1.0,
        do_sample: bool = True,
        **kwargs,
    ):
        super().__init__(bundle)
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.do_sample = do_sample

    @torch.no_grad()
    def __call__(
        self,
        prompts: Union[str, List[str]],
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        do_sample: Optional[bool] = None,
        **kwargs,
    ) -> List[str]:
        """
        Generate text completions.

        Args:
            prompts: single prompt or list of prompts
            max_new_tokens: override default max_new_tokens
            temperature: override default temperature
            do_sample: override default do_sample

        Returns:
            list of generated text strings
        """
        if isinstance(prompts, str):
            prompts = [prompts]

        return self.bundle.generate(
            prompts,
            max_new_tokens=max_new_tokens or self.max_new_tokens,
            temperature=temperature or self.temperature,
            do_sample=do_sample if do_sample is not None else self.do_sample,
            **kwargs,
        )
