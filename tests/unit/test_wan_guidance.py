"""Negative conditioning must influence Wan's denoising trajectory."""

from types import SimpleNamespace

import torch

from hftrainer.models.wan.network.scheduler import FlowMatchEulerDiscreteScheduler
from hftrainer.pipelines.wan.pipeline import WanPipeline


def test_guidance_uses_negative_branch_and_preserves_fp32_state():
    model = torch.nn.Linear(1, 1).to(torch.bfloat16)
    model.config = SimpleNamespace(in_channels=3)
    encoded, predicted, decoded = [], [], []

    def encode(prompts):
        encoded.append(prompts)
        return torch.tensor([1.0 if p == "positive" else 0.0 for p in prompts])

    def predict(latents, timestep, context):
        predicted.append(latents.dtype)
        return torch.ones_like(latents) * context.reshape(-1, 1, 1, 1, 1)

    def decode(latents):
        decoded.append(latents.clone())
        return latents

    bundle = SimpleNamespace(
        transformer=model,
        scheduler=FlowMatchEulerDiscreteScheduler(),
        encode_text=encode,
        predict_noise=predict,
        decode_latent=decode,
        eval=lambda: None,
    )
    pipeline = WanPipeline(
        bundle, num_inference_steps=2, num_frames=1, height=16, width=16
    )
    torch.manual_seed(7)
    pipeline("positive", negative_prompt="negative", guidance_scale=5.0)
    guided = decoded[-1]
    torch.manual_seed(7)
    pipeline("positive", guidance_scale=1.0)
    assert encoded == [["positive"], ["negative"], ["positive"]]
    assert len(predicted) == 6 and all(d == torch.bfloat16 for d in predicted)
    assert guided.dtype == torch.float32
    torch.testing.assert_close(guided - decoded[-1], torch.full_like(guided, -4.0))
