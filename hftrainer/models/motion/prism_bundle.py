"""PRISM motion-generation bundle."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Union

import torch
from einops import rearrange

from hftrainer.models.base_model_bundle import ModelBundle
from hftrainer.models.motion.components.gaussian_distribution import (
    DiagonalGaussianDistributionNd,
)
from hftrainer.registry import MODEL_BUNDLES


def _get_sigmas(scheduler, timesteps, n_dim: int = 4, dtype=torch.float32):
    device = timesteps.device
    sigmas = scheduler.sigmas.to(device=device, dtype=dtype)
    schedule_timesteps = scheduler.timesteps.to(device=device)
    step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]
    sigma = sigmas[step_indices].flatten()
    while sigma.ndim < n_dim:
        sigma = sigma.unsqueeze(-1)
    return sigma


@MODEL_BUNDLES.register_module()
class PrismBundle(ModelBundle):
    """ModelBundle for PRISM text-to-motion / pose-conditioned motion."""

    def __init__(
        self,
        transformer: dict,
        vae: dict,
        tokenizer: dict,
        text_encoder: dict,
        scheduler: dict,
        smpl_pose_processor: dict,
    ):
        super().__init__()
        self._build_modules(
            {
                'transformer': transformer,
                'vae': vae,
                'tokenizer': tokenizer,
                'text_encoder': text_encoder,
                'scheduler': scheduler,
                'smpl_pose_processor': smpl_pose_processor,
            }
        )
        if hasattr(self.scheduler, 'set_timesteps'):
            self.scheduler.set_timesteps(self.scheduler.config.num_train_timesteps)
        self.use_static = bool(getattr(self.vae.config, 'use_static', False))
        self.register_buffer(
            'latents_mean',
            torch.tensor(self.vae.config.latents_mean).view(1, self.vae.config.z_dim, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            'latents_std',
            torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1),
            persistent=False,
        )

    @classmethod
    def _bundle_config_from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        transformer_type: str = 'PrismTransformerMotionModel',
        vae_type: str = 'AutoencoderKLPrism2DTK',
        tokenizer_type: str = 'AutoTokenizer',
        text_encoder_type: str = 'UMT5EncoderModel',
        scheduler_type: str = 'FlowMatchEulerDiscreteScheduler',
        smpl_pose_processor_cfg: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        root = os.path.abspath(os.path.expanduser(pretrained_model_name_or_path))
        processor_cfg_path = os.path.join(root, 'smpl_pose_processor.json')
        if smpl_pose_processor_cfg is None and os.path.isfile(processor_cfg_path):
            with open(processor_cfg_path, 'r', encoding='utf-8') as f:
                smpl_pose_processor_cfg = json.load(f)
        smpl_pose_processor_cfg = smpl_pose_processor_cfg or {
            'type': 'SMPLPoseProcessor',
            'smpl_model': None,
            'smooth_model': None,
        }
        return {
            'transformer': {
                'type': transformer_type,
                'from_pretrained': {'pretrained_model_name_or_path': os.path.join(root, 'transformer')},
            },
            'vae': {
                'type': vae_type,
                'from_pretrained': {'pretrained_model_name_or_path': os.path.join(root, 'vae')},
            },
            'tokenizer': {
                'type': tokenizer_type,
                'from_pretrained': {'pretrained_model_name_or_path': os.path.join(root, 'tokenizer')},
            },
            'text_encoder': {
                'type': text_encoder_type,
                'from_pretrained': {'pretrained_model_name_or_path': os.path.join(root, 'text_encoder')},
            },
            'scheduler': {
                'type': scheduler_type,
                'from_pretrained': {'pretrained_model_name_or_path': os.path.join(root, 'scheduler')},
            },
            'smpl_pose_processor': smpl_pose_processor_cfg,
        }

    def save_pretrained(self, save_directory: str, **kwargs):
        os.makedirs(save_directory, exist_ok=True)
        self.transformer.save_pretrained(os.path.join(save_directory, 'transformer'), **kwargs)
        self.vae.save_pretrained(os.path.join(save_directory, 'vae'), **kwargs)
        self.text_encoder.save_pretrained(os.path.join(save_directory, 'text_encoder'), **kwargs)
        self.tokenizer.save_pretrained(os.path.join(save_directory, 'tokenizer'))
        self.scheduler.save_pretrained(os.path.join(save_directory, 'scheduler'))
        with open(os.path.join(save_directory, 'smpl_pose_processor.json'), 'w', encoding='utf-8') as f:
            json.dump(self.get_module_build_cfg('smpl_pose_processor'), f, ensure_ascii=False, indent=2)
        with open(os.path.join(save_directory, 'config.json'), 'w', encoding='utf-8') as f:
            json.dump({'model_type': 'hftrainer_prism'}, f, ensure_ascii=False, indent=2)

    @torch.no_grad()
    def encode_motion(self, motion: torch.Tensor, fk_chunk_size: int = 8) -> torch.Tensor:
        motion = motion.float()
        if motion.ndim == 2:
            motion = motion.unsqueeze(0)
        batch_size = motion.shape[0]

        if self.use_static:
            if batch_size > fk_chunk_size:
                static_chunks = []
                for idx in range(0, batch_size, fk_chunk_size):
                    static_chunks.append(
                        self.smpl_pose_processor.get_static_joint_mask_from_motion(motion[idx: idx + fk_chunk_size])
                    )
                static_joints = torch.cat(static_chunks, dim=0)
            else:
                static_joints = self.smpl_pose_processor.get_static_joint_mask_from_motion(motion)

        motion = self.smpl_pose_processor.normalize(motion)
        if self.use_static:
            motion = torch.cat([motion, static_joints], dim=-1)
        motion = rearrange(motion, 'b t (j d) -> b t j d', d=6)

        latents = self.vae.encode(motion)
        latents = DiagonalGaussianDistributionNd(latents).mode()
        latents = (latents - self.latents_mean.to(latents)) / self.latents_std.to(latents)
        return latents

    @torch.no_grad()
    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        max_sequence_length: int = 128,
        prompt_drop_rate: float = 0.0,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        device = next(self.text_encoder.parameters()).device
        dtype = dtype or next(self.text_encoder.parameters()).dtype
        prompt = [prompt] if isinstance(prompt, str) else prompt
        if prompt_drop_rate > 0:
            prompt = ['' if torch.rand(1).item() < prompt_drop_rate else p for p in prompt]

        text_inputs = self.tokenizer(
            prompt,
            padding='max_length',
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        input_ids = text_inputs.input_ids.to(device)
        attention_mask = text_inputs.attention_mask.to(device)
        seq_lens = attention_mask.gt(0).sum(dim=1).long()

        outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        prompt_embeds = outputs.last_hidden_state.to(device=device, dtype=dtype)
        prompt_embeds = [emb[:seq_len] for emb, seq_len in zip(prompt_embeds, seq_lens)]
        prompt_embeds = torch.stack(
            [
                torch.cat([emb, emb.new_zeros(max_sequence_length - emb.size(0), emb.size(1))])
                for emb in prompt_embeds
            ],
            dim=0,
        )
        return prompt_embeds

    def create_padding_mask(
        self,
        num_frames: Optional[torch.Tensor],
        batch_size: int,
        latent_frames: int,
        latent_joints: int,
        device: torch.device,
    ) -> torch.Tensor:
        if num_frames is None:
            return torch.ones(batch_size, latent_frames, latent_joints, device=device)

        num_frames = num_frames.to(device)
        scale_factor = self.vae.config.scale_factor_temporal
        num_frames_vae = (num_frames + scale_factor - 1) // scale_factor
        num_frames_vae = torch.clamp(num_frames_vae, min=0, max=latent_frames)
        frame_idx = torch.arange(latent_frames, device=device).unsqueeze(0)
        mask = frame_idx < num_frames_vae.unsqueeze(1)
        return mask.unsqueeze(-1).expand(batch_size, latent_frames, latent_joints).float()

    def create_condition_mask(
        self,
        latents: torch.Tensor,
        frame_condition_rate: float = 0.1,
        condition_num_frames: Union[int, List[int]] = 1,
    ) -> torch.Tensor:
        batch_size, _, latent_frames, latent_joints = latents.shape
        device = latents.device
        if frame_condition_rate <= 0:
            return torch.ones((batch_size, 1, latent_frames, latent_joints), dtype=torch.bool, device=device)

        if isinstance(condition_num_frames, int):
            condition_num_frames = [condition_num_frames]
        cond_candidates = torch.tensor(list(condition_num_frames), dtype=torch.long, device=device)
        idx = torch.randint(0, len(cond_candidates), (batch_size,), device=device)
        num_cond_orig = cond_candidates[idx]
        downsample = self.vae.config.scale_factor_temporal
        num_cond_vae = (num_cond_orig + downsample - 1) // downsample
        num_cond_vae = torch.clamp(num_cond_vae, min=0, max=latent_frames)
        do_condition = torch.rand(batch_size, device=device) < float(frame_condition_rate)
        num_cond_sel = num_cond_vae * do_condition.long()
        frame_idx = torch.arange(latent_frames, device=device).unsqueeze(0)
        cond_frame_mask = frame_idx < num_cond_sel.unsqueeze(1)
        mask = (~cond_frame_mask).unsqueeze(1).unsqueeze(-1)
        return mask.expand(batch_size, 1, latent_frames, latent_joints).to(torch.bool)

    def create_sequence_ts(
        self,
        ori_ts: torch.Tensor,
        condition_frame_mask_vae: torch.Tensor,
        patch_size=(1, 1),
    ) -> torch.Tensor:
        batch_size, _, latent_frames, latent_joints = condition_frame_mask_vae.shape
        post_patch_num_frames = latent_frames // patch_size[0]
        post_patch_num_joints = latent_joints // patch_size[1]
        target_ts = ori_ts.unsqueeze(1).unsqueeze(2).expand(batch_size, post_patch_num_frames, post_patch_num_joints)
        target_ts = torch.where(
            condition_frame_mask_vae[:, 0, :: patch_size[0], :: patch_size[1]],
            target_ts,
            0,
        )
        return target_ts.flatten(1)

    def add_flow_noise(self, latents: torch.Tensor, timesteps: torch.Tensor):
        noise = torch.randn_like(latents)
        sigmas = _get_sigmas(self.scheduler, timesteps, n_dim=latents.ndim, dtype=latents.dtype)
        noisy_latents = (1 - sigmas) * latents + sigmas * noise
        targets = noise - latents
        return noisy_latents, targets

