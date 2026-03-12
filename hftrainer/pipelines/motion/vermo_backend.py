import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from einops import rearrange
import numpy as np
import torch
from hftrainer.models.motion.vermo_task_utils import LOCATABLE_MODALS
from hftrainer.models.motion.vermo_task_utils.modality import (
    Audio,
    Caption,
    Duration,
    FutureMotion,
    Genre,
    Modality,
    Motion,
    Music,
    NumPerson,
    PastMusic,
    PastMotion,
    SpeechScript,
    is_modal,
)
from hftrainer.models.motion.components import (
    VQVAEWanMotion2DTK,
)
from hftrainer.models.motion.components import (
    VQVAEWanMotion1D,
)
from hftrainer.models.motion.components import WavTokenizer
from hftrainer.models.motion.components.motion_processor.smpl_processor import SMPLPoseProcessor

from hftrainer.trainers.motion.vermo_trainer import VermoTrainer
from hftrainer.models.motion.utils import write_json
from hftrainer.models.motion.utils import write_txt
from hftrainer.models.motion.components.utils.geometry.rotation_convert import rotation_6d_to_axis_angle
from transformers import (
    LlamaForCausalLM,
    PreTrainedTokenizer,
)
from mmengine.device import get_device


def _import_librosa():
    try:
        import librosa
    except Exception as exc:
        raise RuntimeError(
            "Audio loading requires librosa and its runtime dependencies "
            "(for example libsndfile)."
        ) from exc
    return librosa


def _import_torchaudio():
    try:
        import torchaudio
    except Exception as exc:
        raise RuntimeError(
            "Audio saving requires torchaudio and matching torch/torchaudio binaries."
        ) from exc
    return torchaudio


class VermoPipeline:
    lm: LlamaForCausalLM

    def __init__(self, vqvae, audio_tokenizer, text_tokenizer, smpl_processor, lm):

        self.device = get_device()
        self.dtype = torch.float32

        self.vqvae: VQVAEWanMotion2DTK = vqvae.to(self.device, self.dtype)
        self.audio_tokenizer: WavTokenizer = audio_tokenizer.to(self.device, self.dtype) if audio_tokenizer is not None else None
        self.text_tokenizer: PreTrainedTokenizer = text_tokenizer
        self.smpl_processor: SMPLPoseProcessor = smpl_processor.to(
            self.device, self.dtype
        )
        self.lm: LlamaForCausalLM = lm.to(self.device, self.dtype)
        self.set_special_tokens()

    def set_special_tokens(self):
        self.cond_bos = "<|begin_of_condition|>"
        self.cond_eos = "<|end_of_condition|>"
        self.task_bos = "<|begin_of_task_template|>"
        self.task_eos = "<|end_of_task_template|>"
        self.output_bos = "<|begin_of_output|>"
        self.output_bos_id = self.text_tokenizer.convert_tokens_to_ids(self.output_bos)

        self.mp_separator = Motion.mp_separator

    def load_audio(self, audio: Union[str, np.ndarray], sr: int = 24000):
        if isinstance(audio, str):
            librosa = _import_librosa()
            audio, _ = librosa.load(audio, sr=sr, mono=True)
        audio = torch.from_numpy(audio).unsqueeze((0)).to(self.device, self.dtype)
        return audio

    def load_motion(self, motion: Union[str, List[str], Dict, List[Dict]]):
        if isinstance(motion, str):
            motion = [motion]

        if isinstance(motion, dict):
            motion = [motion]

        all_person_motion = []
        for m in motion:
            if isinstance(m, str):
                m = self.smpl_processor.load_smplx_dict_from_npz(m)
            else:
                assert isinstance(m, dict)
            all_person_motion.append(m)

        # p t (j d)
        motion_vec = torch.stack(
            [
                self.smpl_processor.smplx_dict_to_motion_vector(p)
                for p in all_person_motion
            ]
        ).to(self.device, self.dtype)
        # p t (j d)
        if self.vqvae.config.use_static:
            static_joints = self.smpl_processor.get_static_joint_mask_from_motion(
                motion_vec
            )
            motion_vec = self.smpl_processor.normalize(motion_vec)
            motion_vec = torch.cat([motion_vec, static_joints], dim=-1)
        else:
            motion_vec = self.smpl_processor.normalize(motion_vec)

        if isinstance(self.vqvae, VQVAEWanMotion1D):
            # 1D: keep [P, T, C]
            return motion_vec, motion_vec.shape[0]
        else:
            # 2D: reshape to [P, T, J, D]
            motion_vec = rearrange(motion_vec, "p t (j d) -> p t j d", d=6)
            return motion_vec, motion_vec.shape[0]

    def prepare_input_str(
        self,
        task_prompt: str,
        num_person: Optional[int] = None,
        caption: Optional[str] = None,
        duration: Optional[float] = None,
        # dance related
        music: Optional[Union[str, np.ndarray]] = None,
        past_music: Optional[Union[str, np.ndarray]] = None,
        genre: Optional[str] = None,
        # speech related
        audio: Optional[Union[str, np.ndarray]] = None,
        speech_script: Optional[str] = None,
        # motion completion related
        past_motion: Optional[Union[str, Dict]] = None,
        future_motion: Optional[Union[str, Dict]] = None,
        # motion understanding related
        motion: Union[str, Dict] = None,
        device: Optional[Union[str, torch.device]] = None,
        dtype=torch.float32,
    ):
        if device is None:
            device = self.device
        # loading from file or numpy array
        music = self.load_audio(music) if music is not None else None
        past_music = self.load_audio(past_music) if past_music is not None else None
        audio = self.load_audio(audio) if audio is not None else None

        if past_motion is not None:
            past_motion, num_person = self.load_motion(past_motion)

        if future_motion is not None:
            future_motion, num_person = self.load_motion(future_motion)

        if motion is not None:
            motion, num_person = self.load_motion(motion)

        user_input = self.task_bos + task_prompt + self.task_eos + self.cond_bos

        if caption is not None:
            user_input = user_input + Caption.bos + caption + Caption.eos
        if num_person is not None:
            user_input = user_input + NumPerson.bos + str(num_person) + NumPerson.eos

        if duration is not None:
            user_input = user_input + Duration.bos + f"{duration:.1f}" + Duration.eos

        if genre is not None:
            user_input = user_input + Genre.bos + genre + Genre.eos

        if music is not None:
            music_ids = self.encode_audio(music.to(device, dtype))
            music_str = Music.index_to_string(music_ids)
            user_input = user_input + Music.bos + music_str + Music.eos

        if past_music is not None:
            past_music_ids = self.encode_audio(past_music.to(device, dtype))
            past_music_str = PastMusic.index_to_string(past_music_ids)
            user_input = user_input + PastMusic.bos + past_music_str + PastMusic.eos

        if audio is not None:
            audio_ids = self.encode_audio(audio.to(device, dtype))
            audio_str = Audio.index_to_string(audio_ids)
            user_input = user_input + Audio.bos + audio_str + Audio.eos

        if speech_script is not None:
            user_input = (
                user_input + SpeechScript.bos + speech_script + SpeechScript.eos
            )

        if past_motion is not None:
            past_motion_ids = self.encode_motion(past_motion.to(device, dtype))
            past_motion_str = PastMotion.index_to_string(past_motion_ids)
            user_input = user_input + PastMotion.bos + past_motion_str + PastMotion.eos

        if future_motion is not None:
            future_motion_ids = self.encode_motion(future_motion.to(device, dtype))
            future_motion_str = FutureMotion.index_to_string(future_motion_ids)
            user_input = (
                user_input + FutureMotion.bos + future_motion_str + FutureMotion.eos
            )

        if motion is not None:
            motion_ids = self.encode_motion(motion.to(device, dtype))
            motion_str = Motion.index_to_string(motion_ids)
            user_input = user_input + Motion.bos + motion_str + Motion.eos

        user_input = user_input + self.cond_eos

        messages = [
            {
                "role": "user",
                "content": user_input,
            },
        ]

        return messages

    def encode_motion(self, motion: torch.Tensor):
        if isinstance(self.vqvae, VQVAEWanMotion1D):
            indices = self.vqvae.encode(motion).indices
            if indices.ndim == 3:
                # RVQ: [B, N, Q] → offset + interleave → [B, N*Q]
                B, N, Q = indices.shape
                cb = self.vqvae.codebook_size
                offsets = torch.arange(Q, device=indices.device) * cb
                indices = (indices + offsets).reshape(B, N * Q)
            return indices.squeeze(0)
        else:
            return self.vqvae.encode(motion, flatten=True).indices.squeeze(0)

    def post_process_motion(
        self,
        x_dec,
        use_static: bool = False,
        use_smooth: bool = False,
        mocap_framerate: float = 30.0,
        gender: str = "neutral",
    ) -> Union[List, Dict]:
        if x_dec.ndim == 4:
            x_dec = rearrange(x_dec, "p t j d -> p t (j d)")
        num_person = x_dec.shape[0]

        vqvae_has_static = getattr(self.vqvae.config, "use_static", False)
        if vqvae_has_static:
            x_dec = x_dec[..., :-6]

        x_dec = self.smpl_processor.denormalize(x_dec)

        transl_abs_rel = x_dec[..., :6]
        transl = self.smpl_processor.inv_convert_transl(transl_abs_rel, use_rollout=True)
        pred_poses = x_dec[..., 6:]

        pred_poses = rearrange(pred_poses, "p t (j d)-> (p t) j d", d=6)

        pred_poses = rotation_6d_to_axis_angle(pred_poses)
        pred_poses = rearrange(pred_poses, "(p t) j d -> p t (j d)", p=num_person)

        if use_static:
            pred_poses = self.smpl_processor.post_hoc_static_refine(
                transl, pred_poses, rot_type="axis_angle"
            )

        pred_smplx_dict_all_person = []
        for t, p in zip(transl, pred_poses):
            pred_smplx_dict = self.smpl_processor.transl_pose_to_smplx_dict(
                t,
                p,
                mocap_framerate=mocap_framerate,
                gender=gender,
                rot_type="axis_angle",
            )

            if use_smooth:
                pred_smplx_dict = self.smpl_processor.smooth_smplx_dict(pred_smplx_dict)
            pred_smplx_dict_all_person.append(pred_smplx_dict)
        if len(pred_smplx_dict_all_person) == 1:
            pred_smplx_dict_all_person = pred_smplx_dict_all_person[0]
        return pred_smplx_dict_all_person

    def encode_audio(self, audio: torch.Tensor):
        audio_ids = self.audio_tokenizer.encode(audio)[1].squeeze(0)
        return audio_ids

    def encode_text(
        self,
        text: str,
    ) -> torch.Tensor:
        # encode the text into batch_encodings
        # including input_ids, attention_mask
        lm_input = self.text_tokenizer(
            [text],
            add_special_tokens=False,
            padding=False,
            return_overflowing_tokens=False,
            return_length=False,
            return_attention_mask=True,
            return_tensors="pt",
        ).to(self.device)
        # remove the unused keys
        lm_input.pop("token_type_ids", None)
        return lm_input

    def locate_modal(self, text: str) -> Dict:
        """
            1, Firstly, fetch special tokens from the output of causal lm
            2, The predicted modal of each sample may differ in each batch,
            once a sample A has modal X, other samples should make dummy index
            to keep synchronization with sample A. We use [0] as the dummy index
        :param batch_text: causal lm predicted text
        :return: Modality -> corresponding sub string in the LLM output text.
        """
        modal_dict = {}
        for modal in LOCATABLE_MODALS:
            match_text: list = modal.locate_modality(text)
            if len(match_text) and len(match_text[0]):
                modal_dict[modal] = match_text[0]
        return modal_dict

    def text_to_output_modal(
        self,
        text: str,
        use_static: bool = False,
        use_smooth: bool = False,
        mocap_framerate: float = 30.0,
        gender: str = "neutral",
    ):
        modal_dict: Dict[Modality, str] = self.locate_modal(text)
        output_dict = {}
        output_modal = text
        for modal, modal_text in modal_dict.items():
            if is_modal(modal, Motion):
                # tensor
                output_modal = self.string2motion(modal_text)
                # to smplx_dict
                output_modal = self.post_process_motion(
                    output_modal,
                    use_static=use_static,
                    use_smooth=use_smooth,
                    mocap_framerate=mocap_framerate,
                    gender=gender,
                )
            elif is_modal(modal, Audio):
                output_modal = self.string2audio(modal_text)
            else:
                output_modal = modal_text
            output_dict[modal] = output_modal
        return output_dict

    def string2motion(self, text) -> torch.Tensor:
        motion_ids = Motion.string_to_index(text, return_tensor=True)
        if motion_ids is None or motion_ids.numel() == 0:
            return None

        motion_ids = motion_ids.squeeze(0).to(dtype=torch.long)

        if isinstance(self.vqvae, VQVAEWanMotion1D):
            return self._decode_1d_rvq(motion_ids)
        else:
            return self._decode_2d(motion_ids)

    def _decode_1d_rvq(self, motion_ids: torch.Tensor) -> torch.Tensor:
        """Decode offset-encoded RVQ indices back to motion for 1D VQ-VAE."""
        cb = self.vqvae.codebook_size
        num_q = getattr(self.vqvae.quantizer, 'num_quantizers', 1)

        if motion_ids.ndim == 2:
            # Multi-person: [P, N_flat] — decode each person separately
            results = []
            for p in range(motion_ids.shape[0]):
                results.append(self._decode_1d_rvq(motion_ids[p]))
            return torch.cat(results, dim=0)  # [P, T, C]

        # motion_ids: [N_flat] — single person, interleaved offset-encoded
        ids = motion_ids.to(device=self.device, dtype=torch.long)

        if num_q > 1:
            # De-interleave and de-offset:
            # ids are interleaved [t0_q0, t0_q1, ..., t0_q5, t1_q0, ...]
            N_flat = ids.shape[0]
            N_flat = N_flat // num_q * num_q  # truncate to multiple of num_q
            ids = ids[:N_flat]
            ids_reshaped = ids.reshape(-1, num_q)  # [N, num_q]
            # Remove per-layer offset
            offsets = torch.arange(num_q, device=ids.device) * cb
            raw_ids = ids_reshaped - offsets  # [N, num_q], values in 0..cb-1

            # Dequantize per-layer and sum (avoids ResidualVQ.codebooks property)
            quantizer = self.vqvae.quantizer
            latent = torch.zeros(raw_ids.shape[0], self.vqvae.z_dim,
                                 device=ids.device, dtype=self.dtype)
            for qi, layer_vq in enumerate(quantizer.layers):
                latent = latent + layer_vq.dequantize(raw_ids[:, qi])  # [N, D]
            latent = latent.unsqueeze(0).permute(0, 2, 1)  # [1, D, N]
        else:
            # Single quantizer: ids are raw indices
            layer_vq = self.vqvae.quantizer.layers[0]
            latent = layer_vq.dequantize(ids).unsqueeze(0)  # [1, N, D]
            latent = latent.permute(0, 2, 1)  # [1, D, N]

        # Decode latent → motion through post_quant_conv + decoder
        decoded = self.vqvae._decode(latent, is_indices=False)  # [1, C, T_out]
        return decoded.permute(0, 2, 1)  # [1, T_out, C]

    def _decode_2d(self, motion_ids: torch.Tensor) -> torch.Tensor:
        """Decode indices for 2D VQ-VAE.
        num_joints is determined by VQ-VAE config (24 if trained with static channel, else 23).
        """
        num_joints = 24 if getattr(self.vqvae.config, "use_static", False) else 23

        if motion_ids.shape[-1] % num_joints != 0:
            motion_ids = motion_ids[..., : motion_ids.shape[-1] // num_joints * num_joints]

        if motion_ids.ndim == 1:
            motion_ids = rearrange(motion_ids, "(t j) -> t j", j=num_joints)
            motion = self.vqvae.decode(
                motion_ids.unsqueeze(0).to(self.device, torch.long),
                is_indices=True,
            )
        else:
            motion_ids = rearrange(motion_ids, "p (t j) -> p t j", j=num_joints)
            motion = self.vqvae.decode(
                motion_ids.to(self.device, torch.long),
                is_indices=True,
            )

        return motion

    def string2audio(self, text) -> torch.Tensor:
        audio_ids = Audio.string_to_index(text, return_tensor=True)
        if audio_ids is None or audio_ids.numel() == 0:
            audio = None
        else:
            audio = self.audio_tokenizer.decode(
                audio_ids.unsqueeze(0).to(device=self.device, dtype=torch.long),
                is_idx=True,
            ).squeeze(0)
        return audio

    def idx2text(self, ids: torch.Tensor) -> str:
        return self.text_tokenizer.decode(ids, skip_special_tokens=False)

    def __call__(
        self,
        task_prompt: str,
        num_person: Optional[int] = None,
        caption: Optional[str] = None,
        duration: Optional[float] = None,
        # dance related
        music: Optional[Union[str, np.ndarray]] = None,
        past_music: Optional[Union[str, np.ndarray]] = None,
        genre: Optional[str] = None,
        # speech related
        audio: Optional[Union[str, np.ndarray]] = None,
        speech_script: Optional[str] = None,
        # motion completion related
        past_motion: Optional[Union[str, Dict]] = None,
        future_motion: Optional[Union[str, Dict]] = None,
        # motion understanding related
        motion: Union[str, Dict] = None,
        do_sample: bool = True,
        temperature: float = 0.8,
        top_k: int = 50,
        max_new_tokens: int = 8192,
        num_beams: int = 1,
        repetition_penalty: float = 1.2,
    ):
        # loading prepare input conditions

        messages = self.prepare_input_str(
            task_prompt=task_prompt,
            num_person=num_person,
            caption=caption,
            duration=duration,
            # dance related
            music=music,
            past_music=past_music,
            genre=genre,
            # speech related
            audio=audio,
            speech_script=speech_script,
            # motion completion related
            past_motion=past_motion,
            future_motion=future_motion,
            # motion understanding related
            motion=motion,
        )

        # 使用与训练时一致的格式：需要手动添加 assistant header
        # 因为自定义的 chat_template 不支持 add_generation_prompt
        lm_input_str = (
            self.text_tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,  # 改为 False，手动添加 assistant header
            )
            + "<|start_header_id|>assistant<|end_header_id|>\n\n"  # 手动添加 assistant header
            + self.output_bos
        )

        # Debug: 打印推理时的输入格式，确保与训练一致
        print("[DEBUG] Inference input format:")
        print(
            repr(lm_input_str[:500]) + "..."
            if len(lm_input_str) > 500
            else repr(lm_input_str)
        )

        lm_input = self.text_tokenizer([lm_input_str], return_tensors="pt").to(
            self.device
        )
        lm_input.pop("token_type_ids", None)

        generate_kwargs = dict(
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            repetition_penalty=repetition_penalty,
        )
        if do_sample:
            generate_kwargs.update(
                do_sample=True,
                temperature=temperature,
                top_k=top_k,
            )
        else:
            generate_kwargs["do_sample"] = False

        generated_ids = self.lm.generate(**lm_input, **generate_kwargs)
        output_ids = generated_ids[0][len(lm_input.input_ids[0]) :].tolist()

        content = self.text_tokenizer.decode(output_ids, skip_special_tokens=False)
        print(content)

        output_dict = self.text_to_output_modal(
            content, use_smooth=False, use_static=False
        )
        output_dict["message"] = messages
        output_dict["response"] = content
        output_dict["output_ids"] = output_ids
        return output_dict

    def save_out_dict(self, output_dict, output_path: str):
        os.makedirs(output_path, exist_ok=True)
        new_dict = {}
        for modal, output_modal in output_dict.items():
            if isinstance(modal, str):
                new_dict[modal] = output_modal
            elif is_modal(modal, Audio):
                assert isinstance(output_modal, torch.Tensor)
                output_modal = output_modal.detach().cpu()
                if output_modal.ndim == 1:
                    output_modal = output_modal.unsqueeze(0)
                torchaudio = _import_torchaudio()
                torchaudio.save(
                    os.path.join(output_path, f"{modal.name}.wav"), output_modal, 24000
                )
            elif is_modal(modal, Motion):
                if isinstance(output_modal, list):
                    for p_idx, p_motion in enumerate(output_modal):
                        self.smpl_processor.save_smplx_npz(
                            os.path.join(output_path, f"{modal.name}_{p_idx}.npz"),
                            p_motion,
                        )
                else:
                    self.smpl_processor.save_smplx_npz(
                        os.path.join(output_path, f"{modal.name}.npz"),
                        output_modal,
                    )
            else:
                write_txt(os.path.join(output_path, f"{modal.name}.txt"), output_modal)
                new_dict[modal.name] = output_modal
        write_json(os.path.join(output_path, "output.json"), new_dict)
        print("save output dict to", output_path)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _sort_checkpoint_key(path: Path) -> Tuple[int, int, str]:
    match = re.search(r"(iter|epoch)_(\d+)\.pth$", path.name)
    if match is None:
        return (2, -1, path.name)
    prefix_rank = 0 if match.group(1) == "iter" else 1
    return (prefix_rank, int(match.group(2)), path.name)


def resolve_checkpoint_path(cfg: str, checkpoint: str = "auto") -> str:
    if checkpoint not in (None, "", "auto"):
        return str(Path(checkpoint).expanduser().resolve())

    from mmengine import Config

    project_root = _project_root()
    cfg_path = Path(cfg)
    cfg_obj = Config.fromfile(str(cfg_path))
    work_dir = cfg_obj.get(
        "work_dir",
        str(project_root / "work_dirs" / cfg_path.stem),
    )
    work_dir = Path(work_dir)
    if not work_dir.is_absolute():
        work_dir = (project_root / work_dir).resolve()

    last_checkpoint = work_dir / "last_checkpoint"
    if last_checkpoint.is_file():
        resolved = last_checkpoint.read_text().strip()
        resolved_path = Path(resolved)
        if not resolved_path.is_absolute():
            resolved_path = (project_root / resolved_path).resolve()
        if resolved_path.is_file():
            return str(resolved_path)

    candidates = sorted(work_dir.glob("iter_*.pth"), key=_sort_checkpoint_key)
    if not candidates:
        candidates = sorted(work_dir.glob("epoch_*.pth"), key=_sort_checkpoint_key)
    if not candidates:
        raise FileNotFoundError(
            f"Cannot resolve checkpoint for cfg={cfg}. "
            f"No last_checkpoint / iter_*.pth / epoch_*.pth under {work_dir}"
        )
    return str(candidates[-1].resolve())


def build_pipeline_from_config(
    cfg: str,
    checkpoint: str = "auto",
    tokenizer_save_path: Optional[str] = None,
):
    from hftrainer.registry import MODELS
    from mmengine import Config
    from mmengine.runner import load_checkpoint

    checkpoint_path = resolve_checkpoint_path(cfg, checkpoint)
    cfg_model = Config.fromfile(cfg)["model"]

    trainer: VermoTrainer = MODELS.build(cfg_model)
    load_checkpoint(trainer, checkpoint_path, map_location="cpu", strict=True)

    if tokenizer_save_path is not None:
        trainer.processor.text_tokenizer.save_pretrained(tokenizer_save_path)

    vqvae = trainer.processor.motion_tokenizer
    smpl_processor = trainer.processor.smpl_pose_processor
    audio_tokenizer = trainer.processor.audio_tokenizer

    print(f"[INFO] Using checkpoint: {checkpoint_path}")
    print(f"[INFO] Using VQ-VAE from trainer: use_static={vqvae.config.use_static}")
    stats_file = getattr(smpl_processor, "stats_file", None)
    if stats_file is not None:
        print(f"[INFO] Using smpl_processor from trainer: stats_file={stats_file}")
    else:
        print(
            "[INFO] Using smpl_processor from trainer: "
            f"do_normalize={getattr(smpl_processor, 'do_normalize', None)}, "
            f"mean_shape={tuple(getattr(smpl_processor, 'mean').shape)}"
        )

    pipeline = VermoPipeline(
        vqvae=vqvae,
        audio_tokenizer=audio_tokenizer,
        smpl_processor=smpl_processor,
        text_tokenizer=trainer.processor.text_tokenizer,
        lm=trainer.lm,
    )
    return pipeline, trainer, checkpoint_path


def main(
    task_prompt: str = "Generate motion sequence from the given caption.",
    num_person: Optional[int] = 1,
    caption: Optional[
        str
    ] = "A person is performing a set of band pull-apart exercises to strengthen their upper back.",
    duration: Optional[float] = None,
    # dance related
    music: Optional[Union[str, np.ndarray]] = None,
    past_music: Optional[Union[str, np.ndarray]] = None,
    genre: Optional[str] = None,
    # speech related
    audio: Optional[Union[str, np.ndarray]] = None,
    speech_script: Optional[str] = None,
    # motion completion related
    past_motion: Optional[Union[str, Dict]] = None,
    future_motion: Optional[Union[str, Dict]] = None,
    # motion understanding related
    motion: Union[str, Dict] = None,
    output_path: str = "./outputs/vermo/",
    cfg="configs/vermo/vermo_sft_16k_llama1b_wavtokenizer.py",
    # vqvae_checkpoint 已移除，现在直接使用 trainer.processor.motion_tokenizer
    checkpoint="auto",
    tokenizer_save_path="checkpoints/vermo_tokenizer_qwen0.6b/",
    do_sample: bool = True,
    temperature: float = 0.8,
    top_k: int = 50,
    max_new_tokens: int = 8192,
    num_beams: int = 1,
    repetition_penalty: float = 1.2,
):
    checkpoint_path = resolve_checkpoint_path(cfg, checkpoint)

    output_path = os.path.join(
        output_path,
        os.path.basename(cfg).replace(".py", ""),
        os.path.basename(checkpoint_path).replace(".pth", ""),
    )
    os.makedirs(output_path, exist_ok=True)
    pipeline, _, _ = build_pipeline_from_config(
        cfg=cfg,
        checkpoint=checkpoint_path,
        tokenizer_save_path=tokenizer_save_path,
    )

    output_dict = pipeline(
        task_prompt=task_prompt,
        num_person=num_person,
        caption=caption,
        duration=duration,
        # dance related
        music=music,
        past_music=past_music,
        genre=genre,
        # speech related
        audio=audio,
        speech_script=speech_script,
        # motion completion related
        past_motion=past_motion,
        future_motion=future_motion,
        # motion understanding related
        motion=motion,
        do_sample=do_sample,
        temperature=temperature,
        top_k=top_k,
        max_new_tokens=max_new_tokens,
        num_beams=num_beams,
        repetition_penalty=repetition_penalty,
    )
    pipeline.save_out_dict(output_dict, output_path)


if __name__ == "__main__":
    import fire

    fire.Fire(main)
