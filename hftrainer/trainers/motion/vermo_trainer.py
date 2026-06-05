"""VerMo trainer."""

from __future__ import annotations

import os
import time
from typing import Any, Dict

import torch

from hftrainer.registry import TRAINERS
from hftrainer.trainers.base_trainer import BaseTrainer
from hftrainer.utils.logger import get_logger


logger = get_logger()


@TRAINERS.register_module()
class VermoTrainer(BaseTrainer):
    """Trainer for VerMo causal multimodal generation."""

    def train_step(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        global_step = int(getattr(getattr(self, 'runner', None), 'global_step', 0))
        debug_steps = int(os.environ.get('VERMO_DEBUG_TIMING_STEPS', '20'))
        debug_timing = (
            os.environ.get('VERMO_DEBUG_TIMING') == '1'
            and global_step < debug_steps
        )
        accelerator = getattr(self, 'accelerator', None)
        is_main = getattr(accelerator, 'is_main_process', True)
        rank = getattr(accelerator, 'process_index', 0)
        should_log_debug = (
            is_main or os.environ.get('VERMO_DEBUG_ALL_RANKS') == '1'
        )

        if not debug_timing:
            outputs = self.bundle.forward_lm(batch)
            return {'loss': outputs.loss, 'loss_lm': outputs.loss.detach()}

        def sync_cuda():
            if torch.cuda.is_available():
                torch.cuda.synchronize()

        sync_cuda()
        t0 = time.perf_counter()
        lm_inputs = self.bundle.process_train(batch)
        sync_cuda()
        t1 = time.perf_counter()

        if should_log_debug:
            input_ids = lm_inputs.get('input_ids')
            labels = lm_inputs.get('labels')
            supervised = int((labels != -100).sum().item()) if labels is not None else 0
            tasks = batch.get('task') if isinstance(batch, dict) else None
            if isinstance(tasks, (list, tuple)):
                tasks = [
                    getattr(task, 'abbr', str(task))
                    for task in list(tasks[:4])
                ]
            try:
                lm = self.bundle.lm
                raw_lm = getattr(lm, 'module', lm)
                embed = raw_lm.get_input_embeddings()
                lm_dtype = next(lm.parameters()).dtype
                vocab_size = int(embed.num_embeddings)
            except StopIteration:
                lm_dtype = None
                vocab_size = None
            input_min = int(input_ids.min().item()) if input_ids is not None else None
            input_max = int(input_ids.max().item()) if input_ids is not None else None
            label_valid = labels[labels != -100] if labels is not None else None
            label_min = int(label_valid.min().item()) if label_valid is not None and label_valid.numel() else None
            label_max = int(label_valid.max().item()) if label_valid is not None and label_valid.numel() else None
            sample_keys = [
                'id', 'key', 'smplx_path', 'motion_path', 'audio_path',
                'music_path', 'speech_path', 'num_person', 'caption',
                'person_captions',
            ]
            meta_bits = []
            if isinstance(batch, dict):
                for key in sample_keys:
                    if key not in batch:
                        continue
                    value = batch[key]
                    if torch.is_tensor(value):
                        meta_bits.append(
                            f'{key}=shape{tuple(value.shape)} '
                            f'min={float(value.min().item()) if value.numel() else None} '
                            f'max={float(value.max().item()) if value.numel() else None}'
                        )
                    else:
                        text = str(value)
                        if len(text) > 600:
                            text = text[:600] + '...'
                        meta_bits.append(f'{key}={text}')
            logger.info(
                f'[VerMo debug rank={rank} step={global_step}] process_train done: '
                f'shape={tuple(input_ids.shape) if input_ids is not None else None}, '
                f'supervised_tokens={supervised}, lm_dtype={lm_dtype}, '
                f'vocab_size={vocab_size}, input_id_range=({input_min},{input_max}), '
                f'label_range=({label_min},{label_max}), '
                f'tasks={tasks}, '
                f'meta={"; ".join(meta_bits)}, '
                f'time={t1 - t0:.2f}s'
            )

        outputs = self.bundle.lm(**lm_inputs)
        sync_cuda()
        t2 = time.perf_counter()

        if should_log_debug:
            logger.info(
                f'[VerMo debug rank={rank} step={global_step}] lm forward done: '
                f'loss={float(outputs.loss.detach().float().cpu()):.6f}, '
                f'time={t2 - t1:.2f}s'
            )
        return {'loss': outputs.loss, 'loss_lm': outputs.loss.detach()}
