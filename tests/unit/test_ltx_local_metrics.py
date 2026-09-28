"""Local loss history must survive disabled W&B and resume-style appends."""
import json
from types import SimpleNamespace
from hftrainer.trainers.ltx_video.native.trainer import LtxvTrainer


def test_metrics_persist_without_wandb_and_append(tmp_path, monkeypatch):
    import hftrainer.trainers.ltx_video.native.trainer as module
    monkeypatch.setattr(module, 'IS_MAIN_PROCESS', True)
    trainer = object.__new__(LtxvTrainer)
    trainer._config = SimpleNamespace(output_dir=str(tmp_path))
    trainer._wandb_run = None
    trainer._log_metrics({'train/global_step': 1, 'train/loss': 1.2})
    trainer._log_metrics({'train/global_step': 2, 'train/loss': 0.8})
    rows = [json.loads(line) for line in (tmp_path / 'metrics.jsonl').read_text().splitlines()]
    assert rows == [{'train/global_step': 1, 'train/loss': 1.2}, {'train/global_step': 2, 'train/loss': 0.8}]
    monkeypatch.setattr(module, 'IS_MAIN_PROCESS', False)
    trainer._log_metrics({'train/global_step': 3, 'train/loss': 0.7})
    assert len((tmp_path / 'metrics.jsonl').read_text().splitlines()) == 2
