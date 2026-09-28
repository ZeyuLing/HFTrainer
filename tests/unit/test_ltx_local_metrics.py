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


def test_repeated_final_save_retains_newest_artifact(tmp_path):
    import torch
    from accelerate.utils import DistributedType
    from safetensors.torch import load_file

    trainer = object.__new__(LtxvTrainer)
    model = torch.nn.Linear(2, 2)
    trainer._transformer = model
    trainer._accelerator = SimpleNamespace(
        distributed_type=DistributedType.NO,
        is_main_process=True,
        wait_for_everyone=lambda: None,
        get_state_dict=lambda module: module.state_dict(),
    )
    trainer._config = SimpleNamespace(
        model=SimpleNamespace(training_mode='full'),
        checkpoints=SimpleNamespace(precision='float32', keep_last_n=1),
        output_dir=str(tmp_path),
    )
    trainer._checkpoint_paths = []
    trainer._build_checkpoint_metadata = lambda: {}
    trainer._save_training_state = lambda path: None
    trainer._global_step = 19
    old = trainer._save_checkpoint()
    trainer._global_step = 20
    final = trainer._save_checkpoint()
    assert not old.exists() and final.exists()
    with torch.no_grad():
        model.weight.fill_(3.)
    repeated = trainer._save_checkpoint()
    assert repeated == final and final.exists()
    assert trainer._checkpoint_paths == [final]
    torch.testing.assert_close(load_file(final)['weight'], model.weight)
