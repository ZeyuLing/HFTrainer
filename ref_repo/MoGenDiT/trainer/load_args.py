import json
from dataclasses import dataclass
from typing import List, Optional


# 定义数据类
@dataclass
class TrainArgs:
    model_name: str
    model_version: str
    save_dir: str
    log_dir: str
    batch_size: int
    lr: float
    weight_decay: float
    save_interval: int
    log_interval: int
    scale_beta: float
    schedule_sampler_type: str
    degrade_rate: float
    seed: int
    consis_loss: bool
    motion_degradation: bool
    global_pose: bool
    train_data: list
    l1_weight_x0: float
    l1_weight_consis: float
    l2_weight_x0: float
    l2_weight_consis: float
    ema_decay: float
    ema_start_step: int
    consis_start_step: int


# 从JSON文件加载数据并转换为数据类
def load_args(json_path: str) -> TrainArgs:
    with open(json_path, "r") as f:
        json_data = json.load(f)

    # 将字典转换为数据类实例
    return TrainArgs(**json_data)
