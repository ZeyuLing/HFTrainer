import json
from dataclasses import dataclass
from typing import List, Optional


# 定义数据类
@dataclass
class GenArgs:
    data_source_path: str
    model_name: str
    model_version: str
    global_pose: bool
    restore_iter: int
    fast_sampling: bool
    caption: str
    denoise_steps: int
    mask_window: int


# 从JSON文件加载数据并转换为数据类
def load_setting(json_path: str) -> GenArgs:
    with open(json_path, "r") as f:
        json_data = json.load(f)

    # 将字典转换为数据类实例
    return GenArgs(**json_data)
