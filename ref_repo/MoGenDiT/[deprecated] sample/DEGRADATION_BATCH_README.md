# 动作退化批处理脚本使用指南

## 概述

de gradation_batch.py 是一个用于批量处理动作文件并应用随机退化的工具。它基于refine_batch.py的框架，但专注于应用各种动作捕捉系统中的常见降质模式。

## 主要功能

1. **递归遍历**：自动遍历输入目录下的所有子文件夹，查找动作文件
2. **多种降质模式**：支持6种不同的动作降质模式
3. **随机段落选择**：在动作序列的随机段落中应用随机降质
4. **可配置参数**：支持命令行参数和JSON配置文件
5. **保持目录结构**：输出文件保持原始目录结构

## 支持的降质模式

1. **关节方向跳变 (orientation_pops)**：模拟标记点混淆导致的关节方向突然变化
2. **姿态扭曲 (pose_twist)**：模拟三角测量噪声导致的局部姿态扭曲
3. **糖果纸扭曲 (candy_wrapper_twist)**：模拟复杂非线性扭曲
4. **帧冻结 (frozen_frame)**：模拟标记点丢失导致的动作停滞
5. **位移漂移 (translation_drift)**：模拟视觉动捕中的深度估计漂移
6. **位移比例失真 (translation_ratio_distortion)**：模拟比例因子估计误差

## 使用方法

### 基本使用

```bash
# 使用默认参数
python sample/degradation_batch.py --input_root /path/to/your/motions

# 指定输出目录和数据集名称
python sample/degradation_batch.py --input_root /path/to/your/motions --output_root ./data/degraded --dataset_name my_dataset

# 覆盖已存在的文件
python sample/degradation_batch.py --input_root /path/to/your/motions --overwrite
```

### 使用配置文件

1. 创建或修改 `sample/degradation_settings.json`
2. 运行脚本：

```bash
python sample/degradation_batch.py --config ./sample/degradation_settings.json
```

### 高级参数

```bash
# 控制随机段落的数量
python sample/degradation_batch.py --input_root /path/to/your/motions --min_segments 3 --max_segments 6

# 控制段落长度
python sample/degradation_batch.py --input_root /path/to/your/motions --min_segment_length 15 --max_segment_length 50

# 调整降质强度
python sample/degradation_batch.py --input_root /path/to/your/motions --strength_multiplier 0.5

# 设置随机种子（确保可重现性）
python sample/degradation_batch.py --input_root /path/to/your/motions --seed 1234

# 控制动作表示参数
python sample/degradation_batch.py --input_root /path/to/your/motions --use_vel false --keep_hand true --global_pose false
```

## 参数说明

### 主要参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--input_root` | str | None | **必需**，输入动作文件的根目录 |
| `--output_root` | str | `./data/degraded_motions` | 输出文件根目录 |
| `--dataset_name` | str | `degraded_dataset` | 数据集名称，用于组织输出目录 |
| `--config` | str | `./sample/degradation_settings.json` | 配置文件路径 |

### 降质参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--min_segments` | int | 2 | 随机退化段落的最小数量 |
| `--max_segments` | int | 5 | 随机退化段落的最大数量 |
| `--min_segment_length` | int | 10 | 段落最小长度（帧数） |
| `--max_segment_length` | int | 30 | 段落最大长度（帧数） |
| `--strength_multiplier` | float | 1.0 | 降质强度乘子 |
| `--seed` | int | 42 | 随机种子 |
| `--overwrite` | flag | False | 覆盖已存在的输出文件 |

### 动作表示参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--use_vel` | bool | True | 是否使用速度信息 |
| `--keep_hand` | bool | False | 是否保留手部关节 |
| `--global_pose` | bool | True | 是否使用全局姿态 |

## 配置文件示例

完整的配置文件示例位于 `sample/degradation_settings.json`，包含以下结构：

```json
{
  "degradation_settings": {
    "input_root": "/path/to/your/motions",
    "output_root": "./data/degraded_motions",
    "dataset_name": "my_dataset",
    
    "degradation_parameters": {
      "min_segments": 2,
      "max_segments": 5,
      "min_segment_length": 10,
      "max_segment_length": 30,
      "strength_multiplier": 1.0,
      "seed": 42
    },
    
    "system": {
      "use_vel": true,
      "keep_hand": false,
      "global_pose": true
    }
  }
}
```

## 输出结构

脚本会保持原始目录结构，输出文件组织如下：

```
output_root/
└── dataset_name/
    ├── folder1/
    │   ├── folder1_file1.npz
    │   ├── folder1_file2.npz
    │   └── ...
    ├── folder2/
    │   ├── folder2_file1.npz
    │   ├── folder2_file2.npz
    │   └── ...
    └── ...
```

## 注意事项

1. **输入格式**：脚本期望输入为.npz格式的SMPL参数文件
2. **设备支持**：自动检测CUDA可用性，优先使用GPU
3. **内存管理**：逐文件处理，避免内存溢出
4. **错误处理**：单个文件处理失败不会影响其他文件的处理
5. **进度显示**：实时显示处理进度和统计信息

## 调试模式

脚本默认显示详细的处理信息，包括：
- 每个文件的维度信息
- 每个段落的降质应用情况
- 处理统计信息

如需减少输出，可以修改脚本中的打印语句。

## 扩展性

脚本设计为模块化，易于扩展：
1. 添加新的降质模式：在 `motion_degradation.py` 中实现新函数
2. 修改降质参数：通过配置文件或命令行参数调整
3. 支持新格式：修改 `NpzMotion` 和 `Npz2VecProcessor` 相关代码