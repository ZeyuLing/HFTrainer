import os
import json
import shutil
from pathlib import Path


def download_npz_files(json_path, base_save_dir, n_sample):
    """
    从JSON文件中读取两种类型的NPZ文件路径并分别下载到不同目录

    参数:
        json_path: JSON文件的路径
        base_save_dir: 基础保存目录（两种类型的文件夹将创建在此目录下）
        n_sample: 每种类型需要下载的样本数量
    """
    # 创建基础目录
    base_save_dir = Path(base_save_dir)
    base_save_dir.mkdir(parents=True, exist_ok=True)

    # 定义两种类型的文件夹名称和对应的JSON字段
    data_types = {
        "center_motions": "center_motion_pths",  # 中心运动数据，文件夹名和JSON字段名
        "edge_motions": "edge_motion_pths",  # 边缘运动数据，文件夹名和JSON字段名
    }

    # 读取JSON文件
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"读取JSON文件失败: {e}")
        return

    # 遍历两种数据类型，分别下载
    for save_folder, json_key in data_types.items():
        # 创建当前类型的保存目录
        save_dir = base_save_dir / save_folder
        save_dir.mkdir(parents=True, exist_ok=True)

        # 获取对应的数据路径列表（截取前n_sample个）
        npz_paths = data.get(json_key, [])[:n_sample]
        if not npz_paths:
            print(f"JSON文件中未找到 {json_key} 字段或该字段为空，跳过该类型下载")
            continue

        # 下载当前类型的文件
        total = len(npz_paths)
        print(f"\n开始下载 {save_folder} 类型文件，共 {total} 个...")
        for i, npz_path in enumerate(npz_paths, 1):
            try:
                # 检查源文件是否存在
                if not os.path.exists(npz_path):
                    print(f"({i}/{total}) 源文件不存在: {npz_path}")
                    continue

                # 获取文件名并构建目标路径
                file_name = os.path.basename(npz_path)
                dest_path = save_dir / file_name

                # 跳过已存在的文件
                if dest_path.exists():
                    print(f"({i}/{total}) 文件已存在，跳过: {file_name}")
                    continue

                # 复制文件
                shutil.copy2(npz_path, dest_path)
                print(f"({i}/{total}) 成功下载: {file_name}")

            except Exception as e:
                print(f"({i}/{total}) 处理失败 {npz_path}: {e}")

    print("\n所有类型文件下载处理完成")


if __name__ == "__main__":
    # 配置参数 - 请根据实际情况修改
    JSON_FILE_PATH = "utils/center_edge_motion_pths.json"  # 包含NPZ路径的JSON文件路径
    SAVE_DIRECTORY = "hunyuan_motion_data"  # 保存NPZ文件的目标目录
    n_sample = 200

    # 执行下载
    download_npz_files(JSON_FILE_PATH, SAVE_DIRECTORY, n_sample)
