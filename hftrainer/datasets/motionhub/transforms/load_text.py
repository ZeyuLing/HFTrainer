import os
import random
from typing import Dict, List, Optional, Tuple, Union
from mmcv.transforms import BaseTransform

from hftrainer.datasets.motionhub.common import hm3d_pattern, read_json, read_txt
from hftrainer.registry import TRANSFORMS


@TRANSFORMS.register_module(force=True)
class LoadHierarchicalCaption(BaseTransform):
    def __init__(self, key="caption", allow_none: bool = False):
        self.key = key
        self.allow_none = allow_none

    def transform(self, results: Dict) -> Dict:
        filename = results.get(f"{self.key}_path")
        if filename is None and self.allow_none:
            return results
        hierarchical_caption = read_json(filename)
        caption_list = []
        granularity_list = []
        for granularity in ["macro", "meso", "micro"]:
            assert (
                granularity in hierarchical_caption
            ), f"{filename} contains no {granularity} captions"
            captions = hierarchical_caption[granularity]
            for caption in captions:
                caption_list.append(caption)
                granularity_list.append(granularity)
        assert len(caption_list) > 0, f"{filename} contains no captions"
        select_idx = random.randint(0, len(caption_list) - 1)
        results["caption"] = caption_list[select_idx]
        results["granularity"] = granularity_list[select_idx]
        results["caption_list"] = caption_list
        results["granularity_list"] = granularity_list
        return results


@TRANSFORMS.register_module(force=True)
class LoadHYMotionCaption(BaseTransform):
    def __init__(self, key="caption", allow_none: bool = False):
        self.key = key
        self.allow_none = allow_none

    def transform(self, results: Dict) -> Dict:
        filename = results.get(f"{self.key}_path")
        if filename is None and self.allow_none:
            return results
        hierarchical_caption = read_json(filename)
        caption_list = []
        granularity_list = []

        # 获取 result 数组
        result_list: List[Dict] = hierarchical_caption.get("result", [])

        # 遍历 result 数组中的每个元素
        for item in result_list:
            # 如果存在 short_caption_rewritten，使用它作为 caption 列表
            if "short_caption_rewritten" in item and isinstance(
                item["short_caption_rewritten"], list
            ):
                # short_caption_rewritten 是一个字符串数组
                for rewritten_caption in item["short_caption_rewritten"]:
                    if (
                        isinstance(rewritten_caption, str)
                        and len(rewritten_caption.strip()) > 0
                    ):
                        caption_list.append(rewritten_caption.strip())
            # 否则使用 short_caption
            elif "short_caption" in item and isinstance(item["short_caption"], str):
                short_caption = item["short_caption"].strip()
                if len(short_caption) > 0:
                    caption_list.append(short_caption)

        assert len(caption_list) > 0, f"{filename} contains no captions"
        select_idx = random.randint(0, len(caption_list) - 1)
        results["caption"] = caption_list[select_idx]
        results["caption_list"] = caption_list
        return results


@TRANSFORMS.register_module(force=True)
class LoadCompatibleCaption(BaseTransform):
    """
    兼容两种 caption 格式的 transform：
    1. LoadHierarchicalCaption 格式：包含 "macro", "meso", "micro" 三个键
    2. LoadHYMotionCaption 格式：包含 "result" 数组
    如果两种格式都不符合，抛出异常。
    """

    def __init__(self, key="caption", allow_none: bool = False):
        self.key = key
        self.allow_none = allow_none

    def _is_hierarchical_format(self, data: Dict) -> bool:
        """判断是否为 LoadHierarchicalCaption 格式（包含 macro, meso, micro）"""
        required_keys = ["macro", "meso", "micro"]
        # 检查所有必需的键都存在且是列表（允许空列表，因为原始实现只检查存在性）
        return all(
            key in data and isinstance(data[key], list) for key in required_keys
        )

    def _is_hymotion_format(self, data: Dict) -> bool:
        """判断是否为 LoadHYMotionCaption 格式（包含 result 数组）"""
        if "result" not in data:
            return False
        result_list = data["result"]
        if not isinstance(result_list, list) or len(result_list) == 0:
            return False
        # 检查 result 数组中的元素是否有 short_caption 或 short_caption_rewritten
        for item in result_list:
            if not isinstance(item, dict):
                continue
            if "short_caption" in item or "short_caption_rewritten" in item:
                return True
        return False

    def transform(self, results: Dict) -> Dict:
        filename = results.get(f"{self.key}_path")
        if filename is None and self.allow_none:
            return results

        hierarchical_caption = read_json(filename)
        caption_list = []
        granularity_list = []

        # 判断格式并处理
        if self._is_hierarchical_format(hierarchical_caption):
            # LoadHierarchicalCaption 格式
            for granularity in ["macro", "meso", "micro"]:
                captions = hierarchical_caption[granularity]
                for caption in captions:
                    caption_list.append(caption)
                    granularity_list.append(granularity)
            assert len(caption_list) > 0, f"{filename} contains no captions"
            select_idx = random.randint(0, len(caption_list) - 1)
            results["caption"] = caption_list[select_idx]
            results["granularity"] = granularity_list[select_idx]
            results["caption_list"] = caption_list
            results["granularity_list"] = granularity_list

        elif self._is_hymotion_format(hierarchical_caption):
            # LoadHYMotionCaption 格式
            result_list: List[Dict] = hierarchical_caption.get("result", [])
            for item in result_list:
                if "short_caption_rewritten" in item and isinstance(
                    item["short_caption_rewritten"], list
                ):
                    for rewritten_caption in item["short_caption_rewritten"]:
                        if (
                            isinstance(rewritten_caption, str)
                            and len(rewritten_caption.strip()) > 0
                        ):
                            caption_list.append(rewritten_caption.strip())
                elif "short_caption" in item and isinstance(item["short_caption"], str):
                    short_caption = item["short_caption"].strip()
                    if len(short_caption) > 0:
                        caption_list.append(short_caption)
            assert len(caption_list) > 0, f"{filename} contains no captions"
            select_idx = random.randint(0, len(caption_list) - 1)
            results["caption"] = caption_list[select_idx]
            results["caption_list"] = caption_list

        else:
            # 两种格式都不符合，抛出异常
            raise ValueError(
                f"{filename} does not match either format:\n"
                f"  - LoadHierarchicalCaption: requires 'macro', 'meso', 'micro' keys\n"
                f"  - LoadHYMotionCaption: requires 'result' array with 'short_caption' or 'short_caption_rewritten'"
            )

        return results


@TRANSFORMS.register_module(force=True)
class LoadHm3dTxt(BaseTransform):

    def __init__(
        self, keys: Union[str, List[str]] = "caption", min_duration=0, sr=None
    ):
        if isinstance(keys, str):
            keys = [keys]
        self.keys = keys

        self.sr = sr
        self.min_duration = min_duration

    def transform(self, results: dict) -> dict:
        """Functions to load humanml3d caption text.

        Args:
            results (dict): Result dict from :obj:``mmcv.BaseDataset``.
        Returns:
            dict: The dict contains loaded caption, token, etc.
        """
        for key in self.keys:
            filename = results.get(f"{key}_path")
            if filename is None or not os.path.exists(filename):
                continue

            caption_list, pos_list, range_list = self.load_caption(filename)
            # 0 <= idx <= num_captions - 1
            select_idx = random.randint(0, len(caption_list) - 1)
            caption = caption_list[select_idx]
            pos = pos_list[select_idx]
            range = range_list[select_idx]

            results[key] = caption
            results[f"{key}_pos"] = pos
            results[f"{key}_range"] = range

            results[f"{key}_list"] = caption_list
            # pos: part of speech
            results[f"{key}_pos_list"] = pos_list
            results[f"{key}_range_list"] = range_list

        return results

    @staticmethod
    def judge_hm3d(content: str):
        """Judge if the content is a humanml3d type caption file
        :param content: content of file
        :return: True or False
        """
        content = content.strip()

        first_line = content.split("\n")[0]
        if hm3d_pattern.match(first_line):
            return True
        return False

    def load_hm3d_caption(self, content: str):
        caption_list = []
        pos_list = []
        range_list = []

        for line in content.split("\n"):
            caption = line.split("#")[0].strip()
            assert len(caption) > 0, content
            pos = line.split("#")[1].strip()

            range = line.split("#")[-2:]
            range = [float(x) for x in range]
            duration = range[1] - range[0]
            # duration == 0 means no crop occurs.
            if 0 < duration < self.min_duration:
                continue

            caption_list.append(caption)
            pos_list.append(pos)
            range_list.append(range)
        return caption_list, pos_list, range_list

    @staticmethod
    def load_pure_caption(content: str):
        caption_list = []
        pos_list = []
        range_list = []
        for line in content.split("\n"):
            caption = line.strip()

            caption_list.append(caption)

            pos_list.append(None)
            range_list.append([0, 0])
        return caption_list, pos_list, range_list

    def load_caption(self, caption_path: str) -> Tuple:
        """
        :param caption_path: txt path of humanml3d caption file.
        :return: caption list, pos list and range list
        """
        try:
            content = read_txt(caption_path).strip()
        except:
            raise Exception(caption_path)
        is_hm3d = self.judge_hm3d(content)
        if is_hm3d:
            caption_list, pos_list, range_list = self.load_hm3d_caption(content)
        else:
            caption_list, pos_list, range_list = self.load_pure_caption(content)

        return caption_list, pos_list, range_list

    def __repr__(self):
        repr_str = f"{self.__class__.__name__}(" f"key={self.key})"

        return repr_str


@TRANSFORMS.register_module(force=True)
class LoadTxt(BaseTransform):
    def __init__(self, key: str = "speech_script", allow_none: bool = False):
        self.key = key
        self.allow_none = allow_none

    def transform(self, results: Dict) -> Optional[Union[Dict, Tuple[List, List]]]:
        filename = results.get(f"{self.key}_path")
        if filename is None and self.allow_none:
            return results

        text = read_txt(filename)

        results[self.key] = text
        return results
