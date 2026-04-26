import os
import random
from typing import Dict, List, Optional, Tuple, Union
from mmcv.transforms import BaseTransform

import torch

from hftrainer.datasets.motion.motionhub.common import hm3d_pattern, read_json, read_txt
from hftrainer.registry import TRANSFORMS


# ---------------------------------------------------------------------------
# Mapping: caption dir name → pre-extracted qwen3 embedding dir name.
# When a caption JSON file lives under dir X, the corresponding .pt embedding
# file (if it exists) lives under the sibling dir CAPTION_TO_QWEN3_DIR[X].
# The .pt file has the same relative path as the .json but with .pt suffix.
# ---------------------------------------------------------------------------
CAPTION_TO_QWEN3_DIR = {
    # Academic / AcademicRetarget / Game / Taobao (and their mirror variants)
    'human_checked_augmented_caption': 'qwen3_augmented',
    'human_checked_augmented_caption_deprecated_mirror_251215': 'qwen3_augmented',
    'human_checked_augmented_caption_mirror': 'qwen3_augmented',
    'human_checked_caption': 'qwen3_human_checked_short',
    'human_checked_caption_deprecated_mirror_251215': 'qwen3_human_checked_short',
    'human_checked_caption_mirror': 'qwen3_human_checked_short',
    'improved_simple_augmented_caption': 'qwen3_improved_simple_short',
    'improved_simple_augmented_caption_deprecated_mirror_251215': 'qwen3_improved_simple_short',
    'improved_simple_caption': 'qwen3_improved_simple_short',
    'improved_simple_caption_deprecated_mirror_251215': 'qwen3_improved_simple_short',
    # Variants with qwen3embedding prefix (older dirs)
    'augmented_caption': 'qwen3embedding_augmented',
    'augmented_caption_deprecated_250905': 'qwen3embedding_augmented',
    'augmented_caption_deprecated_250926': 'qwen3embedding_augmented',
}


def _caption_path_to_embedding_path(caption_path: str) -> Optional[str]:
    """Given an absolute caption .json path, return the corresponding .pt
    pre-extracted embedding path, or None if no mapping is known.

    The .pt file has identical structure to the embedding dict returned by
    HYTextModel.encode():
        data['result'][i] = {
            'caption': str,
            'text_embedding': {
                'text_vec_raw':        Tensor[1, 1, 768],
                'text_ctxt_raw':       Tensor[1, seq, 4096],
                'text_ctxt_raw_length': Tensor[1],
            },
            ...
        }
    """
    # Normalize the path first (resolve ../ segments)
    caption_path = os.path.normpath(caption_path)
    # Walk up the path to find the first path component that matches a known
    # caption dir name.
    parts = caption_path.replace('\\', '/').split('/')
    for i, part in enumerate(parts):
        if part in CAPTION_TO_QWEN3_DIR:
            qwen3_dir = CAPTION_TO_QWEN3_DIR[part]
            new_parts = parts[:i] + [qwen3_dir] + parts[i + 1:]
            pt_path = '/'.join(new_parts)
            # Replace .json → .pt
            if pt_path.endswith('.json'):
                pt_path = pt_path[:-5] + '.pt'
            return pt_path
    return None


@TRANSFORMS.register_module(force=True)
class LoadPreExtractedTextEmbedding(BaseTransform):
    """Load pre-extracted Qwen3+CLIP text embeddings from .pt files.

    For each sample, the caption JSON path (``results['caption_path']``) is
    mapped to a sibling .pt file that contains pre-extracted embeddings.
    If the .pt file exists, the embeddings are loaded directly (bypassing
    online Qwen3-8B inference during training).  If no .pt file is found,
    the transform falls back gracefully: it leaves ``results`` unchanged so
    that the downstream trainer can fall back to online encoding or null
    embeddings.

    The .pt file format expected::

        data['result'][i] = {
            'caption': str,
            'text_embedding': {
                'text_vec_raw':         Tensor[1, 1, 768],   # CLIP-L
                'text_ctxt_raw':        Tensor[1, seq, 4096], # Qwen3
                'text_ctxt_raw_length': Tensor[1],
            },
            ...
        }

    Output keys added to results (when successful):
        ``text_vec_raw``, ``text_ctxt_raw``, ``text_ctxt_raw_length``
    The ``caption`` string is also set (from the chosen embedding item) so
    that existing CFG dropout logic continues to work.

    Args:
        key (str): Key prefix for caption path in results dict.
            Default ``'caption'`` → reads ``results['caption_path']``.
        allow_none (bool): If True, silently skip when caption_path is None.
        fallback_to_caption (bool): If True (default), keep ``caption`` text
            in results even when embedding is found, so text-only fallback
            pipelines still work.
    """

    def __init__(
        self,
        key: str = 'caption',
        allow_none: bool = True,
        fallback_to_caption: bool = True,
        vtxt_dim: int = 768,
        ctxt_dim: int = 4096,
    ):
        self.key = key
        self.allow_none = allow_none
        self.fallback_to_caption = fallback_to_caption
        self.vtxt_dim = vtxt_dim
        self.ctxt_dim = ctxt_dim

    def _fill_null_embedding(self, results: Dict) -> Dict:
        """Fill null (zero) embedding tensors so that every sample in a batch
        has a consistent tensor type for collation.  The trainer's CFG dropout
        (mask_text_cond) will replace these with the *learned* null embeddings
        when cond_mask_prob triggers, but having zeros here prevents mixed
        Tensor/None collation errors.

        Note: we mark null samples with text_ctxt_raw_length=0 so the trainer
        can build a correct attention mask (all False → padding).
        """
        results['text_vec_raw'] = torch.zeros(1, self.vtxt_dim)
        results['text_ctxt_raw'] = torch.zeros(1, self.ctxt_dim)
        results['text_ctxt_raw_length'] = torch.tensor(0)
        results['_text_is_null'] = True
        return results

    def transform(self, results: Dict) -> Dict:
        caption_path = results.get(f'{self.key}_path')
        if caption_path is None:
            if self.allow_none:
                return self._fill_null_embedding(results)
            raise ValueError(
                f"LoadPreExtractedTextEmbedding: '{self.key}_path' not found in results"
            )

        # Derive .pt path from caption JSON path
        pt_path = _caption_path_to_embedding_path(caption_path)
        if pt_path is None or not os.path.exists(pt_path):
            # No pre-extracted embedding available — fill null embedding.
            return self._fill_null_embedding(results)

        try:
            data = torch.load(pt_path, map_location='cpu', weights_only=False)
        except Exception:
            # Corrupted file – fill null embedding
            return self._fill_null_embedding(results)

        result_list = data.get('result', [])
        if not result_list:
            return self._fill_null_embedding(results)

        # Randomly select one caption variant (data augmentation)
        idx = random.randint(0, len(result_list) - 1)
        item = result_list[idx]
        emb = item.get('text_embedding')
        if emb is None:
            return self._fill_null_embedding(results)

        # Unpack: remove the leading batch dim added during extraction
        # Each tensor was saved as [1, ...] from a batch-size-1 encode call.
        text_vec_raw = emb['text_vec_raw'].squeeze(0)          # [1, 768]
        text_ctxt_raw = emb['text_ctxt_raw'].squeeze(0)        # [seq, 4096]
        text_ctxt_raw_length = emb['text_ctxt_raw_length'].squeeze(0)  # scalar

        results['text_vec_raw'] = text_vec_raw
        results['text_ctxt_raw'] = text_ctxt_raw
        results['text_ctxt_raw_length'] = text_ctxt_raw_length

        # Also store caption string (for logging / CFG dropout compatibility)
        if self.fallback_to_caption and 'caption' not in results:
            results['caption'] = item.get('caption', '')

        return results


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
        # NOTE: Some caption files use "short caption" (space) instead of
        # "short_caption" (underscore). Accept both variants.
        for item in result_list:
            # 如果存在 short_caption_rewritten，使用它作为 caption 列表
            rewritten_key = (
                "short_caption_rewritten" if "short_caption_rewritten" in item
                else "short caption_rewritten" if "short caption_rewritten" in item
                else None
            )
            caption_key = (
                "short_caption" if "short_caption" in item
                else "short caption" if "short caption" in item
                else None
            )
            if rewritten_key is not None and isinstance(
                item[rewritten_key], list
            ):
                # short_caption_rewritten 是一个字符串数组
                for rewritten_caption in item[rewritten_key]:
                    if (
                        isinstance(rewritten_caption, str)
                        and len(rewritten_caption.strip()) > 0
                    ):
                        caption_list.append(rewritten_caption.strip())
            # 否则使用 short_caption
            elif caption_key is not None and isinstance(item[caption_key], str):
                short_caption = item[caption_key].strip()
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
        # Also accept "short caption" (space) variant
        for item in result_list:
            if not isinstance(item, dict):
                continue
            if any(k in item for k in ("short_caption", "short_caption_rewritten",
                                        "short caption", "short caption_rewritten")):
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
                rewritten_key = (
                    "short_caption_rewritten" if "short_caption_rewritten" in item
                    else "short caption_rewritten" if "short caption_rewritten" in item
                    else None
                )
                caption_key = (
                    "short_caption" if "short_caption" in item
                    else "short caption" if "short caption" in item
                    else None
                )
                if rewritten_key is not None and isinstance(
                    item[rewritten_key], list
                ):
                    for rewritten_caption in item[rewritten_key]:
                        if (
                            isinstance(rewritten_caption, str)
                            and len(rewritten_caption.strip()) > 0
                        ):
                            caption_list.append(rewritten_caption.strip())
                elif caption_key is not None and isinstance(item[caption_key], str):
                    short_caption = item[caption_key].strip()
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
