from collections import defaultdict
import os
import random
from typing import Dict, List, Optional, Tuple, Type, Union
from tqdm import tqdm

from overrides import override
import mmengine
from hftrainer.models.vermo.task_utils import ALL_TASKS, abbr_list_to_task_list
from hftrainer.models.vermo.task_utils.modality import Audio, Caption, Modality
from hftrainer.models.vermo.task_utils.task_lib.base_task import BaseTask
from hftrainer.models.vermo.task_utils.task_lib.completion_tasks.motion_inbetween import MotionInbetween
from hftrainer.datasets.motionhub.single_agent_dataset import MotionHubSingleAgentDataset
from hftrainer.datasets.motionhub.flexible_collate import flexible_collate
from mmcv.transforms import BaseTransform
from mmengine.logging import print_log
from hftrainer.registry import DATASETS


TASK_BUCKETS = (
    "caption_audio",
    "caption_non_audio",
    "audio_non_caption",
    "motion_non_caption",
)


def task_has_modal(task: Type[BaseTask], modal_cls: Type[Modality]) -> bool:
    try:
        for modal in task.all_modality():
            if isinstance(modal, type) and issubclass(modal, modal_cls):
                return True
            if isinstance(modal, modal_cls):
                return True
    except Exception:
        return False
    return False


def task_bucket_name(task: Type[BaseTask]) -> str:
    has_caption = task_has_modal(task, Caption)
    has_audio = task_has_modal(task, Audio)
    if has_caption and has_audio:
        return "caption_audio"
    if has_caption:
        return "caption_non_audio"
    if has_audio:
        return "audio_non_caption"
    return "motion_non_caption"


TASK_BUCKET_TASKS = {
    bucket: [task for task in ALL_TASKS if task_bucket_name(task) == bucket]
    for bucket in TASK_BUCKETS
}


@DATASETS.register_module(force=True)
class MotionhubMultiTaskMultiAgentDataset(MotionHubSingleAgentDataset):
    collate_fn = staticmethod(flexible_collate)
    SUPPORTED_TASK_MODE = ["auto", "preset"]
    SUPPORTED_TASK_BUCKET_MODE = ["none", "modality"]

    def __init__(
        self,
        motion_key: str = "smplx",
        data_dir: str = "data/motionhub",
        anno_file: str = "data/motionhub/train.json",
        pipeline: Union[Dict, BaseTransform, List[Union[Dict, BaseTransform]]] = None,
        refetch: bool = True,
        verbose: bool = False,
        task_mode: str = "auto",
        task_bucket_mode: str = "none",
        preset_tasks: Optional[List[str]] = None,
        log_task_iter: int = 10000,
        num_person: Optional[int] = None,
    ):
        self.num_person = num_person

        super().__init__(
            motion_key=motion_key,
            data_dir=data_dir,
            anno_file=anno_file,
            pipeline=pipeline,
            refetch=refetch,
            verbose=verbose,
        )
        assert (
            task_mode in self.SUPPORTED_TASK_MODE
        ), f"task_mode must be in {self.SUPPORTED_TASK_MODE}"
        assert (
            task_bucket_mode in self.SUPPORTED_TASK_BUCKET_MODE
        ), f"task_bucket_mode must be in {self.SUPPORTED_TASK_BUCKET_MODE}"

        self.task_mode = task_mode
        self.task_bucket_mode = task_bucket_mode
        if task_mode == "preset":
            self.preset_tasks = abbr_list_to_task_list(preset_tasks)
        self.log_task_iter = log_task_iter

        # ``spawn`` dataloader workers require the dataset to be pickleable.
        # A local lambda breaks pickling, while ``int`` keeps the same zero
        # default behaviour.
        self.task_counter = defaultdict(int)
        # if num_person is not None, then only use num_person agents
        self._task_bucket_mask_cache: List[Optional[int]] = [None] * len(self.data_list)

        self._inject_dataset_into_compose_transforms()

    def _inject_dataset_into_compose_transforms(self):
        """Wire dataset reference into ComposeMultiPerson transforms in the pipeline."""
        from hftrainer.datasets.motionhub.transforms.compose_multi_person import ComposeMultiPerson

        if hasattr(self, "pipeline") and hasattr(self.pipeline, "transforms"):
            for transform in self.pipeline.transforms:
                if isinstance(transform, ComposeMultiPerson):
                    transform.set_dataset(self)

    def load_data_list(self) -> List[dict]:
        """Copied from mmengine.dataset.based_dataset.BaseDataset
        Load annotations from an annotation file named as ``self.ann_file``

        If the annotation file does not follow `OpenMMLab 2.0 format dataset
        <https://mmengine.readthedocs.io/en/latest/advanced_tutorials/basedataset.html>`_ .
        The subclass must override this method for load annotations. The meta
        information of annotation file will be overwritten :attr:`meta_info`
        and ``meta_info`` argument of constructor.

        Returns:
            list[dict]: A list of annotation.
        """  # noqa: E501
        # `self.ann_file` denotes the absolute annotation file path if
        # `self.root=None` or relative path if `self.root=/path/to/data/`.
        annotations = mmengine.load(self.anno_file)
        if not isinstance(annotations, dict):
            raise TypeError(
                f"The annotations loaded from annotation file "
                f"should be a dict, but got {type(annotations)}!"
            )
        if "data_list" not in annotations or "meta_info" not in annotations:
            raise ValueError("Annotation must have data_list and meta_info keys")

        meta_info = annotations["meta_info"]
        for k, v in meta_info.items():
            self._metainfo.setdefault(k, v)

        raw_data_list = annotations["data_list"]

        is_main = True
        try:
            from mmengine import dist

            is_main = (not dist.is_distributed()) or (dist.get_rank() == 0)
        except ImportError:
            is_main = True

        iterator = (
            tqdm(raw_data_list.values(), desc="Loading data_list")
            if is_main
            else raw_data_list.values()
        )

        data_list = []
        for data_info in iterator:
            assert isinstance(data_info, dict)
            # skip multi-person data
            motion_path: Union[str, List[str]] = data_info[f"{self.motion_key}_path"]

            if isinstance(motion_path, str):
                if self.num_person is not None and self.num_person != 1:
                    continue

            if self.num_person is not None:
                if (
                    isinstance(motion_path, list)
                    and len(motion_path) != self.num_person
                ):
                    continue
            data_list.append(data_info)

        print_log(f"Loaded {len(data_list)} samples from {self.anno_file}")
        return data_list

    @override
    def prepare_data(self, idx: int) -> dict:
        raw_idx, extra = self._split_index(idx)
        task_bucket = extra[0] if extra else None

        raw_data_info = self.data_list[raw_idx]

        # load motion
        motion_path = raw_data_info[f"{self.motion_key}_path"]

        # load caption
        caption_path = raw_data_info.get(
            "hierarchical_caption_path", raw_data_info.get("caption_path")
        )
        if caption_path is not None:
            caption_path = os.path.join(self.data_dir, caption_path)
            if not os.path.exists(caption_path):
                caption_path = None

        # multi agent or single agent
        if isinstance(motion_path, str):
            motion_path = os.path.join(self.data_dir, motion_path)
            num_person = 1
        else:
            assert isinstance(motion_path, list), "motion_path must be a list"
            motion_path = [os.path.join(self.data_dir, path) for path in motion_path]
            num_person = len(motion_path)

        # load music
        music_path = raw_data_info.get("music_path", None)
        if music_path is not None:
            music_path = os.path.join(self.data_dir, music_path)
            if not os.path.exists(music_path):
                music_path = None
        # Load speech related
        audio_path = raw_data_info.get("audio_path", None)
        if audio_path is not None:
            audio_path = os.path.join(self.data_dir, audio_path)
            if not os.path.exists(audio_path):
                audio_path = None

        speech_script_path = raw_data_info.get("speech_script_path", None)
        if speech_script_path is not None:
            speech_script_path = os.path.join(self.data_dir, speech_script_path)
            if not os.path.exists(speech_script_path):
                speech_script_path = None

        data_info = {
            "num_person": num_person,
            "motion_path": motion_path,
            "subset": raw_data_info["subset"],
            "fps": raw_data_info["fps"],
            "has_hand": raw_data_info["has_hand"],
            "duration": raw_data_info["duration"],
            "num_frames": raw_data_info["num_frames"],
            # text related
            "caption_path": caption_path,
            # sound related
            "sr": raw_data_info.get("sr"),
            # music related
            "music_path": music_path,
            "genre": raw_data_info.get("genre"),
            # language related
            "language": raw_data_info.get("language"),
            "audio_path": audio_path,
            "speech_script_path": speech_script_path,
            "speaker_id": raw_data_info.get("speaker_id"),
        }
        candidate_tasks = self._resolve_candidate_tasks(task_bucket)
        # determine what tasks can be trained on this data
        available_tasks = self.assign_task_for_data(data_info, candidate_tasks)
        # randomly assign a task for the data info
        if available_tasks:
            task = random.choice(available_tasks)

            data_info["task"] = task
            self.task_counter[task.abbr] += 1

            if sum(self.task_counter.values()) % self.log_task_iter == 0:
                print_log(self.task_counter, logger="current")
            return data_info
        raise ValueError(
            f"No available task in {self.preset_tasks} for data {motion_path}"
        )

    def _resolve_candidate_tasks(
        self, task_bucket: Optional[str] = None
    ) -> List[Type[BaseTask]]:
        if self.task_mode == "preset":
            candidate_tasks = list(self.preset_tasks)
        else:
            candidate_tasks = list(ALL_TASKS)

        if (
            task_bucket is not None
            and self.task_bucket_mode != "none"
            and task_bucket in TASK_BUCKET_TASKS
        ):
            bucket_task_set = set(TASK_BUCKET_TASKS[task_bucket])
            candidate_tasks = [task for task in candidate_tasks if task in bucket_task_set]

        return candidate_tasks

    def _build_bucket_probe_data_info(self, raw_data_info: Dict) -> Dict:
        motion_path = raw_data_info[f"{self.motion_key}_path"]
        num_person = 1 if isinstance(motion_path, str) else len(motion_path)
        return {
            "num_person": num_person,
            "motion_path": motion_path,
            "duration": raw_data_info.get("duration"),
            "num_frames": raw_data_info.get("num_frames"),
            "caption_path": raw_data_info.get(
                "hierarchical_caption_path", raw_data_info.get("caption_path")
            ),
            "music_path": raw_data_info.get("music_path"),
            "genre": raw_data_info.get("genre"),
            "audio_path": raw_data_info.get("audio_path"),
            "speech_script_path": raw_data_info.get("speech_script_path"),
            "sr": raw_data_info.get("sr"),
        }

    def _infer_task_bucket_mask(self, raw_idx: int) -> int:
        if self.task_bucket_mode == "none":
            return 0

        raw_data_info = self.data_list[raw_idx]
        probe_data_info = self._build_bucket_probe_data_info(raw_data_info)
        mask = 0
        for bucket_idx, bucket_name in enumerate(TASK_BUCKETS):
            available = self.assign_task_for_data(
                probe_data_info,
                self._resolve_candidate_tasks(bucket_name),
            )
            if available:
                mask |= 1 << bucket_idx
        return mask

    def get_task_bucket_names(self, raw_idx: int) -> Tuple[str, ...]:
        if self.task_bucket_mode == "none":
            return tuple()

        mask = self._task_bucket_mask_cache[raw_idx]
        if mask is None:
            mask = self._infer_task_bucket_mask(raw_idx)
            self._task_bucket_mask_cache[raw_idx] = mask

        return tuple(
            bucket_name
            for bucket_idx, bucket_name in enumerate(TASK_BUCKETS)
            if mask & (1 << bucket_idx)
        )

    def sample_refetch_index(self, task_bucket: Optional[str] = None) -> int:
        if not task_bucket or self.task_bucket_mode == "none":
            return random.randint(0, len(self.data_list) - 1)

        for _ in range(32):
            idx = random.randint(0, len(self.data_list) - 1)
            if task_bucket in self.get_task_bucket_names(idx):
                return idx

        return random.randint(0, len(self.data_list) - 1)

    def assign_task_for_data(
        self, data_info: Dict, candidate_tasks: Optional[List[Type[Modality]]] = None
    ) -> List[BaseTask]:
        """Assign the task for the data info

        :param data_info: original data info
        :param candidate_tasks: preset tasks, if None, use all tasks
        :return: The task-specific data info.
        """
        tasks = []
        candidate_tasks: List[BaseTask] = candidate_tasks or ALL_TASKS

        for task in candidate_tasks:
            essential_modals: List[Type[Modality]] = task.essential_modality()
            # check if all required modalities exist
            for modal in essential_modals:
                modal_exist = False
                candidate_keys = modal.load_keys

                for key in candidate_keys:
                    if key not in data_info:
                        key = f"{key}_path"
                    if key in data_info and data_info[key] is not None:
                        # because we check the path exist in prepare_data,
                        # we can assume the path is valid here
                        modal_exist = True
                        break
                # if any modality is not found, skip this task
                if not modal_exist:
                    break

            if modal_exist:
                tasks.append(task)

        return tasks
