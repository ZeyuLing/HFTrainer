"""Common MMEngine-style dataset base for HFTrainer."""

from __future__ import annotations

from abc import ABC
from collections.abc import Mapping
from typing import Callable, List, Optional, Sequence, Union

from mmengine.dataset import BaseDataset


class PipelineDataset(BaseDataset, ABC):
    """Thin wrapper over MMEngine BaseDataset with opt-in default pipelines."""

    def __init__(
        self,
        pipeline: Optional[Sequence[Union[dict, Callable]]] = None,
        metainfo: Optional[dict] = None,
        serialize_data: bool = False,
        test_mode: bool = False,
        lazy_init: bool = False,
        max_refetch: int = 1000,
    ):
        # Resolve config dictionaries through HFTrainer's strict local registry
        # before handing callables to MMEngine's Compose. This avoids MMEngine's
        # global transform registry and its dotted-import fallback.
        import hftrainer.datasets.transforms  # noqa: F401
        from hftrainer.registry import TRANSFORMS

        raw_pipeline = (
            list(pipeline) if pipeline is not None else self.build_default_pipeline()
        )
        resolved_pipeline = [
            TRANSFORMS.build(dict(transform))
            if isinstance(transform, Mapping)
            else transform
            for transform in raw_pipeline
        ]

        data_root = getattr(self, 'data_root', '')
        super().__init__(
            ann_file='',
            metainfo=metainfo,
            data_root=data_root,
            data_prefix={},
            serialize_data=serialize_data,
            pipeline=resolved_pipeline,
            test_mode=test_mode,
            lazy_init=lazy_init,
            max_refetch=max_refetch,
        )

    def build_default_pipeline(self) -> List[Union[dict, Callable]]:
        return []
