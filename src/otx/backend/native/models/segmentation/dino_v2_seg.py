# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""DinoV2Seg model implementations."""

from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Literal
from urllib.parse import urlparse

from torch.hub import download_url_to_file

from otx.backend.native.models.base import DataInputParams, DefaultOptimizerCallable, DefaultSchedulerCallable
from otx.backend.native.models.classification.backbones.vision_transformer import VisionTransformerBackbone
from otx.backend.native.models.segmentation.base import OTXSegmentationModel
from otx.backend.native.models.segmentation.heads import FCNHead
from otx.backend.native.models.segmentation.losses import CrossEntropyLossWithIgnore
from otx.backend.native.models.segmentation.segmentors import BaseSegmentationModel
from otx.config.data import TileConfig
from otx.metrics.dice import SegmCallable

if TYPE_CHECKING:
    from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable
    from torch import nn

    from otx.backend.native.schedulers import LRSchedulerListCallable
    from otx.metrics import MetricCallable
    from otx.types.label import LabelInfoTypes


class DinoV2Seg(OTXSegmentationModel):
    """DinoV2Seg for Semantic Segmentation model.

    Args:
        label_info (LabelInfoTypes): Information about the hierarchical labels.
        data_input_params (DataInputParams): Parameters for data input.
        model_name (Literal, optional): Name of the model. Defaults to "dinov2-small-seg".
        optimizer (OptimizerCallable, optional): Callable for the optimizer. Defaults to DefaultOptimizerCallable.
        scheduler (LRSchedulerCallable | LRSchedulerListCallable, optional): Callable for the learning rate scheduler.
        Defaults to DefaultSchedulerCallable.
        metric (MetricCallable, optional): Callable for the metric. Defaults to SegmCallable.
        torch_compile (bool, optional): Flag to indicate whether to use torch.compile. Defaults to False.
        tile_config (TileConfig, optional): Configuration for tiling. Defaults to TileConfig(enable_tiler=False).
    """

    pretrained_weights: ClassVar[dict[str, str]] = {
        "dinov2-small-seg": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth",
    }

    def __init__(
        self,
        label_info: LabelInfoTypes,
        data_input_params: DataInputParams,
        model_name: Literal["dinov2-small-seg"] = "dinov2-small-seg",
        optimizer: OptimizerCallable = DefaultOptimizerCallable,
        scheduler: LRSchedulerCallable | LRSchedulerListCallable = DefaultSchedulerCallable,
        metric: MetricCallable = SegmCallable,  # type: ignore[assignment]
        torch_compile: bool = False,
        tile_config: TileConfig = TileConfig(enable_tiler=False),
    ):
        super().__init__(
            label_info=label_info,
            data_input_params=data_input_params,
            model_name=model_name,
            optimizer=optimizer,
            scheduler=scheduler,
            metric=metric,
            torch_compile=torch_compile,
            tile_config=tile_config,
        )

    def _create_model(self, num_classes: int | None = None) -> nn.Module:
        # initialize backbones
        num_classes = num_classes if num_classes is not None else self.num_classes

        backbone = VisionTransformerBackbone(model_name=self.model_name, img_size=self.data_input_params.input_size)
        backbone.forward = partial(  # type: ignore[method-assign]
            backbone.get_intermediate_layers,
            n=[8, 9, 10, 11],
            reshape=True,
        )
        decode_head = FCNHead(self.model_name, num_classes=num_classes)
        criterion = CrossEntropyLossWithIgnore(ignore_index=self.label_info.ignore_index)  # type: ignore[attr-defined]

        backbone.init_weights()
        if self.model_name in self.pretrained_weights:
            print(f"init weight - {self.pretrained_weights[self.model_name]}")
            parts = urlparse(self.pretrained_weights[self.model_name])
            filename = Path(parts.path).name

            cache_dir = Path.home() / ".cache" / "torch" / "hub" / "checkpoints"
            cache_file = cache_dir / filename
            if not Path.exists(cache_file):
                download_url_to_file(self.pretrained_weights[self.model_name], cache_file, "", progress=True)
            backbone.load_pretrained(checkpoint_path=cache_file)

        # freeze backbone
        for _, v in backbone.named_parameters():
            v.requires_grad = False

        return BaseSegmentationModel(
            backbone=backbone,
            decode_head=decode_head,
            criterion=criterion,
        )

    @property
    def _optimization_config(self) -> dict[str, Any]:
        """PTQ config for DinoV2Seg."""
        return {"model_type": "transformer"}
