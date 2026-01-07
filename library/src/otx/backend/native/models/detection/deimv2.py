# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""DEIM-DFine model implementations."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, Literal

from otx.backend.native.models.base import DataInputParams, DefaultOptimizerCallable, DefaultSchedulerCallable
from otx.backend.native.models.detection.backbones.dinov3sta import DINOv3STAs
from otx.backend.native.models.detection.detectors import DETR
from otx.backend.native.models.detection.heads.deim_decoder import DEIMTransformer
from otx.backend.native.models.detection.losses.deim_loss import DEIMCriterion
from otx.backend.native.models.detection.necks.dfine_hybrid_encoder import HybridEncoder
from otx.backend.native.models.utils.utils import load_checkpoint
from otx.config.data import TileConfig
from otx.metrics.fmeasure import MeanAveragePrecisionFMeasureCallable

from .deim import DEIMDFine

if TYPE_CHECKING:
    from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable

    from otx.backend.native.schedulers import LRSchedulerListCallable
    from otx.metrics import MetricCallable
    from otx.types.label import LabelInfoTypes


class DEIMV2(DEIMDFine):
    """OTX Detection model class for DEIMV2.

    DEIMV2 is an improved version of DEIMV1, which introduces DINOV3 backbone and improved decoder.

    It is based on the DEIMV2 paper: https://arxiv.org/abs/2412.04234
    The original implementation is available at: https://github.com/Intellindust-AI-Lab/DEIMv2/tree/main

    The model should be used with
    :class:`~otx.backend.native.callbacks.aug_scheduler.DataAugSwitch` and
    :class:`~otx.backend.native.callbacks.aug_scheduler.AugmentationSchedulerCallback`
    for dynamic augmentation scheduling.

    Attributes:
        _pretrained_weights (ClassVar[dict[str, str]]): Dictionary containing URLs for pretrained weights.
        input_size_multiplier (int): Multiplier for the input size.

    Args:
        label_info (LabelInfoTypes): Information about the labels.
        data_input_params (DataInputParams | None): Parameters for the image data preprocessing.
            If None, uses _default_preprocessing_params.
        model_name (literal, optional): Name of the model to use. Defaults to "deim_dfine_hgnetv2_x".
        optimizer (OptimizerCallable, optional): Callable for the optimizer. Defaults to DefaultOptimizerCallable.
        scheduler (LRSchedulerCallable | LRSchedulerListCallable, optional): Callable for the learning rate scheduler.
            Defaults to DefaultSchedulerCallable.
        metric (MetricCallable, optional): Callable for the metric. Defaults to MeanAveragePrecisionFMeasureCallable.
        multi_scale (bool, optional): Whether to use multi-scale training. Defaults to False.
        torch_compile (bool, optional): Whether to use torch compile. Defaults to False.
        tile_config (TileConfig, optional): Configuration for tiling. Defaults to TileConfig(enable_tiler=False).
    """

    _pretrained_weights: ClassVar[dict[str, str]] = {
        "deimv2_x": "https://github.com/kprokofi/DEIMv2/releases/download/1.0.0/deimv2_dinov3_x_coco.pth",
        "deimv2_l": "https://github.com/kprokofi/DEIMv2/releases/download/1.0.0/deimv2_dinov3_l_coco.pth",
        "deimv2_m": "https://github.com/kprokofi/DEIMv2/releases/download/1.0.0/deimv2_dinov3_m_coco.pth",
        "deimv2_s": "https://github.com/kprokofi/DEIMv2/releases/download/1.0.0/deimv2_dinov3_s_coco.pth",
    }

    input_size_multiplier = 32

    def __init__(
        self,
        label_info: LabelInfoTypes,
        data_input_params: DataInputParams | None = None,
        model_name: Literal[
            "deimv2_x",
            "deimv2_l",
            "deimv2_m",
            "deimv2_s",
        ] = "deimv2_x",
        optimizer: OptimizerCallable = DefaultOptimizerCallable,
        scheduler: LRSchedulerCallable | LRSchedulerListCallable = DefaultSchedulerCallable,
        metric: MetricCallable = MeanAveragePrecisionFMeasureCallable,
        multi_scale: bool = False,
        torch_compile: bool = False,
        tile_config: TileConfig = TileConfig(enable_tiler=False),
    ) -> None:
        super().__init__(
            model_name=model_name,  # type: ignore[arg-type]
            label_info=label_info,
            data_input_params=data_input_params,
            optimizer=optimizer,
            scheduler=scheduler,
            metric=metric,
            torch_compile=torch_compile,
            tile_config=tile_config,
            multi_scale=multi_scale,
        )

    def _create_model(self, num_classes: int | None = None) -> DETR:
        """Create DEIM-DFine model."""
        num_classes = num_classes if num_classes is not None else self.num_classes
        backbone = DINOv3STAs(model_name=self.model_name)
        encoder = HybridEncoder(model_name=self.model_name)
        decoder = DEIMTransformer(
            model_name=self.model_name,
            num_classes=num_classes,
            eval_spatial_size=self.data_input_params.input_size,
        )

        criterion = DEIMCriterion(
            weight_dict={
                "loss_vfl": 1,
                "loss_bbox": 5,
                "loss_giou": 2,
                "loss_fgl": 0.15,
                "loss_ddf": 1.5,
                "loss_mal": 1.0,
            },
            alpha=0.75,
            gamma=1.5,
            reg_max=32,
            num_classes=num_classes,
        )

        backbone_lr_mapping = {
            "deimv2_x": 0.00001,
            "deimv2_l": 0.0000125,
            "deimv2_m": 0.000025,
            "deimv2_s": 0.000025,
        }

        try:
            backbone_lr = backbone_lr_mapping[self.model_name]
        except KeyError as err:
            msg = f"Unsupported model name: {self.model_name}"
            raise ValueError(msg) from err

        optimizer_configuration = [
            {"params": "^(?=.*.dinov3)(?!.*(?:norm|bn|bias)).*$", "lr": backbone_lr},
            {"params": "^(?=.*.dinov3)(?=.*(?:norm|bn|bias)).*$", "lr": backbone_lr, "weight_decay": 0.0},
            {"params": "^(?=.*(?:sta|encoder|decoder))(?=.*(?:norm|bn|bias)).*$", "weight_decay": 0.0},
        ]

        model = DETR(
            multi_scale=self.multi_scale,
            backbone=backbone,
            encoder=encoder,
            decoder=decoder,
            criterion=criterion,
            num_classes=num_classes,
            optimizer_configuration=optimizer_configuration,
            input_size=self.data_input_params.input_size[0],
        )
        model.init_weights()
        load_checkpoint(model, self._pretrained_weights[self.model_name], map_location="cpu")
        return model

    @property
    def _default_preprocessing_params(self) -> DataInputParams | dict[str, DataInputParams]:
        return DataInputParams(input_size=(640, 640), mean=(123.675, 116.280, 103.530), std=(58.395, 57.120, 57.375))
