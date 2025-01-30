# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""D-Fine model implementations."""

from __future__ import annotations

import copy
import re
from typing import TYPE_CHECKING, Any, Literal

import torch
from torch import Tensor, nn
from torchvision.ops import box_convert
from torchvision.tv_tensors import BoundingBoxFormat

from otx.algo.detection.backbones.hgnetv2 import HGNetv2
from otx.algo.detection.detectors import DETR
from otx.algo.detection.heads.dfine_decoder import DFINETransformer
from otx.algo.detection.losses.dfine_loss import DFINECriterion
from otx.algo.detection.necks.dfine_hybrid_encoder import HybridEncoder
from otx.core.config.data import TileConfig
from otx.core.data.entity.base import OTXBatchLossEntity
from otx.core.data.entity.detection import DetBatchDataEntity, DetBatchPredEntity
from otx.core.exporter.base import OTXModelExporter
from otx.core.exporter.native import OTXNativeModelExporter
from otx.core.metrics.fmeasure import MeanAveragePrecisionFMeasureCallable
from otx.core.model.base import DefaultOptimizerCallable, DefaultSchedulerCallable
from otx.core.model.detection import ExplainableOTXDetModel

if TYPE_CHECKING:
    from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable

    from otx.core.metrics import MetricCallable
    from otx.core.schedulers import LRSchedulerListCallable
    from otx.core.types.label import LabelInfoTypes


PRETRAINED_ROOT: str = "https://github.com/Peterande/storage/releases/download/dfinev1.0/"

PRETRAINED_WEIGHTS: dict[str, str] = {
    "dfine_hgnetv2_n": PRETRAINED_ROOT + "dfine_n_coco.pth",
    "dfine_hgnetv2_s": PRETRAINED_ROOT + "dfine_s_coco.pth",
    "dfine_hgnetv2_m": PRETRAINED_ROOT + "dfine_m_coco.pth",
    "dfine_hgnetv2_l": PRETRAINED_ROOT + "dfine_l_coco.pth",
    "dfine_hgnetv2_x": PRETRAINED_ROOT + "dfine_x_coco.pth",
}


class DFine(ExplainableOTXDetModel):
    """OTX Detection model class for D-Fine."""

    input_size_multiplier = 32
    mean: tuple[float, float, float] = (0.0, 0.0, 0.0)
    std: tuple[float, float, float] = (255.0, 255.0, 255.0)

    def __init__(
        self,
        model_name: Literal[
            "dfine_hgnetv2_n",
            "dfine_hgnetv2_s",
            "dfine_hgnetv2_m",
            "dfine_hgnetv2_l",
            "dfine_hgnetv2_x",
        ],
        label_info: LabelInfoTypes,
        input_size: tuple[int, int] = (640, 640),
        optimizer: OptimizerCallable = DefaultOptimizerCallable,
        scheduler: LRSchedulerCallable | LRSchedulerListCallable = DefaultSchedulerCallable,
        metric: MetricCallable = MeanAveragePrecisionFMeasureCallable,
        multi_scale: bool = False,
        torch_compile: bool = False,
        tile_config: TileConfig = TileConfig(enable_tiler=False),
    ) -> None:
        self.load_from: str = PRETRAINED_WEIGHTS[model_name]
        self.multi_scale = multi_scale
        super().__init__(
            model_name=model_name,
            label_info=label_info,
            input_size=input_size,
            optimizer=optimizer,
            scheduler=scheduler,
            metric=metric,
            torch_compile=torch_compile,
            tile_config=tile_config,
        )

    def _build_model(self, num_classes: int) -> DETR:
        backbone = HGNetv2(model_name=self.model_name)
        encoder = HybridEncoder(model_name=self.model_name)
        decoder = DFINETransformer(
            model_name=self.model_name,
            num_classes=num_classes,
        )
        criterion = DFINECriterion(
            weight_dict={
                "loss_vfl": 1,
                "loss_bbox": 5,
                "loss_giou": 2,
                "loss_fgl": 0.15,
                "loss_ddf": 1.5,
            },
            alpha=0.75,
            gamma=2.0,
            reg_max=32,
            num_classes=num_classes,
        )

        if self.model_name == "dfine_hgnetv2_n":
            backbone_lr = 0.0004
        elif self.model_name == "dfine_hgnetv2_s":
            backbone_lr = 0.0001
        elif self.model_name == "dfine_hgnetv2_m":
            backbone_lr = 0.00002
        elif self.model_name in ("dfine_hgnetv2_l", "dfine_hgnetv2_x"):
            backbone_lr = 0.0000125
        else:
            msg = f"Unsupported model name: {self.model_name}"
            raise ValueError(msg)

        optimizer_configuration = [
            # no weight decay for norm layers in backbone
            {"params": "^(?=.*backbone)(?=.*norm).*$", "weight_decay": 0.0, "lr": backbone_lr},
            # lr for the backbone, but not norm layers is 0.00001
            {"params": "^(?=.*backbone)(?!.*norm).*$", "lr": backbone_lr},
            # no weight decay for norm layers and biases in encoder and decoder layers
            {"params": "^(?=.*(?:encoder|decoder))(?=.*(?:norm|bias)).*$", "weight_decay": 0.0},
        ]

        return DETR(
            multi_scale=None if self.multi_scale else [],
            backbone=backbone,
            encoder=encoder,
            decoder=decoder,
            criterion=criterion,
            num_classes=num_classes,
            optimizer_configuration=optimizer_configuration,
        )

    def _customize_inputs(
        self,
        entity: DetBatchDataEntity,
        pad_size_divisor: int = 32,
        pad_value: int = 0,
    ) -> dict[str, Any]:
        targets: list[dict[str, Any]] = []
        # prepare bboxes for the model
        for bb, ll in zip(entity.bboxes, entity.labels):
            # convert to cxcywh if needed
            if len(scaled_bboxes := bb):
                converted_bboxes = (
                    box_convert(bb, in_fmt="xyxy", out_fmt="cxcywh") if bb.format == BoundingBoxFormat.XYXY else bb
                )
                # normalize the bboxes
                scaled_bboxes = converted_bboxes / torch.tensor(bb.canvas_size[::-1]).tile(2)[None].to(
                    converted_bboxes.device,
                )
            targets.append({"boxes": scaled_bboxes, "labels": ll})

        if self.explain_mode:
            return {"entity": entity}

        return {
            "images": entity.images,
            "targets": targets,
        }

    def _customize_outputs(
        self,
        outputs: list[torch.Tensor] | dict,  # type: ignore[override]
        inputs: DetBatchDataEntity,
    ) -> DetBatchPredEntity | OTXBatchLossEntity:
        if self.training:
            if not isinstance(outputs, dict):
                raise TypeError(outputs)

            losses = OTXBatchLossEntity()
            for k, v in outputs.items():
                if isinstance(v, list):
                    losses[k] = sum(v)
                elif isinstance(v, Tensor):
                    losses[k] = v
                else:
                    msg = "Loss output should be list or torch.tensor but got {type(v)}"
                    raise TypeError(msg)
            return losses

        original_sizes = [img_info.ori_shape for img_info in inputs.imgs_info]
        scores, bboxes, labels = self.model.postprocess(outputs, original_sizes)

        if self.explain_mode:
            if not isinstance(outputs, dict):
                msg = f"Model output should be a dict, but got {type(outputs)}."
                raise ValueError(msg)

            if "feature_vector" not in outputs:
                msg = "No feature vector in the model output."
                raise ValueError(msg)

            if "saliency_map" not in outputs:
                msg = "No saliency maps in the model output."
                raise ValueError(msg)

            saliency_map = outputs["saliency_map"].detach().cpu().numpy()
            feature_vector = outputs["feature_vector"].detach().cpu().numpy()

            return DetBatchPredEntity(
                batch_size=len(outputs),
                images=inputs.images,
                imgs_info=inputs.imgs_info,
                scores=scores,
                bboxes=bboxes,
                labels=labels,
                feature_vector=feature_vector,
                saliency_map=saliency_map,
            )

        return DetBatchPredEntity(
            batch_size=len(outputs),
            images=inputs.images,
            imgs_info=inputs.imgs_info,
            scores=scores,
            bboxes=bboxes,
            labels=labels,
        )

    def configure_optimizers(self) -> tuple[list[torch.optim.Optimizer], list[dict[str, Any]]]:
        """Configure an optimizer and learning-rate schedulers.

        Set up the optimizer and schedulers from the provided inputs.
        Typically, a warmup scheduler is used initially, followed by the main scheduler.

        Returns:
            Two list. The former is a list that contains an optimizer
            The latter is a list of lr scheduler configs which has a dictionary format.
        """
        param_groups = self._get_optim_params(self.model.optimizer_configuration, self.model)
        optimizer = self.optimizer_callable(param_groups)
        schedulers = self.scheduler_callable(optimizer)

        def ensure_list(item: Any) -> list:  # noqa: ANN401
            return item if isinstance(item, list) else [item]

        lr_scheduler_configs = []
        for scheduler in ensure_list(schedulers):
            lr_scheduler_config = {"scheduler": scheduler}
            if hasattr(scheduler, "interval"):
                lr_scheduler_config["interval"] = scheduler.interval
            if hasattr(scheduler, "monitor"):
                lr_scheduler_config["monitor"] = scheduler.monitor
            lr_scheduler_configs.append(lr_scheduler_config)

        return [optimizer], lr_scheduler_configs

    @staticmethod
    def _get_optim_params(cfg: list[dict[str, Any]] | None, model: nn.Module) -> list[dict[str, Any]]:
        """Perform no bias decay and learning rate correction for the modules.

        The configuration dict should consist of regular expression pattern for the model parameters with "params" key.
        Other optimizer parameters can be added as well.

        E.g.:
            cfg = [{"params": "^((?!b).)*$", "lr": 0.01, "weight_decay": 0.0}, ..]
            The above configuration is for the parameters that do not contain "b".

            ^(?=.*a)(?=.*b).*$         means including a and b
            ^((?!b.)*a((?!b).)*$       means including a but not b
            ^((?!b|c).)*a((?!b|c).)*$  means including a but not (b | c)
        """
        if cfg is None:
            return model.parameters()

        cfg = copy.deepcopy(cfg)

        param_groups = []
        visited = []
        for pg in cfg:
            if "params" not in pg:
                msg = f"The 'params' key should be included in the configuration, but got {pg.keys()}"
                raise ValueError(msg)
            pattern = pg["params"]
            params = {k: v for k, v in model.named_parameters() if v.requires_grad and len(re.findall(pattern, k)) > 0}
            pg["params"] = params.values()
            param_groups.append(pg)
            visited.extend(list(params.keys()))

        names = [k for k, v in model.named_parameters() if v.requires_grad]

        if len(visited) < len(names):
            unseen = set(names) - set(visited)
            params = {k: v for k, v in model.named_parameters() if v.requires_grad and k in unseen}
            param_groups.append({"params": params.values()})
            visited.extend(list(params.keys()))

        return param_groups

    @property
    def _exporter(self) -> OTXModelExporter:
        """Creates OTXModelExporter object that can export the model."""
        if self.input_size is None:
            msg = f"Input size attribute is not set for {self.__class__}"
            raise ValueError(msg)

        return OTXNativeModelExporter(
            task_level_export_parameters=self._export_parameters,
            input_size=(1, 3, *self.input_size),
            mean=self.mean,
            std=self.std,
            resize_mode="standard",
            swap_rgb=False,
            via_onnx=False,
            onnx_export_configuration={
                "input_names": ["images"],
                "output_names": ["bboxes", "labels", "scores"],
                "dynamic_axes": {
                    "images": {0: "batch"},
                    "boxes": {0: "batch", 1: "num_dets"},
                    "labels": {0: "batch", 1: "num_dets"},
                    "scores": {0: "batch", 1: "num_dets"},
                },
                "autograd_inlining": False,
                "opset_version": 16,
            },
            output_names=["bboxes", "labels", "scores"],
        )

    @property
    def _optimization_config(self) -> dict[str, Any]:
        """PTQ config for D-FINE."""
        return {
            "model_type": "transformer",
            "advanced_parameters": {
                "activations_range_estimator_params": {
                    "min": {"statistics_type": "QUANTILE", "aggregator_type": "MIN", "quantile_outlier_prob": 1e-4},
                    "max": {"statistics_type": "QUANTILE", "aggregator_type": "MAX", "quantile_outlier_prob": 1e-4},
                },
            },
        }

    @staticmethod
    def _forward_explain_detection(
        self,  # noqa: ANN001
        entity: DetBatchDataEntity,
        mode: str = "tensor",  # noqa: ARG004
    ) -> dict[str, torch.Tensor]:
        """Forward function for explainable detection model."""
        backbone_feats = self.encoder(self.backbone(entity.images))
        predictions = self.decoder(backbone_feats, explain_mode=True)

        raw_logits = DETR.split_and_reshape_logits(
            backbone_feats,
            predictions["raw_logits"],
        )

        saliency_map = self.explain_fn(raw_logits)
        feature_vector = self.feature_vector_fn(backbone_feats)
        predictions.update(
            {
                "feature_vector": feature_vector,
                "saliency_map": saliency_map,
            },
        )

        return predictions
