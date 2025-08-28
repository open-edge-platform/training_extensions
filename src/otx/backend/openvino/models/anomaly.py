# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""OTX Anomaly OpenVINO model.

All anomaly models use the same AnomalyDetection model from ModelAPI.
"""

# TODO(someone): Revisit mypy errors after OTXLitModule deprecation and anomaly refactoring
# mypy: ignore-errors

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import openvino
import torch
from torchvision.transforms.functional import resize

from otx.backend.openvino.models import OVModel
from otx.data.entity.torch import OTXDataBatch, OTXPredBatch
from otx.data.module import OTXDataModule
from otx.metrics.anomaly import AnomalyCallable
from otx.metrics.types import MetricCallable
from otx.types.label import AnomalyLabelInfo
from otx.types.task import OTXTaskType

if TYPE_CHECKING:
    from pathlib import Path

    from model_api.models import Model
    from model_api.models.anomaly import AnomalyResult


class OVAnomalyModel(OVModel):
    """Anomaly OpenVINO model."""

    def __init__(
        self,
        model_path: str,
        async_inference: bool = True,
        max_num_requests: int | None = None,
        use_throughput_mode: bool = True,
        model_api_configuration: dict[str, Any] | None = None,
        metric: MetricCallable = AnomalyCallable,
        task: Literal[
            OTXTaskType.ANOMALY,
            OTXTaskType.ANOMALY_CLASSIFICATION,
            OTXTaskType.ANOMALY_DETECTION,
            OTXTaskType.ANOMALY_SEGMENTATION,
        ] = OTXTaskType.ANOMALY,
        **kwargs,
    ) -> None:
        super().__init__(
            model_path=model_path,
            model_type="AnomalyDetection",
            async_inference=async_inference,
            max_num_requests=max_num_requests,
            use_throughput_mode=use_throughput_mode,
            model_api_configuration=model_api_configuration,
            metric=metric,
        )
        self._task = task

    def _create_model(self) -> Model:
        from model_api.adapters import OpenvinoAdapter, create_core, get_user_config
        from model_api.models import AnomalyDetection

        plugin_config = get_user_config("AUTO", str(self.num_requests), "AUTO")
        if self.use_throughput_mode:
            plugin_config["PERFORMANCE_HINT"] = "THROUGHPUT"

        model_adapter = OpenvinoAdapter(
            create_core(),
            self.model_path,
            max_num_requests=self.num_requests,
            plugin_config=plugin_config,
        )
        return AnomalyDetection.create_model(
            model=model_adapter,
            model_type=self.model_type,
            configuration=self.model_api_configuration,
        )

    def prepare_metric_inputs(
        self,
        preds: OTXPredBatch,  # type: ignore[override]
        inputs: OTXDataBatch,  # type: ignore[override]
    ) -> dict:
        """Convert prediction and input entities to a format suitable for metric computation.

        Args:
            preds (OTXPredBatch): The predicted batch entity containing predicted bboxes.
            inputs (OTXDataBatch): The input batch entity containing ground truth bboxes.

        Returns:
            MetricInput: A dictionary contains 'preds' and 'target' keys
            corresponding to the predicted and target bboxes for metric evaluation.
        """
        score_dict = {
            "pred_scores": torch.tensor(
                [score if label == 1 else 1 - score for score, label in zip(preds.scores, preds.labels)],
            ),
            "labels": torch.tensor(inputs.labels) if inputs.batch_size == 1 else torch.vstack(inputs.labels),
        }
        score_dict["anomaly_maps"] = torch.vstack(preds.masks)
        score_dict["masks"] = torch.vstack(inputs.masks)
        # resize masks and anomaly maps to 256,256 as this is the size used in Anomalib
        score_dict["masks"] = resize(score_dict["masks"], (256, 256))
        score_dict["anomaly_maps"] = resize(score_dict["anomaly_maps"], (256, 256))

        return score_dict

    def optimize(
        self,
        output_dir: Path,
        data_module: OTXDataModule,
        ptq_config: dict[str, Any] | None = None,
        optimized_model_name: str = "optimized_model",
    ) -> Path:
        """Runs NNCF quantization.

        Note:
            The only difference between the base class is that it uses `val_dataloader` instead of `train_dataloader`.

        See ``otx.core.model.base.OVModel.optimize`` for more details.
        """
        import nncf

        output_model_path = output_dir / (optimized_model_name + ".xml")

        def check_if_quantized(model: openvino.Model) -> bool:
            """Checks if OpenVINO model is already quantized."""
            nodes = model.get_ops()
            return any(op.get_type_name() == "FakeQuantize" for op in nodes)

        ov_model = openvino.Core().read_model(self.model_path)

        if check_if_quantized(ov_model):
            msg = "Model is already optimized by PTQ"
            raise RuntimeError(msg)

        val_dataset = data_module.val_dataloader()

        ptq_config_from_ir = self._read_ptq_config_from_ir(ov_model)
        if ptq_config is not None:
            ptq_config_from_ir.update(ptq_config)
            ptq_config = ptq_config_from_ir
        else:
            ptq_config = ptq_config_from_ir

        quantization_dataset = nncf.Dataset(val_dataset, self.transform_fn)  # type: ignore[attr-defined]

        compressed_model = nncf.quantize(  # type: ignore[attr-defined]
            ov_model,
            quantization_dataset,
            **ptq_config,
        )

        openvino.save_model(compressed_model, output_model_path)

        return output_model_path

    def _customize_outputs(self, outputs: list[AnomalyResult], inputs: OTXDataBatch) -> list[AnomalyResult]:
        """Return outputs from the OpenVINO model as is."""
        return OTXPredBatch(
            images=inputs.images,
            batch_size=inputs.batch_size,
            labels=[torch.tensor(0) if output.pred_label == "Normal" else torch.tensor(1) for output in outputs],
            scores=[torch.tensor(output.pred_score) for output in outputs],
            masks=[torch.tensor(output.anomaly_map).unsqueeze(0) / 255.0 for output in outputs],
        )

    def _create_label_info_from_ov_ir(self) -> AnomalyLabelInfo:
        return AnomalyLabelInfo()
