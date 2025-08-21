# Copyright (C) 2023-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Class definition for base model entity used in OTX."""

# mypy: disable-error-code="arg-type"

from __future__ import annotations

import inspect
import logging
import warnings
from abc import abstractmethod
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any, Callable, Literal, Sequence

import torch
from datumaro import LabelCategories
from lightning import LightningModule, Trainer
from torch import Tensor, nn
from torch.optim.lr_scheduler import ConstantLR
from torch.optim.sgd import SGD
from torchmetrics import Metric, MetricCollection

from otx import __version__
from otx.backend.native.optimizers.callable import OptimizerCallableSupportAdaptiveBS
from otx.backend.native.schedulers import (
    LinearWarmupScheduler,
    LinearWarmupSchedulerCallable,
    LRSchedulerListCallable,
    SchedulerCallableSupportAdaptiveBS,
)
from otx.backend.native.utils.utils import (
    ensure_callable,
    is_ckpt_for_finetuning,
    is_ckpt_from_otx_v1,
    remove_state_dict_prefix,
)
from otx.config.data import TileConfig
from otx.data.entity.base import (
    OTXBatchLossEntity,
)
from otx.data.entity.tile import OTXTileBatchDataEntity
from otx.data.entity.torch import OTXDataBatch, OTXPredBatch
from otx.metrics import MetricInput, NullMetricCallable
from otx.types.export import OTXExportFormatType, TaskLevelExportParameters
from otx.types.label import LabelInfo, LabelInfoTypes
from otx.types.precision import OTXPrecisionType
from otx.types.task import OTXTaskType

if TYPE_CHECKING:
    from pathlib import Path

    from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable
    from lightning.pytorch.utilities.types import LRSchedulerTypeUnion, OptimizerLRScheduler
    from torch.optim.lr_scheduler import LRScheduler
    from torch.optim.optimizer import Optimizer, params_t

    from otx.backend.native.exporter.base import OTXModelExporter
    from otx.data.module import OTXDataModule
    from otx.metrics import MetricCallable

logger = logging.getLogger()


@dataclass
class DataInputParams:
    """Parameters of the input data such as input size, mean, and std."""

    input_size: tuple[int, int]
    mean: tuple[float, float, float]
    std: tuple[float, float, float]

    def as_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {"input_size": self.input_size, "mean": self.mean, "std": self.std}

    def as_ncwh(self, batch_size: int = 1) -> tuple[int, int, int, int]:
        """Convert input_size to NCWH format."""
        return (batch_size, 3, *self.input_size)


def _default_optimizer_callable(params: params_t) -> Optimizer:
    return SGD(params=params, lr=0.01)


def _default_scheduler_callable(
    optimizer: Optimizer,
    interval: Literal["epoch", "step"] = "epoch",
    **kwargs,
) -> LRScheduler:
    scheduler = ConstantLR(optimizer=optimizer, **kwargs)
    # NOTE: "interval" attribute should be set to configure the scheduler's step interval correctly
    scheduler.interval = interval
    return scheduler


DefaultOptimizerCallable = _default_optimizer_callable
DefaultSchedulerCallable = _default_scheduler_callable


class OTXModel(LightningModule):
    """Base class for the models used in OTX.

    Args:
        num_classes: Number of classes this model can predict.

    Attributes:
        explain_mode: If true, `self.predict_step()` will produce a XAI output as well
        input_size_multiplier (int):
            multiplier value for input size a model requires. If input_size isn't multiple of this value,
            error is raised.
    """

    _OPTIMIZED_MODEL_BASE_NAME: str = "optimized_model"
    input_size_multiplier: int = 1

    def __init__(
        self,
        label_info: LabelInfoTypes | int | Sequence,
        data_input_params: DataInputParams | dict,
        task: OTXTaskType | None = None,
        model_name: str = "OTXModel",
        optimizer: OptimizerCallable = DefaultOptimizerCallable,
        scheduler: LRSchedulerCallable | LRSchedulerListCallable = DefaultSchedulerCallable,
        metric: MetricCallable = NullMetricCallable,
        torch_compile: bool = False,
        tile_config: TileConfig | dict = TileConfig(enable_tiler=False),
    ) -> None:
        """Initialize the base model with the given parameters.

        Args:
            label_info (LabelInfoTypes | int | Sequence): Information about the labels used in the model.
                If `int` is given, label info will be constructed from number of classes,
                if `Sequence` is given, label info will be constructed from the sequence of label names.
            data_input_params (DataInputParams | dict): Parameters of the input data such as input size, mean, and std.
            model_name (str, optional): Name of the model. Defaults to "OTXModel".
            optimizer (OptimizerCallable, optional): Callable for the optimizer. Defaults to DefaultOptimizerCallable.
            scheduler (LRSchedulerCallable | LRSchedulerListCallable): Callable for the learning rate scheduler.
                Defaults to DefaultSchedulerCallable.
            metric (MetricCallable, optional): Callable for the metric. Defaults to NullMetricCallable.
            torch_compile (bool, optional): Flag to indicate if torch.compile should be used. Defaults to False.
            tile_config (TileConfig, optional): Configuration for tiling. Defaults to TileConfig(enable_tiler=False).

        Returns:
            None
        """
        super().__init__()

        self._label_info = self._dispatch_label_info(label_info)
        if isinstance(data_input_params, dict):
            data_input_params = DataInputParams(**data_input_params)
        self._check_preprocessing_params(data_input_params)
        self.data_input_params = data_input_params
        self.model_name = model_name
        self.model = self._create_model()
        self.optimizer_callable = ensure_callable(optimizer)
        self.scheduler_callable = ensure_callable(scheduler)
        self.metric_callable = ensure_callable(metric)
        self._task = task

        self.torch_compile = torch_compile
        self._explain_mode = False

        # NOTE: To guarantee immutablility of the default value
        if isinstance(tile_config, dict):
            tile_config = TileConfig(**tile_config)
        self._tile_config = tile_config.clone()
        self.save_hyperparameters(
            logger=False,
            ignore=["optimizer", "scheduler", "metric", "label_info", "tile_config", "data_input_params"],
        )

    def training_step(self, batch: OTXDataBatch, batch_idx: int) -> Tensor:
        """Step for model training."""
        train_loss = self.forward(inputs=batch)
        if train_loss is None:
            msg = "Loss is None."
            raise ValueError(msg)

        if isinstance(train_loss, Tensor):
            self.log(
                "train/loss",
                train_loss,
                on_step=True,
                on_epoch=False,
                prog_bar=True,
            )
            return train_loss
        if isinstance(train_loss, dict):
            for k, v in train_loss.items():
                self.log(
                    f"train/{k}",
                    v,
                    on_step=True,
                    on_epoch=False,
                    prog_bar=True,
                )

            total_train_loss = train_loss.get("total_loss", sum(train_loss.values()))
            self.log(
                "train/total_loss",
                total_train_loss,
                on_step=True,
                on_epoch=False,
                prog_bar=True,
            )
            return total_train_loss

        raise TypeError(train_loss)

    def validation_step(self, batch: OTXDataBatch, batch_idx: int) -> OTXPredBatch:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        preds = self.forward(inputs=batch)

        if isinstance(preds, OTXBatchLossEntity):
            raise TypeError(preds)

        metric_inputs = self._convert_pred_entity_to_compute_metric(preds, batch)

        if isinstance(metric_inputs, dict):
            self.metric.update(**metric_inputs)
            return preds

        if isinstance(metric_inputs, list) and all(isinstance(inp, dict) for inp in metric_inputs):
            for inp in metric_inputs:
                self.metric.update(**inp)
            return preds

        raise TypeError(metric_inputs)

    def test_step(self, batch: OTXDataBatch, batch_idx: int) -> OTXPredBatch:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        preds = self.forward(inputs=batch)

        if isinstance(preds, OTXBatchLossEntity):
            raise TypeError(preds)

        metric_inputs = self._convert_pred_entity_to_compute_metric(preds, batch)

        if isinstance(metric_inputs, dict):
            self.metric.update(**metric_inputs)
            return preds

        if isinstance(metric_inputs, list) and all(isinstance(inp, dict) for inp in metric_inputs):
            for inp in metric_inputs:
                self.metric.update(**inp)
            return preds

        raise TypeError(metric_inputs)

    def predict_step(
        self,
        batch: OTXDataBatch | OTXTileBatchDataEntity,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> OTXPredBatch:
        """Step function called during PyTorch Lightning Trainer's predict."""
        if self.explain_mode:
            return self.forward_explain(inputs=batch)

        outputs = self.forward(inputs=batch)

        if isinstance(outputs, OTXBatchLossEntity):
            raise TypeError(outputs)

        return outputs

    def on_validation_start(self) -> None:
        """Called at the beginning of validation."""
        self.configure_metric()

    def on_test_start(self) -> None:
        """Called at the beginning of testing."""
        self.configure_metric()

    def on_validation_epoch_start(self) -> None:
        """Callback triggered when the validation epoch starts."""
        self.metric.reset()

    def on_test_epoch_start(self) -> None:
        """Callback triggered when the test epoch starts."""
        self.metric.reset()

    def on_validation_epoch_end(self) -> None:
        """Callback triggered when the validation epoch ends."""
        self._log_metrics(self.metric, "val")

    def on_test_epoch_end(self) -> None:
        """Callback triggered when the test epoch ends."""
        self._log_metrics(self.metric, "test")

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate, test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.torch_compile and stage == "fit":
            # Set the log_level of this to error due to the numerous warning messages from compile.
            torch._logging.set_logs(dynamo=logging.ERROR)  # noqa: SLF001
            self.model = torch.compile(self.model)
            warnings.warn(
                (
                    "torch model compile has been applied. It may be slower than usual because "
                    "it builds the graph in the initial training."
                ),
                stacklevel=1,
            )

    def configure_optimizers(self) -> OptimizerLRScheduler:
        """Configure an optimizer and learning-rate schedulers.

        Configure an optimizer and learning-rate schedulers
        from the given optimizer and scheduler or scheduler list callable in the constructor.
        Generally, there is two lr schedulers. One is for a linear warmup scheduler and
        the other is the main scheduler working after the warmup period.

        Returns:
            Two list. The former is a list that contains an optimizer
            The latter is a list of lr scheduler configs which has a dictionary format.
        """
        optimizer = self.optimizer_callable(self.parameters())
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

    def configure_metric(self) -> None:
        """Configure the metric."""
        if not callable(self.metric_callable):
            raise TypeError(self.metric_callable)

        metric = self.metric_callable(self.label_info)

        if not isinstance(metric, (Metric, MetricCollection)):
            msg = "Metric should be the instance of `torchmetrics.Metric` or `torchmetrics.MetricCollection`."
            raise TypeError(msg, metric)

        self._metric = metric.to(self.device)

    @property
    def metric(self) -> Metric | MetricCollection:
        """Metric module for this OTX model."""
        return self._metric

    @abstractmethod
    def _convert_pred_entity_to_compute_metric(
        self,
        preds: OTXPredBatch,
        inputs: OTXDataBatch,
    ) -> MetricInput:
        """Convert given inputs to a Python dictionary for the metric computation."""
        raise NotImplementedError

    def _log_metrics(self, meter: Metric, key: Literal["val", "test"], **compute_kwargs) -> None:
        sig = inspect.signature(meter.compute)
        filtered_kwargs = {key: value for key, value in compute_kwargs.items() if key in sig.parameters}
        if removed_kwargs := set(compute_kwargs.keys()).difference(filtered_kwargs.keys()):
            msg = f"These keyword arguments are removed since they are not in the function signature: {removed_kwargs}"
            logger.debug(msg)

        results: dict[str, Tensor] = meter.compute(**filtered_kwargs)

        if not isinstance(results, dict):
            raise TypeError(results)

        if not results:
            msg = f"{meter} has no data to compute metric or there is an error computing metric"
            raise RuntimeError(msg)

        for name, value in results.items():
            log_metric_name = f"{key}/{name}"

            if not isinstance(value, Tensor) or value.numel() != 1:
                msg = f"Log metric name={log_metric_name} is not a scalar tensor. Skip logging it."
                warnings.warn(msg, stacklevel=1)
                continue

            self.log(log_metric_name, value.to(self.device), sync_dist=True, prog_bar=True)

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """Callback on saving checkpoint."""
        if self.torch_compile:
            # If torch_compile is True, a prefix key named _orig_mod. is added to the state_dict. Remove this.
            compiled_state_dict = checkpoint["state_dict"]
            checkpoint["state_dict"] = remove_state_dict_prefix(compiled_state_dict, "_orig_mod.")
        super().on_save_checkpoint(checkpoint)
        checkpoint["hyper_parameters"]["label_info"] = asdict(self.label_info)
        checkpoint["otx_version"] = __version__
        checkpoint["hyper_parameters"]["tile_config"] = asdict(self.tile_config)
        checkpoint["hyper_parameters"]["data_input_params"] = asdict(self.data_input_params)
        checkpoint.pop("datamodule_hparams_name", None)
        checkpoint.pop(
            "datamodule_hyper_parameters",
            None,
        )  # Remove datamodule_hyper_parameters to prevent storing OTX classes

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """Callback on loading checkpoint."""
        super().on_load_checkpoint(checkpoint)
        hyper_parameters = checkpoint.get("hyper_parameters")
        if hyper_parameters:
            if ckpt_label_info := hyper_parameters.get("label_info"):
                self._label_info = self._dispatch_label_info(ckpt_label_info)
            if ckpt_tile_config := hyper_parameters.get("tile_config"):
                if isinstance(ckpt_tile_config, dict):
                    ckpt_tile_config = TileConfig(**ckpt_tile_config)
                self.tile_config = ckpt_tile_config

    def load_state_dict_incrementally(self, ckpt: dict[str, Any], *args, **kwargs) -> None:
        """Load state dict incrementally."""
        ckpt_label_info: LabelInfo | None = (
            ckpt.get("hyper_parameters", {}).get("label_info")
            if not is_ckpt_from_otx_v1(ckpt)
            else self.get_ckpt_label_info_v1(ckpt)
        )

        if ckpt_label_info is None:
            msg = "Checkpoint should have `label_info`."
            raise ValueError(msg, ckpt_label_info)

        ckpt_label_info = self._dispatch_label_info(ckpt_label_info)

        if not hasattr(ckpt_label_info, "label_ids"):
            msg = "Loading checkpoint from OTX < 2.2.1, label_ids are assigned automatically"
            logger.info(msg)
            ckpt_label_info.label_ids = [str(i) for i, _ in enumerate(ckpt_label_info.label_names)]

        if not set(ckpt_label_info.label_names).isdisjoint(self.label_info.label_names):
            msg = (
                "Load model state dictionary incrementally: "
                f"Label info from checkpoint: {ckpt_label_info} -> "
                f"Label info from training data: {self.label_info}"
            )
            logger.info(msg)
            self.register_load_state_dict_pre_hook(
                self.label_info.label_names,
                ckpt_label_info.label_names,
            )

        # Model weights
        state_dict: dict[str, Any] = ckpt.get("state_dict", {}) if not is_ckpt_from_otx_v1(ckpt) else ckpt

        if state_dict is None or state_dict == {}:
            msg = "Checkpoint should have `state_dict`."
            raise ValueError(msg, state_dict)

        self.load_state_dict(state_dict, *args, **kwargs)

    def load_state_dict(self, ckpt: dict[str, Any], *args, **kwargs) -> None:
        """Load state dictionary from checkpoint state dictionary.

        It successfully loads the checkpoint from OTX v1.x and for finetune and for resume.

        If checkpoint's label_info and OTXLitModule's label_info are different,
        load_state_pre_hook for smart weight loading will be registered.
        """
        if is_ckpt_from_otx_v1(ckpt):
            msg = "The checkpoint comes from OTXv1, checkpoint keys will be updated automatically."
            warnings.warn(msg, stacklevel=2)
            state_dict = self.load_from_otx_v1_ckpt(ckpt)
        elif is_ckpt_for_finetuning(ckpt):
            self.on_load_checkpoint(ckpt)
            state_dict = ckpt["state_dict"]
        else:
            state_dict = ckpt
        return super().load_state_dict(state_dict, *args, **kwargs)

    def load_from_otx_v1_ckpt(self, ckpt: dict[str, Any]) -> dict:
        """Load the previous OTX ckpt according to OTX2.0."""
        raise NotImplementedError

    @staticmethod
    def get_ckpt_label_info_v1(ckpt: dict) -> LabelInfo:
        """Generate label info from OTX v1 checkpoint."""
        return LabelInfo.from_dm_label_groups(LabelCategories.from_iterable(ckpt["labels"].keys()))

    @property
    def label_info(self) -> LabelInfo:
        """Get this model label information."""
        return self._label_info

    @label_info.setter
    def label_info(self, label_info: LabelInfoTypes) -> None:
        """Set this model label information."""
        self._set_label_info(label_info)

    def _set_label_info(self, label_info: LabelInfoTypes) -> None:
        """Actual implementation for set this model label information.

        Derived classes should override this function.
        """
        msg = (
            "Assign new label_info to the model. "
            "It is usually not recommended. "
            "Please create a new model instance by giving label_info to its initializer "
            "such as `OTXModel(label_info=label_info, ...)`."
        )
        logger.warning(msg, stacklevel=0)

        new_label_info = self._dispatch_label_info(label_info)

        old_num_classes = self._label_info.num_classes
        new_num_classes = new_label_info.num_classes

        if old_num_classes != new_num_classes:
            msg = (
                f"Given LabelInfo has the different number of classes "
                f"({old_num_classes}!={new_num_classes}). "
                "The model prediction layer is reset to the new number of classes "
                f"(={new_num_classes})."
            )
            logger.warning(msg, stacklevel=0)
            self._reset_prediction_layer(num_classes=new_label_info.num_classes)

        self._label_info = new_label_info

    @property
    def num_classes(self) -> int:
        """Returns model's number of classes. Can be redefined at the model's level."""
        return self.label_info.num_classes

    @property
    def explain_mode(self) -> bool:
        """Get model explain mode."""
        return self._explain_mode

    @explain_mode.setter
    def explain_mode(self, explain_mode: bool) -> None:
        """Set model explain mode."""
        self._explain_mode = explain_mode

    @abstractmethod
    def _create_model(self, num_classes: int | None = None) -> nn.Module:
        """Create a PyTorch model for this class."""

    def _customize_inputs(self, inputs: OTXDataBatch) -> dict[str, Any]:
        """Customize OTX input batch data entity if needed for your model."""
        raise NotImplementedError

    def _customize_outputs(
        self,
        outputs: Any,  # noqa: ANN401
        inputs: OTXDataBatch,
    ) -> OTXPredBatch | OTXBatchLossEntity:
        """Customize OTX output batch data entity if needed for model."""
        raise NotImplementedError

    def forward(
        self,
        inputs: OTXDataBatch,
    ) -> OTXPredBatch | OTXBatchLossEntity:
        """Model forward function."""
        # If customize_inputs is overridden
        if isinstance(inputs, OTXTileBatchDataEntity):
            return self.forward_tiles(inputs)

        outputs = (
            self.model(**self._customize_inputs(inputs))
            if self._customize_inputs != OTXModel._customize_inputs
            else self.model(inputs)
        )

        return (
            self._customize_outputs(outputs, inputs)
            if self._customize_outputs != OTXModel._customize_outputs
            else outputs
        )

    def forward_explain(self, inputs: OTXDataBatch) -> OTXPredBatch:
        """Model forward explain function."""
        msg = "Derived model class should implement this class to support the explain pipeline."
        raise NotImplementedError(msg)

    def forward_for_tracing(self, *args, **kwargs) -> Tensor | dict[str, Tensor]:
        """Model forward function used for the model tracing during model exportation."""
        msg = (
            "Derived model class should implement this class to support the export pipeline. "
            "If it wants to use `otx.core.exporter.native.OTXNativeModelExporter`."
        )
        raise NotImplementedError(msg)

    def get_explain_fn(self) -> Callable:
        """Returns explain function."""
        raise NotImplementedError

    def forward_tiles(
        self,
        inputs: OTXTileBatchDataEntity,
    ) -> OTXPredBatch | OTXBatchLossEntity:
        """Model forward function for tile task."""
        raise NotImplementedError

    def register_load_state_dict_pre_hook(self, model_classes: list[str], ckpt_classes: list[str]) -> None:
        """Register load_state_dict_pre_hook.

        Args:
            model_classes (list[str]): Class names from training data.
            ckpt_classes (list[str]): Class names from checkpoint state dictionary.
        """
        self.model_classes = model_classes
        self.ckpt_classes = ckpt_classes
        self._register_load_state_dict_pre_hook(self.load_state_dict_pre_hook)

    def load_state_dict_pre_hook(self, state_dict: dict[str, torch.Tensor], prefix: str, *args, **kwargs) -> None:
        """Modify input state_dict according to class name matching before weight loading."""
        model2ckpt = self.map_class_names(self.model_classes, self.ckpt_classes)

        for param_name in self._identify_classification_layers():
            model_param = self.state_dict()[param_name].clone()
            ckpt_param = state_dict[prefix + param_name]
            for model_dst, ckpt_dst in enumerate(model2ckpt):
                if ckpt_dst >= 0:
                    model_param[model_dst : model_dst + 1].copy_(
                        ckpt_param[ckpt_dst : ckpt_dst + 1],
                    )

            # Replace checkpoint weight by mixed weights
            state_dict[prefix + param_name] = model_param

    def _identify_classification_layers(self, prefix: str = "model.") -> list[str]:
        """Simple identification of the classification layers."""
        # identify classification layers
        sample_model_dict = self._create_model(num_classes=3).state_dict()
        incremental_model_dict = self._create_model(num_classes=4).state_dict()
        # iterate over the model dict and compare the shapes.
        # Add the key to the list if the shapes are different
        return [
            prefix + key
            for key in sample_model_dict
            if sample_model_dict[key].shape != incremental_model_dict[key].shape
        ]

    @staticmethod
    def map_class_names(src_classes: list[str], dst_classes: list[str]) -> list[int]:
        """Computes src to dst index mapping.

        src2dst[src_idx] = dst_idx
        #  according to class name matching, -1 for non-matched ones
        assert(len(src2dst) == len(src_classes))
        ex)
          src_classes = ['person', 'car', 'tree']
          dst_classes = ['tree', 'person', 'sky', 'ball']
          -> Returns src2dst = [1, -1, 0]
        """
        src2dst = []
        for src_class in src_classes:
            if src_class in dst_classes:
                src2dst.append(dst_classes.index(src_class))
            else:
                src2dst.append(-1)
        return src2dst

    def optimize(self, output_dir: Path, data_module: OTXDataModule, ptq_config: dict[str, Any] | None = None) -> Path:
        """Runs quantization of the model with NNCF.PTQ on the passed data. Works only for OpenVINO models.

        PTQ performs int-8 quantization on the input model, so the resulting model
        comes in mixed precision (some operations, however, remain in FP32).

        Args:
            output_dir (Path): working directory to save the optimized model.
            data_module (OTXDataModule): dataset for calibration of quantized layers.
            ptq_config (dict[str, Any] | None): config for NNCF.PTQ.

        Returns:
            Path: path to the resulting optimized OpenVINO model.
        """
        msg = "Optimization is not implemented for torch models"
        raise NotImplementedError(msg)

    def export(
        self,
        output_dir: Path,
        base_name: str,
        export_format: OTXExportFormatType,
        precision: OTXPrecisionType = OTXPrecisionType.FP32,
        to_exportable_code: bool = False,
    ) -> Path:
        """Export this model to the specified output directory.

        Args:
            output_dir (Path): directory for saving the exported model
            base_name: (str): base name for the exported model file. Extension is defined by the target export format
            export_format (OTXExportFormatType): format of the output model
            precision (OTXExportPrecisionType): precision of the output model
            to_exportable_code (bool): flag to export model in exportable code with demo package

        Returns:
            Path: path to the exported model.
        """
        mode = self.training
        self.eval()

        orig_forward = self.forward
        orig_trainer = self._trainer  # type: ignore[has-type]
        try:
            if self._trainer is None:  # type: ignore[has-type]
                self._trainer = Trainer()
            self.forward = self.forward_for_tracing  # type: ignore[method-assign, assignment]
            return self._exporter.export(
                self,
                output_dir,
                base_name,
                export_format,
                precision,
                to_exportable_code,
            )
        finally:
            self.train(mode)
            self.forward = orig_forward  # type: ignore[method-assign]
            self._trainer = orig_trainer

    @property
    def _exporter(self) -> OTXModelExporter:
        """Defines exporter of the model. Should be overridden in subclasses."""
        msg = (
            "To export this OTXModel, you should implement an appropriate exporter for it. "
            "You can try to reuse ones provided in `otx.core.exporter.*`."
        )
        raise NotImplementedError(msg)

    @property
    def _export_parameters(self) -> TaskLevelExportParameters:
        """Defines export parameters sharable at a task level.

        To export OTXModel which is compatible with ModelAPI,
        you should define an appropriate export parameters for each task.
        This property is usually defined at the task level classes defined in `otx.core.model.*`.
        Please refer to `TaskLevelExportParameters` for more details.

        Returns:
            Collection of exporter parameters that can be defined at a task level.

        Examples:
            This example shows how this property is used at the new model development

            ```python

            class MyDetectionModel(OTXDetectionModel):
                ...

                @property
                def _exporter(self) -> OTXModelExporter:
                    # `self._export_parameters` defined at `OTXDetectionModel`
                    # You can redefine it `MyDetectionModel` if you need
                    return OTXModelExporter(
                        task_level_export_parameters=self._export_parameters,
                        ...
                    )
            ```
        """
        return TaskLevelExportParameters(
            model_type="null",
            task_type="null",
            model_name=self.model_name,
            label_info=self.label_info,
            optimization_config=self._optimization_config,
        )

    def _reset_prediction_layer(self, num_classes: int) -> None:
        """Reset its prediction layer with a given number of classes.

        Args:
            num_classes: Number of classes
        """
        raise NotImplementedError

    @property
    def _optimization_config(self) -> dict[str, str]:
        return {}

    def lr_scheduler_step(self, scheduler: LRSchedulerTypeUnion, metric: Tensor) -> None:
        """It is required to prioritize the warmup lr scheduler than other lr scheduler during a warmup period.

        It will ignore other lr scheduler's stepping if the warmup scheduler is currently activated.
        """
        warmup_schedulers = [
            config.scheduler
            for config in self.trainer.lr_scheduler_configs
            if isinstance(config.scheduler, LinearWarmupScheduler)
        ]

        if not warmup_schedulers:
            # There is no warmup scheduler
            return super().lr_scheduler_step(scheduler=scheduler, metric=metric)

        if len(warmup_schedulers) != 1:
            msg = "No more than one warmup schedulers coexist."
            raise RuntimeError(msg)

        warmup_scheduler = next(iter(warmup_schedulers))

        if scheduler != warmup_scheduler and warmup_scheduler.activated:
            msg = (
                "Warmup lr scheduler is currently activated. "
                "Ignore other schedulers until the warmup lr scheduler is finished"
            )
            logger.debug(msg)
            return None

        return super().lr_scheduler_step(scheduler=scheduler, metric=metric)

    def patch_optimizer_and_scheduler_for_adaptive_bs(self) -> None:
        """Patch optimizer and scheduler for adaptive batch size.

        This is inplace function changing inner states (`optimizer_callable` and `scheduler_callable`).
        Both will be changed to be picklable. In addition, `optimizer_callable` is changed
        to make its hyperparameters gettable.
        """
        if not isinstance(self.optimizer_callable, OptimizerCallableSupportAdaptiveBS):
            self.optimizer_callable = OptimizerCallableSupportAdaptiveBS.from_callable(self.optimizer_callable)

        if not isinstance(self.scheduler_callable, SchedulerCallableSupportAdaptiveBS) and not isinstance(
            self.scheduler_callable,
            LinearWarmupSchedulerCallable,  # LinearWarmupSchedulerCallable natively supports adaptive batch size
        ):
            self.scheduler_callable = SchedulerCallableSupportAdaptiveBS.from_callable(self.scheduler_callable)

    @property
    def tile_config(self) -> TileConfig:
        """Get tiling configurations."""
        return self._tile_config

    @tile_config.setter
    def tile_config(self, tile_config: TileConfig) -> None:
        """Set tiling configurations."""
        msg = (
            "Assign new tile_config to the model. "
            "It is usually not recommended. "
            "Please create a new model instance by giving tile_config to its initializer "
            "such as `OTXModel(..., tile_config=tile_config)`."
        )
        logger.warning(msg, stacklevel=0)

        self._tile_config = tile_config

    def get_dummy_input(self, batch_size: int = 1) -> OTXDataBatch:
        """Generates a dummy input, suitable for launching forward() on it.

        Args:
            batch_size (int, optional): number of elements in a dummy input sequence. Defaults to 1.

        Returns:
            TorchDataBatch: A batch containing randomly generated inference data.
        """
        raise NotImplementedError

    @staticmethod
    def _dispatch_label_info(label_info: LabelInfoTypes) -> LabelInfo:
        if isinstance(label_info, dict):
            if "label_ids" not in label_info:
                # NOTE: This is for backward compatibility
                label_info["label_ids"] = label_info["label_names"]
            return LabelInfo(**label_info)
        if isinstance(label_info, int):
            return LabelInfo.from_num_classes(num_classes=label_info)
        if isinstance(label_info, (list, tuple)) and all(isinstance(name, str) for name in label_info):
            return LabelInfo(
                label_names=label_info,
                label_groups=[label_info],
                label_ids=[str(i) for i in range(len(label_info))],
            )
        if isinstance(label_info, LabelInfo):
            if not hasattr(label_info, "label_ids"):
                # NOTE: This is for backward compatibility
                label_info.label_ids = label_info.label_names
            return label_info

        raise TypeError(label_info)

    def _check_preprocessing_params(self, preprocessing_params: DataInputParams | None) -> None:
        """Check the validity of the preprocessing parameters."""
        if preprocessing_params is None:
            msg = "Data input parameters should not be None."
            raise ValueError(msg)

        input_size = preprocessing_params.input_size
        mean = preprocessing_params.mean
        std = preprocessing_params.std

        if not (len(mean) == 3 and all(isinstance(m, float) for m in mean)):
            msg = f"Mean should be a tuple of 3 float values, but got {mean} instead."
            raise ValueError(msg)
        if not (len(std) == 3 and all(isinstance(s, float) for s in std)):
            msg = f"Std should be a tuple of 3 float values, but got {std} instead."
            raise ValueError(msg)

        if not all(0 <= m <= 255 for m in mean):
            msg = f"Mean values should be in the range [0, 255], but got {mean} instead."
            raise ValueError(msg)
        if not all(0 <= s <= 255 for s in std):
            msg = f"Std values should be in the range [0, 255], but got {std} instead."
            raise ValueError(msg)

        if input_size is not None and (
            input_size[0] % self.input_size_multiplier != 0 or input_size[1] % self.input_size_multiplier != 0
        ):
            msg = f"Input size should be a multiple of {self.input_size_multiplier}, but got {input_size} instead."
            raise ValueError(msg)

    @property
    def task(self) -> OTXTaskType:
        """Get  task type."""
        if self._task is None:
            msg = "Task type is not set. Please set the task type before using this model."
            raise ValueError(msg)
        return self._task
