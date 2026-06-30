# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Classification validators for the getitune data bridge."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

import torch

if TYPE_CHECKING:
    from torchmetrics import Metric, MetricCollection
from torch.utils.data import DataLoader
from ultralytics.models.yolo.classify import ClassificationValidator as _UltralyticsClassificationValidator
from ultralytics.utils import LOGGER

from getitune.backend.ultralytics.data.adapter import UltralyticsDatasetAdapter
from getitune.backend.ultralytics.data.collate import classification_collate_fn, multilabel_collate_fn
from getitune.metrics.accuracy import MultiLabelClsMetricCallable
from getitune.types.label import LabelInfo

from .base import GetiTuneValidatorMixin


class ClassificationValidator(GetiTuneValidatorMixin, _UltralyticsClassificationValidator):
    """Classification validator for the getitune data bridge."""

    _task_kind: ClassVar[str] = "classify"

    def _build_adapter_dataloader(self) -> DataLoader:
        """Build a classification DataLoader from the DataModule's val/test subset."""
        if self._datamodule is None:
            msg = "_build_adapter_dataloader requires a DataModule"
            raise TypeError(msg)
        test_key = self._datamodule.test_subset.subset_name
        val_key = self._datamodule.val_subset.subset_name
        subset = self._datamodule.subsets.get(test_key) or self._datamodule.subsets[val_key]
        adapter = UltralyticsDatasetAdapter(subset, task_kind="classify")
        return DataLoader(
            adapter,
            batch_size=self.args.batch,  # type: ignore[attr-defined]
            shuffle=False,
            collate_fn=classification_collate_fn,
            pin_memory=True,
        )


class MultiLabelClassificationValidator(GetiTuneValidatorMixin, _UltralyticsClassificationValidator):
    """Multi-label classification validator using getitune torchmetrics."""

    _task_kind: ClassVar[str] = "multilabel"

    def init_metrics(self, model: torch.nn.Module) -> None:
        """Initialize multi-label metrics from DataModule label info."""
        raw_names = getattr(model, "names", None)
        names: dict[int, str] = raw_names if isinstance(raw_names, dict) else {}
        self.names = names
        self.nc = len(names)
        label_info = self._label_info_from_names(names)
        self.metric: Metric | MetricCollection = MultiLabelClsMetricCallable(label_info)
        self.metric.to(self.device)  # type: ignore[attr-defined]

    def _label_info_from_names(self, names: dict[int, str]) -> LabelInfo:
        """Build a ``LabelInfo`` with per-label groups for multi-label metrics."""
        if self._datamodule is not None:
            label_info = self._datamodule.label_info
            if all(len(group) == 1 for group in label_info.label_groups):
                return label_info
            return LabelInfo(
                label_names=label_info.label_names,
                label_groups=[[name] for name in label_info.label_names],
                label_ids=label_info.label_ids,
            )
        name_list = list(names.values())
        return LabelInfo(
            label_names=name_list,
            label_groups=[[name] for name in name_list],
            label_ids=[str(i) for i in range(len(name_list))],
        )

    def update_metrics(self, preds: torch.Tensor, batch: dict[str, Any]) -> None:
        """Accumulate sigmoid predictions and multi-hot targets."""
        target = batch["cls"].float()
        self.metric.update(preds=preds, target=target)

    def finalize_metrics(self) -> None:
        """No-op: metric is computed in ``get_stats``."""

    def gather_stats(self) -> None:
        """No-op: torchmetrics state is kept on the local rank."""

    def get_stats(self) -> dict[str, float]:
        """Compute and return multi-label accuracy and mAP."""
        results = self.metric.compute()
        accuracy = self._extract_scalar(results, "accuracy")
        mean_ap = self._extract_scalar(results, "mAP")
        return {"metrics/accuracy": accuracy, "metrics/mAP": mean_ap}

    @staticmethod
    def _extract_scalar(results: dict[str, Any], key: str) -> float:
        """Extract a scalar value from a possibly-nested metric result dict."""
        value = results.get(key)
        if isinstance(value, dict):
            value = value.get(key, value.get("accuracy"))
        if value is None:
            return 0.0
        if isinstance(value, torch.Tensor):
            return float(value.item())
        return float(value)

    def print_results(self) -> None:
        """Print multi-label validation metrics."""
        stats = self.get_stats()
        LOGGER.info(f"{'multi-label accuracy':>20}: {stats['metrics/accuracy']:.3g}")
        LOGGER.info(f"{'multi-label mAP':>20}: {stats['metrics/mAP']:.3g}")

    def _build_adapter_dataloader(self) -> DataLoader:
        """Build a multi-label DataLoader from the DataModule's val/test subset."""
        if self._datamodule is None:
            msg = "_build_adapter_dataloader requires a DataModule"
            raise TypeError(msg)
        test_key = self._datamodule.test_subset.subset_name
        val_key = self._datamodule.val_subset.subset_name
        subset = self._datamodule.subsets.get(test_key) or self._datamodule.subsets[val_key]
        adapter = UltralyticsDatasetAdapter(subset, task_kind="multilabel")
        return DataLoader(
            adapter,
            batch_size=self.args.batch,  # type: ignore[attr-defined]
            shuffle=False,
            collate_fn=multilabel_collate_fn,
            pin_memory=True,
        )
