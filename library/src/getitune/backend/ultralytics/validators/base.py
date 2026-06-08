# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Base validator mixin with shared standalone evaluation logic."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from torch.utils.data import DataLoader
from ultralytics.nn.autobackend import AutoBackend
from ultralytics.utils import RANK, TQDM, callbacks
from ultralytics.utils.checks import check_imgsz
from ultralytics.utils.ops import Profile
from ultralytics.utils.torch_utils import select_device, unwrap_model

from getitune.backend.ultralytics.data.adapter import UltralyticsDatasetAdapter
from getitune.backend.ultralytics.data.collate import collate_fn

if TYPE_CHECKING:
    from ultralytics.engine.trainer import BaseTrainer

    from getitune.data.module import DataModule


class GetiTuneValidatorMixin:
    """Mixin providing shared standalone evaluation logic for Ultralytics validators.

    When ``_datamodule`` is set and called without a trainer, runs
    standalone validation bypassing Ultralytics' YAML data parsing.

    Subclasses must set :attr:`_include_masks` to control whether the adapter
    includes instance masks.
    """

    _datamodule: DataModule | None = None
    _include_masks: bool = False

    def __call__(self, trainer: BaseTrainer | None = None, model: torch.nn.Module | None = None) -> dict:
        """Dispatch to upstream (training) or standalone DataModule validation."""
        if trainer is not None or self._datamodule is None:
            return super().__call__(trainer=trainer, model=model)  # type: ignore[misc]
        return self._run_standalone_eval(model)

    def preprocess(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Move tensors to device."""
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(self.device, non_blocking=True)  # type: ignore[attr-defined]
        if self.args.half:  # type: ignore[attr-defined]
            batch["img"] = batch["img"].half()
        if "masks" in batch and isinstance(batch["masks"], torch.Tensor):
            batch["masks"] = batch["masks"].float()
        return batch

    def _run_standalone_eval(self, model: torch.nn.Module | None) -> dict:
        """Run validation using DataModule without YAML data config.

        Replicates the essential non-training setup from
        ``BaseValidator.__call__`` but skips ``check_det_dataset()``
        which requires a YAML file on disk.
        """
        self.training = False  # type: ignore[attr-defined]
        callbacks.add_integration_callbacks(self)

        raw_dev = self.args.device  # type: ignore[attr-defined]
        if isinstance(raw_dev, torch.device) and raw_dev.type == "xpu":
            dev = raw_dev
        elif isinstance(raw_dev, str) and "xpu" in raw_dev:
            dev = torch.device(raw_dev)
        else:
            dev = select_device(raw_dev) if RANK == -1 else torch.device("cuda", RANK)
        model = AutoBackend(
            model=model or self.args.model,  # pyrefly: ignore[bad-argument-type]  # type: ignore[attr-defined]
            device=dev,  # pyrefly: ignore[bad-argument-type]
            fp16=self.args.half,  # type: ignore[attr-defined]
        )
        self.device = model.device  # type: ignore[attr-defined]
        self.args.half = model.fp16  # type: ignore[attr-defined]
        imgsz = check_imgsz(self.args.imgsz, stride=model.stride)  # type: ignore[attr-defined]

        if self._datamodule is None:
            msg = "Standalone evaluation requires a DataModule"
            raise TypeError(msg)
        li = self._datamodule.label_info
        self.data = {  # type: ignore[attr-defined]
            "nc": li.num_classes,
            "names": dict(enumerate(li.label_names)),
            "channels": 3,
            "val": "",
        }

        self.dataloader = self.dataloader or self._build_adapter_dataloader()  # type: ignore[attr-defined]

        model.eval()
        if isinstance(imgsz, int):
            warmup_imgsz = (1, int(self.data["channels"]), imgsz, imgsz)
        else:
            warmup_imgsz = (1, int(self.data["channels"]), int(imgsz[0]), int(imgsz[1]))
        model.warmup(imgsz=warmup_imgsz)  # type: ignore[attr-defined]

        self.run_callbacks("on_val_start")  # type: ignore[attr-defined]
        dt = tuple(Profile(device=self.device) for _ in range(4))  # type: ignore[attr-defined]
        bar = TQDM(self.dataloader, desc=self.get_desc(), total=len(self.dataloader))  # type: ignore[attr-defined]
        self.init_metrics(unwrap_model(model))  # type: ignore[attr-defined]
        self.jdict = []  # type: ignore[attr-defined]

        for batch_i, raw_batch in enumerate(bar):
            self.run_callbacks("on_val_batch_start")  # type: ignore[attr-defined]
            self.batch_i = batch_i  # type: ignore[attr-defined]
            with dt[0]:
                batch = self.preprocess(raw_batch)
            with dt[1]:
                preds = model(batch["img"])
            with dt[2]:
                pass  # no loss in standalone mode
            with dt[3]:
                preds = self.postprocess(preds)  # type: ignore[attr-defined]
            self.update_metrics(preds, batch)  # type: ignore[attr-defined]
            if self.args.plots and batch_i < 3 and RANK in {-1, 0}:  # type: ignore[attr-defined]
                self.plot_val_samples(batch, batch_i)  # type: ignore[attr-defined]
                self.plot_predictions(batch, preds, batch_i)  # type: ignore[attr-defined]
            self.run_callbacks("on_val_batch_end")  # type: ignore[attr-defined]

        stats: dict = {}
        self.gather_stats()  # type: ignore[attr-defined]
        if RANK in {-1, 0}:
            stats = self.get_stats()  # type: ignore[attr-defined]
            ds_len = len(self.dataloader.dataset)  # pyrefly: ignore[bad-argument-type]  # type: ignore[attr-defined]
            self.speed = dict(zip(self.speed.keys(), (x.t / ds_len * 1e3 for x in dt)))  # type: ignore[attr-defined]
            self.finalize_metrics()  # type: ignore[attr-defined]
            self.print_results()  # type: ignore[attr-defined]
            self.run_callbacks("on_val_end")  # type: ignore[attr-defined]
        return stats

    def _build_adapter_dataloader(self) -> DataLoader:
        """Build a DataLoader from the DataModule's val/test subset."""
        if self._datamodule is None:
            msg = "_build_adapter_dataloader requires a DataModule"
            raise TypeError(msg)
        test_key = self._datamodule.test_subset.subset_name
        val_key = self._datamodule.val_subset.subset_name
        subset = self._datamodule.subsets.get(test_key) or self._datamodule.subsets[val_key]
        adapter = UltralyticsDatasetAdapter(subset, include_masks=self._include_masks)
        return DataLoader(
            adapter,
            batch_size=self.args.batch,  # type: ignore[attr-defined]
            shuffle=False,
            collate_fn=collate_fn,
            pin_memory=True,
        )
