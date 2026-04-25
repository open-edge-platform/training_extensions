# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Detection validator that skips /255 normalization."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from torch.utils.data import DataLoader
from ultralytics.models.yolo.detect import DetectionValidator as _UltralyticsDetectionValidator
from ultralytics.nn.autobackend import AutoBackend
from ultralytics.utils import RANK, TQDM, callbacks
from ultralytics.utils.checks import check_imgsz
from ultralytics.utils.torch_utils import Profile, select_device, unwrap_model

from getitune.backend.ultralytics.data.adapter import UltralyticsDatasetAdapter
from getitune.backend.ultralytics.data.collate import ultralytics_collate_fn

if TYPE_CHECKING:
    from getitune.data.module import DataModule


class DetectionValidator(_UltralyticsDetectionValidator):
    """Detection validator for the getitune data bridge.

    Skips ``/255`` normalization (images are already float32 [0,1]).
    When ``_datamodule`` is set and called without a trainer, runs
    standalone validation bypassing Ultralytics' YAML data parsing.
    """

    _datamodule: DataModule | None = None

    def __call__(self, trainer: object = None, model: object = None) -> dict:
        """Dispatch to upstream (training) or standalone DataModule validation."""
        if trainer is not None or self._datamodule is None:
            return super().__call__(trainer=trainer, model=model)
        return self._run_standalone_eval(model)

    def preprocess(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Move tensors to device; skip ``/255``."""
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(self.device, non_blocking=True)
        if self.args.half:
            batch["img"] = batch["img"].half()
        return batch

    # ------------------------------------------------------------------
    # Standalone DataModule evaluation
    # ------------------------------------------------------------------

    def _run_standalone_eval(self, model: object) -> dict:
        """Run validation using DataModule without YAML data config.

        Replicates the essential non-training setup from
        ``BaseValidator.__call__`` but skips ``check_det_dataset()``
        which requires a YAML file on disk.
        """
        self.training = False
        callbacks.add_integration_callbacks(self)

        dev = select_device(self.args.device) if RANK == -1 else torch.device("cuda", RANK)
        model = AutoBackend(
            model=model or self.args.model,  # pyrefly: ignore[bad-argument-type]
            device=dev,  # pyrefly: ignore[bad-argument-type]
            fp16=self.args.half,
        )
        self.device = model.device
        self.args.half = model.fp16
        imgsz = check_imgsz(self.args.imgsz, stride=model.stride)

        assert self._datamodule is not None  # guaranteed by caller  # noqa: S101
        li = self._datamodule.label_info
        self.data = {
            "nc": li.num_classes,
            "names": dict(enumerate(li.label_names)),
            "channels": 3,
            "val": "",
        }

        self.dataloader = self.dataloader or self._build_adapter_dataloader()

        model.eval()
        model.warmup(imgsz=(1, self.data["channels"], imgsz, imgsz))  # pyrefly: ignore[bad-argument-type]

        # --- validation loop (mirrors upstream BaseValidator.__call__) ---
        self.run_callbacks("on_val_start")
        dt = tuple(Profile(device=self.device) for _ in range(4))
        bar = TQDM(self.dataloader, desc=self.get_desc(), total=len(self.dataloader))
        self.init_metrics(unwrap_model(model))
        self.jdict = []

        for batch_i, raw_batch in enumerate(bar):
            self.run_callbacks("on_val_batch_start")
            self.batch_i = batch_i
            with dt[0]:
                batch = self.preprocess(raw_batch)
            with dt[1]:
                preds = model(batch["img"])
            with dt[2]:
                pass  # no loss in standalone mode
            with dt[3]:
                preds = self.postprocess(preds)
            self.update_metrics(preds, batch)
            if self.args.plots and batch_i < 3 and RANK in {-1, 0}:
                self.plot_val_samples(batch, batch_i)
                self.plot_predictions(batch, preds, batch_i)
            self.run_callbacks("on_val_batch_end")

        stats: dict = {}
        self.gather_stats()
        if RANK in {-1, 0}:
            stats = self.get_stats()
            ds_len = len(self.dataloader.dataset)  # pyrefly: ignore[bad-argument-type]
            self.speed = dict(zip(self.speed.keys(), (x.t / ds_len * 1e3 for x in dt)))
            self.finalize_metrics()
            self.print_results()
            self.run_callbacks("on_val_end")
        return stats

    def _build_adapter_dataloader(self) -> DataLoader:
        """Build a DataLoader from the DataModule's val/test subset."""
        assert self._datamodule is not None  # guaranteed by caller  # noqa: S101
        subset = self._datamodule.subsets.get("test") or self._datamodule.subsets["val"]
        adapter = UltralyticsDatasetAdapter(subset, include_masks=False)
        return DataLoader(
            adapter,
            batch_size=self.args.batch,
            shuffle=False,
            collate_fn=ultralytics_collate_fn,
            pin_memory=True,
        )
