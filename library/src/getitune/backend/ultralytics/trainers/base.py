# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Base trainer mixin with shared logic for the getitune DataModule bridge."""

from __future__ import annotations

import logging
import multiprocessing
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from ultralytics.data.build import InfiniteDataLoader, seed_worker

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

from getitune.backend.ultralytics.data.adapter import UltralyticsDatasetAdapter
from getitune.backend.ultralytics.data.collate import collate_fn

if TYPE_CHECKING:
    from getitune.data.module import DataModule

logger = logging.getLogger(__name__)

_MP_CONTEXT = multiprocessing.get_context("spawn")


class GetiTuneDataBridgeMixin:
    """Shared DataModule bridge logic for Ultralytics trainers."""

    _datamodule: DataModule | None = None
    _use_getitune_data: bool = False
    _include_masks: bool = False
    _progress_fn: Any = None
    _progress_min: float = 0.0
    _progress_max: float = 100.0

    def get_dataset(self) -> dict[str, Any]:
        """Build data config dict from DataModule or fall back to YAML."""
        if not self._use_getitune_data:
            return super().get_dataset()  # type: ignore[misc]

        if self._datamodule is None:
            msg = "DataModule is required when _use_getitune_data=True"
            raise RuntimeError(msg)

        li = self._datamodule.label_info  # type: ignore[union-attr]
        names = dict(enumerate(li.label_names))
        return {
            "train": "datamodule://train",
            "val": "datamodule://val",
            "nc": li.num_classes,
            "names": names,
            "channels": 3,
        }

    def build_dataset(self, img_path: str, mode: str = "train", batch: int | None = None) -> UltralyticsDatasetAdapter:
        """Return adapter wrapping the appropriate DataModule subset.

        Args:
            img_path: Ignored when DataModule is set.
            mode: ``"train"`` or ``"val"``.
            batch: Batch size (unused by adapter).
        """
        if not self._use_getitune_data:
            return super().build_dataset(img_path, mode, batch)  # type: ignore[misc]

        if self._datamodule is None:
            msg = "DataModule is required when _use_getitune_data=True"
            raise RuntimeError(msg)

        subset_key = "train" if mode == "train" else "val"
        vision_dataset = self._datamodule.subsets[subset_key]  # type: ignore[union-attr]
        return UltralyticsDatasetAdapter(vision_dataset, include_masks=self._include_masks)

    def get_dataloader(
        self,
        dataset_path: str,
        batch_size: int = 16,
        rank: int = 0,
        mode: str = "train",
    ) -> DataLoader:
        """Build a DataLoader from the adapter dataset."""
        if not self._use_getitune_data:
            return super().get_dataloader(dataset_path, batch_size, rank, mode)  # type: ignore[misc]

        dataset = self.build_dataset(dataset_path, mode, batch_size)
        shuffle = mode == "train"
        nw = self._effective_num_workers(len(dataset), mode)
        return InfiniteDataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=nw,
            prefetch_factor=4 if nw > 0 else None,
            collate_fn=collate_fn,
            pin_memory=True,
            drop_last=False,
            multiprocessing_context=_MP_CONTEXT if nw > 0 else None,
            persistent_workers=nw > 0,
            worker_init_fn=seed_worker,
        )

    def _effective_num_workers(self, dataset_size: int, mode: str) -> int:
        """Determine optimal worker count based on total training iterations.

        With spawn multiprocessing context (required for Datumaro compatibility),
        worker spawn overhead is significant (~15-20s). However, workers provide
        ~1.5x throughput improvement once warmed up. This method balances spawn
        overhead against throughput gains based on expected training duration.

        Decision is based on total iterations (epochs * batches):
        - <300 iterations: workers=0 (spawn overhead dominates)
        - 300-1000 iterations: workers=2 (moderate benefit)
        - >1000 iterations: use configured workers (spawn overhead amortized)

        For validation, always use workers=0 since validation is typically
        short and spawn overhead would dominate.

        Args:
            dataset_size: Number of samples in the dataset.
            mode: "train" or "val".

        Returns:
            Optimal number of workers.
        """
        configured_workers: int = self.args.workers  # type: ignore[attr-defined]

        # Validation: always use workers=0 (short duration, spawn overhead dominates)
        if mode != "train":
            if configured_workers > 0:
                logger.info("Validation mode: using workers=0 to avoid spawn overhead")
            return 0

        # Estimate total training iterations
        batch_size: int = self.args.batch  # type: ignore[attr-defined]
        epochs: int = self.args.epochs  # type: ignore[attr-defined]
        batches_per_epoch = (dataset_size + batch_size - 1) // batch_size
        total_iterations = epochs * batches_per_epoch

        # Decision thresholds based on empirical benchmarks:
        # - Spawn overhead: ~19s for 4 workers
        # - Throughput: workers=0 ~5.7 it/s, workers=4 ~8.8 it/s
        # - Break-even: ~300 iterations (19s / (1/5.7 - 1/8.8))
        if total_iterations < 300:
            if configured_workers > 0:
                logger.info(f"Short training ({total_iterations} iterations): using workers=0 to avoid spawn overhead")
            return 0

        if total_iterations < 1000:
            effective = min(configured_workers, 2)
            if effective != configured_workers:
                logger.info(
                    f"Medium training ({total_iterations} iterations): using workers={effective} "
                    f"instead of {configured_workers}"
                )
            return effective

        # Long training: use configured workers (spawn overhead amortized)
        if configured_workers > 0:
            logger.info(
                f"Long training ({total_iterations} iterations): using workers={configured_workers} "
                f"(spawn overhead will be amortized)"
            )
        return configured_workers

    def _setup_train(self) -> None:
        """Run parent setup, then fix warmup for small datasets.

        Ultralytics enforces a minimum of 100 warmup iterations regardless
        of dataset size (``max(round(warmup_epochs * nb), 100)``).  For
        small datasets this can make warmup consume an unreasonable fraction
        of total training (e.g. 33+ epochs of warmup with only 3 batches/epoch).

        When the natural warmup (``warmup_epochs * nb``) is below 100 we
        disable the built-in warmup entirely and register a custom
        ``on_train_batch_start`` callback that applies the same LR /
        momentum ramp but respects the natural iteration count.
        """
        super()._setup_train()  # type: ignore[misc]

        if not self._use_getitune_data:
            return

        self._register_progress_callback()

        if self.args.warmup_epochs <= 0:  # type: ignore[attr-defined]
            return

        nb = len(self.train_loader)  # type: ignore[attr-defined]
        natural_nw = round(self.args.warmup_epochs * nb)  # type: ignore[attr-defined]
        if natural_nw < 100:
            logger.info(
                f"Bypassing Ultralytics 100-iteration warmup minimum. "
                f"With {nb} batches/epoch, using natural warmup of "
                f"{natural_nw} iterations ({self.args.warmup_epochs} epochs)."
            )

            warmup_bias_lr = self.args.warmup_bias_lr  # type: ignore[attr-defined]
            warmup_momentum = self.args.warmup_momentum  # type: ignore[attr-defined]
            nbs = self.args.nbs  # type: ignore[attr-defined]
            batch_size = self.batch_size  # type: ignore[attr-defined]
            self.args.warmup_epochs = 0  # type: ignore[attr-defined]

            counter = {"ni": 0}

            def _warmup_callback(trainer: Any) -> None:  # noqa: ANN401
                ni = counter["ni"]
                if ni <= natural_nw:
                    xi = [0, natural_nw]
                    trainer.accumulate = max(1, int(np.interp(ni, xi, [1, nbs / batch_size]).round()))
                    for pg in trainer.optimizer.param_groups:
                        pg["lr"] = np.interp(
                            ni,
                            xi,
                            [
                                warmup_bias_lr if pg.get("param_group") == "bias" else 0.0,
                                pg["initial_lr"] * trainer.lf(trainer.epoch),
                            ],
                        )
                        if "momentum" in pg:
                            pg["momentum"] = np.interp(ni, xi, [warmup_momentum, trainer.args.momentum])
                counter["ni"] += 1

            self.add_callback("on_train_batch_start", _warmup_callback)  # type: ignore[attr-defined]

    def _register_progress_callback(self) -> None:
        """Register a progress-reporting callback for the training loop.

        Bridges the application's progress callable (passed via
        ``_progress_fn``) into the Ultralytics callback system.
        Computes progress as a linear interpolation between ``_progress_min``
        and ``_progress_max`` based on ``current_step / total_steps``.
        """
        if self._progress_fn is None:
            return

        progress_fn = self._progress_fn
        min_p = self._progress_min
        max_p = self._progress_max
        nb = len(self.train_loader)  # type: ignore[attr-defined]
        total_steps = max(1, self.epochs * nb)  # type: ignore[attr-defined]
        step_counter = {"step": 0}

        def _progress_callback(_trainer: Any) -> None:  # noqa: ANN401
            step_counter["step"] += 1
            ratio = step_counter["step"] / total_steps
            progress = min_p + ratio * (max_p - min_p)
            progress_fn(progress)

        self.add_callback("on_train_batch_end", _progress_callback)  # type: ignore[attr-defined]
        logger.info(f"Registered progress callback: {total_steps} total steps, range [{min_p}, {max_p}]")

    def _clear_memory(self, threshold: float | None = None) -> None:
        """Lightweight memory clearing that skips ``gc.collect()``.

        The upstream implementation calls ``gc.collect()`` every epoch which
        forces a full Python garbage-collection cycle.  This is expensive
        with large object graphs (spawn-based DataLoader workers, Datumaro
        caches, etc.) and unnecessary during tight training loops where
        memory pressure is manageable via CUDA cache management alone.

        Falls back to the upstream implementation when running without the
        DataModule bridge (native YOLO data path).
        """
        if not self._use_getitune_data:
            return super()._clear_memory(threshold)  # type: ignore[misc]

        if self.device.type == "cpu":  # type: ignore[attr-defined]
            return None

        if threshold is not None and self._get_memory(fraction=True) <= threshold:  # type: ignore[attr-defined]
            return None

        if self.device.type == "mps":  # type: ignore[attr-defined]
            torch.mps.empty_cache()
        elif self.device.type == "xpu":  # type: ignore[attr-defined]
            torch.xpu.empty_cache()
        else:
            torch.cuda.empty_cache()
        return None

    def set_model_attributes(self) -> None:
        """Set model attributes; disable Ultralytics augmentations when using DataModule."""
        super().set_model_attributes()  # type: ignore[misc]
        if self._use_getitune_data:
            self._disable_ultralytics_augmentations()

    def set_class_weights(self) -> None:
        """Skip class-weight computation (adapter has no ``labels`` attr)."""
        if self._use_getitune_data:
            return
        super().set_class_weights()  # type: ignore[misc]

    def plot_training_labels(self) -> None:
        """Skip label plotting (adapter has no ``labels`` attr)."""
        if self._use_getitune_data:
            return
        super().plot_training_labels()  # type: ignore[misc]

    def auto_batch(self) -> int:
        """Skip auto-batch when using DataModule."""
        if self._use_getitune_data:
            return self.batch_size  # type: ignore[attr-defined]
        return super().auto_batch()  # type: ignore[misc]

    def _disable_ultralytics_augmentations(self) -> None:
        """Zero out Ultralytics augmentation hyperparams.

        All augmentations are handled by the DataModule pipeline, so we
        disable all upstream augmentation parameters to prevent double
        augmentation.
        """
        for attr in (
            "mosaic",
            "mixup",
            "cutmix",
            "copy_paste",
            "hsv_h",
            "hsv_s",
            "hsv_v",
            "flipud",
            "fliplr",
            "degrees",
            "translate",
            "scale",
            "shear",
            "perspective",
            "close_mosaic",
        ):
            if hasattr(self.args, attr):  # type: ignore[attr-defined]
                setattr(self.args, attr, 0.0 if attr != "close_mosaic" else 0)  # type: ignore[attr-defined]
