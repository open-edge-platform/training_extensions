# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Lightweight builder functions for job runnables.

These module-level functions are used as targets for `functools.partial` inside
`RunnableFactory`.  Because the multiprocessing "spawn" start method pickles
these callables, the module that contains them will be imported in every worker
process during unpickling.  By placing them here — rather than in
`app.lifecycle` — we avoid pulling in FastAPI, aiortc, WebRTC and other
web-stack dependencies that are irrelevant to job execution.

Each builder performs its heavy imports lazily (inside the function body) so
that the ML stack is loaded only when the function is actually called.
"""

from typing import Any

from app.core.run import Runnable


def build_trainer(**deps: Any) -> Runnable:
    """Lazily import the training stack and build a trainer runnable in the worker process."""
    from app.execution import GetiTuneTrainer, TrainingDependencies

    return GetiTuneTrainer(training_deps=TrainingDependencies(**deps))


def build_quantizer(**deps: Any) -> Runnable:
    """Lazily import the quantization stack and build a quantizer runnable in the worker process."""
    from app.execution import GetiTuneQuantizer, QuantizationDependencies

    return GetiTuneQuantizer(quantization_deps=QuantizationDependencies(**deps))


def build_export_dataset(**kwargs: Any) -> Runnable:
    """Lazily import and build a dataset-export runnable in the worker process."""
    from app.execution import ExportDataset

    return ExportDataset(**kwargs)


def build_prepare_dataset(**kwargs: Any) -> Runnable:
    """Lazily import and build a prepare-dataset runnable in the worker process."""
    from app.execution import PrepareDataset

    return PrepareDataset(**kwargs)


def build_import_to_project(**kwargs: Any) -> Runnable:
    """Lazily import and build an import-to-project runnable in the worker process."""
    from app.execution import ImportDatasetToProject

    return ImportDatasetToProject(**kwargs)


def build_import_as_new_project(**kwargs: Any) -> Runnable:
    """Lazily import and build an import-as-new-project runnable in the worker process."""
    from app.execution.dataset_import.import_as_new_project import ImportDatasetAsNewProject

    return ImportDatasetAsNewProject(**kwargs)
