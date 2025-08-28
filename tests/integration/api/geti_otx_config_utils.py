# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path

import yaml

from otx.tools.converter import GetiConfigConverter

BASE_MODEL_FILENAME = "model_fp32_xai.pth"

DEFAULT_VALUE = "default_value"
VALUE = "value"
MIN_VALUE = "min_value"
MAX_VALUE = "max_value"
STEP_SIZE = "step_size"
DESCRIPTION = "description"
HEADER = "header"
WARNING = "warning"
EDITABLE = "editable"
VISIBLE_IN_UI = "visible_in_ui"
AFFECTS_OUTCOME_OF = "affects_outcome_of"
UI_RULES = "ui_rules"
TYPE = "type"
OPTIONS = "options"
ENUM_NAME = "enum_name"


def allows_model_template_override(keyword: str) -> bool:
    """Returns True if the metadata element described by `keyword` can be overridden in a model template file.

    Args:
        keyword (str): Name of the metadata key to check.
    Returns:
        bool: True if the key can be overridden, False otherwise.
    """
    overrideable_keys = [
        DEFAULT_VALUE,
        VALUE,
        MIN_VALUE,
        MAX_VALUE,
        DESCRIPTION,
        HEADER,
        EDITABLE,
        WARNING,
        VISIBLE_IN_UI,
        OPTIONS,
        ENUM_NAME,
        UI_RULES,
        AFFECTS_OUTCOME_OF,
    ]
    return keyword in overrideable_keys


def allows_dictionary_values(keyword: str) -> bool:
    """Returns True if the metadata element described by `keyword` allows having a dictionary as its value.

    Args:
        keyword (str): Name of the metadata key to check.

    Returns:
        bool: True if the key allows dictionary values, False otherwise.
    """
    keys_allowing_dictionary_values = [OPTIONS, UI_RULES]
    return keyword in keys_allowing_dictionary_values


class JobType(str, Enum):
    TRAIN = "train"
    OPTIMIZE_NNCF = "optimize_nncf"
    OPTIMIZE_POT = "optimize_pot"


class OptimizationType(str, Enum):
    NNCF = "NNCF"
    POT = "POT"


class ExportFormat(str, Enum):
    BASE_FRAMEWORK = "BASE_FRAMEWORK"
    OPENVINO = "OPENVINO"
    ONNX = "ONNX"


class PrecisionType(str, Enum):
    FP32 = "FP32"
    FP16 = "FP16"
    INT8 = "INT8"


@dataclass
class ExportParameter:
    """
    config.yaml's export_parameters item model.
    """

    export_format: ExportFormat
    precision: PrecisionType = PrecisionType.FP32
    with_xai: bool = False


def str2bool(value: str | bool) -> bool:
    """Convert given value to boolean."""
    if isinstance(value, bool):
        return value

    if isinstance(value, str):
        if value.lower() == "true":
            return True
        if value.lower() == "false":
            return False
        raise ValueError(value)

    raise TypeError(value)


@dataclass
class OTXConfig:
    job_type: JobType
    model_manifest_id: str
    hyper_parameters: dict | None
    export_parameters: list[ExportParameter]
    optimization_type: OptimizationType | None
    sub_task_type: str | None = None

    @classmethod
    def from_yaml_file(cls, config_file_path: Path) -> OTXConfig:
        with Path(config_file_path).open() as fp:
            config: dict = yaml.safe_load(fp)

        return OTXConfig(
            job_type=JobType(config["job_type"]),
            model_manifest_id=config["model_manifest_id"],
            hyper_parameters=config.get("hyperparameters"),
            export_parameters=[
                ExportParameter(
                    export_format=ExportFormat(cfg["format"].upper()),
                    precision=PrecisionType(cfg["precision"].upper()),
                    with_xai=str2bool(cfg["with_xai"]),
                )
                for cfg in config.get("export_models", [])
            ],
            optimization_type=OptimizationType.POT if config["job_type"] == "optimize_pot" else None,
            sub_task_type=config.get("sub_task_type"),
        )

    def to_otx_config(self) -> dict[str, dict]:
        """Convert OTXConfig to OTX2 config format."""
        otx2_config = GetiConfigConverter.convert(asdict(self))

        otx2_config["data"]["data_format"] = "arrow"
        otx2_config["data"]["train_subset"]["subset_name"] = "TRAINING"
        otx2_config["data"]["val_subset"]["subset_name"] = "VALIDATION"
        otx2_config["data"]["test_subset"]["subset_name"] = "TESTING"

        return otx2_config
