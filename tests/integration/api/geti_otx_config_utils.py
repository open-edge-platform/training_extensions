# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

from omegaconf import OmegaConf

from otx.tools.converter import TEMPLATE_ID_DICT, ConfigConverter
from otx.types.task import OTXTaskType

if TYPE_CHECKING:
    from collections.abc import Iterator


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
    config.json's export_parameters item model.
    """

    export_format: ExportFormat
    precision: PrecisionType = PrecisionType.FP32
    with_xai: bool = False


@dataclass(frozen=True)
class OTXConfig:
    job_type: JobType
    model_template_id: str
    hyper_parameters: dict
    export_parameters: list[ExportParameter]
    optimization_type: OptimizationType | None
    sub_task_type: OTXTaskType

    def to_json_file(self, fpath: Path) -> None:
        with fpath.open("w") as fp:
            json.dump(
                {
                    "job_type": self.job_type,
                    "model_template_id": self.model_template_id,
                    "hyperparameters": self.hyper_parameters,
                    "export_parameters": [
                        {"type": param.export_format, "precision": param.precision, "with_xai": param.with_xai}
                        for param in self.export_parameters
                    ],
                    "optimization_type": "NONE" if self.optimization_type is None else self.optimization_type,
                    "sub_task_type": self.sub_task_type,
                },
                fp,
            )

    def to_otx_config(self, work_dir: Path) -> dict[str, dict]:
        fpath = work_dir / "tmp_config.json"
        self.to_json_file(fpath)

        with self.monkeypatch_cls_task_type(override_cls_task_type=self.sub_task_type):
            otx_config = ConfigConverter.convert(fpath)

        otx_config["data"]["input_size"] = tuple(otx_config["data"]["input_size"])  # cast to tuple
        otx_config["data"]["data_format"] = "arrow"
        otx_config["data"]["train_subset"]["subset_name"] = "TRAINING"
        otx_config["data"]["val_subset"]["subset_name"] = "VALIDATION"
        otx_config["data"]["test_subset"]["subset_name"] = "TESTING"

        return otx_config

    @staticmethod
    @contextmanager
    def monkeypatch_cls_task_type(override_cls_task_type: OTXTaskType | None = None) -> Iterator[None]:
        """Monkeypatch classification task type which is fixed as `MULTI_CLASS_CLS` in OTX side.

        This should be improved on the OTX side.

        Args:
            override_cls_task_type: Override classification task type if given. Otherwise, do nothing.

        Yields:
            None: Yields nothing.
        """
        if override_cls_task_type is None:
            yield
            return

        tmp_dict = {}
        for key, value in TEMPLATE_ID_DICT.items():
            if "multi_class_cls" in value:
                tmp_dict[key] = value

                new_value = deepcopy(value)
                model_name = Path(value["model_config_path"]).name
                parent_classification_path = Path("src/otx/recipe/classification/")
                new_value["model_config_path"] = (
                    parent_classification_path / override_cls_task_type.value.lower() / model_name
                )
                TEMPLATE_ID_DICT[key] = new_value

        yield

        # Revert
        for key, value in tmp_dict.items():
            TEMPLATE_ID_DICT[key] = value


def substitute_parameter_overrides(override_dict: dict, parameter_dict: dict):
    """Substitutes parameters form override_dict into parameter_dict.

    Recursively substitutes overridden parameter values specified in `override_dict` into the base set of
    hyper parameters passed in as `parameter_dict`

    Args:
        override_dict (Dict): dictionary containing the parameter overrides
        parameter_dict (Dict): dictionary that contains the base set of hyper parameters, in which the overridden
            values are substituted
    """
    for key, value in override_dict.items():
        if isinstance(value, dict) and not allows_dictionary_values(key):
            if key in parameter_dict:
                substitute_parameter_overrides(value, parameter_dict[key])
            else:
                msg = f"Unable to perform parameter override. Parameter or parameter group named {key}."
                raise ValueError(msg)
        elif allows_model_template_override(key):
            parameter_dict[key] = value
        else:
            msg = f"{key} is not a valid keyword for hyper parameter overrides"
            raise KeyError(msg)


def load_hyper_parameters(model_template_path: Path) -> tuple[str, dict]:
    """Load hyper parameters.

    Loads the actual hyper parameters defined in the file at `base_path`, and performs any overrides specified in
    the `parameter_overrides`.

    Args:
        model_template_path (Path): file path to the model template file in which the HyperParameters live.

    Returns:
        tuple[str, dict]: A tuple containing the model template ID and the loaded hyper parameters.
    """

    model_template = OmegaConf.load(model_template_path)
    model_template = OmegaConf.to_container(model_template)

    base_hyper_parameter_path = model_template_path.parent / model_template["hyper_parameters"]["base_path"]

    config_dict = OmegaConf.load(base_hyper_parameter_path)
    data = OmegaConf.to_container(config_dict)
    if model_template.get("hyper_parameters", {}).get("parameter_overrides"):

        def add_value_key(d: dict) -> None:
            for k, v in list(d.items()):  # Use list to avoid modifying during iteration
                if isinstance(v, dict):
                    add_value_key(v)
                if k == "default_value":
                    d["value"] = v

        add_value_key(model_template["hyper_parameters"]["parameter_overrides"])

        substitute_parameter_overrides(
            model_template["hyper_parameters"]["parameter_overrides"],
            data,
        )
    return (model_template["model_template_id"], data)
