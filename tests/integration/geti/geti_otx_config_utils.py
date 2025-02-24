#

from __future__ import annotations

import json
import os
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

import metadata_keys
from omegaconf import OmegaConf
from sc_sdk.entities.model_template import parse_model_template

from otx.core.types.export import OTXExportFormatType
from otx.core.types.precision import OTXPrecisionType
from otx.core.types.task import OTXTaskType
from otx.tools.converter import TEMPLATE_ID_DICT, ConfigConverter

if TYPE_CHECKING:
    from collections.abc import Iterator


BASE_MODEL_FILENAME = "model_fp32_xai.pth"


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

    def to_mlflow_artifact_fnames(self) -> list[str]:
        fname = "model_"
        precision_name = (
            self.precision.name.lower() + "-pot"
            if self.precision == PrecisionType.INT8
            else self.precision.name.lower()
        )
        fname += precision_name + "_"
        if self.with_xai:
            fname += "xai"
        else:
            fname += "non-xai"

        if self.export_format == ExportFormat.OPENVINO:
            return [
                f"{fname}.bin",
                f"{fname}.xml",
            ]
        if self.export_format == ExportFormat.ONNX:
            return [
                f"{fname}.onnx",
            ]
        if self.export_format == ExportFormat.BASE_FRAMEWORK:
            return [f"{fname}.pth"]
        raise ValueError

    def to_exportable_code_artifact_fname(self) -> str:
        fname = "exportable-code_"
        precision_name = (
            self.precision.name.lower() + "-pot"
            if self.precision == PrecisionType.INT8
            else self.precision.name.lower()
        )
        fname += precision_name + "_"
        if self.with_xai:
            fname += "xai"
        else:
            fname += "non-xai"

        return fname + ".whl"

    def to_otx2_export_format(self) -> OTXExportFormatType:
        if self.export_format == ExportFormat.OPENVINO:
            return OTXExportFormatType.OPENVINO
        if self.export_format == ExportFormat.ONNX:
            return OTXExportFormatType.ONNX

        raise ValueError(self.export_format)

    def to_otx2_precision(self) -> OTXPrecisionType:
        if self.precision == PrecisionType.FP32:
            return OTXPrecisionType.FP32
        if self.precision == PrecisionType.FP16:
            return OTXPrecisionType.FP16

        raise ValueError(self.precision)


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


@dataclass(frozen=True)
class OTXConfig:
    job_type: JobType
    model_template_id: str
    hyper_parameters: dict
    export_parameters: list[ExportParameter]
    optimization_type: OptimizationType | None
    sub_task_type: OTXTaskType | None

    @classmethod
    def from_json_file(cls, config_file_path: Path) -> OTXConfig:
        with open(config_file_path) as fp:
            config: dict = json.load(fp)

        opt_type_name: str | None = config.get("optimization_type")

        if opt_type_name.upper() == "NNCF":
            optimization_type = OptimizationType.NNCF
        elif opt_type_name.upper() == "POT":
            optimization_type = OptimizationType.POT
        else:
            optimization_type = None

        sub_task_type = config.get("sub_task_type")

        if sub_task_type is not None:
            sub_task_type = OTXTaskType(sub_task_type)

        return OTXConfig(
            job_type=JobType(config["job_type"]),
            model_template_id=config["model_template_id"],
            hyper_parameters=config["hyperparameters"],
            export_parameters=[
                ExportParameter(
                    export_format=ExportFormat(cfg["type"].upper()),
                    precision=PrecisionType(cfg["precision"].upper()),
                    with_xai=str2bool(cfg["with_xai"]),
                )
                for cfg in config.get("export_parameters", [])
            ],
            optimization_type=optimization_type,
            sub_task_type=sub_task_type,
        )

    def to_json_file(self, fpath: Path) -> None:
        with fpath.open("w") as fp:
            json.dump(
                {
                    "model_template_id": self.model_template_id,
                    "hyperparameters": self.hyper_parameters,
                },
                fp,
            )

    def to_otx2_config(self, work_dir: Path) -> dict[str, dict]:
        fpath = work_dir / "tmp_config.json"
        self.to_json_file(fpath)

        with self.monkeypatch_cls_task_type(override_cls_task_type=self.sub_task_type):
            otx2_config = ConfigConverter.convert(fpath)

        otx2_config["data"]["data_format"] = "arrow"
        otx2_config["data"]["train_subset"]["subset_name"] = "TRAINING"
        otx2_config["data"]["val_subset"]["subset_name"] = "VALIDATION"
        otx2_config["data"]["test_subset"]["subset_name"] = "TESTING"

        return otx2_config

    @staticmethod
    @contextmanager
    def monkeypatch_cls_task_type(override_cls_task_type: OTXTaskType | None = None) -> Iterator[None]:
        """Monkeypath classification task type which is fixed as `MULTI_CLASS_CLS` in OTX side.

        This should be improved on the OTX side.

        :param override_cls_task_type: Override classification task type if given. Otherwise, do nothing.
        """
        if override_cls_task_type is None:
            yield
            return

        tmp_dict = {}
        for key, value in TEMPLATE_ID_DICT.items():
            if value["task"] == OTXTaskType.MULTI_CLASS_CLS:
                tmp_dict[key] = value

                new_value = deepcopy(value)
                new_value["task"] = override_cls_task_type
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
        if key in metadata_keys.deprecated_keys():
            continue
        if isinstance(value, dict) and not metadata_keys.allows_dictionary_values(key):
            if key in parameter_dict:
                substitute_parameter_overrides(value, parameter_dict[key])
            else:
                raise ValueError(f"Unable to perform parameter override. Parameter or parameter group named {key}.")
        elif metadata_keys.allows_model_template_override(key):
            parameter_dict[key] = value
        else:
            raise KeyError(f"{key} is not a valid keyword for hyper parameter overrides")


def load_hyper_parameters(model_template_path: dict) -> dict:
    """Load hyper parameters.

    Loads the actual hyper parameters defined in the file at `base_path`, and performs any overrides specified in
    the `parameter_overrides`.

    Args:
        model_template_path (str): file path to the model template file in which the HyperParameters live.
    """

    model_template = OmegaConf.load(model_template_path)
    model_template = OmegaConf.to_container(model_template)

    model_template_dir = os.path.dirname(model_template_path)
    base_hyper_parameter_path = os.path.join(
        model_template_dir,
        model_template["hyper_parameters"]["base_path"],
    )

    config_dict = OmegaConf.load(base_hyper_parameter_path)
    data = OmegaConf.to_container(config_dict)
    substitute_parameter_overrides(
        model_template["hyper_parameters"]["parameter_overrides"],
        data,
    )
    return data


if __name__ == "__main__":
    model_template_path = "src/otx/tools/templates/detection/detection/mobilenetv2_atss/template.yaml"
    geti_model_template = parse_model_template(model_template_path)

    model_template = OmegaConf.load(model_template_path)
    hyper_parameters = load_hyper_parameters(model_template_path)

    otx_config = OTXConfig(
        job_type=JobType.TRAIN,
        model_template_id=model_template.model_template_id,
        hyper_parameters=hyper_parameters,
        export_parameters=[
            ExportParameter(
                export_format=ExportFormat.OPENVINO,
                precision=PrecisionType.FP32,
                with_xai=True,
            ),
            ExportParameter(
                export_format=ExportFormat.OPENVINO,
                precision=PrecisionType.FP32,
            ),
            ExportParameter(
                export_format=ExportFormat.OPENVINO,
                precision=PrecisionType.FP16,
            ),
            ExportParameter(
                export_format=ExportFormat.ONNX,
                precision=PrecisionType.FP32,
            ),
        ],
        optimization_type=None,
        sub_task_type=None,
    )

    # otx_config = OTXConfig(
    #     job_type=JobType.TRAIN,
    #     model_template_id=model_template["model_template_id"],
    #     hyper_parameters=model_template["hyper_parameters"],
    #     export_parameters=[
    #         ExportParameter(
    #             export_format=ExportFormat.OPENVINO,
    #             precision=PrecisionType.FP32,
    #             with_xai=True,
    #         ),
    #         ExportParameter(
    #             export_format=ExportFormat.OPENVINO,
    #             precision=PrecisionType.FP32,
    #         ),
    #         ExportParameter(
    #             export_format=ExportFormat.OPENVINO,
    #             precision=PrecisionType.FP16,
    #         ),
    #         ExportParameter(
    #             export_format=ExportFormat.ONNX,
    #             precision=PrecisionType.FP32,
    #         ),
    #     ],
    #     optimization_type=None,
    #     sub_task_type=None,
    # )

    otx_config.to_otx2_config(Path("otx-workspace"))

    print(otx_config)
