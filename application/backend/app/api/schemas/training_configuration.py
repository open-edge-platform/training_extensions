#  Copyright (C) 2026 Intel Corporation
#  SPDX-License-Identifier: Apache-2.0

from enum import StrEnum
from typing import Literal, Union

from pydantic import BaseModel, Field
from pydantic.fields import FieldInfo
from pydantic_core import PydanticUndefined

from app.models.training_configuration import Scalar, TrainingConfiguration


class ConfigurableParameterViewElementType(StrEnum):
    PARAMETER = "parameter"
    PARAMETER_GROUP = "parameter_group"


class ConfigurableParameterView(BaseModel):
    """
    A single configurable parameter that can be customized by the user. Includes metadata such as type, default value,
    and constraints to guide user input and validation.
    """

    type: Literal[ConfigurableParameterViewElementType.PARAMETER] = ConfigurableParameterViewElementType.PARAMETER
    key: str = Field(title="Key to identify the parameter")
    name: str = Field(title="User-friendly name of the parameter")
    description: str = Field(title="Extended description of the parameter", default="")
    value: bool | int | str | float | tuple[float, float] | None = Field(title="Actual value of the parameter")
    default_value: bool | int | str | float | tuple[float, float] | None = Field(title="Default value of the parameter")
    value_type: Literal["bool", "int", "float", "str", "float_range"] = Field(title="Type of the parameter value")
    min_value: int | float | None = Field(
        default=None, title="Minimum value for numeric parameters. None if unbounded or not applicable"
    )
    max_value: int | float | None = Field(
        default=None, title="Maximum value for numeric parameters. None if unbounded or not applicable"
    )


class ConfigurableParameterGroupView(BaseModel):
    """A group of related parameters, which can contain both individual parameters and nested parameter groups."""

    type: Literal[ConfigurableParameterViewElementType.PARAMETER_GROUP] = (
        ConfigurableParameterViewElementType.PARAMETER_GROUP
    )
    key: str = Field(title="Key to identify the parameter group")
    name: str = Field(title="User-friendly name of the parameter group")
    description: str = Field(title="Extended description of the parameter group", default="")
    parameters: list[Union[ConfigurableParameterView, "ConfigurableParameterGroupView"]] = Field(
        title="List of parameters in the group"
    )


class TrainingConfigurationView(BaseModel):
    """
    Configuration of training parameters, structured as a list of parameter groups and parameters that can be tuned by
    the user to control the training process.
    """

    parameters: list[ConfigurableParameterView | ConfigurableParameterGroupView] = Field(
        title="Training configuration parameters",
        description=(
            "List of configurable parameters and parameter groups for training. "
            "Parameters are organized hierarchically into groups based on their functions and relative similarities. "
            "Each parameter includes metadata such as type, default value, and constraints to guide user input."
        ),
    )

    @classmethod
    def _extract_constraints(cls, field_info: FieldInfo) -> tuple[float | None, float | None]:
        """Extract min/max constraints from field metadata."""
        min_value = None
        max_value = None

        if hasattr(field_info, "metadata"):
            for constraint in field_info.metadata:
                if hasattr(constraint, "ge"):
                    min_value = constraint.ge
                elif hasattr(constraint, "gt"):
                    min_value = constraint.gt
                if hasattr(constraint, "le"):
                    max_value = constraint.le
                elif hasattr(constraint, "lt"):
                    max_value = constraint.lt

        return min_value, max_value

    @classmethod
    def _get_value_type(cls, field_info: FieldInfo) -> Literal["bool", "int", "float", "str", "float_range"]:
        """Determine the value type from field annotation."""
        annotation = field_info.annotation

        # Handle Optional/Union types by extracting the non-None type
        if hasattr(annotation, "__origin__"):
            from typing import get_args, get_origin

            origin = get_origin(annotation)
            if origin is Union:
                args = [arg for arg in get_args(annotation) if arg is not type(None)]
                if args:
                    annotation = args[0]
                    origin = get_origin(annotation)

            # Detect tuple[float, float]
            if origin is tuple:
                args = get_args(annotation)
                if len(args) == 2 and all(a is float for a in args):
                    return "float_range"

        # Map Python types to string representations
        if annotation is bool:
            return "bool"
        if annotation is int:
            return "int"
        if annotation is float:
            return "float"
        if annotation is str:
            return "str"
        # Default to str for unknown types
        return "str"

    @classmethod
    def _field_to_configurable_parameter(
        cls, key: str, value: Scalar | None, field_info: FieldInfo
    ) -> ConfigurableParameterView:
        """Convert a single field to ConfigurableParameterView."""
        min_value, max_value = cls._extract_constraints(field_info)
        value_type = cls._get_value_type(field_info)

        # Handle PydanticUndefined default values
        if field_info.default is PydanticUndefined:
            default_value = value
        else:
            default_value = field_info.default

        if field_info.title is None:
            raise ValueError(
                f"Field '{key}' is missing a title in its FieldInfo, "
                f"which is required to associate a user-friendly name to the parameter."
            )

        return ConfigurableParameterView(
            key=key,
            name=field_info.title,
            description=field_info.description or "",
            value=value,
            default_value=default_value,
            value_type=value_type,
            min_value=min_value,
            max_value=max_value,
        )

    @classmethod
    def _model_to_parameter_group(
        cls, key: str, model: BaseModel, field_info: FieldInfo
    ) -> ConfigurableParameterGroupView:
        """Convert a Pydantic model to a ConfigurableParameterGroupView with nested parameters."""
        parameters: list[ConfigurableParameterView | ConfigurableParameterGroupView] = []

        for field_name, child_field_info in type(model).model_fields.items():
            # Skip validation-only fields
            if isinstance(child_field_info.json_schema_extra, dict) and child_field_info.json_schema_extra.get(
                "validation_only"
            ):
                continue

            value = getattr(model, field_name)

            # Skip parameters with null values
            if value is None:
                continue

            if isinstance(value, BaseModel):
                # Nested model -> create a nested parameter group
                nested_group = cls._model_to_parameter_group(field_name, value, child_field_info)
                # Only add the group if it has parameters (not empty due to null filtering)
                if nested_group.parameters:
                    parameters.append(nested_group)
            else:
                # Scalar value -> create a parameter
                parameters.append(cls._field_to_configurable_parameter(field_name, value, child_field_info))

        if field_info.title is None:
            raise ValueError(
                f"Field '{key}' is missing a title in its FieldInfo, "
                f"which is required to associate a user-friendly name to the parameter group."
            )

        return ConfigurableParameterGroupView(
            key=key,
            name=field_info.title,
            description=field_info.description if field_info.description else "",
            parameters=parameters,
        )

    @classmethod
    def _merge_parameter_groups(cls, *groups: ConfigurableParameterGroupView) -> ConfigurableParameterGroupView:
        """Merge multiple parameter groups with the same key into one."""
        if not groups:
            raise ValueError("At least one group is required")

        # Use the first group as base
        base = groups[0]
        merged_parameters: list[ConfigurableParameterView | ConfigurableParameterGroupView] = []

        # Collect all parameters by key for merging
        params_by_key: dict[str, list[ConfigurableParameterView | ConfigurableParameterGroupView]] = {}
        for group in groups:
            for param in group.parameters:
                if param.key not in params_by_key:
                    params_by_key[param.key] = []
                params_by_key[param.key].append(param)

        # Merge parameters with same key, or just add unique ones
        for key, params in params_by_key.items():
            if len(params) == 1:
                merged_parameters.append(params[0])
            # Multiple params with same key - they should all be groups to merge
            elif all(isinstance(p, ConfigurableParameterGroupView) for p in params):
                merged_parameters.append(cls._merge_parameter_groups(*params))  # type: ignore
            else:
                # If not all groups, just take the first one
                merged_parameters.append(params[0])

        return ConfigurableParameterGroupView(
            key=base.key,
            name=base.name,
            description=base.description,
            parameters=merged_parameters,
        )

    @classmethod
    def from_training_configuration(cls, config: TrainingConfiguration) -> "TrainingConfigurationView":
        """Convert TrainingConfiguration to TrainingConfigurationView."""

        # Convert task-level dataset_preparation to a parameter group
        task_dataset_prep = cls._model_to_parameter_group(
            "dataset_preparation",
            config.task_level_parameters.dataset_preparation,
            type(config.task_level_parameters).model_fields["dataset_preparation"],
        )

        # Convert model-level dataset_preparation to a parameter group
        model_dataset_prep = cls._model_to_parameter_group(
            "dataset_preparation",
            config.algo_level_parameters.dataset_preparation,
            type(config.algo_level_parameters).model_fields["dataset_preparation"],
        )

        # Merge both dataset_preparation groups
        merged_dataset_prep = cls._merge_parameter_groups(task_dataset_prep, model_dataset_prep)

        # Convert training parameters to a parameter group
        training_group = cls._model_to_parameter_group(
            "training",
            config.algo_level_parameters.training,
            type(config.algo_level_parameters).model_fields["training"],
        )

        # Return with direct list of parameter groups
        return cls(parameters=[merged_dataset_prep, training_group])
