# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from pydantic import BaseModel
from pydantic.fields import FieldInfo

from app.configuration_tools.training_configuration import TrainingConfiguration


def _get_field_metadata(field_info: FieldInfo, value: Any) -> dict[str, Any]:
    """
    Extract metadata from a Pydantic field.

    Args:
        field_info: Pydantic FieldInfo object
        value: The actual value of the field

    Returns:
        Dictionary with metadata (type, description, default_value, constraints, etc.)
    """
    # Basic metadata
    metadata = {
        "type": _infer_type(field_info, value),
        "description": field_info.description or "",
        "value": value,
        "default_value": field_info.default
        if field_info.default is not None
        else field_info.default_factory()
        if callable(field_info.default_factory)
        else None,
    }

    # Extract constraints from field metadata
    if hasattr(field_info, "metadata"):
        for constraint in field_info.metadata:
            if hasattr(constraint, "gt"):
                metadata["min_value"] = (
                    constraint.gt
                    if constraint.gt is not None
                    else (constraint.ge if hasattr(constraint, "ge") and constraint.ge is not None else None)
                )
                metadata["max_value"] = None
            if hasattr(constraint, "lt"):
                metadata["max_value"] = (
                    constraint.lt
                    if constraint.lt is not None
                    else (constraint.le if hasattr(constraint, "le") and constraint.le is not None else None)
                )
                metadata["min_value"] = None

    # Check for allowed_values in json_schema_extra
    if field_info.json_schema_extra and isinstance(field_info.json_schema_extra, dict):
        if "allowed_values" in field_info.json_schema_extra:
            metadata["allowed_values"] = field_info.json_schema_extra["allowed_values"]
            metadata["type"] = "enum"
        if "default_value" in field_info.json_schema_extra:
            metadata["default_value"] = field_info.json_schema_extra["default_value"]

    return metadata


def _infer_type(field_info: FieldInfo, value: Any) -> str:
    """Infer the type string from field annotation and value."""
    annotation = field_info.annotation

    # Handle Optional types
    if hasattr(annotation, "__origin__"):
        # Extract non-None type from Union
        args = getattr(annotation, "__args__", ())
        non_none_types = [arg for arg in args if arg is not type(None)]
        if non_none_types:
            annotation = non_none_types[0]

    # Map Python types to string representations
    if annotation == bool or isinstance(value, bool):
        return "bool"
    if annotation == int or isinstance(value, int):
        return "int"
    if annotation == float or isinstance(value, float):
        return "float"
    if annotation == str or isinstance(value, str):
        return "str"
    if hasattr(annotation, "__origin__") and annotation.__origin__ == list:
        return "list"

    return "str"


def _convert_nested_model(model: BaseModel) -> dict[str, Any]:
    """Convert nested Pydantic models to the flattened parameter format."""
    result: dict[str, Any] = {}

    for field_name, field_info in model.model_fields.items():
        value = getattr(model, field_name)

        if value is None:
            continue  # Skip fields with None value

        # Handle nested BaseModel
        if isinstance(value, BaseModel):
            nested_result = _convert_nested_model(value)
            result[field_name] = _flatten_to_list_format(nested_result)
        else:
            # Create parameter metadata
            param = {
                "key": field_name,
                "name": field_info.title or field_name.replace("_", " ").title(),
            }
            param.update(_get_field_metadata(field_info, value))
            result[field_name] = [param]

    return result


def _flatten_to_list_format(nested_dict: dict[str, Any]) -> list[Any]:
    """Convert nested dictionary to list format with proper handling of sub-parameters."""
    result = []

    for key, value in nested_dict.items():
        if isinstance(value, list):
            # Already in parameter list format
            result.extend(value)
        elif isinstance(value, dict):
            # Check if this is a nested group (all values are lists or dicts)
            all_nested = all(isinstance(v, list | dict) for v in value.values())
            if all_nested:
                # Convert nested structure
                nested_list = _flatten_to_list_format(value)
                result.append({key: nested_list})
            else:
                result.append({key: value})

    return result


def convert_training_configuration_to_rest(config: TrainingConfiguration) -> dict[str, Any]:
    """
    Convert TrainingConfiguration Pydantic model to REST API format.

    Args:
        config: TrainingConfiguration Pydantic model instance

    Returns:
        Dictionary in the format matching output.json
    """
    result: dict[str, Any] = {}

    if config.model_manifest_id:
        result["model_manifest_id"] = config.model_manifest_id

    result["global_parameters"] = _convert_nested_model(config.global_parameters)
    result["hyperparameters"] = _convert_nested_model(config.hyperparameters)

    return result


def convert_rest_to_training_configuration(rest_data: dict[str, Any]) -> dict[str, Any]:
    """
    Convert REST API format back to TrainingConfiguration format.

    Args:
        rest_data: Dictionary in output.json format with lists of parameter objects

    Returns:
        Dictionary suitable for TrainingConfiguration.model_validate()
    """
    result: dict[str, Any] = {}

    for section_key, section_value in rest_data.items():
        if section_key == "model_manifest_id":
            continue

        # Handle empty sections - but still include them for required fields
        if not section_value:
            # For required fields like evaluation, provide empty dict
            if section_key == "evaluation":
                result[section_key] = {}
            continue

        if isinstance(section_value, list):
            # Convert list format back to nested dict
            section_dict = _convert_list_to_nested_dict(section_value)
            if section_dict:  # Only add if not empty
                result[section_key] = section_dict
        elif isinstance(section_value, dict):
            # Handle nested dictionary structures
            converted_dict = _convert_dict_recursively(section_value)
            if converted_dict:  # Only add if not empty
                result[section_key] = converted_dict
        else:
            result[section_key] = section_value

    # Ensure required fields are present
    if "evaluation" not in result:
        result["evaluation"] = {}

    return result


def _convert_dict_recursively(data: dict[str, Any]) -> dict[str, Any]:
    """Recursively convert nested dictionary structures that may contain REST format lists."""

    result = {}
    for key, value in data.items():
        if isinstance(value, list):
            # Convert list format back to nested dict
            converted = _convert_list_to_nested_dict(value)
            if converted:  # Only add if not empty
                result[key] = converted
        elif isinstance(value, dict):
            # Recursively handle nested dicts
            converted = _convert_dict_recursively(value)
            if converted:  # Only add if not empty
                result[key] = converted
        else:
            result[key] = value

    return result


def _convert_list_to_nested_dict(param_list: list[Any]) -> dict[str, Any]:
    """Convert parameter list format back to nested dictionary."""
    result = {}

    for item in param_list:
        if isinstance(item, dict):
            # Check if this is a parameter object with 'key' and 'value'
            if "key" in item and "value" in item:
                result[item["key"]] = item["value"]
            else:
                # This is a nested group - should have one key with a list value
                for group_key, group_value in item.items():
                    if isinstance(group_value, list):
                        result[group_key] = _convert_list_to_nested_dict(group_value)
                    else:
                        result[group_key] = group_value

    return result
