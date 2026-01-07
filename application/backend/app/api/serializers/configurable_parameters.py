# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


from typing import Any

from pydantic import BaseModel

PYDANTIC_BASE_TYPES_MAPPING = {
    "integer": "int",
    "number": "float",
    "boolean": "bool",
    "string": "str",
    "array": "array",
}
PYDANTIC_ANY_OF = "anyOf"

BasicType = int | float | str | bool


class ConfigurableParametersConverter:
    """
    Base class for converting configurable parameters to REST views.

    This class provides methods to transform Pydantic models and their fields
    into REST-compatible dictionary representations.
    """

    @staticmethod
    def _parameter_to_rest(
        key: str, rest_type: str, value: BasicType, json_schema: dict, default_value_override: BasicType | None = None
    ) -> dict[str, Any]:
        """
        Convert a single parameter to its REST representation.

        Args:
            key (str): The parameter name/key.
            rest_type (str): The parameter type (int, float, string, or boolean).
            value (BasicType): The parameter value (int, float, string, or boolean).
            json_schema (dict): The JSON schema for the parameter from the Pydantic model.
            default_value_override (BasicType | None): Optional override for the default value.

        Returns:
            dict[str, Any]: Dictionary containing the REST representation of the parameter.
        """
        default = default_value_override if default_value_override is not None else json_schema.get("default_value")
        default_value = default if default is not None else json_schema.get("default")
        rest_view = {
            "key": key,
            "name": json_schema.get("title"),
            "type": rest_type,
            "description": json_schema.get("description"),
            "value": value,
            "default_value": default_value,
        }
        # optional parameter may contain `'anyOf': [{'exclusiveMinimum': 0, 'type': 'integer'}, {'type': 'null'}]`
        type_any_of = json_schema.get(PYDANTIC_ANY_OF, [{}])[0]
        rest_view["type"] = PYDANTIC_BASE_TYPES_MAPPING.get(json_schema.get("type", type_any_of.get("type")))
        if rest_view["type"] in ["int", "float"]:
            # min/max values can be contained in "minimum", "exclusiveMinimum" or 'anyOf.*'
            rest_view["min_value"] = json_schema.get(
                "minimum",
                json_schema.get("exclusiveMinimum", type_any_of.get("minimum", type_any_of.get("exclusiveMinimum"))),
            )
            rest_view["max_value"] = json_schema.get(
                "maximum",
                json_schema.get("exclusiveMaximum", type_any_of.get("maximum", type_any_of.get("exclusiveMaximum"))),
            )
        if "allowed_values" in json_schema:
            # If the parameter has allowed values, the parameter is an enum
            rest_view["type"] = "enum"
            rest_view["allowed_values"] = json_schema["allowed_values"]
            # Remove numeric constraints for enum parameters
            rest_view.pop("min_value", None)
            rest_view.pop("max_value", None)
        return rest_view

    @classmethod
    def configurable_parameters_to_rest(
        cls, configurable_parameters: BaseModel, default_config: dict[str, Any] | None = None
    ) -> dict[str, Any] | list[dict[str, Any]]:
        """
        Convert a Pydantic model of configurable parameters to its REST representation.

        This method processes a Pydantic model containing configuration parameters and transforms it
        into a REST view. It handles both simple fields and nested models:

        - Simple fields (int, float, str, bool) are converted to a list of dictionaries with metadata
            including key, name, description, value, type, and constraints
        - Nested Pydantic models are processed recursively and maintained as nested structures

        The return format depends on the content:
        - If only simple parameters exist: returns a list of parameter dictionaries
        - If only nested models exist: returns a dictionary mapping nested model names to their contents
        - If both exist: returns a list containing parameter dictionaries and nested model dictionary

        Args:
            configurable_parameters (BaseModel): Pydantic model containing configurable parameters.
            default_config (dict[str, Any] | None): Optional default configuration to use for setting
                "default_value" in the REST view.

        Returns:
            dict[str, Any] | list[dict[str, Any]]: REST representation as either a dictionary of nested models,
                a list of parameter dictionaries, or a combined list of both.
        """
        nested_params: dict[str, Any] = {}
        list_params: list[dict[str, Any]] = []
        default_config = default_config or {}

        json_model = configurable_parameters.model_json_schema()
        for field_name, field_info in type(configurable_parameters).model_fields.items():
            field = getattr(configurable_parameters, field_name)
            default_field = default_config.get(field_name)

            # Update schema with any extra JSON schema information
            schema = json_model["properties"][field_name] | (field_info.json_schema_extra or {})
            # optional parameter may contain `'anyOf': [{'exclusiveMinimum': 0, 'type': 'integer'}, {'type': 'null'}]`
            type_any_of = schema.get(PYDANTIC_ANY_OF, [{}])[0]
            pydantic_type = schema.get("type", type_any_of.get("type"))

            if field is None or schema.get("validation_only", False):
                # Do not show None values in the REST view. None parameters means they are not supported
                continue

            if pydantic_type in PYDANTIC_BASE_TYPES_MAPPING:
                # If the field is a simple type, convert directly to REST view
                list_params.append(
                    cls._parameter_to_rest(
                        key=field_name,
                        rest_type=PYDANTIC_BASE_TYPES_MAPPING[pydantic_type],
                        value=field,
                        json_schema=schema,
                        default_value_override=default_field,
                    )
                )
            else:
                # If the field is a nested Pydantic model, process it recursively
                nested_params[field_name] = cls.configurable_parameters_to_rest(
                    configurable_parameters=field, default_config=default_field
                )

        # Return combined or individual results based on content
        if nested_params and list_params:
            return [*list_params, nested_params]
        return list_params or nested_params

    @classmethod
    def configurable_parameters_from_rest(
        cls, configurable_parameters_rest: dict[str, Any] | list[dict[str, Any]]
    ) -> dict[str, Any]:
        """
        Convert a REST representation back to a dictionary for Pydantic model creation.

        This method performs the reverse operation of configurable_parameters_to_rest:
        - For a list of parameter dictionaries, it extracts the key-value pairs
        - For a dictionary of nested models, it processes each nested model recursively
        - For a mixed list containing both, it handles both types

        Args:
            configurable_parameters_rest (dict[str, Any] | list[dict[str, Any]]): REST representation
                as a dictionary or list.

        Returns:
            dict[str, Any]: Dictionary representation suitable for Pydantic model instantiation.

        Raises:
            ValueError: If attempting to set reserved parameters starting with "allowed_values_".
        """
        # If the input is a list (of parameters or mixed)
        if isinstance(configurable_parameters_rest, list):
            result = {}

            for item in configurable_parameters_rest:
                # If this is a parameter entry (has a "key" field)
                if isinstance(item, dict) and "key" in item:
                    key = item["key"]
                    value = item["value"]
                    result[key] = value
                    if key.startswith("allowed_values_"):
                        # `allowed_values_` is a reserved prefix used for validation
                        raise ValueError(f"Cannot set reserved parameter '{key}' directly.")
                # If it's a dictionary without a "key" field, it must contain nested models
                elif isinstance(item, dict):
                    # Process each nested model recursively and merge with result
                    nested_result = cls.configurable_parameters_from_rest(item)
                    result.update(nested_result)

            return result

        # If the input is a dictionary (of nested models or other fields)
        if isinstance(configurable_parameters_rest, dict):
            result = {}

            for key, value in configurable_parameters_rest.items():
                # If the value is a complex structure, process it recursively
                if isinstance(value, dict | list):
                    result[key] = cls.configurable_parameters_from_rest(value)
                else:
                    # Simple value, keep as is
                    result[key] = value

            return result

        # If it's neither a list nor a dictionary, return it as is
        return configurable_parameters_rest
