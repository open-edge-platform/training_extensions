# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from copy import deepcopy
from functools import cache
from typing import Any, Optional, cast, get_args

from pydantic import BaseModel, create_model
from pydantic.fields import FieldInfo, PydanticUndefined


@cache  # avoids creating many classes with same name
def partial_model(model: type[BaseModel]) -> type[BaseModel]:
    """
    Creates a new Pydantic model class with all fields made optional.

    This decorator transforms a Pydantic model into a "partial" version where all fields
    are wrapped with Optional and have None as default value. The resulting model
    is particularly useful for:

    - Handling PATCH operations in REST APIs where only some fields need updating
    - Representing partial data structures during validation
    - Supporting incremental configuration changes

    The new model class inherits from the original model, with its name prefixed
    with "Partial" (e.g., "PartialProjectConfiguration").

    Args:
        model (type[BaseModel]): The Pydantic model class to make partial.

    Returns:
        type[BaseModel]: A new Pydantic model class with all fields optional.
    """

    @cache
    def make_field_optional(field: FieldInfo) -> tuple[Any, FieldInfo]:
        """
        Convert a Pydantic field to an optional field.

        Args:
            field (FieldInfo): The field information to make optional.

        Returns:
            tuple[Any, FieldInfo]: A tuple containing the optional annotation and updated field info.
        """
        # use json_schema_extra to store the default value, since default has to be None
        default_value = None if field.default is PydanticUndefined else field.default
        if isinstance(field.json_schema_extra, dict):
            field.json_schema_extra["default_value"] = field.json_schema_extra.get("default_value", default_value)
        else:
            field.json_schema_extra = {"default_value": default_value}
        field.default = None
        field.default_factory = None
        field.annotation = Optional[field.annotation]  # type: ignore[assignment] # noqa: UP007
        return field.annotation, field

    partial_fields: dict[str, FieldInfo | tuple[Any, FieldInfo]] = {}
    for field_name, field_info in model.model_fields.items():
        new_field = deepcopy(field_info)

        is_already_optional = not new_field.exclude and not new_field.is_required()
        if is_already_optional and (optional_annotation := get_args(new_field.annotation)):
            # Field is already optional, but still need to handle nested fields
            field_type, _ = optional_annotation  # tuple (annotation_type, None)
            new_field.annotation = field_type
            partial_fields[field_name] = new_field
        if type(new_field.annotation) is type(BaseModel):
            partial_inner_model = partial_model(cast("type[BaseModel]", new_field.annotation))  # type: ignore[arg-type]
            partial_fields[field_name] = (
                partial_inner_model | None,
                FieldInfo(annotation=partial_inner_model, default=None),
            )
        else:
            partial_fields[field_name] = make_field_optional(new_field)

    return create_model(  # type: ignore[call-overload]
        f"Partial{model.__name__}",
        __base__=model,
        __module__=model.__module__,
        **partial_fields,
    )
