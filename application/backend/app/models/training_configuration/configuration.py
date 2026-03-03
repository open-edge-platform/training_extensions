# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from pydantic import BaseModel, Field

from .dataset_preparation import AlgoLevelDatasetPreparationParameters, TaskLevelDatasetPreparationParameters
from .evaluation import TaskLevelEvaluationParameters
from .training import AlgoLevelTrainingParameters

type ParamValueType = bool | str | float | int | tuple[float, float]


class TaskLevelParameters(BaseModel):
    """
    Configurable parameters that apply at the task level and are relevant for all models, regardless of their specific
    architecture.
    """

    dataset_preparation: TaskLevelDatasetPreparationParameters = Field(
        default_factory=TaskLevelDatasetPreparationParameters,
        title="Dataset preparation",
        description="Configurable parameters related to the training data, such as augmentations and filters.",
    )
    evaluation: TaskLevelEvaluationParameters = Field(
        default_factory=TaskLevelEvaluationParameters,
        title="Evaluation parameters",
        description="Configurable parameters related to the model evaluation.",
    )


class AlgoLevelParameters(BaseModel):
    """Configurable parameters that are specific to a particular algorithm or model architecture."""

    dataset_preparation: AlgoLevelDatasetPreparationParameters = Field(
        default_factory=AlgoLevelDatasetPreparationParameters,
        title="Dataset preparation",
        description="Configurable parameters related to the training data, such as augmentations and filters.",
    )
    training: AlgoLevelTrainingParameters = Field(
        # No default_factory here because AlgoLevelTrainingParameters has required fields without defaults
        title="Training",
        description="Configurable parameters related to the learning phase (hyperparameters).",
    )


class TrainingConfiguration(BaseModel):
    """Configuration for model training"""

    task_level_parameters: TaskLevelParameters = Field(title="Task-level configurable parameters")
    algo_level_parameters: AlgoLevelParameters = Field(title="Algorithm-level configurable parameters")

    def _resolve_parameter_path(self, path: str) -> tuple[BaseModel, str]:
        """
        Resolve a dot-notation path to find the parent object and field name.

        Returns a tuple of (parent_object, field_name) where the field should be updated.
        Raises ValueError if the path cannot be resolved.
        """
        parts = path.split(".")
        if len(parts) < 2:
            raise ValueError(f"Invalid path '{path}': must contain at least one dot separator")

        # The path starts with a top-level section (e.g., "dataset_preparation", "training")
        # We need to find which parameter level contains this section
        top_section = parts[0]

        # Determine which parameter level(s) to search
        task_level = self.task_level_parameters
        algo_level = self.algo_level_parameters

        # Try to find the path in task-level or model-level parameters
        remaining_parts = parts[1:]

        # Check if top_section exists in task_level_parameters
        if hasattr(task_level, top_section):
            task_section = getattr(task_level, top_section)
            resolved = TrainingConfiguration._try_resolve_path(task_section, remaining_parts)
            if resolved is not None:
                return resolved

        # Check if top_section exists in model_level_parameters
        if hasattr(algo_level, top_section):
            model_section = getattr(algo_level, top_section)
            resolved = TrainingConfiguration._try_resolve_path(model_section, remaining_parts)
            if resolved is not None:
                return resolved

        raise ValueError(f"Cannot resolve path '{path}': path not found in configuration")

    @staticmethod
    def _try_resolve_path(obj: BaseModel, path_parts: list[str]) -> tuple[BaseModel, str] | None:
        """
        Try to resolve remaining path parts starting from obj.

        Returns (parent_object, field_name) if successful, None otherwise.
        """
        if not path_parts:
            return None

        current = obj
        for part in path_parts[:-1]:
            if not hasattr(current, part):
                return None
            next_obj = getattr(current, part)
            if not isinstance(next_obj, BaseModel):
                return None
            current = next_obj

        # Check if the final field exists
        final_field = path_parts[-1]
        if not hasattr(current, final_field):
            return None

        # Verify the field exists in the model's fields
        if final_field not in type(current).model_fields:
            return None

        return current, final_field

    def apply_updates(self, updates: dict[str, ParamValueType]) -> "TrainingConfiguration":
        """
        Apply a dictionary of updates to this TrainingConfiguration instance.

        Args:
            updates: Dictionary with dot-notation keys and new values

        Returns:
            Self for method chaining

        Raises:
            ValueError: If any key in updates cannot be resolved
        """
        # First, validate all paths exist before applying any updates
        resolved_updates: list[tuple[BaseModel, str, ParamValueType]] = []
        for path, value in updates.items():
            parent_obj, field_name = self._resolve_parameter_path(path)
            resolved_updates.append((parent_obj, field_name, value))

        # Apply all updates (modifies in place)
        for parent_obj, field_name, value in resolved_updates:
            setattr(parent_obj, field_name, value)

        return self
