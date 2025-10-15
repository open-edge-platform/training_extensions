# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from enum import Enum, auto

from app.schemas.project import TaskType


class DefaultCategory(str, Enum):
    """Model optimization preference categories.

    Defines the trade-off preference between accuracy and inference speed:
    - DEFAULT: The recommended model choice when no specific optimization preference is specified
    - ACCURACY: Models optimized for highest prediction quality, potentially at the cost of speed
    - SPEED: Models optimized for fastest inference time, potentially at the cost of accuracy
    - BALANCE: Models that provide a balanced trade-off between accuracy and speed
    """

    DEFAULT = auto()
    ACCURACY = auto()
    SPEED = auto()
    BALANCE = auto()


class DefaultModels:
    """
    Provides the recommended model architectures for different computer vision tasks.

    This class maps task types to recommended model architectures based on optimization
    preferences (accuracy, speed, or balance), offering a curated selection of pre-configured
    models suitable for various computer vision applications.
    """

    default_models_by_task: dict[TaskType, dict[DefaultCategory, str | None]] = {
        TaskType.CLASSIFICATION: {
            DefaultCategory.DEFAULT: "Custom_Image_Classification_EfficientNet-B0",
            DefaultCategory.ACCURACY: "Custom_Image_Classification_EfficientNet-V2-S",
            DefaultCategory.SPEED: "Custom_Image_Classification_MobileNet-V3-large-1x",
            DefaultCategory.BALANCE: "Custom_Image_Classification_EfficientNet-B0",
        },
        TaskType.DETECTION: {
            DefaultCategory.DEFAULT: "Custom_Object_Detection_Gen3_ATSS",
            DefaultCategory.ACCURACY: "Object_Detection_DFine_X",
            DefaultCategory.SPEED: "Object_Detection_YOLOX_S",
            DefaultCategory.BALANCE: "Custom_Object_Detection_Gen3_ATSS",
        },
        TaskType.INSTANCE_SEGMENTATION: {
            DefaultCategory.DEFAULT: "Custom_Counting_Instance_Segmentation_MaskRCNN_EfficientNetB2B",
            DefaultCategory.ACCURACY: "Custom_Counting_Instance_Segmentation_MaskRCNN_SwinT_FP16",
            DefaultCategory.SPEED: "Custom_Counting_Instance_Segmentation_MaskRCNN_EfficientNetB2B",
            DefaultCategory.BALANCE: "Custom_Instance_Segmentation_MaskRCNN_ResNet50_v2",
        },
    }

    @classmethod
    def get_model_by_type(cls, task_type: str, default_type: DefaultCategory) -> str | None:
        """
        Retrieve a recommended model architecture for a specific task and optimization preference.

        Args:
            task_type: The computer vision task category as a string (e.g., 'DETECTION', 'CLASSIFICATION').
            default_type: The optimization preference indicating whether to prioritize accuracy,
                        speed, or balanced performance.

        Returns:
            str | None: The name of the recommended model architecture, or None if no model
                    is available for the given task type (e.g., VISUAL_PROMPTING).

        Raises:
            ValueError: If the task_type is not a supported TaskType value or if the default_type
                    is not available for the given task type.
        """
        try:
            task_type_str = task_type.upper()
            _task_type = TaskType[task_type_str]
        except KeyError:
            raise ValueError(f"Unknown task type: `{task_type}`")
        if default_type not in cls.default_models_by_task[_task_type]:
            raise ValueError(f"Unknown default type: {default_type} for task: {task_type}")
        return cls.default_models_by_task[_task_type][default_type]

    @classmethod
    def get_default_model(cls, task_type: str) -> str | None:
        """
        Gets the default model architecture for a specific computer vision task.

        This is a convenience method that retrieves the default (balanced) model
        recommendation for the given task type by delegating to get_model_by_type.

        Args:
            task_type: The computer vision task category as a string
                    (e.g., 'DETECTION', 'CLASSIFICATION').

        Returns:
            str | None: The name of the default model architecture, or None if no
                    default model is available for the given task type.

        Raises:
            ValueError: If the task_type is not a supported TaskType value.
        """
        return cls.get_model_by_type(task_type, DefaultCategory.DEFAULT)

    @classmethod
    def get_accuracy_model(cls, task_type: str) -> str | None:
        """
        Gets the model architecture that prioritizes prediction quality over inference speed.

        This method retrieves the accuracy-optimized model recommendation for the given
        task type by delegating to get_model_by_type with the ACCURACY preference.

        Args:
            task_type: The computer vision task category as a string
                    (e.g., 'DETECTION', 'CLASSIFICATION').

        Returns:
            str | None: The name of the accuracy-optimized model architecture, or None
                    if no accuracy model is available for the given task type.

        Raises:
            ValueError: If the task_type is not a supported TaskType value.
        """
        return cls.get_model_by_type(task_type, DefaultCategory.ACCURACY)

    @classmethod
    def get_speed_model(cls, task_type: str) -> str | None:
        """
        Gets the model architecture that prioritizes inference speed over prediction quality.

        This method retrieves the speed-optimized model recommendation for the given
        task type by delegating to get_model_by_type with the SPEED preference.

        Args:
            task_type: The computer vision task category as a string
                    (e.g., 'DETECTION', 'CLASSIFICATION').

        Returns:
            str | None: The name of the speed-optimized model architecture, or None
                    if no speed model is available for the given task type.

        Raises:
            ValueError: If the task_type is not a supported TaskType value.
        """
        return cls.get_model_by_type(task_type, DefaultCategory.SPEED)

    @classmethod
    def get_balanced_model(cls, task_type: str) -> str | None:
        """
        Gets the model architecture that offers a compromise between accuracy and speed.

        This method retrieves the balanced model recommendation for the given task type
        by delegating to get_model_by_type with the BALANCE preference.

        Args:
            task_type: The computer vision task category as a string
                    (e.g., 'DETECTION', 'CLASSIFICATION').

        Returns:
            str | None: The name of the balanced model architecture, or None
                    if no balanced model is available for the given task type.

        Raises:
            ValueError: If the task_type is not a supported TaskType value.
        """
        return cls.get_model_by_type(task_type, DefaultCategory.BALANCE)

    @classmethod
    def get_default_models_per_task(cls) -> dict[str, str | None]:
        """
        Gets the default model for each task type.

        Returns:
            dict[str, str | None]: A dictionary mapping task type names (in lowercase)
            to their default model names. Returns None as value for task types that
            don't have a default model.

        Example:
            >>> get_default_models_per_task()
            {
                'detection': 'yolox',
                'classification': 'efficientnet_b0',
                'segmentation': 'segformer',
                'visual_prompting': None
            }
        """
        return {task_type.name.lower(): cls.get_default_model(task_type.name.lower()) for task_type in TaskType}
