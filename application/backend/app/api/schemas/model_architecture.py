# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


from pydantic import BaseModel, Field

from app.models import TaskType
from app.supported_models.model_manifest import Capabilities, ModelManifestDeprecationStatus, ModelStats


class ModelArchitectureCategory:
    BALANCE = "balance"
    SPEED = "speed"
    ACCURACY = "accuracy"


RECOMMENDED_MODEL_ARCHITECTURES = {
    TaskType.CLASSIFICATION: {
        ModelArchitectureCategory.BALANCE: "Custom_Image_Classification_EfficinetNet-B0",
        ModelArchitectureCategory.ACCURACY: "Custom_Image_Classification_EfficientNet-V2-S",
        ModelArchitectureCategory.SPEED: "Custom_Image_Classification_MobileNet-V3-large-1x",
    },
    TaskType.DETECTION: {
        ModelArchitectureCategory.BALANCE: "Custom_Object_Detection_Gen3_ATSS",
        ModelArchitectureCategory.ACCURACY: "Object_Detection_DFine_X",
        ModelArchitectureCategory.SPEED: "Object_Detection_YOLOX_S",
    },
    TaskType.INSTANCE_SEGMENTATION: {
        ModelArchitectureCategory.BALANCE: "Custom_Instance_Segmentation_MaskRCNN_ResNet50_v2",
        ModelArchitectureCategory.ACCURACY: "Custom_Counting_Instance_Segmentation_MaskRCNN_SwinT_FP16",
        ModelArchitectureCategory.SPEED: "Custom_Counting_Instance_Segmentation_MaskRCNN_EfficientNetB2B",
    },
}


class ModelArchitectureView(BaseModel):
    """Simplified model architecture information for API responses"""

    id: str = Field(title="Model architecture ID", description="Unique identifier for the model architecture")
    task: TaskType = Field(title="Task Type", description="Type of machine learning task addressed by the model")
    name: str = Field(title="Model architecture name", description="Friendly name of the model architecture")
    description: str = Field(title="Description", description="Detailed description of the model capabilities")
    capabilities: Capabilities = Field(
        title="Model Capabilities", description="Special capabilities supported by the model"
    )
    stats: ModelStats = Field(title="Model Statistics", description="Statistics about the model")
    support_status: ModelManifestDeprecationStatus = Field(
        title="Support Status", description="Current support level (active, deprecated, or obsolete)"
    )


class TopPicks(BaseModel):
    """Top picks for model architectures based on categories"""

    balance: str = Field(title="Balance Model", description="Model architecture that balances accuracy and speed")
    speed: str = Field(title="Speed Model", description="Model architecture optimized for speed")
    accuracy: str = Field(title="Accuracy Model", description="Model architecture optimized for accuracy")


class ModelArchitectures(BaseModel):
    """Model architectures response"""

    model_architectures: list[ModelArchitectureView] = Field(
        title="Model Architectures", description="List of available model architectures"
    )

    top_picks: TopPicks | None = Field(
        title="Top Picks", description="Recommended model architectures for different categories"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "model_architectures": [
                    {
                        "id": "Object_Detection_Deim_DFine_M",
                        "task": "detection",
                        "name": "Deim-DFine-M",
                        "description": "DEIM is an advanced training framework designed to enhance the matching"
                        " mechanism in DETRs, enabling faster convergence and improved accuracy.",
                        "capabilities": {"xai": True, "tiling": True},
                        "stats": {
                            "gigaflops": 57,
                            "trainable_parameters": 19,
                            "performance_ratings": {"accuracy": 2, "training_time": 2, "inference_speed": 2},
                        },
                        "support_status": "active",
                    },
                ],
                "top_picks": {
                    "balance": "Custom_Object_Detection_Gen3_ATSS",
                    "speed": "Object_Detection_YOLOX_S",
                    "accuracy": "Object_Detection_DFine_X",
                },
            }
        }
    }
