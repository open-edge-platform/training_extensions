# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


from pydantic import BaseModel, Field

from app.models import TaskType
from app.supported_models.model_manifest import Capabilities, ModelManifest, ModelManifestDeprecationStatus, ModelStats


class ModelArchitecture(BaseModel):
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

    @classmethod
    def from_manifest(cls, manifest: ModelManifest) -> "ModelArchitecture":
        """Create a ModelArchitecture from a ModelManifest, excluding unwanted fields"""
        return cls(
            id=manifest.id,
            task=manifest.task,
            name=manifest.name,
            description=manifest.description,
            capabilities=manifest.capabilities,
            stats=manifest.stats,
            support_status=manifest.support_status,
        )
