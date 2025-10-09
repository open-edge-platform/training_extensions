# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
from functools import cache
from importlib import resources

from . import manifests
from .model_manifest import ModelManifest
from .parser import parse_manifest


class ManifestNotFoundException(Exception):
    """Exception raised when a model manifest is not found."""

    def __init__(self, model_manifest_id: str):
        super().__init__(f"Model manifest with ID {model_manifest_id} not found.")


class SupportedModels:
    @classmethod
    def get_model_manifest_by_id(cls, model_manifest_id: str) -> ModelManifest:
        """
        Retrieve a specific model manifest by its ID.

        Args:
            model_manifest_id: The unique identifier of the model manifest to retrieve.

        Returns:
            ModelManifest: The ModelManifest object corresponding to the given ID.

        Raises:
            ManifestNotFoundException: If the model manifest with the given ID does not exist
                                    in the available model manifests.
        """
        model_manifests = cls.get_model_manifests()
        if model_manifest_id not in model_manifests:
            raise ManifestNotFoundException(model_manifest_id=model_manifest_id)
        return model_manifests[model_manifest_id]

    @staticmethod
    @cache
    def get_model_manifests() -> dict[str, ModelManifest]:
        """
        Find and load all model manifest files in the manifests directory.

        Scans the manifests directory hierarchy and loads YAML configuration files
        following a structured format where base configurations are inherited and
        overridden by more specific configurations.

        Returns:
            dict[str, ModelManifest]: A dictionary mapping model manifest IDs to their
            corresponding ModelManifest objects.

        Note:
            Directory structure follows:
            - manifests/base.yaml: Base configuration for all models
            - manifests/task/base.yaml: Base configuration for specific task types
            - manifests/task/model.yaml: Model-specific configurations
        """
        # Get the manifests directory path
        manifests_dir = resources.files(manifests)
        root_base_yaml = "base.yaml"

        # Find all model-specific YAML files
        model_manifests = {}

        # Iterate through task type directories
        for task_dir in [d for d in manifests_dir.iterdir() if os.path.isdir(str(d))]:
            task_base_yaml = os.path.join(task_dir.name, "base.yaml")

            # Find all model files in this task directory
            for file in task_dir.iterdir():
                if file.name.endswith(".yaml") and file.name != "base.yaml":
                    model_yaml_path = os.path.join(task_dir.name, file.name)

                    # Build dependency chain: base -> task_base -> model
                    dependency_chain = [root_base_yaml, task_base_yaml, model_yaml_path]

                    # Parse manifest with all dependencies in order
                    manifest = parse_manifest(*dependency_chain, relative=True)
                    model_manifests[manifest.id] = manifest

        return model_manifests
