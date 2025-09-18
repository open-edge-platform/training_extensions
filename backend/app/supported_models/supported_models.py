# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
from functools import cache
from importlib import resources

from . import manifests
from .model_manifest import ModelManifest, NullModelManifest
from .parser import parse_manifest


class SupportedModels:
    @classmethod
    def get_model_manifest_by_id(cls, model_manifest_id: str) -> ModelManifest:
        """
        Retrieve a specific model manifest by its ID.
        """
        model_manifests = cls.get_model_manifests()
        if model_manifest_id not in model_manifests:
            return NullModelManifest()
        return model_manifests[model_manifest_id]

    @staticmethod
    @cache
    def get_model_manifests() -> dict[str, ModelManifest]:
        """
        Find and load all model manifest files in the manifests directory.

        Structure:
        - manifests/base.yaml: Base configuration for all models
        - manifests/task/base.yaml: Base configuration for specific task type
        - manifests/task/model.yaml: Model-specific configuration

        :return: A dictionary mapping model manifest IDs to their corresponding ModelManifest objects.
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
