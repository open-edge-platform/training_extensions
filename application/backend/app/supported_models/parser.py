# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from importlib import resources

import hiyapyco

from . import manifests
from .model_manifest import ModelManifest


def parse_manifest(*manifest_sources, relative: bool = True) -> ModelManifest:
    """
    Parse model manifest YAML files and merge them into an Algorithm object.

    This function takes multiple manifest source paths, merges them using hierarchical
    YAML configuration, and converts the merged result into a structured Algorithm model.

    Args:
        *manifest_sources: Variable length list of YAML manifest file paths.
            Files are merged in order, with later files overriding values from earlier files.
        relative: If True, the paths are treated as relative to the package where the
            manifest files are stored. If False, they are treated as absolute paths.

    Returns:
        ModelManifest: A populated Algorithm object containing the parsed manifest data.

    Note:
        Uses hiyapyco library for YAML merging with substitution method and interpolation.
        Fails if any of the specified manifest files cannot be found.
    """
    if relative:
        manifest_sources = tuple(str(resources.files(manifests).joinpath(path)) for path in manifest_sources)
    yaml_manifest = hiyapyco.load(
        *manifest_sources,
        method=hiyapyco.METHOD_SUBSTITUTE,
        interpolate=True,
        failonmissingfiles=True,
        none_behavior=hiyapyco.NONE_BEHAVIOR_OVERRIDE,
    )
    return ModelManifest(**yaml_manifest)
