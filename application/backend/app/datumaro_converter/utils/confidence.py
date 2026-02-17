# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Sequence

from app.models import DatasetItemAnnotation


def validate_confidence_consistency(annotations: Sequence[DatasetItemAnnotation]) -> bool:
    """Validates that all or none annotations have confidence scores."""
    any_with = any(ann.confidences is not None for ann in annotations)
    all_with = all(ann.confidences is not None for ann in annotations)

    if any_with and not all_with:
        raise ValueError("Either all or none of the annotations must have confidence scores")

    return all_with
