# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from datumaro.experimental.fields import Subset

from app.models import DatasetItemSubset


class SubsetConverter:
    """Knows how to convert between subset types."""

    _MAPPING = {
        DatasetItemSubset.TRAINING: Subset.TRAINING,
        DatasetItemSubset.VALIDATION: Subset.VALIDATION,
        DatasetItemSubset.TESTING: Subset.TESTING,
        DatasetItemSubset.UNASSIGNED: Subset.UNASSIGNED,
    }

    @classmethod
    def to_datumaro(cls, subset: DatasetItemSubset) -> Subset:
        """Converts to Datumaro subset."""
        result = cls._MAPPING.get(subset)
        if result is None:
            raise ValueError(f"Unknown subset type: {subset}")
        return result
