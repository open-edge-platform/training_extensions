# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

from datumaro.experimental import Sample, import_dataset

from app.datumaro_converter import (
    DetectionImportExportSample,
    InstanceSegmentationImportExportSample,
    MulticlassClassificationImportExportSample,
    MultilabelClassificationImportExportSample,
)

ANNOTATION_TYPE_TO_SAMPLE: dict[str, type[Sample]] = {
    "bounding_box": DetectionImportExportSample,
    "multilabel": MultilabelClassificationImportExportSample,
    "single_label": MulticlassClassificationImportExportSample,
    "polygon": InstanceSegmentationImportExportSample,
}


def test_import_dataset(archive: Path) -> None:
    """Verify that each regression zip archive can be imported by datumaro."""
    dataset = import_dataset(archive)
    annotation_type, _ = archive.name.split("-")
    sample_type = ANNOTATION_TYPE_TO_SAMPLE[annotation_type]
    dataset = dataset.convert_to_schema(sample_type)
    assert len(dataset) > 0
