# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from datetime import datetime
from enum import StrEnum
from uuid import UUID

from pydantic import Field, model_validator

from app.db.schema import ModelRevisionDB, ModelVariantDB
from app.models import DatasetItemSubset, EvaluationResult
from app.models.base import BaseEntity


class ModelFormat(StrEnum):
    OPENVINO = "openvino"
    ONNX = "onnx"
    PYTORCH = "pytorch"


class ModelPrecision(StrEnum):
    FP16 = "fp16"
    FP32 = "fp32"
    INT8 = "int8"


class TrainingStatus(StrEnum):
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    FAILED = "failed"
    SUCCESSFUL = "successful"


class TrainingInfo(BaseEntity):
    """Information about the training process of a model revision."""

    status: TrainingStatus = TrainingStatus.NOT_STARTED
    label_schema_revision: dict = Field(default_factory=dict)
    start_time: datetime | None = None
    end_time: datetime | None = None
    dataset_revision_id: UUID | None = None

    @model_validator(mode="before")
    @classmethod
    def populate_training_info(cls, data: object) -> object:
        if isinstance(data, ModelRevisionDB):
            return {
                "status": data.training_status,
                "label_schema_revision": data.label_schema_revision,
                "start_time": data.training_started_at,
                "end_time": data.training_finished_at,
                "dataset_revision_id": data.training_dataset_id,
            }
        return data


class ModelVariant(BaseEntity):
    """
    Represents a specific variant of a model revision, such as a particular format and precision combination.

    A model variant is a first-class entity with its own ID, format, precision, and evaluation results.
    Variants are stored in the database and on the filesystem under the model's variants directory.

    Attributes:
        id: Unique identifier for the model variant.
        model_revision_id: UUID of the parent model revision.
        format: The format of the model variant (e.g., 'openvino', 'onnx', 'pytorch').
        precision: The precision of the model variant (e.g., 'fp16', 'fp32', 'int8').
        weights_size: Size of the model weights files in bytes.
        evaluations: List of evaluation results for this variant.
        quantization_info: Info about the quantization process (only for quantized variants).
        files_deleted: Flag indicating whether the variant files have been deleted from storage.
    """

    id: UUID
    model_revision_id: UUID
    format: ModelFormat
    precision: ModelPrecision
    weights_size: int = 0
    evaluations: list[EvaluationResult] = []
    quantization_info: dict | None = None
    files_deleted: bool = False

    @model_validator(mode="before")
    @classmethod
    def populate_model_variant(cls, data: object) -> object:
        if isinstance(data, ModelVariantDB):
            return {
                "id": data.id,
                "model_revision_id": data.model_revision_id,
                "format": data.format,
                "precision": data.precision,
                "weights_size": 0,  # Computed at runtime
                "quantization_info": data.quantization_info,
                "files_deleted": data.files_deleted,
                "evaluations": [
                    EvaluationResult(
                        model_revision_id=UUID(data.model_revision_id),
                        model_variant_id=UUID(e.model_variant_id),
                        dataset_revision_id=UUID(e.dataset_revision_id),
                        subset=DatasetItemSubset(e.subset),
                        metrics={m.metric: m.score for m in e.metric_scores},
                    )
                    for e in data.evaluations
                ],
            }
        return data


class ModelRevision(BaseEntity):
    """
    Represents a specific revision of a machine learning model.

    A model revision tracks a particular version of a model, including its architecture, relationship to other
    revisions, training information, and file storage status.

    Attributes:
        id: Unique identifier for the model revision.
        name: User friendly name to identify a model
        architecture: Identifier of the model architecture (e.g., 'object-detection-rt-detr-r50').
        parent_revision: UUID of the parent revision if this is derived from another revision,
            None if this is the initial revision.
        training_info: Details about the training process, including status, configuration, and associated dataset.
            None if training hasn't started.
        variants: List of model variants (different formats/precisions/quanizations) for this revision.
        files_deleted: Flag indicating whether the model files have been deleted from storage.
    """

    id: UUID
    name: str
    architecture: str
    parent_revision: UUID | None = None
    training_info: TrainingInfo | None = None
    variants: list[ModelVariant] = []
    training_configuration: dict | None = None
    files_deleted: bool = False

    @model_validator(mode="before")
    @classmethod
    def populate_model_revision(cls, data: object) -> object:
        if isinstance(data, ModelRevisionDB):
            return {
                "id": data.id,
                "name": data.name,
                "architecture": data.architecture,
                "parent_revision": data.parent_revision,
                "files_deleted": data.files_deleted,
                "variants": [ModelVariant.model_validate(v) for v in data.variants],
                "training_info": TrainingInfo.model_validate(data),
                "training_configuration": data.training_configuration,
            }
        return data
