#  Copyright (C) 2026 Intel Corporation
#  SPDX-License-Identifier: Apache-2.0

from enum import StrEnum

from app.models import TaskType


class ModelArchitectureCategory(StrEnum):
    BALANCE = "balance"
    SPEED = "speed"
    ACCURACY = "accuracy"


RECOMMENDED_MODEL_ARCHITECTURES = {
    TaskType.CLASSIFICATION: {
        ModelArchitectureCategory.BALANCE: "image-classification-efficientnet-b0",
        ModelArchitectureCategory.ACCURACY: "image-classification-efficientnet-v2-s",
        ModelArchitectureCategory.SPEED: "image-classification-mobilenet-v3-large",
    },
    TaskType.DETECTION: {
        ModelArchitectureCategory.BALANCE: "object-detection-atss-mobilenet-v2",
        ModelArchitectureCategory.ACCURACY: "object-detection-d-fine-x",
        ModelArchitectureCategory.SPEED: "object-detection-yolox-s",
    },
    TaskType.INSTANCE_SEGMENTATION: {
        ModelArchitectureCategory.BALANCE: "instance-segmentation-mask-rcnn-resnet50",
        ModelArchitectureCategory.ACCURACY: "instance-segmentation-mask-rcnn-swin-t",
        ModelArchitectureCategory.SPEED: "instance-segmentation-mask-rcnn-efficientnet-b2",
    },
}
