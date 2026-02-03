# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest

from app.supported_models import SupportedModels


class TestSupportedModels:
    def test_get_model_manifests(self) -> None:
        # test that the model manifests can be retrieved without errors
        model_manifests = SupportedModels.get_model_manifests()

        assert len(model_manifests) > 0

    @pytest.mark.parametrize(
        "model_manifest_id, expected_task",
        [
            ("image-classification-efficientnet-v2-s", "classification"),
            ("object-detection-atss-mobilenet-v2", "detection"),
            ("instance-segmentation-mask-rcnn-efficientnet-b2", "instance_segmentation"),
        ],
    )
    def test_get_model_manifest_by_id(self, model_manifest_id, expected_task) -> None:
        model_manifest = SupportedModels.get_model_manifest_by_id(model_manifest_id)

        assert model_manifest.id == model_manifest_id
        assert model_manifest.task == expected_task
