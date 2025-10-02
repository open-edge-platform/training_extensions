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
            ("Custom_Image_Classification_EfficientNet-V2-S", "classification"),
            ("Custom_Object_Detection_Gen3_ATSS", "detection"),
            ("Custom_Counting_Instance_Segmentation_MaskRCNN_EfficientNetB2B", "instance_segmentation"),
        ],
    )
    def test_get_model_manifest_by_id(self, model_manifest_id, expected_task) -> None:
        model_manifest = SupportedModels.get_model_manifest_by_id(model_manifest_id)

        assert model_manifest.id == model_manifest_id
        assert model_manifest.task == expected_task
