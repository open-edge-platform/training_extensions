# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest

from app.supported_models import NullModelManifest, SupportedModels


class TestSupportedModels:
    def test_get_model_manifests(self) -> None:
        # test that the model manifests can be retrieved without errors
        model_manifests = SupportedModels.get_model_manifests()

        assert len(model_manifests) > 0

    @pytest.mark.parametrize(
        "model_manifest_id, expected_task",
        [
            ("ote_anomaly_padim", "anomaly"),
            ("ote_anomaly_stfpm", "anomaly"),
            ("Custom_Image_Classification_EfficientNet-V2-S", "classification"),
            ("Custom_Object_Detection_Gen3_ATSS", "detection"),
            ("Custom_Counting_Instance_Segmentation_MaskRCNN_EfficientNetB2B", "instance_segmentation"),
            ("Keypoint_Detection_RTMPose_Tiny", "keypoint_detection"),
            ("Custom_Rotated_Detection_via_Instance_Segmentation_MaskRCNN_ResNet50", "rotated_detection"),
            ("Custom_Rotated_Detection_via_Instance_Segmentation_MaskRCNN_EfficientNetB2B", "rotated_detection"),
            ("Custom_Semantic_Segmentation_Lite-HRNet-18-mod2_OCR", "segmentation"),
            ("visual_prompting_model", "visual_prompting"),
        ],
    )
    def test_get_model_manifest_by_id(self, model_manifest_id, expected_task) -> None:
        model_manifest = SupportedModels.get_model_manifest_by_id(model_manifest_id)

        if model_manifest_id == "ote_anomaly_padim":
            # checks that None override is active
            assert model_manifest.hyperparameters.training.max_epochs is None
        assert not isinstance(model_manifest, NullModelManifest)
        assert model_manifest.id == model_manifest_id
        assert model_manifest.task == expected_task
