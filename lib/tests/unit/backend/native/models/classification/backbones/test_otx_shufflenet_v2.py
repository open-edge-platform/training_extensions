# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import torch

from otx.backend.native.models.classification.backbones.shufflenet_v2 import ShuffleNetV2Backbone


class TestOTXShuffleNetV2:
    def test_forward_large(self):
        model = ShuffleNetV2Backbone(model_name="shufflenetv2_large")
        output = model(torch.randn(1, 3, 224, 224))
        assert output[0].shape == torch.Size([1, 1024, 7, 7])

    def test_forward_small(self):
        model = ShuffleNetV2Backbone(model_name="shufflenetv2_small")
        output = model(torch.randn(1, 3, 224, 224))
        assert output[0].shape == torch.Size([1, 1024, 7, 7])

    def test_input_size_respected(self):
        model = ShuffleNetV2Backbone(model_name="shufflenetv2_large", input_size=(64, 64))
        assert model.in_size == (64, 64)
        # Small input: first conv stride=1 instead of 2
        output = model(torch.randn(1, 3, 64, 64))
        assert output[0].shape[0] == 1
        assert output[0].shape[1] == 1024

    def test_invalid_model_name(self):
        import pytest

        with pytest.raises(ValueError, match="Unknown ShuffleNetV2 model"):
            ShuffleNetV2Backbone(model_name="invalid_model")
