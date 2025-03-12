# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
import torch
from torch import nn

from otx.algo.visual_prompting.backbones.tiny_vit import TinyViT
from otx.algo.visual_prompting.decoders.sam_mask_decoder import SAMMaskDecoder
from otx.algo.visual_prompting.encoders.sam_prompt_encoder import SAMPromptEncoder
from otx.algo.visual_prompting.losses.sam_loss import SAMCriterion
from otx.algo.visual_prompting.sam import SAM
from otx.core.model.base import DataInputParams


class TestSAM:
    @pytest.fixture()
    def sam(self) -> SAM:
        return SAM(
            model_name="tiny_vit",
            data_input_params=DataInputParams((1024, 1024), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
        )

    def test_initialization(self, mocker) -> None:
        mock_freeze_networks = mocker.patch.object(SAM, "freeze_networks")
        mock_load_state_dict = mocker.patch.object(SAM, "load_state_dict")

        sam = SAM(
            model_name="tiny_vit",
            data_input_params=DataInputParams((1024, 1024), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
        )
        assert sam.model_name == "tiny_vit"
        assert sam.image_size == 1024
        assert sam.image_embedding_size == 64
        assert sam.use_stability_score is False
        assert sam.return_single_mask is True
        assert sam.return_extra_metrics is False
        assert sam.stability_score_offset == 1.0

        mock_load_state_dict.assert_called_once_with(checkpoint=sam.pretrained_weights["tiny_vit"])
        mock_freeze_networks.assert_called_once_with(True, True, False)

    def test_create_model(self, sam: SAM) -> None:
        segment_anything = sam._create_model()
        assert segment_anything is not None
        assert isinstance(segment_anything, torch.nn.Module)
        assert segment_anything.__class__.__name__ == "SegmentAnything"

        assert isinstance(segment_anything.image_encoder, TinyViT)
        assert isinstance(segment_anything.prompt_encoder, SAMPromptEncoder)
        assert isinstance(segment_anything.mask_decoder, SAMMaskDecoder)
        assert isinstance(segment_anything.criterion, SAMCriterion)

    def test_load_state_dict_success(self, mocker) -> None:
        mock_load_state_dict_from_url = mocker.patch("torch.hub.load_state_dict_from_url")
        mock_state_dict = {
            "image_encoder.norm_head.weight": torch.tensor([1.0]),
            "image_encoder.norm_head.bias": torch.tensor([1.0]),
            "image_encoder.head.weight": torch.tensor([1.0]),
            "image_encoder.head.bias": torch.tensor([1.0]),
            "some_other_key": torch.tensor([1.0]),
        }
        mock_load_state_dict_from_url.return_value = mock_state_dict

        # Mock only nn.Module's load_state_dict
        mock_module_load_state_dict = mocker.patch.object(nn.Module, "load_state_dict")

        # Create a test class that inherits from nn.Module and SAM
        class TestSam(SAM, nn.Module):
            def __init__(self):
                super().__init__(data_input_params=DataInputParams((1024, 1024), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)))
                self.some_param = nn.Parameter(torch.randn(1))

        # Create an instance of the test class
        test_sam = TestSam()

        # Call load_state_dict (this will use SAM's implementation)
        test_sam.load_state_dict(checkpoint="https://example.com/checkpoint.pth")

        # Verify that load_state_dict_from_url was called
        mock_load_state_dict_from_url.assert_called_with("https://example.com/checkpoint.pth")

        # Verify that nn.Module's load_state_dict was called with the expected arguments
        expected_state_dict = {
            k: v
            for k, v in mock_state_dict.items()
            if k
            not in [
                "image_encoder.norm_head.weight",
                "image_encoder.norm_head.bias",
                "image_encoder.head.weight",
                "image_encoder.head.bias",
            ]
        }
        mock_module_load_state_dict.assert_called_with(expected_state_dict, True, False)

    def test_load_state_dict_failure(self, mocker) -> None:
        sam = SAM(data_input_params=DataInputParams((1024, 1024), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)))

        with pytest.raises(ValueError, match="Invalid checkpoint type or format: <class 'str'>: invalid_url"):
            sam.load_state_dict(checkpoint="invalid_url")

    @pytest.mark.parametrize("freeze_image_encoder", [True, False])
    @pytest.mark.parametrize("freeze_prompt_encoder", [True, False])
    @pytest.mark.parametrize("freeze_mask_decoder", [True, False])
    def test_freeze_networks(
        self,
        freeze_image_encoder: bool,
        freeze_prompt_encoder: bool,
        freeze_mask_decoder: bool,
    ) -> None:
        class MockModel:
            def __init__(self):
                self.image_encoder = nn.Linear(10, 10)
                self.prompt_encoder = nn.Linear(10, 10)
                self.mask_decoder = nn.Linear(10, 10)

        sam = SAM(data_input_params=DataInputParams((1024, 1024), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)))

        sam.freeze_networks(
            freeze_image_encoder=freeze_image_encoder,
            freeze_prompt_encoder=freeze_prompt_encoder,
            freeze_mask_decoder=freeze_mask_decoder,
        )

        for param in sam.model.image_encoder.parameters():
            assert param.requires_grad != freeze_image_encoder

        for param in sam.model.prompt_encoder.parameters():
            assert param.requires_grad != freeze_prompt_encoder

        for param in sam.model.mask_decoder.parameters():
            assert param.requires_grad != freeze_mask_decoder

    def test_forward_for_tracing(self, mocker) -> None:
        sam = SAM(data_input_params=DataInputParams((1024, 1024), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)))
        mock_forward_for_tracing = mocker.patch.object(sam.model, "forward_for_tracing")

        image_embeddings = torch.zeros((1, 256, 64, 64))
        point_coords = torch.zeros((1, 10, 2))
        point_labels = torch.zeros((1, 10))
        mask_input = torch.zeros((1, 1, 256, 256))
        has_mask_input = torch.zeros((1, 1))
        ori_shape = torch.zeros((1, 2))

        sam.forward_for_tracing(
            image_embeddings=image_embeddings,
            point_coords=point_coords,
            point_labels=point_labels,
            mask_input=mask_input,
            has_mask_input=has_mask_input,
            ori_shape=ori_shape,
        )

        mock_forward_for_tracing.assert_called_once_with(
            image_embeddings=image_embeddings,
            point_coords=point_coords,
            point_labels=point_labels,
            mask_input=mask_input,
            has_mask_input=has_mask_input,
            ori_shape=ori_shape,
        )
