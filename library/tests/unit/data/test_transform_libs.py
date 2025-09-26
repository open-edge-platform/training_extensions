# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytest
import torch
from lightning.pytorch.cli import instantiate_class
from omegaconf import OmegaConf
from torchvision.transforms import v2

from otx.config.data import SubsetConfig
from otx.data.dataset.classification import HLabelInfo
from otx.data.dataset.instance_segmentation import OTXInstanceSegDataset
from otx.data.transform_libs.torchvision import (
    TorchVisionTransformLib,
)
from otx.types.image import ImageColorChannel


class TestTorchVisionTransformLib:
    @pytest.fixture(params=["from_dict", "from_obj", "from_compose"])
    def fxt_config(self, request) -> list[dict[str, Any]]:
        if request.param == "from_compose":
            return v2.Compose(
                [
                    v2.RandomResizedCrop(size=(224, 224), antialias=True),
                    v2.RandomHorizontalFlip(p=0.5),
                    v2.ToDtype(torch.float32),
                    v2.Normalize(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]),
                ],
            )
        prefix = "torchvision.transforms.v2"
        cfg = f"""
        transforms:
          - class_path: {prefix}.RandomResizedCrop
            init_args:
                size: [224, 224]
                antialias: True
          - class_path: {prefix}.RandomHorizontalFlip
            init_args:
                p: 0.5
          - class_path: {prefix}.ToDtype
            init_args:
                dtype: ${{as_torch_dtype:torch.float32}}
          - class_path: {prefix}.Normalize
            init_args:
                mean: [123.675, 116.28, 103.53]
                std: [58.395, 57.12, 57.375]
        """
        created = OmegaConf.create(cfg)
        if request.param == "from_obj":
            return SubsetConfig(
                batch_size=1,
                subset_name="dummy",
                transforms=[instantiate_class(args=(), init=transform) for transform in created.transforms],
            )
        return created

    def test_transform(
        self,
        mocker,
        fxt_config,
        fxt_dataset_and_data_entity_cls,
        fxt_mock_dm_subset,
        fxt_mock_hlabelinfo,
    ) -> None:
        transform = TorchVisionTransformLib.generate(fxt_config)
        assert isinstance(transform, v2.Compose)

        dataset_cls, data_entity_cls, kwargs = fxt_dataset_and_data_entity_cls
        if dataset_cls == OTXInstanceSegDataset:
            pytest.skip(
                "Instance segmentation task are not suitible for torchvision transform",
            )
        mocker.patch.object(HLabelInfo, "from_dm_label_groups", return_value=fxt_mock_hlabelinfo)
        dataset = dataset_cls(
            dm_subset=fxt_mock_dm_subset,
            transforms=transform,
            **kwargs,
        )
        dataset.num_classes = 1

        item = dataset[0]
        assert isinstance(item, data_entity_cls)

    def test_transform_enable_flag(self) -> None:
        prefix = "torchvision.transforms.v2"
        cfg_str = f"""
        transforms:
          - class_path: {prefix}.RandomResizedCrop
            init_args:
                size: [224, 224]
                antialias: True
          - class_path: {prefix}.RandomHorizontalFlip
            init_args:
                p: 0.5
          - class_path: {prefix}.ToDtype
            init_args:
                dtype: ${{as_torch_dtype:torch.float32}}
          - class_path: {prefix}.Normalize
            init_args:
                mean: [123.675, 116.28, 103.53]
                std: [58.395, 57.12, 57.375]
        """
        cfg_org = OmegaConf.create(cfg_str)

        cfg = deepcopy(cfg_org)
        cfg.transforms[0].enable = False  # Remove 1st
        transform = TorchVisionTransformLib.generate(cfg)
        assert len(transform.transforms) == 3
        assert "RandomResizedCrop" not in repr(transform)

        cfg = deepcopy(cfg_org)
        cfg.transforms[1].enable = False  # Remove 2nd
        transform = TorchVisionTransformLib.generate(cfg)
        assert len(transform.transforms) == 3
        assert "RandomHorizontalFlip" not in repr(transform)

        cfg = deepcopy(cfg_org)
        cfg.transforms[2].enable = True  # No effect
        transform = TorchVisionTransformLib.generate(cfg)
        assert len(transform.transforms) == 4
        assert "ToDtype" in repr(transform)

    @pytest.fixture()
    def fxt_config_w_input_size(self) -> list[dict[str, Any]]:
        cfg = """
        input_size:
        - 300
        - 200
        transforms:
          - class_path: otx.data.transform_libs.torchvision.RandomResize
            init_args:
                scale: $(input_size) * 0.5
          - class_path: otx.data.transform_libs.torchvision.RandomCrop
            init_args:
                crop_size: $(input_size)
          - class_path: otx.data.transform_libs.torchvision.RandomResize
            init_args:
                scale: $(input_size) * 1.1
        """
        return OmegaConf.create(cfg)

    def test_configure_input_size(self, fxt_config_w_input_size):
        transform = TorchVisionTransformLib.generate(fxt_config_w_input_size)
        assert isinstance(transform, v2.Compose)
        assert transform.transforms[0].scale == (150, 100)  # RandomResize gets sequence of integer
        assert transform.transforms[1].crop_size == (300, 200)  # RandomCrop gets sequence of integer
        assert transform.transforms[2].scale == (round(300 * 1.1), round(200 * 1.1))  # check round

    def test_configure_input_size_none(self, fxt_config_w_input_size):
        """Check input size is None but transform has $(ipnput_size)."""
        fxt_config_w_input_size.input_size = None
        with pytest.raises(RuntimeError, match="input_size is set to None"):
            TorchVisionTransformLib.generate(fxt_config_w_input_size)

    def test_eval_input_size_str(self):
        assert TorchVisionTransformLib._eval_input_size_str("2") == 2
        assert TorchVisionTransformLib._eval_input_size_str("(2, 3)") == (2, 3)
        assert TorchVisionTransformLib._eval_input_size_str("2*3") == 6
        assert TorchVisionTransformLib._eval_input_size_str("(2, 3) *3") == (6, 9)
        assert TorchVisionTransformLib._eval_input_size_str("(5, 5) / 2") == (2, 2)
        assert TorchVisionTransformLib._eval_input_size_str("(10, 11) * -0.5") == (-5, -6)

    @pytest.mark.parametrize("input_str", ["1+1", "1+-5", "rm fake", "hoho"])
    def test_eval_input_size_str_wrong_value(self, input_str):
        with pytest.raises(SyntaxError):
            assert TorchVisionTransformLib._eval_input_size_str(input_str)

    @pytest.fixture(params=["RGB", "BGR"])
    def fxt_image_color_channel(self, request) -> ImageColorChannel:
        return ImageColorChannel(request.param)

    def test_image_info(
        self,
        mocker,
        fxt_config,
        fxt_dataset_and_data_entity_cls,
        fxt_mock_dm_subset,
        fxt_image_color_channel,
        fxt_mock_hlabelinfo,
    ) -> None:
        transform = TorchVisionTransformLib.generate(fxt_config)
        assert isinstance(transform, v2.Compose)

        dataset_cls, data_entity_cls, kwargs = fxt_dataset_and_data_entity_cls
        if dataset_cls == OTXInstanceSegDataset:
            pytest.skip(
                "Instance segmentation task are not suitible for torchvision transform",
            )
        mocker.patch.object(HLabelInfo, "from_dm_label_groups", return_value=fxt_mock_hlabelinfo)
        dataset = dataset_cls(
            dm_subset=fxt_mock_dm_subset,
            transforms=transform,
            image_color_channel=fxt_image_color_channel,
            **kwargs,
        )
        dataset.num_classes = 1

        item = dataset[0]
        assert item.img_info.img_shape == item.image.shape[1:]

        if fxt_image_color_channel == ImageColorChannel.RGB:
            r_pixel = 255.0 * (0.229 * item.image[0, 0, 0] + 0.485)
            g_pixel = 255.0 * (0.224 * item.image[1, 0, 0] + 0.456)
            b_pixel = 255.0 * (0.225 * item.image[2, 0, 0] + 0.406)
        else:
            b_pixel = 255.0 * (0.229 * item.image[0, 0, 0] + 0.485)
            g_pixel = 255.0 * (0.224 * item.image[1, 0, 0] + 0.456)
            r_pixel = 255.0 * (0.225 * item.image[2, 0, 0] + 0.406)

        assert torch.allclose(r_pixel, torch.tensor(2.0))
        assert torch.allclose(g_pixel, torch.tensor(1.0))
        assert torch.allclose(b_pixel, torch.tensor(0.0))
