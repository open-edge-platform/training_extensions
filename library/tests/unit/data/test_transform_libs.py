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
from otx.data.transform_libs.torchvision import (
    TorchVisionTransformLib,
)


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
                transforms=[instantiate_class(args=(), init=transform) for transform in created.transforms],
            )
        return created

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

    @pytest.fixture
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
