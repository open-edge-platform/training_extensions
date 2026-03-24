# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Implementation of ShuffleNetV2.

Original paper:
- 'ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design,'
  https://arxiv.org/abs/1807.11164.
"""

from __future__ import annotations

import math
from typing import Any, ClassVar

import torch
from torch import nn


def channel_shuffle(x: torch.Tensor, groups: int) -> torch.Tensor:
    """Channel shuffle operation from ShuffleNet.

    Rearranges channels so that information flows between groups.

    Args:
        x: Input tensor of shape (batch, channels, height, width).
        groups: Number of groups for the shuffle.

    Returns:
        Tensor with shuffled channels.
    """
    batch_size, num_channels, height, width = x.size()
    channels_per_group = num_channels // groups
    x = x.view(batch_size, groups, channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    return x.view(batch_size, -1, height, width)


class InvertedResidual(nn.Module):
    """ShuffleNetV2 building block (inverted residual with channel shuffle).

    When stride=1: channel split -> right branch (1x1 conv, DW conv, 1x1 conv) -> concat -> channel shuffle.
    When stride=2: both branches active -> concat -> channel shuffle (doubles channels).

    Args:
        inp: Number of input channels.
        oup: Number of output channels.
        stride: Stride for depthwise convolution (1 or 2).
    """

    def __init__(self, inp: int, oup: int, stride: int) -> None:
        super().__init__()

        if stride not in (1, 2):
            msg = f"Stride must be 1 or 2, got {stride}"
            raise ValueError(msg)

        self.stride = stride
        branch_features = oup // 2

        if self.stride == 2:
            # Left branch: depthwise conv + 1x1 conv (processes all input channels)
            self.branch1 = nn.Sequential(
                # Depthwise conv
                nn.Conv2d(inp, inp, kernel_size=3, stride=self.stride, padding=1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                # Pointwise conv
                nn.Conv2d(inp, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(branch_features),
                nn.ReLU(inplace=True),
            )
        else:
            self.branch1 = nn.Sequential()

        # Right branch
        inp_right = inp if self.stride == 2 else branch_features
        self.branch2 = nn.Sequential(
            # Pointwise conv
            nn.Conv2d(inp_right, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
            # Depthwise conv
            nn.Conv2d(branch_features, branch_features, kernel_size=3, stride=self.stride, padding=1, groups=branch_features, bias=False),
            nn.BatchNorm2d(branch_features),
            # Pointwise conv
            nn.Conv2d(branch_features, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)
            out = torch.cat((x1, self.branch2(x2)), dim=1)
        else:
            out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)
        return channel_shuffle(out, 2)


class ShuffleNetV2FeatureExtractor(nn.Module):
    """ShuffleNetV2 feature extractor.

    Args:
        stages_repeats: Number of repeats for each stage.
        stages_out_channels: Output channels for each stage.
            [input_channels, stage2, stage3, stage4, conv5].
        input_size: Spatial input size as (H, W).
    """

    def __init__(
        self,
        stages_repeats: list[int],
        stages_out_channels: list[int],
        input_size: tuple[int, int] = (224, 224),
    ) -> None:
        super().__init__()

        if len(stages_repeats) != 3:
            msg = f"Expected 3 stage repeats, got {len(stages_repeats)}"
            raise ValueError(msg)
        if len(stages_out_channels) != 5:
            msg = f"Expected 5 stage output channel counts, got {len(stages_out_channels)}"
            raise ValueError(msg)

        self.in_size = input_size
        input_channels = 3
        output_channels = stages_out_channels[0]

        stride = 1 if input_size[0] < 100 else 2

        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Build stages
        self.stage2: nn.Sequential
        self.stage3: nn.Sequential
        self.stage4: nn.Sequential
        input_channels = output_channels
        for i, (repeats, out_ch) in enumerate(
            zip(stages_repeats, stages_out_channels[1:4]),
        ):
            seq = [InvertedResidual(input_channels, out_ch, stride=2)]
            for _ in range(repeats - 1):
                seq.append(InvertedResidual(out_ch, out_ch, stride=1))
            setattr(self, f"stage{i + 2}", nn.Sequential(*seq))
            input_channels = out_ch

        output_channels = stages_out_channels[-1]
        self.conv5 = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )

        # Collect feature layers into a single Sequential for compatibility
        self.features = nn.Sequential(
            self.conv1,
            self.maxpool,
            self.stage2,
            self.stage3,
            self.stage4,
            self.conv5,
        )

        self._initialize_weights()

    def extract_features(self, x: torch.Tensor) -> tuple[torch.Tensor]:
        """Extract features from input tensor."""
        y = self.features(x)
        return (y,)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor]:
        """Forward pass."""
        return self.extract_features(x)

    def _initialize_weights(self) -> None:
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


class ShuffleNetV2Backbone:
    """Factory class for ShuffleNetV2 backbone.

    Creates a ShuffleNetV2FeatureExtractor based on the specified model variant.

    Args:
        model_name: Model variant name. One of "shufflenetv2_small" (x0.5) or
            "shufflenetv2_large" (x1.0).
        input_size: Spatial input size as (H, W). Defaults to (224, 224).

    Returns:
        ShuffleNetV2FeatureExtractor instance.
    """

    SHUFFLENETV2_CFG: ClassVar[dict[str, Any]] = {
        "shufflenetv2_small": {
            # x0.5 variant
            "stages_repeats": [4, 8, 4],
            "stages_out_channels": [24, 48, 96, 192, 1024],
            "out_channels": 1024,
        },
        "shufflenetv2_large": {
            # x1.0 variant
            "stages_repeats": [4, 8, 4],
            "stages_out_channels": [24, 116, 232, 464, 1024],
            "out_channels": 1024,
        },
    }

    def __new__(
        cls,
        model_name: str = "shufflenetv2_large",
        input_size: tuple[int, int] = (224, 224),
        **kwargs,
    ) -> ShuffleNetV2FeatureExtractor:
        """Create a new ShuffleNetV2 feature extractor.

        Args:
            model_name: Model variant name. Defaults to "shufflenetv2_large".
            input_size: Spatial input size. Defaults to (224, 224).
            **kwargs: Additional keyword arguments (unused, for API compatibility).

        Returns:
            A ShuffleNetV2FeatureExtractor instance.
        """
        if model_name not in cls.SHUFFLENETV2_CFG:
            msg = f"Unknown ShuffleNetV2 model: {model_name}. Available: {list(cls.SHUFFLENETV2_CFG.keys())}"
            raise ValueError(msg)

        cfg = cls.SHUFFLENETV2_CFG[model_name]
        return ShuffleNetV2FeatureExtractor(
            stages_repeats=cfg["stages_repeats"],
            stages_out_channels=cfg["stages_out_channels"],
            input_size=input_size,
        )
