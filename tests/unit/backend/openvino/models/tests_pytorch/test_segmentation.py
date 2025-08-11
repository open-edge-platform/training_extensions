# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for PyTorch SegmentationWrapper with OpenVINO optimization for OTX."""

import unittest
import torch

from otx.backend.openvino.models.pytorch import SegmentationWrapper

class DummySegmentation(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 2, kernel_size=1)
    def forward(self, x):
        return self.conv(x)

class TestSegmentationWrapper(unittest.TestCase):
    def setUp(self):
        self.model = DummySegmentation()
        self.wrapper = SegmentationWrapper(self.model)
        self.x = torch.randn(8, 3, 32, 32)
        self.y = torch.randint(0, 2, (8, 32, 32))
        self.loader = torch.utils.data.DataLoader(list(zip(self.x, self.y)), batch_size=2)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.wrapper.model.parameters())

    def test_fit_and_score(self):
        self.wrapper.fit(self.loader, self.criterion, self.optimizer, num_epochs=1)
        score = self.wrapper.score(self.loader)
        self.assertIsInstance(score, float)

    def test_predict(self):
        preds = self.wrapper.predict(self.x)
        self.assertTrue(isinstance(preds, torch.Tensor) or hasattr(preds, "__array__"))

if __name__ == "__main__":
    unittest.main()