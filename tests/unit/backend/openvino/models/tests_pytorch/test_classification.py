# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for PyTorch ClassificationWrapper with OpenVINO optimization for OTX."""

import unittest
import torch
import numpy as np

from otx.backend.openvino.models.pytorch import ClassificationWrapper

class DummyClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(8, 3)
    def forward(self, x):
        return self.fc(x)

class TestClassificationWrapper(unittest.TestCase):
    def setUp(self):
        self.model = DummyClassifier()
        self.wrapper = ClassificationWrapper(self.model)
        self.x = torch.randn(16, 8)
        self.y = torch.randint(0, 3, (16,))
        self.loader = torch.utils.data.DataLoader(list(zip(self.x, self.y)), batch_size=4)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.wrapper.model.parameters())

    def test_fit_and_score(self):
        self.wrapper.fit(self.loader, self.criterion, self.optimizer, num_epochs=1)
        acc = self.wrapper.score(self.loader)
        self.assertIsInstance(acc, float)

    def test_predict(self):
        preds = self.wrapper.predict(self.x)
        self.assertTrue(isinstance(preds, torch.Tensor) or hasattr(preds, "__array__"))

if __name__ == "__main__":
    unittest.main()