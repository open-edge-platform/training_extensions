# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for PyTorch RegressionWrapper with OpenVINO optimization for OTX."""

import unittest
import torch
import sys
import os
import numpy as np

from otx.backend.openvino.models.pytorch import RegressionWrapper

class DummyRegressor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(5, 1)
    def forward(self, x):
        return self.fc(x)

class TestRegressionWrapper(unittest.TestCase):
    def setUp(self):
        self.model = DummyRegressor()
        self.wrapper = RegressionWrapper(self.model)
        self.x = torch.randn(20, 5)
        self.y = torch.randn(20, 1)
        self.loader = torch.utils.data.DataLoader(list(zip(self.x, self.y)), batch_size=4)
        self.criterion = torch.nn.MSELoss()
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