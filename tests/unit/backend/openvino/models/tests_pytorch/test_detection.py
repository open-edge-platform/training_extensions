# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for PyTorch DetectionWrapper with OpenVINO optimization for OTX."""

import unittest
import torch

from otx.backend.openvino.models.pytorch import DetectionWrapper

class DummyDetection(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(10, 4)
    def forward(self, x):
        return self.fc(x)

class TestDetectionWrapper(unittest.TestCase):
    def setUp(self):
        self.model = DummyDetection()
        self.wrapper = DetectionWrapper(self.model)
        self.x = torch.randn(12, 10)
        self.y = torch.randn(12, 4)
        self.loader = torch.utils.data.DataLoader(list(zip(self.x, self.y)), batch_size=3)
        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.wrapper.model.parameters())

    def test_fit_and_score(self):
        self.wrapper.fit(self.loader, self.criterion, self.optimizer, num_epochs=1)
        # Use MSE as metric for detection/regression tasks
        def mse_metric(preds, targets):
            return torch.nn.functional.mse_loss(preds, targets).item()
        score = self.wrapper.score(self.loader, metric_fn=mse_metric)
        self.assertIsInstance(score, float)

    def test_predict(self):
        preds = self.wrapper.predict(self.x)
        self.assertTrue(isinstance(preds, torch.Tensor) or hasattr(preds, "__array__"))

if __name__ == "__main__":
    unittest.main()