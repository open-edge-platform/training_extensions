# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit test for Ridge model wrapper with OpenVINO optimization for OTX."""

import unittest
from timeit import default_timer as timer
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn import metrics

from otx.backend.openvino.models.sklearn import Ridge

class TestRidge(unittest.TestCase):
    def test_ridge(self):
        x, y = fetch_california_housing(return_X_y=True)
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

        params = {}

        model = Ridge(**params, use_openvino=True)
        start = timer()
        model.fit(x_train, y_train)
        patched_time = timer() - start
        patched_pred = model.predict(x_test)
        patched_r2 = metrics.r2_score(y_test, patched_pred)

        from sklearn.linear_model import Ridge as SkRidge
        base_model = SkRidge(**params)
        start = timer()
        base_model.fit(x_train, y_train)
        base_time = timer() - start
        base_pred = base_model.predict(x_test)
        base_r2 = metrics.r2_score(y_test, base_pred)

        print(f"Patched (sklearnex) fit time: {patched_time:.4f}s | R2: {patched_r2:.4f}")
        print(f"Original sklearn fit time: {base_time:.4f}s | R2: {base_r2:.4f}")
        print(f"Speedup: {base_time/patched_time:.1f}x")

        self.assertAlmostEqual(patched_r2, base_r2, delta=0.01)

if __name__ == "__main__":
    unittest.main()
