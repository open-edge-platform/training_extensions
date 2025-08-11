# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit test for KMeans model wrapper with OpenVINO optimization for OTX."""

import unittest
from timeit import default_timer as timer
from sklearnex import unpatch_sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from otx.backend.openvino.models.sklearn import KMeans

class TestKMeans(unittest.TestCase):
    def test_kmeans(self):
        x, y = load_iris(return_X_y=True)
        x_train, x_test, _, _ = train_test_split(x, y, test_size=0.2, random_state=42)
        params = {"n_clusters": 3, "random_state": 42}
        model = KMeans(**params, use_openvino=True)
        start = timer()
        model.fit(x_train)
        patched_time = timer() - start
        patched_pred = model.predict(x_test)
        unpatch_sklearn()
        from sklearn.cluster import KMeans as SkKMeans
        base_model = SkKMeans(**params)
        start = timer()
        base_model.fit(x_train)
        base_time = timer() - start
        base_pred = base_model.predict(x_test)
        print(f"Patched (sklearnex) fit time: {patched_time:.4f}s")
        print(f"Original sklearn fit time: {base_time:.4f}s")
        print(f"Speedup: {base_time/patched_time:.1f}x")
        self.assertEqual(len(set(patched_pred)), len(set(base_pred)))
if __name__ == "__main__":
    unittest.main()
