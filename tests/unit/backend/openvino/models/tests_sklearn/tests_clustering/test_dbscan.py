# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit test for DBSCAN model wrapper with OpenVINO optimization for OTX."""

import unittest
from timeit import default_timer as timer
from sklearnex import unpatch_sklearn
from sklearn.datasets import load_iris
from otx.backend.openvino.models.sklearn import DBSCAN

class TestDBSCAN(unittest.TestCase):
    def test_dbscan(self):
        x, y = load_iris(return_X_y=True)
        params = {"eps": 0.5, "min_samples": 5}
        model = DBSCAN(**params, use_openvino=True)
        start = timer()
        model.fit(x)
        patched_time = timer() - start
        patched_pred = model.predict(x)
        unpatch_sklearn()
        from sklearn.cluster import DBSCAN as SkDBSCAN
        base_model = SkDBSCAN(**params)
        start = timer()
        base_pred = base_model.fit_predict(x)
        base_time = timer() - start
        print(f"Patched (sklearnex) fit time: {patched_time:.4f}s")
        print(f"Original sklearn fit time: {base_time:.4f}s")
        print(f"Speedup: {base_time/patched_time:.1f}x")
        self.assertEqual(len(patched_pred), len(base_pred))
if __name__ == "__main__":
    unittest.main()
