# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit test for IncrementalPCA model wrapper with OpenVINO optimization for OTX."""

import unittest
from timeit import default_timer as timer
import numpy as np
from sklearnex import unpatch_sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from otx.backend.openvino.models.sklearn import IncrementalPCA

class TestIncrementalPCA(unittest.TestCase):
    def test_incremental_pca(self):
        x, y = load_iris(return_X_y=True)
        x_train, x_test, _, _ = train_test_split(x, y, test_size=0.2, random_state=42)
        params = {"n_components": 2}
        model = IncrementalPCA(**params, use_openvino=True)
        start = timer()
        model.fit(x_train)
        patched_time = timer() - start
        patched_trans = model.transform(x_test)
        unpatch_sklearn()
        from sklearn.decomposition import IncrementalPCA as SkIPCA
        base_model = SkIPCA(**params)
        start = timer()
        base_model.fit(x_train)
        base_time = timer() - start
        base_trans = base_model.transform(x_test)
        print(f"Patched (sklearnex) fit time: {patched_time:.4f}s")
        print(f"Original sklearn fit time: {base_time:.4f}s")
        print(f"Speedup: {base_time/patched_time:.1f}x")
        np.testing.assert_allclose(patched_trans, base_trans, rtol=1e-2, atol=1e-2)
if __name__ == "__main__":
    unittest.main()
