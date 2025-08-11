# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit test for TSNE model wrapper with OpenVINO optimization for OTX."""

import unittest
from timeit import default_timer as timer
import numpy as np
from sklearnex import unpatch_sklearn
from sklearn.datasets import load_iris
from otx.backend.openvino.models.sklearn import TSNE

class TestTSNE(unittest.TestCase):
    def test_tsne(self):
        x, y = load_iris(return_X_y=True)
        params = {"n_components": 2, "perplexity": 5, "n_iter": 250, "random_state": 42}
        model = TSNE(**params, use_openvino=True)
        start = timer()
        patched_trans = model.fit_transform(x)
        patched_time = timer() - start
        unpatch_sklearn()
        from sklearn.manifold import TSNE as SkTSNE
        base_model = SkTSNE(**params)
        start = timer()
        base_trans = base_model.fit_transform(x)
        base_time = timer() - start
        print(f"Patched (sklearnex) fit time: {patched_time:.4f}s")
        print(f"Original sklearn fit time: {base_time:.4f}s")
        print(f"Speedup: {base_time/patched_time:.1f}x")
        self.assertEqual(patched_trans.shape, base_trans.shape)
if __name__ == "__main__":
    unittest.main()
