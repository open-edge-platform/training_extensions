# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit test for KNeighborsClassifier model wrapper with OpenVINO optimization for OTX."""

import unittest
from timeit import default_timer as timer
from sklearnex import unpatch_sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn import metrics
from otx.backend.openvino.models.sklearn import KNeighborsClassifier

class TestKNeighborsClassifier(unittest.TestCase):
    def test_kneighbors_classifier(self):
        x, y = load_iris(return_X_y=True)
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
        params = {"n_neighbors": 5, "n_jobs": -1}
        model = KNeighborsClassifier(**params, use_openvino=True)
        start = timer()
        model.fit(x_train, y_train)
        patched_time = timer() - start
        patched_pred = model.predict(x_test)
        patched_acc = metrics.accuracy_score(y_test, patched_pred)
        unpatch_sklearn()
        from sklearn.neighbors import KNeighborsClassifier as SkKNC
        base_model = SkKNC(**params)
        start = timer()
        base_model.fit(x_train, y_train)
        base_time = timer() - start
        base_pred = base_model.predict(x_test)
        base_acc = metrics.accuracy_score(y_test, base_pred)
        print(f"Patched (sklearnex) fit time: {patched_time:.4f}s | Acc: {patched_acc:.4f}")
        print(f"Original sklearn fit time: {base_time:.4f}s | Acc: {base_acc:.4f}")
        print(f"Speedup: {base_time/patched_time:.1f}x")
        self.assertAlmostEqual(patched_acc, base_acc, delta=0.01)
if __name__ == "__main__":
    unittest.main()
