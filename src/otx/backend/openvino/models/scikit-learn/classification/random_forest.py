# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Class definition for the Random Forest classifier with OpenVINO optimization for OTX."""

from sklearnex import patch_sklearn
import joblib
from time import time
from sklearn.ensemble import RandomForestClassifier as SkModel
from sklearn.metrics import accuracy_score

class RandomForestClassifier:
    def __init__(self, *args, use_openvino=True, **kwargs):
        """
        Initialize the RandomForestClassifier wrapper.

        Args:
            *args: Positional arguments for sklearn's RandomForestClassifier.
            use_openvino (bool): Whether to enable OpenVINO optimizations.
            **kwargs: Keyword arguments for sklearn's RandomForestClassifier.
        """
        self.use_openvino = use_openvino
        self._patched = False

        if self.use_openvino:
            try:
                patch_sklearn()
                self._patched = True
                print("✅ sklearnex patch applied successfully.")
            except Exception:
                print("⚠️ sklearnex patch failed.")

        self.model = SkModel(*args, **kwargs)
        print("📦 RandomForestClassifier model initialized.")

        if self.use_openvino:
            self._warn_if_not_fully_supported(**kwargs)

    def _warn_if_not_fully_supported(self, **kwargs):
        """
        Warns if any parameter is not fully supported for OpenVINO optimization.

        Args:
            **kwargs: Keyword arguments passed to the model.
        """
        unsupported = []
        if kwargs.get("criterion", "gini") != "gini":
            unsupported.append("criterion ≠ 'gini'")
        if kwargs.get("ccp_alpha", 0) != 0:
            unsupported.append("ccp_alpha ≠ 0")
        if kwargs.get("warm_start", False):
            unsupported.append("warm_start = True")
        if unsupported:
            print("⚠️ The following parameters are not supported by OpenVINO optimization and may fall back to sklearn:")
            for u in unsupported:
                print(f"   - {u}")

    def fit(self, X, y):
        """
        Fit the random forest classifier.

        Args:
            X (array-like): Training data.
            y (array-like): Target values.
        """
        start = time()
        self.model.fit(X, y)
        elapsed = time() - start
        print(f"🚀 Training completed in {elapsed:.4f} seconds.")

    def predict(self, X):
        """
        Predict class labels for samples in X.

        Args:
            X (array-like): Input data.

        Returns:
            array: Predicted class labels.
        """
        return self.model.predict(X)

    def predict_proba(self, X):
        """
        Probability estimates for samples in X.

        Args:
            X (array-like): Input data.

        Returns:
            array: Probability estimates.
        """
        return self.model.predict_proba(X)

    def score(self, X, y):
        """
        Returns the mean accuracy on the given test data and labels.

        Args:
            X (array-like): Test samples.
            y (array-like): True labels for X.

        Returns:
            float: Mean accuracy.
        """
        acc = self.model.score(X, y)
        print(f"📊 Model score: {acc:.4f}")
        return acc

    def evaluate(self, X, y):
        """
        Evaluate the model and print inference time.

        Args:
            X (array-like): Test samples.
            y (array-like): True labels for X.

        Returns:
            float: Mean accuracy.
        """
        start = time()
        y_pred = self.predict(X)
        elapsed = time() - start
        acc = accuracy_score(y, y_pred)
        print(f"📈 Accuracy: {acc:.4f} | Inference time: {elapsed:.4f} seconds.")
        return acc

    def save_model(self, path="rf_model.joblib"):
        """
        Save the trained model to a file.

        Args:
            path (str): Path to save the model.
        """
        joblib.dump(self.model, path)
        print(f"💾 Model saved to {path}")

    def load_model(self, path="rf_model.joblib"):
        """
        Load a model from a file.

        Args:
            path (str): Path to the saved model.
        """
        self.model = joblib.load(path)
        print(f"📂 Model loaded from {path}")

    def convert_to_ir(self, X_train, model_name="rf_model"):
        """
        Not supported: Exporting RandomForestClassifier to IR via neural network is not possible.

        Args:
            X_train (array-like): Training data (unused).
            model_name (str): Model name (unused).
        """
        print("❌ Export to IR via neural network is not supported for RandomForestClassifier.")