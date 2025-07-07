# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""K-Nearest Neighbors Classifier model wrapper with OpenVINO optimization for OTX."""

from sklearnex import patch_sklearn
import joblib
from time import time
from sklearn.neighbors import KNeighborsClassifier as SkModel
from sklearn.metrics import classification_report, accuracy_score

class KNeighborsClassifier:
    def __init__(self, *args, use_openvino=True, **kwargs):
        """
        Initialize the KNeighborsClassifier wrapper.

        Args:
            *args: Positional arguments for sklearn's KNeighborsClassifier.
            use_openvino (bool): Whether to enable OpenVINO optimizations.
            **kwargs: Keyword arguments for sklearn's KNeighborsClassifier.
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
        print("📦 KNeighborsClassifier model initialized.")

    def fit(self, X, y):
        """
        Fit the KNeighborsClassifier model.

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
        Evaluate the model and print inference time and classification report.

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
        print(classification_report(y, y_pred))
        return acc

    def save_model(self, path="knn_model.joblib"):
        """
        Save the trained model to a file.

        Args:
            path (str): Path to save the model.
        """
        joblib.dump(self.model, path)
        print(f"💾 Model saved to {path}")

    def load_model(self, path="knn_model.joblib"):
        """
        Load a model from a file.

        Args:
            path (str): Path to the saved model.
        """
        self.model = joblib.load(path)
        print(f"📂 Model loaded from {path}")

    def convert_to_ir(self, X_train, model_name="knn_model"):
        """
        Not supported: Exporting KNeighborsClassifier to IR via neural network is not possible.

        Args:
            X_train (array-like): Training data (unused).
            model_name (str): Model name (unused).
        """
        print("❌ Export to IR via neural network is not supported for KNeighborsClassifier.")