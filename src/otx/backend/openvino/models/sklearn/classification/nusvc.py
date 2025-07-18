# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Class definition for the Nu-Support Vector Classifier with OpenVINO optimization for OTX."""

import joblib
import numpy as np
from time import time
from sklearnex.svm import NuSVC as SkNuSVC
from sklearn.metrics import accuracy_score
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

class NuSVC:
    def __init__(self, *args, use_openvino=True, **kwargs):
        """
        Initialize the NuSVC wrapper.

        Args:
            *args: Positional arguments for sklearn's NuSVC.
            use_openvino (bool): Whether to enable OpenVINO optimizations.
            **kwargs: Keyword arguments for sklearn's NuSVC.
        """
        self.use_openvino = use_openvino
        self._ir_model = None
        self.model = SkNuSVC(*args, **kwargs)
        print("📦 NuSVC model initialized (sklearnex version).")

    def fit(self, X, y):
        """
        Fit the NuSVC model.

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

        Raises:
            AttributeError: If probability estimates are not supported.
        """
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)
        else:
            raise AttributeError("This NuSVC model does not support probability estimates.")

    def score(self, X, y):
        """
        Returns the mean accuracy on the given test data and labels.

        Args:
            X (array-like): Test samples.
            y (array-like): True labels for X.

        Returns:
            float: Mean accuracy.
        """
        acc = accuracy_score(y, self.predict(X))
        print(f"📊 Accuracy: {acc:.4f}")
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
        acc = self.score(X, y)
        elapsed = time() - start
        print(f"📈 Inference time: {elapsed:.4f} seconds.")
        return acc

    def save_model(self, path="nusvc_model.joblib"):
        """
        Save the trained model to a file.

        Args:
            path (str): Path to save the model.
        """
        joblib.dump(self.model, path)
        print(f"💾 Model saved to {path}")

    def load_model(self, path="nusvc_model.joblib"):
        """
        Load a model from a file.

        Args:
            path (str): Path to the saved model.
        """
        self.model = joblib.load(path)
        print(f"📂 Model loaded from {path}")

    def convert_to_ir(self, X_train, model_name="nusvc_model"):
        """
        Not supported: Exporting NuSVC to IR via neural network is not possible.

        Args:
            X_train (array-like): Training data (unused).
            model_name (str): Model name (unused).
        """
        print("❌ Export to IR via neural network is not supported for NuSVC.")