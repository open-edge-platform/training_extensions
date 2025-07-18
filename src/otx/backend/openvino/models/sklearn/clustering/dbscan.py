# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Class definition for the DBSCAN clustering model with OpenVINO optimization for OTX."""

import joblib
from time import time
from sklearnex.cluster import DBSCAN as SkDBSCAN

class DBSCAN:
    def __init__(self, *args, use_openvino=True, **kwargs):
        """
        Initialize the DBSCAN wrapper.

        Args:
            *args: Positional arguments for sklearn's DBSCAN.
            use_openvino (bool): Whether to enable OpenVINO optimizations.
            **kwargs: Keyword arguments for sklearn's DBSCAN.
        """
        self.use_openvino = use_openvino
        self._ir_model = None

        self.model = SkDBSCAN(*args, **kwargs)
        print("📦 DBSCAN model initialized (sklearnex version).")

    def fit(self, X, y=None):
        """
        Fit the DBSCAN model.

        Args:
            X (array-like): Training data.
            y (ignored): Not used, present for API consistency.
        """
        start = time()
        self.model.fit(X)
        elapsed = time() - start
        print(f"🚀 Training completed in {elapsed:.4f} seconds.")

    def predict(self, X):
        """
        Predict cluster labels for samples in X.

        Args:
            X (array-like): Input data.

        Returns:
            array: Predicted labels.
        """
        return self.model.fit_predict(X)

    def evaluate(self, X):
        """
        Evaluate the model by predicting cluster labels for X.

        Args:
            X (array-like): Input data.

        Returns:
            array: Predicted labels.
        """
        labels = self.predict(X)
        print(f"📊 Predicted labels: {labels[:10]} ...")
        return labels

    def save_model(self, path="dbscan_model.joblib"):
        """
        Save the trained model to a file.

        Args:
            path (str): Path to save the model.
        """
        joblib.dump(self.model, path)
        print(f"💾 Model saved to {path}")

    def load_model(self, path="dbscan_model.joblib"):
        """
        Load a model from a file.

        Args:
            path (str): Path to the saved model.
        """
        self.model = joblib.load(path)
        print(f"📂 Model loaded from {path}")

    def _check_export_support(self, X_train=None):
        """
        Check if the model and input are supported for ONNX/IR conversion.

        Args:
            X_train (array-like): Training data.

        Raises:
            ValueError: If input is a sparse matrix.
        """
        if hasattr(X_train, "toarray"):
            raise ValueError("❌ Sparse matrix input is not supported for ONNX/IR conversion.")
        print("✅ Model and input are supported for ONNX/IR conversion.")

    def convert_to_ir(self, X_train, model_name="dbscan_model"):
        """
        Export DBSCAN to OpenVINO IR (not implemented).

        Args:
            X_train (array-like): Training data for shape reference.
            model_name (str): Base name for the exported model files.
        """
        self._check_export_support(X_train)
        print("⚠️ DBSCAN does not support ONNX/IR conversion directly. Skipping export.")