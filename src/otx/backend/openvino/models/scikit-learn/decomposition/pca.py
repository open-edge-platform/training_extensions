# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Class definition for the PCA model with OpenVINO optimization for OTX."""

from sklearnex import patch_sklearn
import joblib
from time import time
from sklearn.decomposition import PCA as SkPCA

class PCA:
    def __init__(self, *args, use_openvino=True, **kwargs):
        """
        Initialize the PCA wrapper.

        Args:
            *args: Positional arguments for sklearn's PCA.
            use_openvino (bool): Whether to enable OpenVINO optimizations.
            **kwargs: Keyword arguments for sklearn's PCA.
        """
        self.use_openvino = use_openvino
        self._patched = False
        self._ir_model = None

        if self.use_openvino:
            try:
                patch_sklearn()
                self._patched = True
                print("✅ sklearnex patch applied successfully.")
            except ImportError:
                print("⚠️ sklearnex is not installed. Running without OpenVINO optimization.")

        self.model = SkPCA(*args, **kwargs)
        print("📦 PCA model initialized.")

    def fit(self, X, y=None):
        """
        Fit the PCA model.

        Args:
            X (array-like): Training data.
            y (ignored): Not used, present for API consistency.
        """
        start = time()
        self.model.fit(X)
        elapsed = time() - start
        print(f"🚀 Training completed in {elapsed:.4f} seconds.")

    def transform(self, X):
        """
        Apply the dimensionality reduction on X.

        Args:
            X (array-like): Input data.

        Returns:
            array: Transformed data.
        """
        return self.model.transform(X)

    def evaluate(self, X):
        """
        Evaluate the model by transforming X.

        Args:
            X (array-like): Input data.

        Returns:
            array: Transformed data.
        """
        X_trans = self.transform(X)
        print(f"📊 Transformed shape: {X_trans.shape}")
        return X_trans

    def save_model(self, path="pca_model.joblib"):
        """
        Save the trained model to a file.

        Args:
            path (str): Path to save the model.
        """
        joblib.dump(self.model, path)
        print(f"💾 Model saved to {path}")

    def load_model(self, path="pca_model.joblib"):
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

    def convert_to_ir(self, X_train, model_name="pca_model"):
        """
        Export PCA to OpenVINO IR (not implemented).

        Args:
            X_train (array-like): Training data for shape reference.
            model_name (str): Base name for the exported model files.
        """
        self._check_export_support(X_train)
        print("⚠️ PCA OpenVINO IR export is not implemented yet.")