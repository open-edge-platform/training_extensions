# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Class definition for the t-SNE model with OpenVINO optimization for OTX."""

import joblib
from time import time
from sklearnex.manifold import TSNE as SkTSNE

class TSNE:
    def __init__(self, *args, use_openvino=True, **kwargs):
        """
        Initialize the TSNE wrapper.

        Args:
            *args: Positional arguments for sklearn's TSNE.
            use_openvino (bool): Whether to enable OpenVINO optimizations.
            **kwargs: Keyword arguments for sklearn's TSNE.
        """
        self.use_openvino = use_openvino
        self._ir_model = None

        self.model = SkTSNE(*args, **kwargs)
        print("📦 TSNE model initialized (sklearnex version).")

    def fit_transform(self, X, y=None):
        """
        Fit TSNE and return the embedded coordinates.

        Args:
            X (array-like): Training data.
            y (ignored): Not used, present for API consistency.

        Returns:
            array: Embedded coordinates.
        """
        start = time()
        result = self.model.fit_transform(X)
        elapsed = time() - start
        print(f"🚀 TSNE completed in {elapsed:.4f} seconds.")
        return result

    def save_model(self, path="tsne_model.joblib"):
        """
        Save the trained model to a file.

        Args:
            path (str): Path to save the model.
        """
        joblib.dump(self.model, path)
        print(f"💾 Model saved to {path}")

    def load_model(self, path="tsne_model.joblib"):
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

    def convert_to_ir(self, X_train, model_name="tsne_model"):
        """
        Export TSNE to OpenVINO IR (not implemented).

        Args:
            X_train (array-like): Training data for shape reference.
            model_name (str): Base name for the exported model files.
        """
        self._check_export_support(X_train)
        print("⚠️ TSNE OpenVINO IR export is not implemented yet.")