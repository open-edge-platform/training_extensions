# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Class definition for the Nearest Neighbors model with OpenVINO optimization for OTX."""

from sklearnex import patch_sklearn
import joblib
from time import time
from sklearn.neighbors import NearestNeighbors as SkModel

class NearestNeighbors:
    def __init__(self, *args, use_openvino=True, **kwargs):
        """
        Initialize the NearestNeighbors wrapper.

        Args:
            *args: Positional arguments for sklearn's NearestNeighbors.
            use_openvino (bool): Whether to enable OpenVINO optimizations.
            **kwargs: Keyword arguments for sklearn's NearestNeighbors.
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
        print("📦 NearestNeighbors model initialized.")

    def fit(self, X, y=None):
        """
        Fit the NearestNeighbors model.

        Args:
            X (array-like): Training data.
            y (ignored): Not used, present for API consistency.
        """
        start = time()
        self.model.fit(X)
        elapsed = time() - start
        print(f"🚀 Training completed in {elapsed:.4f} seconds.")

    def kneighbors(self, X, n_neighbors=5, return_distance=True):
        """
        Find the K-neighbors of a point.

        Args:
            X (array-like): The query point or points.
            n_neighbors (int): Number of neighbors to get.
            return_distance (bool): Whether to return distances.

        Returns:
            tuple: (distances, indices) if return_distance else indices
        """
        return self.model.kneighbors(X, n_neighbors=n_neighbors, return_distance=return_distance)

    def save_model(self, path="nearest_neighbors_model.joblib"):
        """
        Save the trained model to a file.

        Args:
            path (str): Path to save the model.
        """
        joblib.dump(self.model, path)
        print(f"💾 Model saved to {path}")

    def load_model(self, path="nearest_neighbors_model.joblib"):
        """
        Load a model from a file.

        Args:
            path (str): Path to the saved model.
        """
        self.model = joblib.load(path)
        print(f"📂 Model loaded from {path}")

    def convert_to_ir(self, X_train, model_name="nearest_neighbors"):
        """
        Not supported: Exporting NearestNeighbors to IR via neural network is not possible.

        Args:
            X_train (array-like): Training data (unused).
            model_name (str): Model name (unused).
        """
        print("❌ Export to IR via neural network is not supported for NearestNeighbors.")