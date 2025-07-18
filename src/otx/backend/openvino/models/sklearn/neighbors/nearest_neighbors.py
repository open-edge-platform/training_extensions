# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Class definition for the Nearest Neighbors model with OpenVINO optimization for OTX."""

import os
import joblib
from time import time
from sklearnex.neighbors import NearestNeighbors as SkModel
from sklearn.metrics import pairwise_distances

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
        self._ir_model = None

        self.model = SkModel(*args, **kwargs)
        print("📦 NearestNeighbors model initialized (sklearnex version).")

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

    def kneighbors(self, X, n_neighbors=None, return_distance=True):
        """
        Find the K-neighbors of a point.

        Args:
            X (array-like): Input data.
            n_neighbors (int, optional): Number of neighbors to get.
            return_distance (bool): Whether to return distances.

        Returns:
            distances, indices: Arrays representing distances and indices of neighbors.
        """
        return self.model.kneighbors(X, n_neighbors=n_neighbors, return_distance=return_distance)

    def score(self, X, y=None):
        """
        Not supported: NearestNeighbors does not have a score method.

        Args:
            X (array-like): Input data.
            y (ignored): Not used.

        Returns:
            None
        """
        print("⚠️ NearestNeighbors does not support scoring.")
        return None

    def evaluate(self, X, n_neighbors=None):
        """
        Evaluate the model by finding neighbors for X.

        Args:
            X (array-like): Input data.
            n_neighbors (int, optional): Number of neighbors to get.

        Returns:
            indices: Indices of neighbors.
        """
        start = time()
        _, indices = self.kneighbors(X, n_neighbors=n_neighbors)
        elapsed = time() - start
        print(f"📈 Neighbors found in {elapsed:.4f} seconds.")
        print(f"🔎 Indices of first 5 queries: {indices[:5]}")
        return indices

    def save_model(self, path="nearestneighbors_model.joblib"):
        """
        Save the trained model to a file.

        Args:
            path (str): Path to save the model.
        """
        joblib.dump(self.model, path)
        print(f"💾 Model saved to {path}")

    def load_model(self, path="nearestneighbors_model.joblib"):
        """
        Load a model from a file.

        Args:
            path (str): Path to the saved model.
        """
        self.model = joblib.load(path)
        print(f"📂 Model loaded from {path}")

    def convert_to_ir(self, X_train, model_name="nearestneighbors"):
        """
        Not supported: Exporting NearestNeighbors to IR via neural network is not possible.

        Args:
            X_train (array-like): Training data (unused).
            model_name (str): Model name (unused).
        """
        print("❌ Export to IR via neural network is not supported for NearestNeighbors.")