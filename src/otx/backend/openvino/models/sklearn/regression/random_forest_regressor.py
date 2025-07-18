# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Class definition for the Random Forest regressor with OpenVINO optimization for OTX."""

import os
import joblib
from time import time
from sklearnex.ensemble import RandomForestRegressor as SkModel
from sklearn.metrics import r2_score

class RandomForestRegressor:
    def __init__(self, *args, use_openvino=True, **kwargs):
        """
        Initialize the RandomForestRegressor wrapper.

        Args:
            *args: Positional arguments for sklearn's RandomForestRegressor.
            use_openvino (bool): Whether to enable OpenVINO optimizations.
            **kwargs: Keyword arguments for sklearn's RandomForestRegressor.
        """
        self.use_openvino = use_openvino
        self._ir_model = None
        self.model = SkModel(*args, **kwargs)
        print("📦 RandomForestRegressor model initialized (OpenVINO version).")

    def fit(self, X, y):
        """
        Fit the RandomForestRegressor model.

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
        Predict target values for samples in X.

        Args:
            X (array-like): Input data.

        Returns:
            array: Predicted values.
        """
        return self.model.predict(X)

    def score(self, X, y):
        """
        Returns the R2 score on the given test data and labels.

        Args:
            X (array-like): Test samples.
            y (array-like): True values for X.

        Returns:
            float: R2 score.
        """
        score = r2_score(y, self.predict(X))
        print(f"📊 R2 Score: {score:.4f}")
        return score

    def evaluate(self, X, y):
        """
        Evaluate the model and print inference time.

        Args:
            X (array-like): Test samples.
            y (array-like): True values for X.

        Returns:
            float: R2 score.
        """
        start = time()
        score = self.score(X, y)
        elapsed = time() - start
        print(f"📈 Inference time: {elapsed:.4f} seconds.")
        return score

    def save_model(self, path="randomforestregressor_model.joblib"):
        """
        Save the trained model to a file.

        Args:
            path (str): Path to save the model.
        """
        joblib.dump(self.model, path)
        print(f"💾 Model saved to {path}")

    def load_model(self, path="randomforestregressor_model.joblib"):
        """
        Load a model from a file.

        Args:
            path (str): Path to the saved model.
        """
        self.model = joblib.load(path)
        print(f"📂 Model loaded from {path}")

    def convert_to_ir(self, X_train, model_name="random_forest_regressor"):
        """
        Not supported: Exporting RandomForestRegressor to IR via neural network is not possible.

        Args:
            X_train (array-like): Training data (unused).
            model_name (str): Model name (unused).
        """
        print("❌ Export to IR via neural network is not supported for RandomForestRegressor.")