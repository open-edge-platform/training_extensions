# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Class definition for the Support Vector Classifier with OpenVINO optimization for OTX."""

import joblib
from time import time
from sklearnex.svm import SVC as SkSVC
from sklearn.metrics import classification_report

class SVC:
    def __init__(self, *args, use_openvino=True, **kwargs):
        """
        Initialize the SVC wrapper.

        Args:
            *args: Positional arguments for sklearn's SVC.
            use_openvino (bool): Whether to enable OpenVINO optimizations.
            **kwargs: Keyword arguments for sklearn's SVC.
        """
        self.use_openvino = use_openvino
        self.model = SkSVC(*args, **kwargs)
        print("📦 SVC model initialized (sklearnex version).")

    def fit(self, X, y):
        """
        Fit the SVC model.

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

    def evaluate(self, X, y):
        """
        Evaluate the model and print a classification report.

        Args:
            X (array-like): Test samples.
            y (array-like): True labels for X.

        Returns:
            dict: Classification report as a dictionary.
        """
        y_pred = self.predict(X)
        report = classification_report(y, y_pred, output_dict=True)
        print(f"📊 Classification report:\n{classification_report(y, y_pred)}")
        return report

    def save_model(self, path="svc_model.joblib"):
        """
        Save the trained model to a file.

        Args:
            path (str): Path to save the model.
        """
        joblib.dump(self.model, path)
        print(f"💾 Model saved to {path}")

    def load_model(self, path="svc_model.joblib"):
        """
        Load a model from a file.

        Args:
            path (str): Path to the saved model.
        """
        self.model = joblib.load(path)
        print(f"📂 Model loaded from {path}")

    def convert_to_ir(self, X_train, model_name="svc_model"):
        """
        Not supported: Exporting SVC to IR via neural network is not possible.

        Args:
            X_train (array-like): Training data (unused).
            model_name (str): Model name (unused).
        """
        print("❌ Export to IR via neural network is not supported for SVC.")