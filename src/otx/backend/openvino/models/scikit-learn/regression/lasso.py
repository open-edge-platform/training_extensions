# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Class definition for the Lasso regression model with OpenVINO optimization for OTX."""

from sklearnex import patch_sklearn
import os
import joblib
import numpy as np
from time import time
from sklearn.linear_model import Lasso as SkModel
from sklearn.metrics import r2_score
from sklearn.neural_network import MLPRegressor
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import subprocess
import zipfile
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

class Lasso:
    def __init__(self, *args, use_openvino=True, **kwargs):
        """
        Initialize the Lasso wrapper.

        Args:
            *args: Positional arguments for sklearn's Lasso.
            use_openvino (bool): Whether to enable OpenVINO optimizations.
            **kwargs: Keyword arguments for sklearn's Lasso.
        """
        self.use_openvino = use_openvino
        self._patched = False
        self._ir_model = None

        if self.use_openvino:
            try:
                patch_sklearn()
                self._patched = True
                print("✅ sklearnex patch applied successfully.")
            except Exception:
                print("⚠️ sklearnex patch failed.")

        self.model = SkModel(*args, **kwargs)
        print("📦 Lasso model initialized.")

    def fit(self, X, y):
        """
        Fit the lasso regression model.

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

    def save_model(self, path="lasso_model.joblib"):
        """
        Save the trained model to a file.

        Args:
            path (str): Path to save the model.
        """
        joblib.dump(self.model, path)
        print(f"💾 Model saved to {path}")

    def load_model(self, path="lasso_model.joblib"):
        """
        Load a model from a file.

        Args:
            path (str): Path to the saved model.
        """
        self.model = joblib.load(path)
        print(f"📂 Model loaded from {path}")

    def convert_to_ir(self, X_train, model_name="lasso"):
        """
        Export Lasso to OpenVINO IR via an equivalent MLPRegressor.
        The function creates an ONNX file, converts it to IR, and zips the IR files.

        Args:
            X_train (array-like): Training data for shape reference.
            model_name (str): Base name for the exported model files.

        Returns:
            str: Path to the zipped IR model.
        """
        if hasattr(self.model, "coef_") and hasattr(self.model, "intercept_"):
            print("🔄 Creating equivalent neural network for export...")
            mlp = MLPRegressor(hidden_layer_sizes=(), max_iter=1)
            mlp.fit(X_train, self.model.predict(X_train))
            mlp.coefs_ = [self.model.coef_.reshape(-1, 1) if len(self.model.coef_.shape) == 1 else self.model.coef_.T]
            mlp.intercepts_ = [np.array([self.model.intercept_]).flatten()]
            mlp.n_outputs_ = self.model.coef_.shape[0] if len(self.model.coef_.shape) > 1 else 1
            mlp.out_activation_ = "identity"

            initial_type = [('input', FloatTensorType([None, X_train.shape[1]]))]
            onnx_model = convert_sklearn(mlp, initial_types=initial_type)

            onnx_path = f"{model_name}.onnx"
            with open(onnx_path, "wb") as f:
                f.write(onnx_model.SerializeToString())
            print(f"📦 ONNX model saved to {onnx_path}")

            # Convert ONNX to IR using Model Optimizer (Python subprocess)
            ir_output_dir = f"{model_name}_ir"
            os.makedirs(ir_output_dir, exist_ok=True)

            command = [
                "mo",
                "--input_model", onnx_path,
                "--output_dir", ir_output_dir,
                "--input_shape", f"[1,{X_train.shape[1]}]",
            ]

            print("🧠 Converting ONNX to IR with OpenVINO Model Optimizer...")
            subprocess.run(command, check=True)
            print("✅ IR conversion completed!")

            # Zip IR files
            zip_path = f"{model_name}_ir.zip"
            with zipfile.ZipFile(zip_path, 'w') as zipf:
                zipf.write(f"{ir_output_dir}/model.xml", arcname="model.xml")
                zipf.write(f"{ir_output_dir}/model.bin", arcname="model.bin")
            print(f"📦 IR zip file saved at: {zip_path}")

            return zip_path
        else:
            print("❌ Model not trained or not supported for export via equivalent neural network.")