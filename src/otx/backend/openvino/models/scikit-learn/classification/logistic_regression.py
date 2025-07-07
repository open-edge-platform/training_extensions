# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Class definition for the Logistic Regression classifier with OpenVINO optimization for OTX."""

from sklearnex import patch_sklearn
import os
import joblib
from time import time
from sklearn.linear_model import LogisticRegression as SkModel
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import warnings
from sklearn.exceptions import ConvergenceWarning
import subprocess
import zipfile

warnings.filterwarnings("ignore", category=ConvergenceWarning)

class LogisticRegression:
    def __init__(self, *args, use_openvino=True, **kwargs):
        """
        Initialize the LogisticRegression wrapper.

        Args:
            *args: Positional arguments for sklearn's LogisticRegression.
            use_openvino (bool): Whether to enable OpenVINO optimizations.
            **kwargs: Keyword arguments for sklearn's LogisticRegression.
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
        print("📦 LogisticRegression model initialized.")

        if self.use_openvino:
            self._warn_if_not_fully_supported(**kwargs)

    def _warn_if_not_fully_supported(self, **kwargs):
        """
        Warns if any parameter is not fully supported for OpenVINO optimization.

        Args:
            **kwargs: Keyword arguments passed to the model.
        """
        unsupported = []
        if kwargs.get("penalty", "l2") != "l2":
            unsupported.append("penalty ≠ 'l2'")
        if kwargs.get("dual", False):
            unsupported.append("dual = True")
        if kwargs.get("intercept_scaling", 1) != 1:
            unsupported.append("intercept_scaling ≠ 1")
        if kwargs.get("class_weight", None) is not None:
            unsupported.append("class_weight ≠ None")
        if kwargs.get("solver", "lbfgs") not in ["lbfgs", "newton-cg"]:
            unsupported.append("solver not in ['lbfgs', 'newton-cg']")
        if kwargs.get("multi_class", "ovr") != "ovr":
            unsupported.append("multi_class ≠ 'ovr'")
        if kwargs.get("warm_start", False):
            unsupported.append("warm_start = True")
        if kwargs.get("l1_ratio", None) is not None:
            unsupported.append("l1_ratio ≠ None")

        if unsupported:
            print("⚠️ Unsupported for OpenVINO optimization (may fallback):")
            for u in unsupported:
                print(f"   - {u}")

    def fit(self, X, y):
        """
        Fit the logistic regression model.

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
        if self._ir_model:
            return self._predict_ir(X)
        return self.model.predict(X)

    def predict_proba(self, X):
        """
        Probability estimates for samples in X.

        Args:
            X (array-like): Input data.

        Returns:
            array: Probability estimates.
        """
        if self._ir_model:
            return self._predict_ir(X, proba=True)
        return self.model.predict_proba(X)

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

    def save_model(self, path="logreg_model.joblib"):
        """
        Save the trained model to a file.

        Args:
            path (str): Path to save the model.
        """
        joblib.dump(self.model, path)
        print(f"💾 Model saved to {path}")

    def load_model(self, path="logreg_model.joblib"):
        """
        Load a model from a file.

        Args:
            path (str): Path to the saved model.
        """
        self.model = joblib.load(path)
        print(f"📂 Model loaded from {path}")

    def convert_to_ir(self, X_train, model_name="logreg_model"):
        """
        Export LogisticRegression to OpenVINO IR via an equivalent MLPClassifier.
        The function creates an ONNX file, converts it to IR, and zips the IR files.

        Args:
            X_train (array-like): Training data for shape reference.
            model_name (str): Base name for the exported model files.

        Returns:
            str: Path to the zipped IR model.
        """
        if hasattr(self.model, "coef_") and hasattr(self.model, "intercept_"):
            print("🔄 Creating equivalent neural network for export...")
            mlp = MLPClassifier(hidden_layer_sizes=(), max_iter=1)
            mlp.fit(X_train, self.model.predict(X_train))
            mlp.coefs_ = [self.model.coef_.T]
            mlp.intercepts_ = [self.model.intercept_]
            mlp.n_outputs_ = self.model.coef_.shape[0] if len(self.model.coef_.shape) > 1 else 1
            mlp.out_activation_ = "logistic" if mlp.n_outputs_ == 1 else "softmax"

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

    def _predict_ir(self, X, proba=False):
        """
        Placeholder for IR inference.

        Args:
            X (array-like): Input data.
            proba (bool): If True, return probabilities.

        Raises:
            NotImplementedError: Always, as IR inference is not implemented.
        """
        raise NotImplementedError("IR inference not implemented for LogisticRegression.")