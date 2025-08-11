import torch
import numpy as np
import warnings
from .base_wrapper import BaseWrapper
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    mean_absolute_percentage_error,
    explained_variance_score,
    max_error,
)

class RegressionWrapper(BaseWrapper):
    """Wrapper for regression tasks with built-in metrics."""

    def evaluate_mse(self, test_loader, device="cpu"):
        """Evaluate Mean Squared Error."""
        y_true, y_pred = self._collect_predictions(test_loader, device)
        return mean_squared_error(y_true, y_pred)

    def evaluate_rmse(self, test_loader, device="cpu"):
        """Evaluate Root Mean Squared Error."""
        mse = self.evaluate_mse(test_loader, device)
        return np.sqrt(mse)

    def evaluate_mae(self, test_loader, device="cpu"):
        """Evaluate Mean Absolute Error."""
        y_true, y_pred = self._collect_predictions(test_loader, device)
        return mean_absolute_error(y_true, y_pred)

    def evaluate_r2(self, test_loader, device="cpu"):
        """Evaluate R² Score."""
        y_true, y_pred = self._collect_predictions(test_loader, device)
        return r2_score(y_true, y_pred)

    def evaluate_mape(self, test_loader, device="cpu"):
        """Evaluate Mean Absolute Percentage Error."""
        y_true, y_pred = self._collect_predictions(test_loader, device)
        return mean_absolute_percentage_error(y_true, y_pred)

    def evaluate_explained_variance(self, test_loader, device="cpu"):
        """Evaluate Explained Variance Score."""
        y_true, y_pred = self._collect_predictions(test_loader, device)
        return explained_variance_score(y_true, y_pred)

    def evaluate_max_error(self, test_loader, device="cpu"):
        """Evaluate Maximum Residual Error."""
        y_true, y_pred = self._collect_predictions(test_loader, device)
        return max_error(y_true, y_pred)

    def evaluate_all_metrics(self, test_loader, device="cpu"):
        """Evaluate all regression metrics at once."""
        y_true, y_pred = self._collect_predictions(test_loader, device)
        return {
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred),
            'mape': mean_absolute_percentage_error(y_true, y_pred),
            'explained_variance': explained_variance_score(y_true, y_pred),
            'max_error': max_error(y_true, y_pred)
        }

    def _collect_predictions(self, test_loader, device):
        """Helper to collect true and predicted values."""
        if test_loader is None:
            raise ValueError("test_loader cannot be None")
        if device.startswith("cuda") and not torch.cuda.is_available():
            warnings.warn("CUDA not available, falling back to CPU")
            device = "cpu"
        y_true, y_pred = [], []
        self.model.eval()
        self.model.to(device)
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                predictions = self.model(x)
                if predictions.dim() > 1 and predictions.size(1) == 1:
                    predictions = predictions.squeeze(1)
                if y.dim() > 1 and y.size(1) == 1:
                    y = y.squeeze(1)
                y_true.extend(y.cpu().numpy())
                y_pred.extend(predictions.cpu().numpy())
        if not y_true or not y_pred:
            raise ValueError("No predictions collected from test_loader")
        return np.array(y_true), np.array(y_pred)