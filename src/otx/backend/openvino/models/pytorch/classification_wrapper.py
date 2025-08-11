# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Classification wrapper for PyTorch models with OpenVINO optimization for OTX."""

import torch
from .base_wrapper import BaseWrapper
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    log_loss,
)

class ClassificationWrapper(BaseWrapper):
    """Wrapper for classification tasks with built-in metrics."""

    def evaluate_accuracy(self, test_loader, device="cpu"):
        """Evaluate classification accuracy."""
        y_true, y_pred = self._collect_preds(test_loader, device)
        return accuracy_score(y_true, y_pred)

    def evaluate_f1(self, test_loader, device="cpu", average="macro"):
        """Evaluate F1 score."""
        y_true, y_pred = self._collect_preds(test_loader, device)
        return f1_score(y_true, y_pred, average=average)

    def evaluate_precision(self, test_loader, device="cpu", average="macro"):
        """Evaluate precision score."""
        y_true, y_pred = self._collect_preds(test_loader, device)
        return precision_score(y_true, y_pred, average=average)

    def evaluate_recall(self, test_loader, device="cpu", average="macro"):
        """Evaluate recall score."""
        y_true, y_pred = self._collect_preds(test_loader, device)
        return recall_score(y_true, y_pred, average=average)

    def evaluate_confusion_matrix(self, test_loader, device="cpu"):
        """Return confusion matrix."""
        y_true, y_pred = self._collect_preds(test_loader, device)
        return confusion_matrix(y_true, y_pred)

    def evaluate_classification_report(self, test_loader, device="cpu"):
        """Return classification report as a string."""
        y_true, y_pred = self._collect_preds(test_loader, device)
        return classification_report(y_true, y_pred)

    def evaluate_roc_auc(self, test_loader, device="cpu", average="macro", multi_class="ovr"):
        """Evaluate ROC AUC score (for multi-class, needs probability output)."""
        y_true, y_score = self._collect_probs(test_loader, device)
        return roc_auc_score(y_true, y_score, average=average, multi_class=multi_class)

    def evaluate_log_loss(self, test_loader, device="cpu"):
        """Evaluate log loss (cross-entropy loss)."""
        y_true, y_score = self._collect_probs(test_loader, device)
        return log_loss(y_true, y_score)

    def _collect_preds(self, test_loader, device):
        """Helper to collect true and predicted labels."""
        y_true, y_pred = [], []
        self.model.eval()
        self.model.to(device)
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                logits = self.model(x)
                preds = logits.argmax(dim=1)
                y_true.extend(y.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())
        return y_true, y_pred

    def _collect_probs(self, test_loader, device):
        """Helper to collect true labels and predicted probabilities."""
        y_true, y_score = [], []
        self.model.eval()
        self.model.to(device)
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                logits = self.model(x)
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                y_true.extend(y.cpu().numpy())
                y_score.extend(probs)
        return y_true, y_score
    
    def evaluate_top_k_accuracy(self, test_loader, device="cpu", k=5):
        """Evaluate Top-K accuracy (default k=5)."""
        y_true, y_pred_topk = [], []
        self.model.eval()
        self.model.to(device)
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                logits = self.model(x)
                topk = logits.topk(k, dim=1).indices
                # Check if true label is in top-k predictions
                correct = [int(label in topk_row.cpu().numpy()) for label, topk_row in zip(y, topk)]
                y_true.extend([1]*len(correct))
                y_pred_topk.extend(correct)
        # Top-K accuracy is the mean of correct predictions
        return sum(y_pred_topk) / len(y_pred_topk) if y_pred_topk else 0.0