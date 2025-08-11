import torch
from .base_wrapper import BaseWrapper

class DetectionWrapper(BaseWrapper):
    """Wrapper for object detection tasks."""

    def evaluate_map(self, test_loader, metric_fn, device="cpu"):
        """
        Evaluate mean Average Precision (mAP).
        metric_fn must accept (preds, targets) and return mAP for the batch.
        """
        self.model.eval()
        self.model.to(device)
        all_maps = []
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                preds = self.model(x)
                batch_map = metric_fn(preds, y)
                all_maps.append(batch_map)
        return sum(all_maps) / len(all_maps) if all_maps else 0.0

    def evaluate_precision(self, test_loader, metric_fn, device="cpu"):
        """
        Evaluate precision for object detection.
        metric_fn must accept (preds, targets) and return precision for the batch.
        """
        self.model.eval()
        self.model.to(device)
        all_precisions = []
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                preds = self.model(x)
                batch_precision = metric_fn(preds, y)
                all_precisions.append(batch_precision)
        return sum(all_precisions) / len(all_precisions) if all_precisions else 0.0

    def evaluate_recall(self, test_loader, metric_fn, device="cpu"):
        """
        Evaluate recall for object detection.
        metric_fn must accept (preds, targets) and return recall for the batch.
        """
        self.model.eval()
        self.model.to(device)
        all_recalls = []
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                preds = self.model(x)
                batch_recall = metric_fn(preds, y)
                all_recalls.append(batch_recall)
        return sum(all_recalls) / len(all_recalls) if all_recalls else 0.0

    def evaluate_f1(self, test_loader, metric_fn, device="cpu"):
        """
        Evaluate F1 score for object detection.
        metric_fn must accept (preds, targets) and return F1 score for the batch.
        """
        self.model.eval()
        self.model.to(device)
        all_f1s = []
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                preds = self.model(x)
                batch_f1 = metric_fn(preds, y)
                all_f1s.append(batch_f1)
        return sum(all_f1s) / len(all_f1s) if all_f1s else 0.0

    def evaluate_iou(self, test_loader, metric_fn, device="cpu"):
        """
        Evaluate mean IoU for object detection.
        metric_fn must accept (preds, targets) and return IoU for the batch.
        """
        self.model.eval()
        self.model.to(device)
        all_ious = []
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                preds = self.model(x)
                batch_iou = metric_fn(preds, y)
                all_ious.append(batch_iou)
        return sum(all_ious) / len(all_ious) if all_ious else 0.0

    def evaluate_ap_per_class(self, test_loader, metric_fn, device="cpu"):
        """
        Evaluate AP (Average Precision) per class.
        metric_fn must accept (preds, targets) and return AP per class for the batch.
        Returns a dict: {class_idx: AP}
        """
        self.model.eval()
        self.model.to(device)
        all_ap_dicts = []
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                preds = self.model(x)
                batch_ap_dict = metric_fn(preds, y)
                all_ap_dicts.append(batch_ap_dict)
        # Aggregate AP per class
        if not all_ap_dicts:
            return {}
        keys = all_ap_dicts[0].keys()
        avg_ap = {k: sum(d[k] for d in all_ap_dicts) / len(all_ap_dicts) for k in keys}
        return avg_ap

    def evaluate_detection_report(self, test_loader, metric_fn, device="cpu"):
        """
        Evaluate detection report (TP, FP, FN, etc).
        metric_fn must accept (preds, targets) and return a dict with report for the batch.
        Returns a dict: {metric_name: value}
        """
        self.model.eval()
        self.model.to(device)
        all_reports = []
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                preds = self.model(x)
                batch_report = metric_fn(preds, y)
                all_reports.append(batch_report)
        # Aggregate reports
        if not all_reports:
            return {}
        keys = all_reports[0].keys()
        avg_report = {k: sum(d[k] for d in all_reports) / len(all_reports) for k in keys}