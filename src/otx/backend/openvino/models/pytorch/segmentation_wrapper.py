import torch
import numpy as np
from .base_wrapper import BaseWrapper

def iou_score(pred, target, num_classes):
    pred = pred.argmax(dim=1).cpu().numpy()
    target = target.cpu().numpy()
    ious = []
    for cls in range(num_classes):
        pred_cls = (pred == cls)
        target_cls = (target == cls)
        intersection = np.logical_and(pred_cls, target_cls).sum()
        union = np.logical_or(pred_cls, target_cls).sum()
        if union == 0:
            ious.append(float('nan'))
        else:
            ious.append(intersection / union)
    return np.nanmean(ious)

def dice_score(pred, target, num_classes):
    pred = pred.argmax(dim=1).cpu().numpy()
    target = target.cpu().numpy()
    dices = []
    for cls in range(num_classes):
        pred_cls = (pred == cls)
        target_cls = (target == cls)
        intersection = np.logical_and(pred_cls, target_cls).sum()
        dice = (2. * intersection) / (pred_cls.sum() + target_cls.sum() + 1e-8)
        dices.append(dice)
    return np.nanmean(dices)

class SegmentationWrapper(BaseWrapper):
    """Wrapper for semantic segmentation tasks."""

    def evaluate_iou(self, test_loader, num_classes, device="cpu"):
        """Evaluate mean IoU score."""
        scores = []
        self.model.eval()
        self.model.to(device)
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                preds = self.model(x)
                scores.append(iou_score(preds, y, num_classes))
        return np.nanmean(scores)

    def evaluate_dice(self, test_loader, num_classes, device="cpu"):
        """Evaluate mean Dice score."""
        scores = []
        self.model.eval()
        self.model.to(device)
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                preds = self.model(x)
                scores.append(dice_score(preds, y, num_classes))
        return np.nanmean(scores)