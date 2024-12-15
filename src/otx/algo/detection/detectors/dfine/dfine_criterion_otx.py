from __future__ import annotations

from typing import Callable

import torch
import torch.distributed
import torch.nn.functional as F
from torch import nn
from torchvision.ops import box_convert

from otx.algo.common.utils.bbox_overlaps import bbox_overlaps
from otx.algo.detection.losses.rtdetr_loss import DetrCriterion

from .dfine_utils import bbox2distance


class _DFINECriterion(DetrCriterion):
    """This class computes the loss for D-FINE."""

    def __init__(
        self,
        weight_dict: dict[str, int | float],
        alpha: float = 0.2,
        gamma: float = 2.0,
        num_classes: int = 80,
        reg_max: int = 32,
    ):
        super().__init__(
            weight_dict=weight_dict,
            alpha=alpha,
            gamma=gamma,
            num_classes=num_classes,
        )
        self.reg_max = reg_max

    def loss_local(self, outputs, targets, indices, num_boxes, T=5):
        """Compute Fine-Grained Localization (FGL) Loss and Decoupled Distillation Focal (DDF) Loss."""
        losses = {}
        if "pred_corners" in outputs:
            idx = self._get_src_permutation_idx(indices)
            target_boxes = torch.cat([t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0)

            pred_corners = outputs["pred_corners"][idx].reshape(-1, (self.reg_max + 1))
            ref_points = outputs["ref_points"][idx].detach()
            with torch.no_grad():
                target_corners, weight_right, weight_left = bbox2distance(
                    ref_points,
                    box_convert(target_boxes, in_fmt="cxcywh", out_fmt="xyxy"),
                    self.reg_max,
                    outputs["reg_scale"],
                    outputs["up"],
                )

            ious = torch.diag(
                bbox_overlaps(
                    box_convert(outputs["pred_boxes"][idx], in_fmt="cxcywh", out_fmt="xyxy"),
                    box_convert(target_boxes, in_fmt="cxcywh", out_fmt="xyxy"),
                ),
            )
            weight_targets = ious.unsqueeze(-1).repeat(1, 1, 4).reshape(-1).detach()

            losses["loss_fgl"] = self.unimodal_distribution_focal_loss(
                pred_corners,
                target_corners,
                weight_right,
                weight_left,
                weight_targets,
                avg_factor=num_boxes,
            )

            if "teacher_corners" in outputs and outputs["teacher_corners"] is not None:
                pred_corners = outputs["pred_corners"].reshape(-1, (self.reg_max + 1))
                target_corners = outputs["teacher_corners"].reshape(-1, (self.reg_max + 1))
                if torch.equal(pred_corners, target_corners):
                    losses["loss_ddf"] = pred_corners.sum() * 0
                else:
                    weight_targets_local = outputs["teacher_logits"].sigmoid().max(dim=-1)[0]

                    mask = torch.zeros_like(weight_targets_local, dtype=torch.bool)
                    mask[idx] = True
                    mask = mask.unsqueeze(-1).repeat(1, 1, 4).reshape(-1)

                    weight_targets_local[idx] = ious.reshape_as(weight_targets_local[idx]).to(
                        weight_targets_local.dtype,
                    )
                    weight_targets_local = weight_targets_local.unsqueeze(-1).repeat(1, 1, 4).reshape(-1).detach()

                    loss_match_local = (
                        weight_targets_local
                        * (T**2)
                        * (
                            nn.KLDivLoss(reduction="none")(
                                F.log_softmax(pred_corners / T, dim=1),
                                F.softmax(target_corners.detach() / T, dim=1),
                            )
                        ).sum(-1)
                    )
                    # NOTE: Avoid the influence of batch size per GPU
                    # TODO(Eugene): does it matter training with 1 GPU?
                    if "is_dn" not in outputs:
                        batch_scale = 8 / outputs["pred_boxes"].shape[0]
                        self.num_pos, self.num_neg = (
                            (mask.sum() * batch_scale) ** 0.5,
                            ((~mask).sum() * batch_scale) ** 0.5,
                        )
                    loss_match_local1 = loss_match_local[mask].mean() if mask.any() else 0
                    loss_match_local2 = loss_match_local[~mask].mean() if (~mask).any() else 0
                    losses["loss_ddf"] = (loss_match_local1 * self.num_pos + loss_match_local2 * self.num_neg) / (
                        self.num_pos + self.num_neg
                    )

        return losses

    def _get_go_indices(self, indices, indices_aux_list):
        """Get a matching union set across all decoder layers."""
        results = []
        for indices_aux in indices_aux_list:
            indices = [
                (torch.cat([idx1[0], idx2[0]]), torch.cat([idx1[1], idx2[1]]))
                for idx1, idx2 in zip(indices.copy(), indices_aux.copy())
            ]

        for ind in [torch.cat([idx[0][:, None], idx[1][:, None]], 1) for idx in indices]:
            unique, counts = torch.unique(ind, return_counts=True, dim=0)
            count_sort_indices = torch.argsort(counts, descending=True)
            unique_sorted = unique[count_sort_indices]
            column_to_row = {}
            for idx in unique_sorted:
                row_idx, col_idx = idx[0].item(), idx[1].item()
                if row_idx not in column_to_row:
                    column_to_row[row_idx] = col_idx
            final_rows = torch.tensor(list(column_to_row.keys()), device=ind.device)
            final_cols = torch.tensor(list(column_to_row.values()), device=ind.device)
            results.append((final_rows.long(), final_cols.long()))
        return results

    @property
    def _available_losses(self) -> tuple[Callable]:
        return (
            self.loss_boxes,
            self.loss_labels_vfl,
            self.loss_local,
        )

    def forward(
        self,
        outputs: dict[str, torch.Tensor],
        targets: list[dict[str, torch.Tensor]],
    ):
        outputs_without_aux = {k: v for k, v in outputs.items() if "aux" not in k}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        # Get the matching union set across all decoder layers.
        indices_aux_list, cached_indices, cached_indices_enc = [], [], []
        for i, aux_outputs in enumerate(outputs["aux_outputs"] + [outputs["pre_outputs"]]):
            indices_aux = self.matcher(aux_outputs, targets)
            cached_indices.append(indices_aux)
            indices_aux_list.append(indices_aux)
        for i, aux_outputs in enumerate(outputs["enc_aux_outputs"]):
            indices_enc = self.matcher(aux_outputs, targets)
            cached_indices_enc.append(indices_enc)
            indices_aux_list.append(indices_enc)
        indices_go = self._get_go_indices(indices, indices_aux_list)

        num_boxes_go = sum(len(x[0]) for x in indices_go)
        num_boxes_go = torch.as_tensor(
            [num_boxes_go],
            dtype=torch.float,
            device=next(iter(outputs.values())).device,
        )
        num_boxes_go = torch.clamp(num_boxes_go, min=1).item()

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        num_boxes = torch.clamp(num_boxes, min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self._available_losses:
            indices_in = indices_go if loss in ["boxes", "local"] else indices
            num_boxes_in = num_boxes_go if loss in ["boxes", "local"] else num_boxes
            l_dict = loss(outputs, targets, indices_in, num_boxes_in)
            losses.update(l_dict)

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                aux_outputs["up"], aux_outputs["reg_scale"] = outputs["up"], outputs["reg_scale"]
                for loss in self._available_losses:
                    indices_in = indices_go if loss in ["boxes", "local"] else cached_indices[i]
                    num_boxes_in = num_boxes_go if loss in ["boxes", "local"] else num_boxes
                    l_dict = loss(aux_outputs, targets, indices_in, num_boxes_in)
                    l_dict = {k + f"_aux_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        # In case of auxiliary traditional head output at first decoder layer.
        if "pre_outputs" in outputs:
            aux_outputs = outputs["pre_outputs"]
            for loss in self._available_losses:
                indices_in = indices_go if loss in ["boxes", "local"] else cached_indices[-1]
                num_boxes_in = num_boxes_go if loss in ["boxes", "local"] else num_boxes
                l_dict = loss(aux_outputs, targets, indices_in, num_boxes_in)
                l_dict = {k + "_pre": v for k, v in l_dict.items()}
                losses.update(l_dict)

        # In case of encoder auxiliary losses.
        if "enc_aux_outputs" in outputs:
            enc_targets = targets
            for i, aux_outputs in enumerate(outputs["enc_aux_outputs"]):
                for loss in self._available_losses:
                    indices_in = indices_go if loss == "boxes" else cached_indices_enc[i]
                    num_boxes_in = num_boxes_go if loss == "boxes" else num_boxes
                    l_dict = loss(aux_outputs, enc_targets, indices_in, num_boxes_in)
                    l_dict = {k + f"_enc_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        # In case of cdn auxiliary losses. For dfine
        if "dn_outputs" in outputs:
            assert "dn_meta" in outputs, ""
            indices_dn = self.get_cdn_matched_indices(outputs["dn_meta"], targets)
            dn_num_boxes = num_boxes * outputs["dn_meta"]["dn_num_group"]
            dn_num_boxes = dn_num_boxes if dn_num_boxes > 0 else 1

            for i, aux_outputs in enumerate(outputs["dn_outputs"]):
                aux_outputs["is_dn"] = True
                aux_outputs["up"], aux_outputs["reg_scale"] = outputs["up"], outputs["reg_scale"]
                for loss in self._available_losses:
                    l_dict = loss(aux_outputs, targets, indices_dn, dn_num_boxes)
                    l_dict = {k + f"_dn_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

            # In case of auxiliary traditional head output at first decoder layer.
            if "dn_pre_outputs" in outputs:
                aux_outputs = outputs["dn_pre_outputs"]
                for loss in self._available_losses:
                    l_dict = loss(aux_outputs, targets, indices_dn, dn_num_boxes)
                    l_dict = {k + "_dn_pre": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses

    def unimodal_distribution_focal_loss(
        self,
        preds,
        targets,
        weight_right,
        weight_left,
        iou_weight=None,
        reduction="sum",
        avg_factor=None,
    ):
        dis_left = targets.long()
        dis_right = dis_left + 1

        loss = F.cross_entropy(preds, dis_left, reduction="none") * weight_left.reshape(-1) + F.cross_entropy(
            preds,
            dis_right,
            reduction="none",
        ) * weight_right.reshape(-1)

        if iou_weight is not None:
            iou_weight = iou_weight.float()
            loss = loss * iou_weight

        if avg_factor is not None:
            loss = loss.sum() / avg_factor
        elif reduction == "mean":
            loss = loss.mean()
        elif reduction == "sum":
            loss = loss.sum()

        return loss
