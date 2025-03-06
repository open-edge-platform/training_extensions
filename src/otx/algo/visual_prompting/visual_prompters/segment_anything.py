# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
"""Segment Anything model for the OTX visual prompting."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F  # noqa: N812
from torchvision import tv_tensors

from otx.algo.visual_prompting.utils.postprocess import postprocess_masks
from otx.core.data.entity.base import Points


class SegmentAnything(nn.Module):
    """Visual prompting model class for Segment Anything."""

    def __init__(
        self,
        image_encoder: nn.Module,
        prompt_encoder: nn.Module,
        mask_decoder: nn.Module,
        criterion: nn.Module,
        image_size: int = 1024,
        mask_threshold: float = 0.0,
        use_stability_score: bool = False,
        return_single_mask: bool = False,
        return_extra_metrics: bool = False,
        stability_score_offset: float = 1.0,
    ) -> None:
        super().__init__()

        self.image_size = image_size
        self.mask_threshold = mask_threshold
        self.use_stability_score = use_stability_score
        self.return_single_mask = return_single_mask
        self.return_extra_metrics = return_extra_metrics
        self.stability_score_offset = stability_score_offset

        self.image_encoder = image_encoder
        self.prompt_encoder = prompt_encoder
        self.mask_decoder = mask_decoder
        self.criterion = criterion

    def forward(
        self,
        images: tv_tensors.Image,
        ori_shapes: list[Tensor],
        bboxes: list[tv_tensors.BoundingBoxes | None],
        points: list[tuple[Points, Tensor] | None],
        gt_masks: list[tv_tensors.Mask] | None = None,
    ) -> Tensor | tuple[list[Tensor], list[Tensor]]:
        """Forward method for SAM training/validation/prediction.

        Args:
            images (tv_tensors.Image): Images with shape (B, C, H, W).
            ori_shapes (List[Tensor]): List of original shapes per image.
            bboxes (List[tv_tensors.BoundingBoxes], optional): A Nx4 array given a box prompt to the model,
                in XYXY format.
            points (List[Tuple[Points, Tensor]], optional): Point coordinates and labels to embed.
                Point coordinates are BxNx2 arrays of point prompts to the model.
                Each point is in (X,Y) in pixels. Labels are BxN arrays of labels for the point prompts.
                1 indicates a foreground point and 0 indicates a background point.
            gt_masks (List[tv_tensors.Mask], optional): Ground truth masks for loss calculation.

        Returns:
            (Tensor): Calculated loss values.
            (Tuple[List[Tensor], List[Tensor]]): Tuple of list with predicted masks with shape (B, 1, H, W)
                and List with IoU predictions with shape (N, 1).
        """
        image_embeddings = self.image_encoder(images)
        pred_masks = []
        ious = []
        for idx, embedding in enumerate(image_embeddings):
            low_res_masks, iou_predictions = [], []
            for prompt in [bboxes[idx], points[idx]]:
                if prompt is None:
                    continue

                sparse_embeddings, dense_embeddings = self.prompt_encoder(
                    points=prompt if isinstance(prompt[0], Points) else None,
                    boxes=prompt if isinstance(prompt, tv_tensors.BoundingBoxes) else None,
                    masks=None,
                )
                _low_res_masks, _iou_predictions = self.mask_decoder(
                    image_embeddings=embedding.unsqueeze(0),
                    image_pe=self.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=False,  # when given multiple prompts. if there is single prompt True would be better. # noqa: E501
                )
                low_res_masks.append(_low_res_masks)
                iou_predictions.append(_iou_predictions.squeeze(1))

            pred_masks.append(torch.cat(low_res_masks, dim=0))
            ious.append(torch.cat(iou_predictions, dim=0))

        if self.training:
            return self.criterion(pred_masks, gt_masks, ious, ori_shapes)

        post_processed_pred_masks: list[Tensor] = []
        for pred_mask, ori_shape in zip(pred_masks, ori_shapes):
            post_processed_pred_mask = postprocess_masks(pred_mask, self.image_size, ori_shape)
            post_processed_pred_masks.append(post_processed_pred_mask.squeeze(1).sigmoid())
        return post_processed_pred_masks, ious

    @torch.no_grad()
    def forward_for_tracing(
        self,
        image_embeddings: Tensor,
        point_coords: Tensor,
        point_labels: Tensor,
        mask_input: Tensor,
        has_mask_input: Tensor,
        ori_shape: Tensor,
    ) -> tuple[Tensor, ...]:
        """Forward method for SAM inference (export/deploy).

        Args:
            image_embeddings (Tensor): The image embedding with a batch index of length 1.
                If it is a zero tensor, the image embedding will be computed from the image.
            point_coords (Tensor): Coordinates of sparse input prompts,
                corresponding to both point inputs and box inputs.
                Boxes are encoded using two points, one for the top-left corner and one for the bottom-right corner.
                Coordinates must already be transformed to long-side 1024. Has a batch index of length 1.
            point_labels (Tensor): Labels for the sparse input prompts.
                0 is a negative input point, 1 is a positive input point,
                2 is a top-left box corner, 3 is a bottom-right box corner, and -1 is a padding point.
                If there is no box input, a single padding point with label -1 and
                coordinates (0.0, 0.0) should be concatenated.
            mask_input (Tensor): A mask input to the model with shape 1x1x256x256.
                This must be supplied even if there is no mask input. In this case, it can just be zeros.
            has_mask_input (Tensor): An indicator for the mask input.
                1 indicates a mask input, 0 indicates no mask input.
                This input has 1x1 shape due to supporting openvino input layout.
            ori_shape (Tensor): The size of the input image in (H,W) format, before any transformation.
                This input has 1x2 shape due to supporting openvino input layout.
        """
        sparse_embedding = self._embed_points(point_coords, point_labels)
        dense_embedding = self._embed_masks(mask_input, has_mask_input)

        masks, scores = self.mask_decoder.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embedding,
            dense_prompt_embeddings=dense_embedding,
        )

        if self.use_stability_score:
            scores = self.calculate_stability_score(
                masks,
                self.mask_threshold,
                self.stability_score_offset,
            )

        if self.return_single_mask:
            masks, scores = self.select_masks(masks, scores, point_coords.shape[1])

        upscaled_masks = postprocess_masks(masks, self.image_size, ori_shape)

        if self.return_extra_metrics:
            stability_scores = self.calculate_stability_score(
                upscaled_masks,
                self.mask_threshold,
                self.stability_score_offset,
            )
            areas = (upscaled_masks > self.mask_threshold).sum(-1).sum(-1)
            return upscaled_masks, scores, stability_scores, areas, masks

        return upscaled_masks, scores, masks

    def _embed_points(self, point_coords: Tensor, point_labels: Tensor) -> Tensor:
        """Embed sparse input prompts.

        Args:
            point_coords (Tensor): Coordinates of sparse input prompts,
                corresponding to both point inputs and box inputs. Boxes are encoded using two points,
                one for the top-left corner and one for the bottom-right corner.
                Coordinates must already be transformed to long-side 1024. Has a batch index of length 1.
            point_labels (Tensor): Labels for the sparse input prompts.
                0 is a negative input point, 1 is a positive input point,
                2 is a top-left box corner, 3 is a bottom-right box corner, and -1 is a padding point.
                If there is no box input, a single padding point with label -1 and
                coordinates (0.0, 0.0) should be concatenated.

        Returns:
            point_embedding (Tensor): The embedded sparse input prompts.
        """
        point_coords = point_coords + 0.5
        point_coords = point_coords / self.image_size
        point_embedding = self.prompt_encoder.pe_layer._pe_encoding(point_coords)  # noqa: SLF001
        point_labels = point_labels.unsqueeze(-1).expand_as(point_embedding)

        point_embedding = point_embedding * (point_labels != -1)
        point_embedding = point_embedding + self.prompt_encoder.not_a_point_embed.weight * (point_labels == -1)

        for i in range(self.prompt_encoder.num_point_embeddings):
            point_embedding = point_embedding + self.prompt_encoder.point_embeddings[i].weight * (point_labels == i)

        return point_embedding

    def _embed_masks(self, input_mask: Tensor, has_mask_input: Tensor) -> Tensor:
        """Embed the mask input.

        Args:
            input_mask (Tensor): A mask input to the model with shape 1x1x256x256.
                This must be supplied even if there is no mask input. In this case, it can just be zeros.
            has_mask_input (Tensor): An indicator for the mask input.
                1 indicates a mask input, 0 indicates no mask input.

        Returns:
            mask_embedding (Tensor): The embedded mask input.
        """
        mask_embedding = has_mask_input * self.prompt_encoder.mask_downscaling(input_mask)
        return mask_embedding + (1 - has_mask_input) * self.prompt_encoder.no_mask_embed.weight.reshape(
            1,
            -1,
            1,
            1,
        )

    def calculate_stability_score(self, masks: Tensor, mask_threshold: float, threshold_offset: float = 1.0) -> Tensor:
        """Computes the stability score for a batch of masks.

        The stability score is the IoU between the binary masks obtained
        by thresholding the predicted mask logits at high and low values.

        Args:
            masks (Tensor): A batch of predicted masks with shape BxHxW.
            mask_threshold (float): The threshold used to binarize the masks.
            threshold_offset (float, optional): The offset used to compute the stability score.

        Returns:
            stability_scores (Tensor): The stability scores for the batch of masks.
        """
        # One mask is always contained inside the other.
        # Save memory by preventing unnecessary cast to torch.int64
        intersections = (
            (masks > (mask_threshold + threshold_offset)).sum(-1, dtype=torch.int16).sum(-1, dtype=torch.int32)
        )
        unions = (masks > (mask_threshold - threshold_offset)).sum(-1, dtype=torch.int16).sum(-1, dtype=torch.int32)
        return intersections / unions

    def select_masks(self, masks: Tensor, iou_preds: Tensor, num_points: int) -> tuple[Tensor, Tensor]:
        """Selects the best mask from a batch of masks.

        Args:
            masks (Tensor): A batch of predicted masks with shape BxMxHxW.
            iou_preds (Tensor): A batch of predicted IoU scores with shape BxM.
            num_points (int): The number of points in the input.

        Returns:
            masks (Tensor): The selected masks with shape Bx1xHxW.
            iou_preds (Tensor): The selected IoU scores with shape Bx1.
        """
        # Determine if we should return the multi-click mask or not from the number of points.
        # The reweighting is used to avoid control flow.
        score_reweight = torch.tensor([[1000] + [0] * (self.mask_decoder.num_mask_tokens - 1)]).to(
            iou_preds.device,
        )
        score = iou_preds + (num_points - 2.5) * score_reweight
        best_idx = torch.argmax(score, dim=1)
        masks = masks[torch.arange(masks.shape[0]), best_idx, :, :].unsqueeze(1)
        iou_preds = iou_preds[torch.arange(masks.shape[0]), best_idx].unsqueeze(1)

        return masks, iou_preds


class PromptGetter(nn.Module):
    """Prompt getter for zero-shot learning."""

    default_threshold_reference = 0.3
    default_threshold_target = 0.65

    def __init__(self, image_size: int, downsizing: int = 64) -> None:
        super().__init__()
        self.image_size = image_size
        self.downsizing = downsizing

        self.zero_tensor = torch.tensor(0)

    def set_default_thresholds(self, default_threshold_reference: float, default_threshold_target: float) -> None:
        """Set default thresholds."""
        self.default_threshold_reference = default_threshold_reference
        self.default_threshold_target = default_threshold_target

    def get_prompt_candidates(
        self,
        image_embeddings: Tensor,
        reference_feats: Tensor,
        used_indices: Tensor,
        ori_shape: Tensor,
        threshold: float = 0.0,
        num_bg_points: int = 1,
    ) -> tuple[dict[int, Tensor], dict[int, Tensor]]:
        """Get prompt candidates."""
        total_points_scores: dict[int, Tensor] = {}
        total_bg_coords: dict[int, Tensor] = {}
        for label in map(int, used_indices):
            points_scores, bg_coords = self(
                image_embeddings=image_embeddings,
                reference_feat=reference_feats[label],
                ori_shape=ori_shape,
                threshold=threshold,
                num_bg_points=num_bg_points,
            )

            total_points_scores[label] = points_scores
            total_bg_coords[label] = bg_coords

        return total_points_scores, total_bg_coords

    def forward(
        self,
        image_embeddings: Tensor,
        reference_feat: Tensor,
        ori_shape: Tensor,
        threshold: float = 0.0,
        num_bg_points: int = 1,
    ) -> tuple[Tensor, Tensor]:
        """Get prompt candidates from given reference and target features."""
        target_feat = image_embeddings.squeeze()  # (256, 64, 64)
        c_feat, h_feat, w_feat = target_feat.shape
        target_feat = target_feat / target_feat.norm(dim=0, keepdim=True)
        target_feat = target_feat.reshape(c_feat, h_feat * w_feat)

        sim = reference_feat @ target_feat
        sim = sim.reshape(1, 1, h_feat, w_feat)
        sim = postprocess_masks(sim, self.image_size, ori_shape)

        threshold = (threshold == 0) * self.default_threshold_target + threshold
        points_scores, bg_coords = self._point_selection(
            mask_sim=sim[0, 0],
            ori_shape=ori_shape,
            threshold=threshold,
            num_bg_points=num_bg_points,
        )

        return points_scores, bg_coords

    def _point_selection(
        self,
        mask_sim: Tensor,
        ori_shape: Tensor,
        threshold: float = 0.0,
        num_bg_points: int = 1,
    ) -> tuple[Tensor, Tensor]:
        """Select point used as point prompts."""
        _, w_sim = mask_sim.shape

        # Top-last point selection
        bg_indices = mask_sim.flatten().topk(num_bg_points, largest=False)[1]
        bg_x = (bg_indices // w_sim).unsqueeze(0)
        bg_y = bg_indices - bg_x * w_sim
        bg_coords = torch.cat((bg_y, bg_x), dim=0).permute(1, 0)
        bg_coords = bg_coords.to(torch.float32)

        point_coords = torch.where(mask_sim > threshold)
        fg_coords_scores = torch.stack(point_coords[::-1] + (mask_sim[point_coords],), dim=0).T

        # to handle empty tensor
        len_fg_coords_scores = len(fg_coords_scores)
        fg_coords_scores = F.pad(fg_coords_scores, (0, 0, 0, max(0, 1 - len_fg_coords_scores)), value=-1)

        ratio = self.image_size / ori_shape.max()
        width = (ori_shape[1] * ratio).to(torch.int64)
        n_w = width // self.downsizing

        # get grid numbers
        idx_grid = (
            fg_coords_scores[:, 1] * ratio // self.downsizing * n_w + fg_coords_scores[:, 0] * ratio // self.downsizing
        )
        idx_grid_unique = torch.unique(
            idx_grid.to(torch.int64),
        )  # unique op only supports INT64, INT8, FLOAT, STRING in ORT

        # get matched indices
        matched_matrix = idx_grid.unsqueeze(-1) == idx_grid_unique  # (totalN, uniqueN)

        # sample fg_coords_scores matched by matched_matrix
        matched_grid = fg_coords_scores.unsqueeze(1) * matched_matrix.unsqueeze(-1)

        matched_indices = matched_grid[..., -1].topk(k=1, dim=0, largest=True)[1][0].to(torch.int64)
        points_scores = matched_grid[matched_indices].diagonal().T

        # sort by the highest score
        sorted_points_scores_indices = torch.argsort(points_scores[:, -1], descending=True).to(torch.int64)
        points_scores = points_scores[sorted_points_scores_indices]

        return points_scores, bg_coords
