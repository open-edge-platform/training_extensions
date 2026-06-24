# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Detection-to-track association utilities (IoU + assignment)."""

from getitrack.matching.iou import (
    fuse_score,
    iou_distance,
    iou_matrix,
    linear_assignment,
)

__all__ = ["fuse_score", "iou_distance", "iou_matrix", "linear_assignment"]
