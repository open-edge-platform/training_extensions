# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Motion models (Kalman filter) and bbox-format conversion helpers."""

from getitrack.motion.kalman import KalmanFilter, xyah_to_xyxy, xyxy_to_xyah

__all__ = ["KalmanFilter", "xyah_to_xyxy", "xyxy_to_xyah"]
