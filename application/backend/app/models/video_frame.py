# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from uuid import UUID

from app.models import BaseEntity


class VideoFrame(BaseEntity):
    id: UUID
    video_id: UUID
    timestamp: float
