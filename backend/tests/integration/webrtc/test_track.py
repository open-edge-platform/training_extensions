# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import queue

import numpy as np
import pytest
from av import VideoFrame

from app.webrtc.stream import InferenceVideoStreamTrack


@pytest.mark.asyncio
async def test_track_receives_video_frame_from_numpy_array():
    q = queue.Queue()
    arr = np.zeros((480, 640, 3), dtype=np.uint8)
    q.put(arr)

    track = InferenceVideoStreamTrack(q)
    frame = await track.recv()

    assert isinstance(frame, VideoFrame)
    assert frame.width == 640
    assert frame.height == 480
