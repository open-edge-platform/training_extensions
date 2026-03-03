// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useVideoPlayer } from '../video-player-provider.component';
import { formatDurationText } from './time-utils';

export const VideoDuration = () => {
    const { videoFrame } = useVideoPlayer();
    const currentTime = videoFrame.frame_number / videoFrame.fps;
    const endTime = videoFrame.duration;

    const currentFormattedTime = formatDurationText(currentTime);
    const endFormattedTime = formatDurationText(endTime);

    return (
        <span aria-label={'Video duration'}>
            {currentFormattedTime} / {endFormattedTime}
        </span>
    );
};
