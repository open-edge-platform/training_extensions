// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useVideoPlayer } from '../video-player-provider.component';
import { formatDurationText } from './time-utils';

export const VideoDuration = () => {
    const { videoRef } = useVideoPlayer();
    // TODO: use video player state
    const currentTime = videoRef.current?.currentTime ?? 0;
    const endTime = videoRef.current?.duration ?? 0;

    const currentFormattedTime = formatDurationText(currentTime);
    const endFormattedTime = formatDurationText(endTime);

    return (
        <span aria-label={'Video duration'}>
            {currentFormattedTime} / {endFormattedTime}
        </span>
    );
};
