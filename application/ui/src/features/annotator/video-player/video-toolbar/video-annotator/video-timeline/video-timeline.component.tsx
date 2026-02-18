// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useVideoPlayer } from '../../../video-player-provider.component';
import { VideoPlayerSlider } from './video-player-slider/video-player-slider.component';

export const VideoTimeline = () => {
    const { videoFrame } = useVideoPlayer();

    return (
        <>
            <VideoPlayerSlider media={videoFrame} />
        </>
    );
};
