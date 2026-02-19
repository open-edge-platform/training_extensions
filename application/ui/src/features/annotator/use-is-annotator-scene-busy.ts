// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useVideoPlayer } from './video-player/video-player-provider.component';

export const useIsAnnotatorSceneBusy = () => {
    const { isPlaying } = useVideoPlayer();

    return isPlaying;
};
