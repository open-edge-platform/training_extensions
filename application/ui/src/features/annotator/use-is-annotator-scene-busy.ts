// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useVideoPlayerContext } from './video-player/video-player-provider.component';

export const useIsAnnotatorSceneBusy = () => {
    const context = useVideoPlayerContext();

    return context?.isPlaying ?? false;
};
