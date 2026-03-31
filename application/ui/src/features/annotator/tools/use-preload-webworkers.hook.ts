// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useIntelligentScissorsWorker } from './hooks/use-intelligent-scissors-worker.hook';
import { useSegmentAnythingWorker } from './segment-anything-tool/use-segment-anything.hook';

export const usePreloadWebworkers = (enabled = true) => {
    useSegmentAnythingWorker('SEGMENT_ANYTHING_ENCODER', enabled);
    useSegmentAnythingWorker('SEGMENT_ANYTHING_DECODER', enabled);
    // useSSIMWorker(enabled);
    useIntelligentScissorsWorker(enabled);
};
