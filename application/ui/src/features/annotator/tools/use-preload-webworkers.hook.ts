// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useAvailableTools } from './annotator-tools/use-available-tools';
import { useIntelligentScissorsWorker } from './hooks/use-intelligent-scissors-worker.hook';
import { useSegmentAnythingWorker } from './segment-anything-tool/use-segment-anything.hook';
import { useSSIMWorker } from './ssim-tool/use-ssim.hook';

export const usePreloadWebworkers = () => {
    const availableTools = useAvailableTools();
    const tools = new Set(availableTools.map((tool) => tool.type));

    useSegmentAnythingWorker('SEGMENT_ANYTHING_ENCODER', tools.has('sam'));
    useSegmentAnythingWorker('SEGMENT_ANYTHING_DECODER', tools.has('sam'));
    useSSIMWorker(tools.has('ssim'));
    useIntelligentScissorsWorker(tools.has('magnetic-lasso'));
};
