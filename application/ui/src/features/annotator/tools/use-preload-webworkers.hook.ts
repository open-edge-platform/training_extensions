// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useAvailableTools } from './annotator-tools/use-available-tools';
import { useIntelligentScissorsWorker } from './hooks/use-intelligent-scissors-worker.hook';
import { useSegmentAnythingWorker } from './segment-anything-tool/use-segment-anything.hook';
// TODO: Disable for 3.0, enable for 3.1 after improvements (needs a sidebar to tweak threshold)
// import { useSSIMWorker } from './ssim-tool/use-ssim.hook';

/**
 * Eagerly boot the annotator-tool web workers (SAM, magnetic-lasso) as
 * soon as the annotator mounts, so the heavy WASM + ONNX init runs in the
 * background and the corresponding tools are responsive the moment the user
 * picks them.
 */
export const usePreloadWebworkers = () => {
    const availableTools = useAvailableTools();
    const tools = new Set(availableTools.map((tool) => tool.type));

    useSegmentAnythingWorker(tools.has('sam'));
    // TODO: Disable for 3.0, enable for 3.1 after improvements (needs a sidebar to tweak threshold)
    // useSSIMWorker(tools.has('ssim'));
    useIntelligentScissorsWorker(tools.has('magnetic-lasso'));
};
