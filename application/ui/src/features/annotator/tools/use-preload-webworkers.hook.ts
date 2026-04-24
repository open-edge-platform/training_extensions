// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useEffect } from 'react';

import { useQueryClient } from '@tanstack/react-query';

import { useAvailableTools } from './annotator-tools/use-available-tools';
import { useIntelligentScissorsWorker } from './hooks/use-intelligent-scissors-worker.hook';
import { useSegmentAnythingWorker } from './segment-anything-tool/use-segment-anything.hook';
import { useSSIMWorker } from './ssim-tool/use-ssim.hook';

export const usePreloadWebworkers = () => {
    const availableTools = useAvailableTools();
    const tools = new Set(availableTools.map((tool) => tool.type));
    const queryClient = useQueryClient();

    useSegmentAnythingWorker(tools.has('sam'));
    useSSIMWorker(tools.has('ssim'));
    useIntelligentScissorsWorker(tools.has('magnetic-lasso'));

    // Tear down every annotator-tool worker when the annotator unmounts.
    useEffect(() => {
        return () => {
            const workerQueries = queryClient.getQueryCache().findAll({ queryKey: ['workers'] });

            for (const query of workerQueries) {
                const data = query.state.data as { worker?: unknown } | undefined;

                if (data?.worker instanceof Worker) {
                    data.worker.terminate();
                }
            }

            queryClient.removeQueries({ queryKey: ['workers'] });
        };
    }, [queryClient]);
};
