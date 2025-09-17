// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useQuery } from '@tanstack/react-query';
import { wrap } from 'comlink';

export const useSegmentAnythingWorkerQuery = (
    algorithmType: 'SEGMENT_ANYTHING_DECODER' | 'SEGMENT_ANYTHING_ENCODER'
) => {
    return useQuery({
        queryKey: ['workers', algorithmType],
        queryFn: async () => {
            const segmentAnythingWorker = wrap(
                new Worker(new URL('../webworkers/segment-anything.worker', import.meta.url), {
                    type: 'module',
                })
            );

            // @ts-expect-error build exists on every worker
            return segmentAnythingWorker.build();
        },
        staleTime: Infinity,
    });
};
