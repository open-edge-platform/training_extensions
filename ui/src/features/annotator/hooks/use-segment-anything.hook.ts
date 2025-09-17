// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useQuery } from '@tanstack/react-query';
import { wrap } from 'comlink';

export const useSegmentAnythingWorkerQuery = () => {
    return useQuery({
        queryKey: ['workers', 'segment-anything'],
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
