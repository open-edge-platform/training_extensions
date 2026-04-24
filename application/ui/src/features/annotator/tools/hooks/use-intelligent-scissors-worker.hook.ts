// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useQuery } from '@tanstack/react-query';
import { Remote, wrap } from 'comlink';

import type {
    IntelligentScissorsWorkerApi,
    IntelligentScissorsWorkerInstance,
} from '../../webworkers/intelligent-scissors.worker.interface';

type IntelligentScissorsRemoteInstance = Remote<IntelligentScissorsWorkerInstance>;

export const useIntelligentScissorsWorker = (enabled = true) => {
    const { data, isLoading, isSuccess, isError } = useQuery<{
        worker: Worker;
        instance: IntelligentScissorsRemoteInstance;
    }>({
        queryKey: ['workers', 'INTELLIGENT_SCISSORS'],
        queryFn: async () => {
            const worker = new Worker(new URL('../../webworkers/intelligent-scissors.worker', import.meta.url), {
                type: 'module',
            });
            const intelligentScissorsWorker = wrap<IntelligentScissorsWorkerApi>(worker);
            const instance = await intelligentScissorsWorker.build();

            return { worker, instance };
        },
        staleTime: Infinity,
        enabled,
    });

    return { worker: data?.instance, isLoading, isSuccess, isError };
};
