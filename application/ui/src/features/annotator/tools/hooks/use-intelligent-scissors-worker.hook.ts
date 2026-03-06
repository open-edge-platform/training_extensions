// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useQuery } from '@tanstack/react-query';
import { Remote, wrap } from 'comlink';

import type {
    IntelligentScissorsWorkerApi,
    IntelligentScissorsWorkerInstance,
} from '../../webworkers/intelligent-scissors.worker.interface';

type IntelligentScissorsRemoteInstance = Remote<IntelligentScissorsWorkerInstance>;

export const useIntelligentScissorsWorker = () => {
    const { data, isLoading, isSuccess, isError } = useQuery<IntelligentScissorsRemoteInstance>({
        queryKey: ['workers', 'INTELLIGENT_SCISSORS'],
        queryFn: async () => {
            const baseWorker = new Worker(new URL('../../webworkers/intelligent-scissors.worker', import.meta.url), {
                type: 'module',
            });
            const intelligentScissorsWorker = wrap<IntelligentScissorsWorkerApi>(baseWorker);

            return intelligentScissorsWorker.build();
        },
        staleTime: Infinity,
    });

    return { worker: data, isLoading, isSuccess, isError };
};
