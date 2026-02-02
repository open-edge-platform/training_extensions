// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { IntelligentScissors } from '@geti/smart-tools';
import { useQuery } from '@tanstack/react-query';
import { Remote, wrap } from 'comlink';

export const useIntelligentScissorsWorker = () => {
    const { data, isLoading, isSuccess, isError } = useQuery<Remote<IntelligentScissors>>({
        queryKey: ['workers', 'INTELLIGENT_SCISSORS'],
        queryFn: async () => {
            const baseWorker = new Worker(new URL('../../webworkers/intelligent-scissors.worker', import.meta.url), {
                type: 'module',
            });
            const intelligentScissorsWorker = wrap(baseWorker);

            // @ts-expect-error build exists on every worker
            return intelligentScissorsWorker.build();
        },
        staleTime: Infinity,
    });

    return { worker: data, isLoading, isSuccess, isError };
};
