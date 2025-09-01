// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useQuery } from '@tanstack/react-query';
import { wrap } from 'comlink';

import { AlgorithmType } from './algorithm.interface';

export const getWorker = (): Worker => {
    return new Worker(new URL('./segment-anything.worker', import.meta.url), { type: 'module' });
};

export const useLoadAIWebworker = <T extends AlgorithmType>(algorithmType: T) => {
    const { data, isLoading, isSuccess, isError } = useQuery({
        queryKey: ['workers', algorithmType],
        queryFn: async () => {
            const baseWorker = getWorker();
            const worker = wrap(baseWorker);

            // @ts-expect-error build exists on every worker
            return worker.build();
        },
        staleTime: Infinity,
    });

    return { worker: data, isLoading, isSuccess, isError };
};
