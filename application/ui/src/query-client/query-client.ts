// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { toast } from '@geti/ui';
import { MutationCache, QueryClient } from '@tanstack/react-query';

import { paths } from '../api/openapi-spec';
import { Meta, QueryKey } from './query-client.interface';

declare module '@tanstack/react-query' {
    interface Register {
        mutationMeta: Meta;
        queryMeta: Meta;
    }
}

const TOAST_DURATION = 5000;

const getErrorMessage = (error: unknown): string => {
    if (!error || typeof error !== 'object') {
        return 'An unexpected error occurred. Please try again.';
    }
    if ('detail' in error && typeof error.detail === 'string') {
        return error.detail;
    }
    if ('message' in error && typeof error.message === 'string') {
        const message = error.message;
        if (error instanceof TypeError && message.includes('Failed to fetch')) {
            return 'Network error. Please check your connection and try again.';
        }
        return message;
    }
    return 'An unexpected error occurred. Please try again.';
};

export const queryClient = new QueryClient({
    defaultOptions: {
        queries: {
            retry: false,
        },
        mutations: {
            retry: false,
        },
    },
    mutationCache: new MutationCache({
        onSuccess: (_data, _variables, _context, mutation) => {
            const meta = mutation.meta;
            const invalidateQueries = meta?.invalidateQueries;

            if (invalidateQueries) {
                invalidateQueries.forEach((queryKey) => {
                    queryClient.invalidateQueries({ queryKey });
                });
            }
        },
        onError: (error) => {
            toast({
                type: 'error',
                message: getErrorMessage(error),
                duration: TOAST_DURATION,
            });
        },
    }),
});

/**
 * Returns the provided query key.
 * Helper to centralize construction and typing of React Query keys.
 * It acts as an identity function.
 *
 * @param queryKey - The query key to return.
 * @returns The same query key.
 */
export const getQueryKey = (queryKey: QueryKey<paths>): QueryKey<paths> => {
    return queryKey;
};
