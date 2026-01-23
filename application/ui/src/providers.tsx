// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { toast } from '@geti/ui';
import { ThemeProvider } from '@geti/ui/theme';
import { MutationCache, QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { RouterProvider } from 'react-router';

import type { paths } from './api/openapi-spec';
import { router } from './router';

type HttpMethod = 'get';
type PathsWithMethod<M extends HttpMethod> = {
    [P in keyof paths]: M extends keyof paths[P] ? P : never;
}[keyof paths];
type QueryKey = [HttpMethod, PathsWithMethod<HttpMethod>];
type MutationMeta = {
    invalidateQueries?: QueryKey[];
    errorMessage?: string;
};

declare module '@tanstack/react-query' {
    interface Register {
        mutationMeta: MutationMeta;
    }
}

const TOAST_DURATION = 5000;

const getErrorMessage = (error: unknown, customMessage?: string): string => {
    if (customMessage) {
        return customMessage;
    }

    if (error && typeof error === 'object') {
        if ('detail' in error && typeof error.detail === 'string') {
            return error.detail;
        }

        if ('message' in error && typeof error.message === 'string') {
            if (error instanceof TypeError && error.message.includes('Failed to fetch')) {
                return 'Network error. Please check your connection and try again.';
            }

            return error.message;
        }
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
        onError: (error, _variables, _context, mutation) => {
            toast({
                type: 'error',
                message: getErrorMessage(error, mutation.meta?.errorMessage),
                duration: TOAST_DURATION,
            });
        },
    }),
});

export const Providers = () => {
    return (
        <QueryClientProvider client={queryClient}>
            <ThemeProvider router={router}>
                <RouterProvider router={router} />
            </ThemeProvider>
        </QueryClientProvider>
    );
};
