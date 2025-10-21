// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Toast } from '@geti/ui';
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
};

declare module '@tanstack/react-query' {
    interface Register {
        mutationMeta: MutationMeta;
    }
}

export const queryClient = new QueryClient({
    defaultOptions: {
        queries: {
            gcTime: 30 * 60 * 1000,
            staleTime: 5 * 60 * 1000,
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
    }),
});

export const Providers = () => {
    return (
        <QueryClientProvider client={queryClient}>
            <ThemeProvider router={router}>
                <RouterProvider router={router} />
                <Toast />
            </ThemeProvider>
        </QueryClientProvider>
    );
};
