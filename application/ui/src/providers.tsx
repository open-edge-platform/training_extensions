// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Toast } from '@geti/ui';
import { ThemeProvider } from '@geti/ui/theme';
import { MutationCache, QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { RouterProvider } from 'react-router';

import { router } from './router';

// Type for mutation meta with query invalidation
export type MutationMeta = {
    invalidateQueries?: Array<string | string[]>;
};

export const queryClient = new QueryClient({
    defaultOptions: {
        queries: {
            gcTime: 30 * 60 * 1000,
            staleTime: 5 * 60 * 1000,
        },
    },
    mutationCache: new MutationCache({
        onSuccess: (_data, _variables, _context, mutation) => {
            const meta = mutation.meta as MutationMeta | undefined;
            const invalidateQueries = meta?.invalidateQueries;

            if (invalidateQueries) {
                invalidateQueries.forEach((queryKey) => {
                    const key = Array.isArray(queryKey) ? queryKey : [queryKey];
                    queryClient.invalidateQueries({ queryKey: key });
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
