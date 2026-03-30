// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Toast } from '@geti/ui';
import { ThemeProvider } from '@geti/ui/theme';
import { QueryClientProvider } from '@tanstack/react-query';
import { RouterProvider } from 'react-router';

import { queryClient } from './query-client/query-client';
import { router } from './router';

export const Providers = () => {
    return (
        <QueryClientProvider client={queryClient}>
            <ThemeProvider router={router}>
                <RouterProvider
                    router={router}
                    future={{
                        v7_startTransition: true,
                    }}
                />
                <div data-react-aria-top-layer='true'>
                    <Toast />
                </div>
            </ThemeProvider>
        </QueryClientProvider>
    );
};
