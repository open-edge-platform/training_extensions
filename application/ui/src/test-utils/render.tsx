// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Suspense, type ReactNode } from 'react';

import { IntelBrandedLoading, Toast } from '@geti/ui';
import { ThemeProvider } from '@geti/ui/theme';
import { QueryClientProvider } from '@tanstack/react-query';
import {
    render as rtlRender,
    renderHook as rtlRenderHook,
    RenderOptions as RTLRenderOptions,
} from '@testing-library/react';
import { createMemoryRouter, RouterProvider } from 'react-router';

import { paths } from '../constants/paths';
import { queryClient } from '../query-client/query-client';

export interface RenderOptions extends RTLRenderOptions {
    route?: string;
    path?: string;
}

export const TestProviders = ({ children }: { children: ReactNode }) => {
    return (
        <QueryClientProvider client={queryClient}>
            <ThemeProvider>
                <Suspense fallback={<IntelBrandedLoading />}>{children}</Suspense>
                <Toast />
            </ThemeProvider>
        </QueryClientProvider>
    );
};

const createTestRouter = (children: ReactNode, options: RenderOptions) => {
    const route = options.route ?? paths.project.details({ projectId: '123' });
    const path = options.path ?? paths.project.details.pattern;

    return createMemoryRouter(
        [
            {
                path,
                element: <TestProviders>{children}</TestProviders>,
            },
        ],
        {
            initialEntries: [route],
            initialIndex: 0,
        }
    );
};

export const render = (ui: ReactNode, options: RenderOptions = {}) => {
    const Wrapper = ({ children }: { children: ReactNode }) => {
        const router = createTestRouter(children, options);

        return <RouterProvider router={router} />;
    };

    return rtlRender(ui, { wrapper: Wrapper, ...options });
};

export const renderHook = <TProps, TResult>(callback: (props: TProps) => TResult, options: RenderOptions = {}) => {
    const Wrapper = ({ children }: { children: ReactNode }) => {
        const router = createTestRouter(children, options);

        return <RouterProvider router={router} />;
    };

    return rtlRenderHook(callback, { wrapper: Wrapper });
};
