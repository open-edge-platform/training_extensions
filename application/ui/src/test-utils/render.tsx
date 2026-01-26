// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Suspense, type ReactNode } from 'react';

import { IntelBrandedLoading, Toast } from '@geti/ui';
import { ThemeProvider } from '@geti/ui/theme';
import { QueryClientProvider } from '@tanstack/react-query';
import { render, RenderOptions } from '@testing-library/react';
import { createMemoryRouter, RouterProvider } from 'react-router';

import { paths } from '../constants/paths';
import { queryClient } from '../providers';

interface Options extends RenderOptions {
    route: string;
    path: string;
}

export const TestProviders = ({
    children,
    client = queryClient,
}: {
    children: ReactNode;
    client?: typeof queryClient;
}) => {
    return (
        <QueryClientProvider client={client}>
            <ThemeProvider>
                <Suspense fallback={<IntelBrandedLoading />}>{children}</Suspense>
                <Toast />
            </ThemeProvider>
        </QueryClientProvider>
    );
};

const customRender = (
    ui: ReactNode,
    options: Options = { route: paths.project.index({}), path: paths.project.index.pattern }
) => {
    const router = createMemoryRouter(
        [
            {
                path: options.path,
                element: <TestProviders>{ui}</TestProviders>,
            },
        ],
        {
            initialEntries: [options.route],
            initialIndex: 0,
        }
    );

    return render(<RouterProvider router={router} />);
};

// eslint-disable-next-line import/export
export * from '@testing-library/react';
// eslint-disable-next-line import/export
export { customRender as render };
