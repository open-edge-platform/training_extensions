// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Suspense } from 'react';

import { Loading } from '@geti/ui';
import { redirect } from 'react-router';
import { createBrowserRouter } from 'react-router-dom';

import { ZoomProvider } from './components/zoom/zoom';
import { paths } from './constants/paths';
import { WebRTCConnectionProvider } from './features/inference/stream/web-rtc-connection-provider';
import { ProjectList } from './features/project/list/project-list.component';
import { Layout } from './layout';
import { Dataset } from './routes/dataset/dataset.component';
import { SelectedDataProvider } from './routes/dataset/provider';
import { ErrorPage } from './routes/error-page/error-page';
import { Inference } from './routes/inference/inference';
import { Models } from './routes/models/models';
import { CreateProject } from './routes/project/create-project';
import { ViewProject } from './routes/project/view-project';

export const router = createBrowserRouter([
    {
        path: paths.project.index.pattern,
        element: (
            <Suspense fallback={<Loading mode='fullscreen' />}>
                <ProjectList />
            </Suspense>
        ),
    },
    {
        path: paths.project.new.pattern,
        element: <CreateProject />,
    },
    {
        path: paths.project.details.pattern,
        element: <ViewProject />,
    },
    {
        path: paths.root.pattern,
        element: (
            <Suspense fallback={<Loading mode='fullscreen' />}>
                <Layout />
            </Suspense>
        ),
        errorElement: <ErrorPage />,
        children: [
            {
                index: true,
                loader: () => {
                    // TODO: If there is no project configured then redirect to new project creation
                    // else redirect to inference
                    return redirect(paths.project.index({}));
                },
            },
            {
                path: paths.project.inference.pattern,
                element: (
                    <WebRTCConnectionProvider>
                        <ZoomProvider>
                            <Inference />
                        </ZoomProvider>
                    </WebRTCConnectionProvider>
                ),
            },
            {
                path: paths.project.dataset.pattern,
                element: (
                    <ZoomProvider>
                        <SelectedDataProvider>
                            <Dataset />
                        </SelectedDataProvider>
                    </ZoomProvider>
                ),
            },
            {
                path: paths.project.models.pattern,
                element: <Models />,
            },
        ],
    },
]);
