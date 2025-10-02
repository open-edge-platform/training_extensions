// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Suspense } from 'react';

import { Loading } from '@geti/ui';
import { createBrowserRouter, Navigate } from 'react-router-dom';

import { $api } from './api/client';
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

const Redirect = () => {
    let path = paths.project.index({});

    const { data: projects } = $api.useSuspenseQuery('get', '/api/projects');

    // No projects -> Go to create project
    if (!projects || projects.length === 0) {
        path = paths.project.new({});

        // Only 1 project -> Redirect to the inference page
    } else if (projects.length === 1) {
        const projectId = projects[0].id;

        if (projectId) {
            path = paths.project.inference({ projectId });
        } else {
            path = paths.project.new({});
        }
    } else {
        // More than 1 project -> Load index page (/projects)
        path = paths.project.index({});
    }

    return <Navigate to={path} replace />;
};

export const router = createBrowserRouter([
    {
        path: paths.root.pattern,
        element: (
            <Suspense fallback={<Loading />}>
                <Redirect />
            </Suspense>
        ),
    },
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
        element: (
            <Suspense fallback={<Loading mode='fullscreen' />}>
                <Layout />
            </Suspense>
        ),
        errorElement: <ErrorPage />,
        children: [
            {
                index: true,
                element: <ViewProject />,
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
