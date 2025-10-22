// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Suspense } from 'react';

import { Loading } from '@geti/ui';
import { createBrowserRouter, Navigate, Outlet } from 'react-router-dom';

import { paths } from './constants/paths';
import { SelectedDataProvider } from './features/dataset/selected-data-provider.component';
import { WebRTCConnectionProvider } from './features/inference/stream/web-rtc-connection-provider';
import { ProjectList } from './features/project/list/project-list.component';
import { useProjects } from './hooks/api/project.hook';
import { Layout } from './layout';
import { Dataset } from './routes/dataset/dataset.component';
import { ErrorPage } from './routes/error-page/error-page';
import { Inference } from './routes/inference/inference';
import { Models } from './routes/models/models';
import { CreateProject } from './routes/project/create-project';
import { ViewProject } from './routes/project/view-project';

const Redirect = () => {
    let path = paths.project.index({});

    const { data: projects } = useProjects();

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
        // More than 1 project -> Redirect the active one
        // And if none are active, redirect to /projects
        const projectWithActivePipeline = projects.find((project) => Boolean(project.active_pipeline));

        if (projectWithActivePipeline) {
            path = paths.project.inference({ projectId: projectWithActivePipeline.id });
        } else {
            path = paths.project.index({});
        }
    }

    return <Navigate to={path} replace />;
};

export const router = createBrowserRouter([
    {
        path: paths.root.pattern,
        element: (
            <Suspense fallback={<Loading mode='fullscreen' />}>
                <Outlet />
            </Suspense>
        ),
        errorElement: <ErrorPage />,
        children: [
            {
                index: true,
                element: <Redirect />,
            },
            {
                path: paths.project.index.pattern,
                element: <ProjectList />,
            },
            {
                path: paths.project.new.pattern,
                element: <CreateProject />,
            },
            {
                path: paths.project.details.pattern,
                element: (
                    <WebRTCConnectionProvider>
                        <Layout />
                    </WebRTCConnectionProvider>
                ),
                children: [
                    {
                        index: true,
                        element: <ViewProject />,
                    },
                    {
                        path: paths.project.inference.pattern,
                        element: <Inference />,
                    },
                    {
                        path: paths.project.dataset.pattern,
                        element: (
                            <SelectedDataProvider>
                                <Dataset />
                            </SelectedDataProvider>
                        ),
                    },
                    {
                        path: paths.project.models.pattern,
                        element: <Models />,
                    },
                ],
            },
        ],
    },
]);
