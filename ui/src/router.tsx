// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Suspense } from 'react';

import { Loading } from '@geti/ui';
import { redirect } from 'react-router';
import { createBrowserRouter } from 'react-router-dom';
import { path } from 'static-path';

import { ZoomProvider } from './components/zoom/zoom';
import { WebRTCConnectionProvider } from './features/inference/stream/web-rtc-connection-provider';
import { ProjectDetails } from './features/project/project-details.component';
import { Layout } from './layout';
import { Dataset } from './routes/dataset/dataset.component';
import { SelectedDataProvider } from './routes/dataset/provider';
import { ErrorPage } from './routes/error-page/error-page';
import { Inference } from './routes/inference/inference';
import { Labels } from './routes/labels/labels';
import { Models } from './routes/models/models';
import { CreateProject } from './routes/project/create-project';
import { EditProject } from './routes/project/edit-project';

const root = path('/');
const project = root.path('/project');
const inference = root.path('/inference');
const dataset = root.path('/dataset');
const models = root.path('/models');
const labels = root.path('/labels');

export const paths = {
    root,
    project: {
        index: project,
        new: project.path('/new'),
        edit: project.path('/edit/:projectId'),
    },
    inference: {
        index: inference,
    },
    dataset: {
        index: dataset,
    },
    models: {
        index: models,
    },
    labels: {
        index: labels,
    },
};

export const router = createBrowserRouter([
    {
        path: paths.project.new.pattern,
        element: <CreateProject />,
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
                    return redirect(paths.project.new({}));
                },
            },
            {
                path: paths.project.index.pattern,
                children: [
                    {
                        index: true,
                        path: paths.project.index.pattern,
                        element: <ProjectDetails />,
                    },
                    {
                        path: paths.project.edit.pattern,
                        element: <EditProject />,
                    },
                ],
            },
            {
                path: paths.inference.index.pattern,
                element: (
                    <WebRTCConnectionProvider>
                        <ZoomProvider>
                            <Inference />
                        </ZoomProvider>
                    </WebRTCConnectionProvider>
                ),
            },
            {
                path: paths.dataset.index.pattern,
                element: (
                    <ZoomProvider>
                        <SelectedDataProvider>
                            <Dataset />
                        </SelectedDataProvider>
                    </ZoomProvider>
                ),
            },
            {
                path: paths.models.index.pattern,
                element: <Models />,
            },
            {
                path: paths.labels.index.pattern,
                element: <Labels />,
            },
        ],
    },
]);
