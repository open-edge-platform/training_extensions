// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Suspense } from 'react';

import { Loading } from '@geti/ui';
import { redirect } from 'react-router';
import { createBrowserRouter } from 'react-router-dom';
import { path } from 'static-path';

import { ZoomProvider } from './components/zoom/zoom';
import { WebRTCConnectionProvider } from './features/inference/stream/web-rtc-connection-provider';
import { Layout } from './layout';
import { Dataset } from './routes/dataset/dataset.component';
import { SelectedDataProvider } from './routes/dataset/provider';
import { Inference } from './routes/inference/inference';
import { Model } from './routes/pipeline/model';

const root = path('/');
const pipeline = root.path('/pipeline');
const inference = root.path('/inference');
const dataset = root.path('/dataset');

export const paths = {
    root,
    pipeline: {
        index: pipeline,
        model: pipeline.path('/model'),
    },
    inference: {
        index: inference,
    },
    dataset: {
        index: dataset,
    },
};

export const router = createBrowserRouter([
    {
        path: paths.root.pattern,
        element: (
            <Suspense fallback={<Loading mode='fullscreen' />}>
                <Layout />
            </Suspense>
        ),
        errorElement: <div>Oh no</div>,
        children: [
            {
                index: true,
                loader: () => {
                    // TODO: if no pipeline configured then redirect to create-pipeline
                    // else redirect to inference
                    return redirect(paths.pipeline.model({}));
                },
            },
            {
                path: paths.pipeline.index.pattern,
                children: [
                    {
                        index: true,
                        path: paths.pipeline.model.pattern,
                        element: <Model />,
                    },
                ],
            },
            {
                path: paths.inference.index.pattern,
                element: (
                    <WebRTCConnectionProvider>
                        <Inference />
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
        ],
    },
]);
