import { Suspense } from 'react';

import { Loading } from '@geti/ui';
import { redirect } from 'react-router';
import { createBrowserRouter } from 'react-router-dom';
import { path } from 'static-path';

import { Layout } from './layout';
import { LiveFeed } from './routes/live-feed/live-feed';
import { EditPipelineLayout } from './routes/pipeline/edit-pipeline-layout';
import { Index as PipelineIndex } from './routes/pipeline/index';
import { Input as PipelineInput } from './routes/pipeline/input';
import { Model as PipelineModel } from './routes/pipeline/model';
import { Output as PipelineOutput } from './routes/pipeline/output';

const root = path('/');
const pipeline = root.path('/pipeline');
const liveFeed = root.path('/live-feed');
const dataCollection = root.path('/data-collection');

export const paths = {
    root,
    pipeline: {
        index: pipeline,
        input: pipeline.path('/input'),
        model: pipeline.path('/model'),
        output: pipeline.path('/output'),
    },
    liveFeed: {
        index: liveFeed,
    },
    dataCollection: {
        index: dataCollection,
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
                    // TODO: if no pipeline configured then redirect to input
                    // else redirect to live-feed
                    return redirect('/pipeline/input');
                },
            },
            {
                path: paths.pipeline.index.pattern,
                children: [
                    {
                        index: true,
                        path: paths.pipeline.index.pattern,
                        element: <PipelineIndex />,
                    },
                    {
                        element: <EditPipelineLayout />,
                        children: [
                            {
                                path: paths.pipeline.input.pattern,
                                element: <PipelineInput />,
                            },

                            {
                                path: paths.pipeline.model.pattern,
                                element: <PipelineModel />,
                            },
                            {
                                path: paths.pipeline.output.pattern,
                                element: <PipelineOutput />,
                            },
                        ],
                    },
                ],
            },
            {
                path: paths.liveFeed.index.pattern,
                element: <LiveFeed />,
            },
            {
                path: paths.dataCollection.index.pattern,
                element: <div>Data collection</div>,
            },
        ],
    },
]);
