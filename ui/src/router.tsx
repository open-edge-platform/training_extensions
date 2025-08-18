import { Suspense } from 'react';

import { Loading } from '@geti/ui';
import { redirect } from 'react-router';
import { createBrowserRouter } from 'react-router-dom';
import { path } from 'static-path';

import { ZoomProvider } from './components/zoom/zoom';
import { Layout } from './layout';
import { DataCollection } from './routes/data-collection/data-collection.component';
import { SelectedDataProvider } from './routes/data-collection/provider';
import { LiveFeed } from './routes/live-feed/live-feed';
import { EditPipelineLayout } from './routes/pipeline/edit-pipeline-layout';
import { Index as PipelineIndex } from './routes/pipeline/index';
import { Model as PipelineModel } from './routes/pipeline/model';
import { Sink as PipelineSink } from './routes/pipeline/sink';
import { Source as PipelineSource } from './routes/pipeline/source';

const root = path('/');
const pipeline = root.path('/pipeline');
const liveFeed = root.path('/live-feed');
const dataCollection = root.path('/data-collection');

export const paths = {
    root,
    pipeline: {
        index: pipeline,
        source: pipeline.path('/source'),
        model: pipeline.path('/model'),
        sink: pipeline.path('/sink'),
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
                    // TODO: if no pipeline configured then redirect to source
                    // else redirect to live-feed
                    return redirect('/pipeline/source');
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
                                path: paths.pipeline.source.pattern,
                                element: <PipelineSource />,
                            },

                            {
                                path: paths.pipeline.model.pattern,
                                element: <PipelineModel />,
                            },
                            {
                                path: paths.pipeline.sink.pattern,
                                element: <PipelineSink />,
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
                element: (
                    <ZoomProvider>
                        <SelectedDataProvider>
                            <DataCollection />
                        </SelectedDataProvider>
                    </ZoomProvider>
                ),
            },
        ],
    },
]);
