// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

import { createNetworkFixture, NetworkFixture } from '@msw/playwright';
import { expect, test as testBase } from '@playwright/test';
import { HttpResponse } from 'msw';

import { handlers, http } from '../src/api/utils';
import { StreamPage } from './inference/stream-page';
import { JobsPage } from './jobs/jobs-page';
import { ModelsPage } from './models/models-page';

const filename = fileURLToPath(import.meta.url);
const dirname = path.dirname(filename);

interface Fixtures {
    network: NetworkFixture;
    streamPage: StreamPage;
    modelsPage: ModelsPage;
    jobsPage: JobsPage;
}

const test = testBase.extend<Fixtures>({
    network: createNetworkFixture({
        initialHandlers: [
            ...handlers,
            http.get('/health', ({ response }) => {
                return response(200).json({
                    status: 'ok',
                });
            }),
            http.get('/api/system/metrics/memory', ({ response }) => {
                return response(200).json({});
            }),
            http.get('/api/projects/{project_id}/models', ({ response }) => {
                return response(200).json([]);
            }),
            http.get('/api/projects/{project_id}/pipeline', ({ response }) => {
                return response(200).json({
                    project_id: 'id-1',
                    status: 'idle',
                    source: null,
                    sink: null,
                    model: null,
                    device: 'cpu',
                });
            }),
            http.post('/api/projects/{project_id}/pipeline:enable', () => {
                return HttpResponse.json(null, { status: 204 });
            }),
            http.get('/api/projects', ({ response }) => {
                return response(200).json([
                    {
                        id: 'id-1',
                        name: 'Project 1',
                        task: {
                            task_type: 'detection',
                            exclusive_labels: false,
                            labels: [],
                        },
                        active_pipeline: false,
                    },
                ]);
            }),
            http.get('/api/projects/{project_id}', () => {
                return HttpResponse.json({
                    id: '123',
                    name: 'Test Project',
                    task: {
                        task_type: 'detection',
                        exclusive_labels: false,
                        labels: [
                            { id: '1', color: 'red', name: 'person' },
                            { id: '2', color: 'blue', name: 'car' },
                        ],
                    },
                    active_pipeline: true,
                });
            }),
            http.delete('/api/projects/{project_id}', () => {
                return HttpResponse.json(null, { status: 204 });
            }),
            http.post('/api/webrtc/offer', ({ response }) => {
                return response(200).json({
                    type: 'answer',
                    sdp: 'v=0\r\no=- 0 0 IN IP4 127.0.0.1\r\n',
                } as never);
            }),
            http.post('/api/webrtc/input_hook', ({ response }) => {
                // Schema is empty, so we return an empty object
                return response(200).json({} as never);
            }),
            http.get('/api/projects/{project_id}/dataset/items/{dataset_item_id}/thumbnail', ({}) => {
                const sampleImagePath = path.resolve(dirname, './assets/candy-thumbnail.png');
                const sampleImageBuffer = fs.readFileSync(sampleImagePath);

                return HttpResponse.arrayBuffer(sampleImageBuffer.buffer, {
                    headers: { 'content-type': 'image/png' },
                });
            }),
            http.get('/api/system/devices/inference', ({ response }) => {
                return response(200).json([
                    { type: 'cpu', name: 'CPU' },
                    { type: 'xpu', name: 'XPU' },
                ]);
            }),
        ],
    }),
    streamPage: async ({ page }, use) => {
        const streamPage = new StreamPage(page);

        await use(streamPage);
    },
    modelsPage: async ({ page }, use) => {
        const modelsPage = new ModelsPage(page);

        await use(modelsPage);
    },
    jobsPage: async ({ page }, use) => {
        const jobsPage = new JobsPage(page);

        await use(jobsPage);
    },
});

export { expect, http, test };
