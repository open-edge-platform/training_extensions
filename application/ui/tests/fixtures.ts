// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

import { createNetworkFixture, NetworkFixture } from '@msw/playwright';
import { expect, test as testBase } from '@playwright/test';
import { HttpResponse } from 'msw';

import { handlers, http } from '../src/api/utils';

const filename = fileURLToPath(import.meta.url);
const dirname = path.dirname(filename);

interface Fixtures {
    network: NetworkFixture;
}

const test = testBase.extend<Fixtures>({
    network: createNetworkFixture({
        initialHandlers: [
            ...handlers,
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
                    data_collection_policies: [],
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
        ],
    }),
});

export { expect, http, test };
