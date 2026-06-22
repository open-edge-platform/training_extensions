// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

import { getMockedDatasetRevision } from 'mocks/mock-dataset-revision';
import { getMockedModel } from 'mocks/mock-model';
import { getMockedProject } from 'mocks/mock-project';
import { HttpResponse } from 'msw';

import { expect, http, test } from '../fixtures';

const filename = fileURLToPath(import.meta.url);
const dirname = path.dirname(filename);
const candyPngBuffer = fs.readFileSync(path.resolve(dirname, '../assets/candy.png'));

const ITEM_ID = 'a1b2c3d4-e5f6-7890-abcd-ef1234567890';
const DATASET_REVISION_ID = 'dataset-1';
const MODEL_ID = 'model-1';

const mockedModel = getMockedModel({
    id: MODEL_ID,
    name: 'YOLOX Model v1',
    architecture: 'Object_Detection_YOLOX_X',
    training_info: {
        status: 'successful',
        label_schema_revision: { labels: [] },
        start_time: '2025-01-10T10:00:00.000000+00:00',
        end_time: '2025-01-10T12:30:00.000000+00:00',
        dataset_revision_id: DATASET_REVISION_ID,
    },
});

const mockedDatasetRevision = getMockedDatasetRevision({
    id: DATASET_REVISION_ID,
    name: 'Dataset Revision 1',
    item_counts: { training: 1, validation: 0, testing: 0, total: 1 },
});

const mockedTrainingItem = {
    id: ITEM_ID,
    format: 'jpg' as const,
    width: 1000,
    height: 750,
    subset: 'training' as const,
};

const mockedSegmentationProject = getMockedProject({
    task: {
        exclusive_labels: true,
        task_type: 'instance_segmentation',
        labels: [{ id: 'label-seg', name: 'car', color: '#2424a0', hotkey: 'B' }],
    },
});

test.describe('Subset Gallery — read-only dialog', () => {
    test('opens, toolbar controls work, and segmentation polygon annotations render without crash', async ({
        page,
        network,
        modelsPage,
    }) => {
        network.use(
            http.get('/api/projects/{project_id}', () => HttpResponse.json(mockedSegmentationProject)),
            http.get('/api/projects/{project_id}/models', () => HttpResponse.json([mockedModel])),
            http.get('/api/projects/{project_id}/models/{model_id}', ({ params }) => {
                if (params.model_id === MODEL_ID) {
                    return HttpResponse.json(mockedModel);
                }

                return new HttpResponse(null, { status: 404 });
            }),
            http.get('/api/projects/{project_id}/dataset_revisions', () => HttpResponse.json([mockedDatasetRevision])),
            http.get('/api/projects/{project_id}/dataset_revisions/{dataset_revision_id}/items', ({ request }) => {
                const url = new URL(request.url);
                const subset = url.searchParams.get('subset');
                const items = subset === 'training' ? [mockedTrainingItem] : [];

                return HttpResponse.json({
                    items,
                    pagination: { offset: 0, limit: 20, count: items.length, total: items.length },
                });
            }),
            http.get(
                '/api/projects/{project_id}/dataset_revisions/{dataset_revision_id}/items/{dataset_item_id}/thumbnail',
                () =>
                    HttpResponse.arrayBuffer(candyPngBuffer.buffer, {
                        headers: { 'content-type': 'image/png' },
                    })
            ),
            http.get('/api/projects/{project_id}/dataset/media/{media_id}/binary', () =>
                HttpResponse.arrayBuffer(candyPngBuffer.buffer, {
                    headers: { 'content-type': 'image/png' },
                })
            ),
            http.get('/api/projects/{project_id}/dataset/media/{media_id}/annotations', () =>
                HttpResponse.json({
                    annotations: [
                        {
                            shape: {
                                type: 'polygon',
                                points: [
                                    { x: 10, y: 20 },
                                    { x: 110, y: 20 },
                                    { x: 110, y: 70 },
                                ],
                            },
                            labels: [{ id: 'label-seg' }],
                        },
                    ],
                    user_reviewed: true,
                    subset: 'training',
                })
            )
        );

        await modelsPage.goto();
        await modelsPage.expandModel('YOLOX Model v1');
        await modelsPage.clickTrainingDatasetsTab();

        await expect(page.getByAltText('training item')).toBeVisible();
        await page.getByAltText('training item').dblclick();

        const dialog = page.getByRole('dialog');
        await expect(dialog).toBeVisible();

        await expect(dialog.getByLabel('annotation polygon').first()).toBeVisible();

        const toggleFocusButton = page.getByRole('button', { name: 'Toggle focus' });
        await toggleFocusButton.click();
        await expect(toggleFocusButton).toBeVisible();

        await page.getByRole('button', { name: 'Settings' }).click();
        await expect(page.getByRole('heading', { name: 'Settings' })).toBeVisible();

        await page.getByRole('button', { name: 'Close settings' }).click();
        await expect(page.getByRole('heading', { name: 'Settings' })).toBeHidden();

        await page.getByRole('button', { name: 'Close', exact: true }).click();
        await expect(dialog).toBeHidden();
    });
});
