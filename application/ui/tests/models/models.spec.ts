// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { getMockedExtendedModel, getMockedModel } from 'mocks/mock-model';
import { getMockedProject } from 'mocks/mock-project';
import { HttpResponse } from 'msw';

import { expect, http, test } from '../fixtures';

const mockedModels = [
    getMockedModel({
        id: 'model-1',
        name: 'YOLOX Model v1',
        architecture: 'Object_Detection_YOLOX_X',
        training_info: {
            status: 'successful',
            label_schema_revision: { labels: [] },
            configuration: {},
            start_time: '2025-01-10T10:00:00.000000+00:00',
            end_time: '2025-01-10T12:30:00.000000+00:00',
            dataset_revision_id: 'dataset-1',
        },
    }),
    getMockedModel({
        id: 'model-2',
        name: 'YOLOX Model v2',
        architecture: 'Object_Detection_YOLOX_X',
        parent_revision: 'model-1',
        training_info: {
            status: 'successful',
            label_schema_revision: { labels: [] },
            configuration: {},
            start_time: '2025-01-11T10:00:00.000000+00:00',
            end_time: '2025-01-11T14:00:00.000000+00:00',
            dataset_revision_id: 'dataset-1',
        },
    }),
    getMockedModel({
        id: 'model-3',
        name: 'SSD Model',
        architecture: 'Object_Detection_SSD',
        training_info: {
            status: 'successful',
            label_schema_revision: { labels: [] },
            configuration: {},
            start_time: '2025-01-12T08:00:00.000000+00:00',
            end_time: '2025-01-12T10:00:00.000000+00:00',
            dataset_revision_id: 'dataset-2',
        },
    }),
];

test.describe('Models', () => {
    test.beforeEach(({ network }) => {
        network.use(
            http.get('/api/projects/{project_id}', () => {
                return HttpResponse.json(getMockedProject({ id: 'id-1' }));
            }),
            http.get('/api/projects/{project_id}/models', () => {
                return HttpResponse.json(mockedModels);
            }),
            http.get('/api/projects/{project_id}/models/{model_id}', ({ params }) => {
                const foundModel = mockedModels.find((model) => model.id === params.model_id);

                if (foundModel) {
                    return HttpResponse.json(getMockedExtendedModel(foundModel));
                }

                return new HttpResponse(null, { status: 404 });
            }),
            http.patch('/api/projects/{project_id}/models/{model_id}', async ({ request, params }) => {
                const body = (await request.json()) as { name: string };
                const model = mockedModels.find((model) => model.id === params.model_id);

                if (model) {
                    return HttpResponse.json({ ...model, name: body.name });
                }

                return new HttpResponse(null, { status: 404 });
            }),
            http.delete('/api/projects/{project_id}/models/{model_id}', () => {
                return HttpResponse.json(null, { status: 204 });
            })
        );
    });

    test('displays models list', async ({ modelsPage }) => {
        await modelsPage.goto();

        await expect(modelsPage.getModelByName('YOLOX Model v1')).toBeVisible();
        await expect(modelsPage.getModelByName('YOLOX Model v2')).toBeVisible();
        await expect(modelsPage.getModelByName('SSD Model')).toBeVisible();
    });

    test('can expand model to show details tabs', async ({ modelsPage, page }) => {
        await modelsPage.goto();

        // Expand the first model
        await modelsPage.expandModel('YOLOX Model v1');

        const detailsTabs = page.getByRole('tablist', { name: 'Model details' });
        await expect(detailsTabs).toBeVisible();

        await expect(page.getByRole('tab', { name: 'Model variants' })).toBeVisible();
        await expect(page.getByRole('tab', { name: 'Model metrics' })).toBeVisible();
        await expect(page.getByRole('tab', { name: 'Training parameters' })).toBeVisible();
        await expect(page.getByRole('tab', { name: 'Training datasets' })).toBeVisible();
    });

    test('can search models by name', async ({ modelsPage }) => {
        await modelsPage.goto();

        await modelsPage.searchModels('SSD');

        await expect(modelsPage.getModelByName('SSD Model')).toBeVisible();
        await expect(modelsPage.getModelByName('YOLOX Model v1')).toBeHidden();
    });

    test('shows empty results when search has no matches', async ({ modelsPage, page }) => {
        await modelsPage.goto();

        await modelsPage.searchModels('NonExistentModel');

        await expect(page.getByRole('heading', { name: 'No models found' })).toBeVisible();
    });

    test('can change group by option', async ({ modelsPage, page }) => {
        await modelsPage.goto();

        await modelsPage.selectGroupBy('architecture');

        await expect(page.getByRole('heading', { name: 'Object_Detection_YOLOX_X', level: 2 })).toBeVisible();
        await expect(page.getByRole('heading', { name: 'Object_Detection_SSD', level: 2 })).toBeVisible();
    });

    test('can change sort order', async ({ modelsPage }) => {
        await modelsPage.goto();

        await modelsPage.selectSortBy('name');

        // Sorted alphabetically: v1 comes before v2
        const modelNames = await modelsPage.getModelNamesInOrder();
        const v1Index = modelNames.indexOf('YOLOX Model v1');
        const v2Index = modelNames.indexOf('YOLOX Model v2');

        expect(v1Index).toBeLessThan(v2Index);
    });

    test('can toggle pin active model', async ({ modelsPage, network }) => {
        // Set model-1 as the active model
        network.use(
            http.get('/api/projects/{project_id}/pipeline', () => {
                return HttpResponse.json({
                    project_id: 'id-1',
                    status: 'idle',
                    source: null,
                    sink: null,
                    model: {
                        id: 'model-1',
                        name: 'YOLOX Model v1',
                        architecture: 'Object_Detection_YOLOX_X',
                        files_deleted: false,
                    },
                    device: 'cpu',
                });
            })
        );

        await modelsPage.goto();

        await modelsPage.togglePinActiveModel();

        const modelNames = await modelsPage.getModelNamesInOrder();
        expect(modelNames[0]).toContain('YOLOX Model v1');
    });

    test('can rename a model', async ({ modelsPage, network }) => {
        network.use(
            http.patch('/api/projects/{project_id}/models/{model_id}', async ({ request }) => {
                const body = (await request.json()) as { name: string };

                return HttpResponse.json(getMockedModel({ id: 'model-1', name: body.name }));
            }),
            http.get('/api/projects/{project_id}/models', () => {
                return HttpResponse.json([
                    getMockedModel({ id: 'model-1', name: 'Renamed Model' }),
                    ...mockedModels.slice(1),
                ]);
            })
        );

        await modelsPage.goto();

        await modelsPage.openModelMenu();
        await modelsPage.clickRenameAction();
        await modelsPage.renameModel('Renamed Model');

        await expect(modelsPage.getModelByName('Renamed Model')).toBeVisible();
    });

    test('can delete a model', async ({ modelsPage, network }) => {
        await modelsPage.goto();

        await expect(modelsPage.getModelByName('YOLOX Model v1')).toBeVisible();

        network.use(
            http.get('/api/projects/{project_id}/models', () => {
                return HttpResponse.json(mockedModels.slice(1));
            })
        );

        await modelsPage.openModelMenu();
        await modelsPage.clickDeleteAction();
        await modelsPage.confirmDelete();

        await expect(modelsPage.getModelByName('YOLOX Model v1')).toBeHidden();
    });
});
