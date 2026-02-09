// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { getMockedDatasetRevision } from 'mocks/mock-dataset-revision';
import { getMockedExtendedModel, getMockedModel } from 'mocks/mock-model';
import { HttpResponse } from 'msw';

import { expect, http, test } from '../fixtures';

const mockedModel = getMockedModel({
    id: 'model-1',
    name: 'YOLOX Model v1',
    architecture: 'Object_Detection_YOLOX_X',
    training_info: {
        status: 'successful',
        label_schema_revision: { labels: [] },
        start_time: '2025-01-10T10:00:00.000000+00:00',
        end_time: '2025-01-10T12:30:00.000000+00:00',
        dataset_revision_id: 'dataset-1',
    },
    variants: [
        {
            format: 'openvino',
            precision: 'int8',
            weights_size: 52428800,
        },
        {
            format: 'openvino',
            precision: 'fp32',
            weights_size: 209715200,
        },
        {
            format: 'pytorch',
            precision: 'fp32',
            weights_size: 209715200,
        },
        {
            format: 'onnx',
            precision: 'fp32',
            weights_size: 209715200,
        },
    ],
});

const mockedDatasetRevision = getMockedDatasetRevision({
    id: 'dataset-1',
    name: 'Dataset Revision 1',
    item_counts: {
        training: 70,
        validation: 20,
        testing: 10,
        total: 100,
    },
});

const mockedExtendedModel = getMockedExtendedModel(mockedModel);

test.describe('Model Details', () => {
    test.beforeEach(({ network }) => {
        network.use(
            http.get('/api/projects/{project_id}/models', () => {
                return HttpResponse.json([mockedModel]);
            }),
            http.get('/api/projects/{project_id}/models/{model_id}', ({ params }) => {
                if (params.model_id === 'model-1') {
                    return HttpResponse.json(mockedExtendedModel);
                }

                return new HttpResponse(null, { status: 404 });
            }),
            http.get('/api/projects/{project_id}/dataset_revisions', () => {
                return HttpResponse.json([mockedDatasetRevision]);
            }),
            http.get('/api/projects/{project_id}/dataset_revisions/{dataset_revision_id}/items', ({ request }) => {
                const url = new URL(request.url);
                const subset = url.searchParams.get('subset');

                const countsBySubset: Record<string, number> = {
                    training: 70,
                    validation: 20,
                    testing: 10,
                };
                const count = subset && subset in countsBySubset ? countsBySubset[subset] : 0;
                const total = count;

                return HttpResponse.json({
                    items: [],
                    pagination: { offset: 0, limit: 20, count, total },
                });
            }),
            http.get('/api/projects/{project_id}/models/{model_id}/binary', () => {
                return HttpResponse.arrayBuffer(new ArrayBuffer(1024), {
                    headers: {
                        'content-type': 'application/zip',
                        'content-disposition': 'attachment; filename="model.zip"',
                    },
                });
            })
        );
    });

    test.describe('Model Variants', () => {
        test('shows no variants message when model has no variants', async ({ network, page, modelsPage }) => {
            const modelWithoutVariants = getMockedModel({
                id: 'model-1',
                name: 'YOLOX Model v1',
                variants: [],
            });

            network.use(
                http.get('/api/projects/{project_id}/models', () => {
                    return HttpResponse.json([modelWithoutVariants]);
                }),
                http.get('/api/projects/{project_id}/models/{model_id}', ({ params }) => {
                    if (params.model_id === 'model-1') {
                        return HttpResponse.json(getMockedExtendedModel(modelWithoutVariants));
                    }

                    return new HttpResponse(null, { status: 404 });
                })
            );

            await modelsPage.goto();
            await modelsPage.expandModel('YOLOX Model v1');
            await page.getByRole('tab', { name: 'Model variants' }).click();

            await expect(page.getByText('There are no model variants available.')).toBeVisible();
        });

        test('displays model variants in separate tabs for each format', async ({ page, modelsPage }) => {
            await modelsPage.goto();
            await modelsPage.expandModel('YOLOX Model v1');
            await page.getByRole('tab', { name: 'Model variants' }).click();

            const variantsTabList = page.getByRole('tablist', { name: 'Model variants' });
            await expect(variantsTabList).toBeVisible();

            const formats = ['openvino', 'pytorch', 'onnx'];

            for (const format of formats) {
                await page.getByLabel(`${format} tab`).click();

                const table = page.getByLabel(`Model variants for ${mockedModel.id}`);
                await expect(table).toBeVisible();
            }
        });

        test('can download model variant', async ({ page, network, modelsPage }) => {
            let downloadFormat: string | undefined;

            network.use(
                http.get('/api/projects/{project_id}/models/{model_id}/binary', ({ request }) => {
                    const url = new URL(request.url);
                    downloadFormat = url.searchParams.get('format') ?? undefined;

                    if (!downloadFormat) {
                        return new HttpResponse(null, { status: 400 });
                    }

                    return HttpResponse.arrayBuffer(new ArrayBuffer(1024), {
                        headers: {
                            'content-type': 'application/zip',
                            'content-disposition': `attachment; filename="model-model-1-${downloadFormat}.zip"`,
                        },
                    });
                })
            );

            await modelsPage.goto();
            await modelsPage.expandModel('YOLOX Model v1');
            await page.getByRole('tab', { name: 'Model variants' }).click();

            const downloadButton = page.getByLabel(/Download.*model/i).first();
            await expect(downloadButton).toBeEnabled();

            await downloadButton.click();

            await expect.poll(() => downloadFormat).not.toBeUndefined();

            expect(['openvino', 'pytorch', 'onnx']).toContain(downloadFormat);
        });
    });

    test.describe('Training Datasets', () => {
        test('shows dataset not found message when dataset revision is missing', async ({
            network,
            page,
            modelsPage,
        }) => {
            network.use(
                http.get('/api/projects/{project_id}/dataset_revisions', () => {
                    return HttpResponse.json([]);
                })
            );

            await modelsPage.goto();
            await modelsPage.expandModel('YOLOX Model v1');
            await modelsPage.clickTrainingDatasetsTab();

            await expect(page.getByText('No dataset revision found for this model')).toBeVisible();
        });

        test('shows deleted dataset files message when dataset files are deleted', async ({
            network,
            page,
            modelsPage,
        }) => {
            const deletedDatasetRevision = getMockedDatasetRevision({
                id: 'dataset-1',
                name: 'Dataset Revision 1',
                files_deleted: true,
            });

            network.use(
                http.get('/api/projects/{project_id}/dataset_revisions', () => {
                    return HttpResponse.json([deletedDatasetRevision]);
                })
            );

            await modelsPage.goto();
            await modelsPage.expandModel('YOLOX Model v1');
            await modelsPage.clickTrainingDatasetsTab();

            await expect(page.getByText('The files for this dataset revision have been deleted.')).toBeVisible();
        });

        test('shows correct subset counts and percentages', async ({ page, modelsPage }) => {
            await modelsPage.goto();
            await modelsPage.expandModel('YOLOX Model v1');
            await modelsPage.clickTrainingDatasetsTab();

            const trainingHeading = page.getByRole('heading', { name: /Training/ });
            const validationHeading = page.getByRole('heading', { name: /Validation/ });
            const testingHeading = page.getByRole('heading', { name: /Testing/ });

            await expect(trainingHeading).toBeVisible();
            await expect(validationHeading).toBeVisible();
            await expect(testingHeading).toBeVisible();

            await expect(page.getByText('(70)')).toBeVisible();
            await expect(page.getByText('(20)')).toBeVisible();
            await expect(page.getByText('(10)')).toBeVisible();
        });
    });
});
