// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { getMockedDatasetRevision } from 'mocks/mock-dataset-revision';
import { getMockedJob } from 'mocks/mock-job';
import { getMockedModel } from 'mocks/mock-model';
import { getMockedTrainingConfiguration } from 'mocks/mock-training-configuration';
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
            id: 'variant-openvino-fp32',
            format: 'openvino',
            precision: 'fp32',
            weights_size: 209715200,
            evaluations: [
                {
                    dataset_revision_id: 'dataset-1',
                    subset: 'testing',
                    metrics: [{ name: 'Accuracy', value: 0.82, primary: true }],
                },
            ],
            files_deleted: false,
        },
        {
            id: 'variant-openvino-int8',
            format: 'openvino',
            precision: 'int8',
            weights_size: 52428800,
            evaluations: [
                {
                    dataset_revision_id: 'dataset-1',
                    subset: 'testing',
                    metrics: [{ name: 'Accuracy', value: 0.79, primary: true }],
                },
            ],
            files_deleted: false,
        },
        {
            id: 'variant-pytorch-fp32',
            format: 'pytorch',
            precision: 'fp32',
            weights_size: 209715200,
            evaluations: [],
            files_deleted: false,
        },
        {
            id: 'variant-onnx-fp32',
            format: 'onnx',
            precision: 'fp32',
            weights_size: 209715200,
            evaluations: [],
            files_deleted: false,
        },
    ],
});

const mockedDatasetRevision = getMockedDatasetRevision({
    id: 'dataset-1',
    name: 'Dataset Revision 1',
    item_counts: {
        training: 150,
        validation: 200,
        testing: 150,
        total: 500,
    },
});

test.describe('Model Details', () => {
    test.beforeEach(({ network }) => {
        network.use(
            http.get('/api/projects/{project_id}/models', () => {
                return HttpResponse.json([mockedModel]);
            }),
            http.get('/api/projects/{project_id}/models/{model_id}', ({ params }) => {
                if (params.model_id === 'model-1') {
                    return HttpResponse.json(mockedModel);
                }

                return new HttpResponse(null, { status: 404 });
            }),
            http.get('/api/projects/{project_id}/models/{model_id}/training_metrics', ({ params }) => {
                if (params.model_id === 'model-1') {
                    return HttpResponse.json({
                        training_metrics: [
                            {
                                header: 'Learning rate (SGD)',
                                key: 'lr-sgd',
                                type: 'line',
                                value: {
                                    x_axis_label: 'Epoch',
                                    y_axis_label: 'Learning rate',
                                    line_data: [
                                        {
                                            header: 'Learning rate (SGD)',
                                            key: 'lr-sgd',
                                            points: [
                                                { type: 'point', x: 1, y: 0.01 },
                                                { type: 'point', x: 2, y: 0.008 },
                                                { type: 'point', x: 3, y: 0.006 },
                                            ],
                                        },
                                    ],
                                },
                            },
                        ],
                    });
                }

                return new HttpResponse(null, { status: 404 });
            }),
            http.get('/api/projects/{project_id}/models/{model_id}/training_configuration', ({ params }) => {
                if (params.model_id === 'model-1') {
                    return HttpResponse.json({
                        parameters: getMockedTrainingConfiguration(),
                    });
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
            http.get('/api/projects/{project_id}/models/{model_id}/variants/{model_variant_id}/binary', () => {
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
                training_info: {
                    ...getMockedModel().training_info,
                    dataset_revision_id: mockedDatasetRevision.id,
                },
            });

            network.use(
                http.get('/api/projects/{project_id}/models', () => {
                    return HttpResponse.json([modelWithoutVariants]);
                }),
                http.get('/api/projects/{project_id}/models/{model_id}', ({ params }) => {
                    if (params.model_id === 'model-1') {
                        return HttpResponse.json(getMockedModel(modelWithoutVariants));
                    }

                    return new HttpResponse(null, { status: 404 });
                })
            );

            await modelsPage.goto();
            await modelsPage.expandModel('YOLOX Model v1');
            await page.getByRole('tab', { name: 'Model variants' }).click();

            await expect(page.getByText('No available model variants.')).toBeVisible();
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
            let modelVariantId: string | undefined;

            network.use(
                http.get(
                    '/api/projects/{project_id}/models/{model_id}/variants/{model_variant_id}/binary',
                    ({ params }) => {
                        modelVariantId = params.model_variant_id;

                        if (!modelVariantId) {
                            return new HttpResponse(null, { status: 400 });
                        }

                        return HttpResponse.arrayBuffer(new ArrayBuffer(1024), {
                            headers: {
                                'content-type': 'application/zip',
                                'content-disposition': `attachment; filename="model-model-1-${modelVariantId}.zip"`,
                            },
                        });
                    }
                )
            );

            await modelsPage.goto();
            await modelsPage.expandModel('YOLOX Model v1');
            await page.getByRole('tab', { name: 'Model variants' }).click();

            await page
                .getByLabel(/Download model /i)
                .first()
                .click();

            await expect.poll(() => modelVariantId).not.toBeUndefined();
        });

        test('can start quantization with custom parameters', async ({ network, page, modelsPage }) => {
            let submittedJobBody: Record<string, unknown> | null = null;

            network.use(
                http.post('/api/jobs', async ({ request }) => {
                    submittedJobBody = await request.json();

                    return HttpResponse.json(
                        getMockedJob({
                            job_id: 'quantize-job-1',
                            job_type: 'quantize',
                        }),
                        { status: 201 }
                    );
                }),
                http.get('/api/projects/{project_id}/dataset_revisions', () => {
                    return HttpResponse.json([mockedDatasetRevision]);
                })
            );

            await modelsPage.goto();
            await modelsPage.expandModel('YOLOX Model v1');
            await modelsPage.clickModelVariantsTab();

            await modelsPage.openQuantizationDialog();

            const dialog = modelsPage.getQuantizationDialog();
            await expect(dialog).toBeVisible();
            await expect(dialog.getByRole('heading', { name: /Quantize model to INT8/ })).toBeVisible();

            await modelsPage.getNoMaximumCheckbox().click();
            await modelsPage.getAccuracyDropInput().fill('5');
            await modelsPage.getCalibrationSizeInput().fill('300');

            await modelsPage.submitQuantization();

            await expect(modelsPage.getToast('Quantization job started.')).toBeVisible();

            expect(submittedJobBody).toMatchObject({
                job_type: 'quantize',
                project_id: 'id-1',
                parameters: {
                    model_id: 'model-1',
                    max_drop: 0.05,
                    max_calibration_subset_size: 300,
                },
            });

            await expect(modelsPage.getQuantizationDialog()).toBeHidden();

            const variantsTable = page.getByLabel(`Model variants for ${mockedModel.id}`);
            await expect(variantsTable.getByRole('gridcell', { name: 'INT8' })).toBeVisible();

            // Size: INT8 is -75% smaller than the FP32 baseline
            await expect(variantsTable.getByTestId('model-variant-value-size-int8')).toContainText('52.4 MB');
            await expect(variantsTable.getByTestId('model-variant-delta-size')).toContainText('-75%');

            // Accuracy: INT8 is 79% vs FP32 baseline 82% -> -4% drop
            await expect(variantsTable.getByTestId('model-variant-value-accuracy-int8')).toContainText('79%');
            await expect(variantsTable.getByTestId('model-variant-delta-accuracy')).toContainText('-4%');
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

    test.describe('Training Metrics', () => {
        test('renders evaluations and training metric graphs', async ({ page, modelsPage }) => {
            await modelsPage.goto();
            await modelsPage.expandModel('YOLOX Model v1');
            await page.getByRole('tab', { name: 'Model metrics' }).click();

            await expect(page.getByRole('heading', { name: 'Accuracy' })).toBeVisible();
            await expect(page.getByRole('heading', { name: 'Learning rate (SGD)' })).toBeVisible();
        });

        test('shows error message when training metrics fail to load', async ({ network, page, modelsPage }) => {
            network.use(
                http.get('/api/projects/{project_id}/models/{model_id}/training_metrics', () => {
                    return new HttpResponse(null, { status: 500 });
                })
            );

            await modelsPage.goto();
            await modelsPage.expandModel('YOLOX Model v1');
            await page.getByRole('tab', { name: 'Model metrics' }).click();

            await expect(page.getByText('Failed to load training metrics')).toBeVisible();
        });
    });

    test.describe('Training Parameters', () => {
        test('shows learning parameters, filters and augmentations sections', async ({ page, modelsPage }) => {
            await modelsPage.goto();
            await modelsPage.expandModel('YOLOX Model v1');
            await page.getByRole('tab', { name: 'Training parameters' }).click();

            await expect(page.getByRole('heading', { name: 'LEARNING PARAMETERS' })).toBeVisible();
            await expect(page.getByRole('heading', { name: 'FILTERS' })).toBeVisible();
            await expect(page.getByRole('heading', { name: 'AUGMENTATIONS' })).toBeVisible();

            await expect(page.getByTestId('Box-LEARNING PARAMETERS').getByText('Maximum epochs')).toBeVisible();
            await expect(page.getByTestId('Box-LEARNING PARAMETERS').getByText('200')).toBeVisible();
        });
    });
});
