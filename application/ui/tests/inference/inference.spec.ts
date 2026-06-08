// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { getMockedPipeline } from 'mocks/mock-pipeline';
import { getMockedProject } from 'mocks/mock-project';
import { HttpResponse } from 'msw';

import { expect, http, test } from '../fixtures';
import { stepConfigureInferenceSourceAndSink } from '../workflows/workflow-steps';

test.describe('Inference', () => {
    test.beforeEach(({ network }) => {
        network.use(
            http.get('/api/projects/{project_id}', () => {
                return HttpResponse.json(getMockedProject({ id: 'id-1' }));
            }),
            http.get('/api/projects/{project_id}/pipeline', ({ response }) => {
                return response(200).json(getMockedPipeline({ status: 'idle' }));
            }),
            http.get('/api/sources', () => {
                return HttpResponse.json([]);
            }),
            http.get('/api/sinks', () => {
                return HttpResponse.json([]);
            }),
            http.get('/api/system/devices/camera', () => {
                return HttpResponse.json([
                    {
                        index: 1,
                        name: 'FaceTime HD Camera',
                    },
                ]);
            }),
            http.post('/api/sources', () => {
                return HttpResponse.json(
                    {
                        id: 'generated-source-id',
                        name: 'Default Source',
                        source_type: 'usb_camera',
                        device_id: 0,
                    },
                    { status: 201 }
                );
            }),
            http.post('/api/sinks', () => {
                return HttpResponse.json(
                    {
                        id: 'generated-sink-id',
                        name: 'Default Sink',
                        sink_type: 'folder',
                        rate_limit: 5,
                        folder_path: '/default/path',
                        output_formats: ['predictions'],
                    },
                    { status: 201 }
                );
            })
        );
    });

    test('Inference workflow', async ({ streamPage, page, network }) => {
        await test.step('starts stream', async () => {
            await page.goto('/projects/id-1/inference');

            await streamPage.startStream();

            expect(streamPage.isConnected()).toBeTruthy();
        });

        await test.step('toggles pipeline', async () => {
            await page.goto('/projects/id-1/inference');

            await expect(page.getByRole('switch', { name: /Enable pipeline/i })).toBeEnabled();

            network.use(
                http.post('/api/projects/{project_id}/pipeline:enable', () => {
                    return HttpResponse.json(null, { status: 204 });
                }),
                http.get('/api/projects/{project_id}/pipeline', ({ response }) => {
                    return response(200).json(getMockedPipeline({ status: 'running' }));
                })
            );

            await page.getByRole('switch', { name: /Enable pipeline/i }).click();

            await expect(page.getByRole('switch', { name: 'Disable Pipeline' })).toBeEnabled();
            network.use(
                http.post('/api/projects/{project_id}/pipeline:disable', () => {
                    return HttpResponse.json(null, { status: 204 });
                }),
                http.get('/api/projects/{project_id}/pipeline', ({ response }) => {
                    return response(200).json(getMockedPipeline({ status: 'idle' }));
                })
            );

            await page.getByRole('switch', { name: 'Disable Pipeline' }).click();

            await expect(page.getByRole('switch', { name: /Enable pipeline/i })).toBeEnabled();
        });

        await test.step('updates data collection policy', async () => {
            await page.goto('/projects/id-1/inference');

            // Open both tabs just to make sure everything works
            await page.getByRole('button', { name: 'Toggle Model statistics tab' }).click();
            await expect(page.getByText('Model statistics', { exact: true })).toBeVisible();

            await page.getByRole('button', { name: 'Toggle Data collection policy' }).click();
            await expect(page.getByRole('heading', { name: 'Data collection' })).toBeVisible();

            await expect(page.getByRole('switch', { name: 'Toggle auto capturing' })).not.toBeChecked();

            network.use(
                http.patch('/api/projects/{project_id}/pipeline', () => {
                    return HttpResponse.json({
                        project_id: '',
                        status: 'idle',
                        device: 'images_folder',
                    });
                }),
                http.get('/api/projects/{project_id}/pipeline', ({ response }) => {
                    return response(200).json(
                        getMockedPipeline({
                            data_collection: {
                                max_dataset_size: 500,
                                policies: [
                                    {
                                        type: 'fixed_rate',
                                        enabled: true,
                                        rate: 12,
                                    },
                                    {
                                        type: 'confidence_threshold',
                                        enabled: false,
                                        confidence_threshold: 0.5,
                                        min_sampling_interval: 2.5,
                                    },
                                ],
                            },
                        })
                    );
                })
            );

            network.use(
                http.get('/api/projects/{project_id}/pipeline', ({ response }) => {
                    return response(200).json(
                        getMockedPipeline({
                            data_collection: {
                                max_dataset_size: 700,
                                policies: [
                                    {
                                        type: 'fixed_rate',
                                        enabled: true,
                                        rate: 12,
                                    },
                                    {
                                        type: 'confidence_threshold',
                                        enabled: false,
                                        confidence_threshold: 0.5,
                                        min_sampling_interval: 2.5,
                                    },
                                ],
                            },
                        })
                    );
                })
            );

            const maxDatasetSizeField = page.getByRole('textbox', { name: 'Size' });

            await maxDatasetSizeField.fill('700');
            await expect(maxDatasetSizeField).toHaveValue('700');

            await page.getByRole('switch', { name: 'Toggle auto capturing' }).click();
            await expect(page.getByRole('switch', { name: 'Toggle auto capturing' })).toBeChecked();

            network.use(
                http.get('/api/projects/{project_id}/pipeline', ({ response }) => {
                    return response(200).json(
                        getMockedPipeline({
                            data_collection: {
                                max_dataset_size: 500,
                                policies: [
                                    {
                                        type: 'fixed_rate',
                                        enabled: true,
                                        rate: 20,
                                    },
                                    {
                                        type: 'confidence_threshold',
                                        enabled: false,
                                        confidence_threshold: 0.5,
                                        min_sampling_interval: 2.5,
                                    },
                                ],
                            },
                        })
                    );
                })
            );

            const framesField = page.getByRole('textbox', { name: 'Frames' });
            const secondsField = page.getByRole('textbox', { name: 'Seconds' });

            await expect(framesField).toBeEnabled();
            await expect(secondsField).toBeEnabled();

            await framesField.fill('20');
            await expect(framesField).toHaveValue('20');

            await expect(page.getByRole('switch', { name: 'Confidence threshold' })).not.toBeChecked();

            network.use(
                http.get('/api/projects/{project_id}/pipeline', ({ response }) => {
                    return response(200).json(
                        getMockedPipeline({
                            data_collection: {
                                max_dataset_size: 500,
                                policies: [
                                    {
                                        type: 'fixed_rate',
                                        enabled: true,
                                        rate: 20,
                                    },
                                    {
                                        type: 'confidence_threshold',
                                        enabled: true,
                                        confidence_threshold: 0.5,
                                        min_sampling_interval: 2.5,
                                    },
                                ],
                            },
                        })
                    );
                })
            );

            await page.getByRole('switch', { name: 'Confidence threshold' }).click();
            await expect(page.getByRole('switch', { name: 'Confidence threshold' })).toBeChecked();

            network.use(
                http.get('/api/projects/{project_id}/pipeline', ({ response }) => {
                    return response(200).json(
                        getMockedPipeline({
                            data_collection: {
                                max_dataset_size: 500,
                                policies: [
                                    {
                                        type: 'fixed_rate',
                                        enabled: true,
                                        rate: 20,
                                    },
                                    {
                                        type: 'confidence_threshold',
                                        enabled: true,
                                        confidence_threshold: 0.7,
                                        min_sampling_interval: 2.5,
                                    },
                                ],
                            },
                        })
                    );
                })
            );

            const confidenceSlider = page.getByRole('slider', { name: 'Threshold' });
            await expect(confidenceSlider).toBeVisible();
            await expect(confidenceSlider).toBeEnabled();
            await confidenceSlider.fill('0.7');
            await expect(confidenceSlider).toHaveValue('0.7');
        });

        await test.step('updates input and output source', async () => {
            network.use(
                http.get('/api/projects/{project_id}/pipeline', ({ response }) => {
                    return response(200).json(getMockedPipeline({ source: null, sink: null }));
                }),
                http.post('/api/sources', () => {
                    return HttpResponse.json(
                        {
                            id: 'generated-source-id',
                            name: 'My Source',
                            source_type: 'usb_camera',
                            device_id: 1,
                        },
                        { status: 201 }
                    );
                }),
                http.post('/api/sinks', () => {
                    return HttpResponse.json(
                        {
                            id: 'generated-sink-id',
                            name: 'My Sink',
                            sink_type: 'folder',
                            rate_limit: 5,
                            folder_path: 'e2e-output',
                            output_formats: ['predictions'],
                        },
                        { status: 201 }
                    );
                }),
                http.patch('/api/sources/{source_id}', () => {
                    return HttpResponse.json({});
                }),
                http.patch('/api/sinks/{sink_id}', () => {
                    return HttpResponse.json({});
                })
            );
            await page.goto('/projects/id-1/inference');

            const usbCamera = 'My Source';

            network.use(
                http.get('/api/sources', () => {
                    return HttpResponse.json([
                        {
                            id: '1',
                            name: usbCamera,
                            source_type: 'usb_camera',
                            device_id: 1,
                        },
                    ]);
                })
            );

            network.use(
                http.get('/api/sinks', () => {
                    return HttpResponse.json([
                        {
                            id: '1',
                            name: 'My Sink',
                            sink_type: 'folder',
                            folder_path: 'e2e-output',
                            rate_limit: 5,
                            output_formats: ['predictions'],
                        },
                    ]);
                })
            );

            await stepConfigureInferenceSourceAndSink(page);

            await page.getByLabel('Pipeline configuration tabs').getByText('Input').click();
            await expect(page.getByText(usbCamera)).toBeVisible();
            await expect(page.getByText('Device: FaceTime HD Camera')).toBeVisible();

            await page.getByLabel('Pipeline configuration tabs').getByText('Output').click();
            await expect(page.getByText('My Sink')).toBeVisible();
            await expect(page.getByText('Folder path: e2e-output')).toBeVisible();
            await expect(page.getByText('Rate limit: 5 samples every 1 second')).toBeVisible();
            await expect(page.getByText('Output formats: predictions')).toBeVisible();
        });
    });

    test('shows stream only for projects with enabled pipeline', async ({ page, network, streamPage }) => {
        const projectWithEnabledPipeline = getMockedProject({
            id: 'enabled-project-id',
            name: 'Enabled project',
            active_pipeline: true,
        });

        const projectWithDisabledPipeline = getMockedProject({
            id: 'disabled-project-id',
            name: 'Disabled project',
            active_pipeline: false,
        });

        network.use(
            http.get('/api/projects', ({ response }) => {
                return response(200).json([projectWithEnabledPipeline, projectWithDisabledPipeline]);
            }),
            http.get('/api/projects/{project_id}', ({ params }) => {
                return HttpResponse.json(
                    params.project_id === projectWithEnabledPipeline.id
                        ? projectWithEnabledPipeline
                        : projectWithDisabledPipeline
                );
            }),
            http.get('/api/projects/{project_id}/pipeline', ({ params, response }) => {
                return response(200).json(
                    getMockedPipeline({
                        project_id: params.project_id,
                        status: params.project_id === projectWithEnabledPipeline.id ? 'running' : 'idle',
                    })
                );
            })
        );

        await page.goto(`/projects/${projectWithEnabledPipeline.id}/inference`);
        await expect(streamPage.getStartStreamButton()).toBeVisible();

        await streamPage.startStream();
        await expect(streamPage.getStartStreamButton()).toBeHidden();

        await page.getByRole('button', { name: `Selected project ${projectWithEnabledPipeline.name}` }).click();
        await expect(page.getByRole('dialog')).toBeVisible();

        await page.getByRole('listitem').filter({ hasText: projectWithDisabledPipeline.name }).click();

        await expect(page).toHaveURL(new RegExp(`/projects/${projectWithDisabledPipeline.id}/dataset$`));

        await page.keyboard.press('Escape');
        await expect(page.getByRole('dialog')).toBeHidden();

        await page.getByRole('tab', { name: 'Inference' }).click();

        await expect(page.getByLabel('Enable pipeline to start stream')).toBeVisible();
        await expect(page.getByRole('switch', { name: /Enable pipeline/i })).toBeVisible();
    });

    test('resets stream state when re-opening inference page', async ({ streamPage, page, network }) => {
        network.use(
            http.get('/api/projects/{project_id}/pipeline', ({ response }) => {
                return response(200).json(getMockedPipeline({ project_id: 'id-1', status: 'running' }));
            })
        );

        await page.goto('/projects/id-1/inference');

        await expect(streamPage.getStartStreamButton()).toBeVisible();
        await streamPage.startStream();

        await expect(streamPage.getStartStreamButton()).toBeHidden();

        await page.getByRole('tab', { name: 'Models' }).click();

        await expect(page).toHaveURL(new RegExp(`/projects/id-1/models$`));
        await page.getByRole('tab', { name: 'Inference' }).click();

        await expect(streamPage.getStartStreamButton()).toBeVisible();
    });
});
