// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { getMockedPipeline } from 'mocks/mock-pipeline';
import { getMockedProject } from 'mocks/mock-project';
import { HttpResponse } from 'msw';

import { expect, http, test } from '../fixtures';

test.beforeEach(({ network }) => {
    network.use(
        http.get('/api/projects/{project_id}', () => {
            return HttpResponse.json(getMockedProject({ id: 'id-1' }));
        }),
        http.get('/api/projects/{project_id}/pipeline', ({ response }) => {
            return response(200).json(getMockedPipeline({ status: 'running' }));
        }),
        http.get('/api/sources', () => {
            return HttpResponse.json([]);
        }),
        http.get('/api/sinks', () => {
            return HttpResponse.json([]);
        }),
        http.post('/api/sources', () => {
            return HttpResponse.json(
                {
                    id: 'generated-source-id',
                    name: 'Default Source',
                    source_type: 'webcam',
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
        }),
        http.patch('/api/projects/{project_id}/pipeline', () => {
            return HttpResponse.json({});
        })
    );
});

test('Inference', async ({ streamPage, page, network }) => {
    await test.step('starts stream', async () => {
        await page.goto('/projects/id-1/inference');

        await streamPage.startStream();

        expect(streamPage.isConnected()).toBeTruthy();
    });

    await test.step('updates pipeline status', async () => {
        await page.goto('/projects/id-1/inference');

        await page.getByRole('switch', { name: 'Disable pipeline' }).click();

        network.use(
            http.get('/api/projects/{project_id}/pipeline', ({ response }) => {
                return response(200).json(getMockedPipeline({ status: 'idle' }));
            })
        );

        await page.reload();

        await expect(page.getByText('Enable pipeline')).toBeVisible();
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
                return HttpResponse.json({});
            }),
            http.get('/api/projects/{project_id}/pipeline', ({ response }) => {
                return response(200).json(
                    getMockedPipeline({
                        data_collection_policies: [
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
                    })
                );
            })
        );

        await page.getByRole('switch', { name: 'Toggle auto capturing' }).click();
        await expect(page.getByRole('switch', { name: 'Toggle auto capturing' })).toBeChecked();

        network.use(
            http.get('/api/projects/{project_id}/pipeline', ({ response }) => {
                return response(200).json(
                    getMockedPipeline({
                        data_collection_policies: [
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
                    })
                );
            })
        );

        const rateSlider = page.getByRole('slider', { name: 'Rate' });
        await expect(rateSlider).toBeVisible();
        await expect(rateSlider).toBeEnabled();
        await rateSlider.fill('20');
        await expect(rateSlider).toHaveValue('20');

        await expect(page.getByRole('switch', { name: 'Confidence threshold' })).not.toBeChecked();

        network.use(
            http.get('/api/projects/{project_id}/pipeline', ({ response }) => {
                return response(200).json(
                    getMockedPipeline({
                        data_collection_policies: [
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
                        data_collection_policies: [
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
            http.patch('/api/sources/{source_id}', () => {
                return HttpResponse.json({});
            }),
            http.patch('/api/sinks/{sink_id}', () => {
                return HttpResponse.json({});
            })
        );
        await page.goto('/projects/id-1/inference');

        await page.getByRole('button', { name: 'Pipeline configuration' }).click();
        await page.getByRole('button', { name: 'Webcam' }).click();

        await page.locator('input[name="name"]').fill('New Webcam');
        await page.getByLabel('webcam device id').fill('1');

        network.use(
            http.get('/api/sources', () => {
                return HttpResponse.json([
                    {
                        id: '1',
                        name: 'New Webcam',
                        source_type: 'webcam',
                        device_id: 1,
                    },
                ]);
            })
        );

        await page.getByRole('button', { name: 'Apply' }).click();

        // Click outside the dialog to close it
        await page.click('body', { position: { x: 10, y: 10 } });

        await page.getByRole('button', { name: 'Pipeline configuration' }).click();

        await expect(page.locator('input[name="name"]')).toHaveValue('New Webcam');
        await expect(page.locator('input[name="device_id"]')).toHaveValue('1');

        // Go to output tab
        await page.getByLabel('Dataset import tabs').getByText('Output').click();

        await page.getByRole('button', { name: 'Folder' }).click();
        await page.locator('input[name="name"]').fill('New Folder');
        await page.locator('input[aria-roledescription="Number field"]').fill('5');

        await page.locator('input[name="folder_path"]').fill('some/path');
        await page.locator('input[name="output_formats"][value="predictions"]').click();

        network.use(
            http.get('/api/sinks', () => {
                return HttpResponse.json([
                    {
                        id: '1',
                        name: 'New Folder',
                        sink_type: 'folder',
                        folder_path: 'some/path',
                        rate_limit: 5,
                        output_formats: ['predictions'],
                    },
                ]);
            })
        );

        await page.getByRole('button', { name: 'Apply' }).click();

        // Click outside the dialog to close it
        await page.click('body', { position: { x: 10, y: 10 } });

        await page.getByRole('button', { name: 'Pipeline configuration' }).click();
        await page.getByLabel('Dataset import tabs').getByText('Output').click();

        await expect(page.locator('input[name="name"]')).toHaveValue('New Folder');

        await expect(page.locator('input[aria-roledescription="Number field"]')).toHaveValue('5');
        await expect(page.locator('input[name="folder_path"]')).toHaveValue('some/path');
        await expect(page.locator('input[name="output_formats"][value="predictions"]')).toBeChecked();
    });
});
