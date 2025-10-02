// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { getMockedProject } from 'mocks/mock-project';
import { HttpResponse } from 'msw';

import { expect, http, test } from './fixtures';

test.describe('Inference', () => {
    test.beforeEach(({ network }) => {
        network.use(
            http.get('/api/projects/{project_id}', () => {
                return HttpResponse.json(getMockedProject({ id: 'id-1' }));
            }),
            http.get('/api/projects/{project_id}/pipeline', ({ response }) => {
                return response(200).json({
                    project_id: 'id-1',
                    status: 'running',
                    source: null,
                    sink: null,
                    model: null,
                    data_collection_policies: [],
                });
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

    test('starts stream', async ({ page }) => {
        await page.goto('/projects/id-1/inference');

        await expect(page.getByLabel('Idle')).toBeVisible();

        await page.getByLabel('Start stream').click();

        // TODO: fix the stream mock and update this to "Connected"
        await expect(page.getByLabel('Connecting')).toBeVisible();
    });

    test('updates pipeline status', async ({ page, network }) => {
        await page.goto('/projects/id-1/inference');

        await page.getByRole('switch', { name: 'Disable pipeline' }).click();

        network.use(
            http.get('/api/projects/{project_id}/pipeline', ({ response }) => {
                return response(200).json({
                    project_id: 'id-1',
                    status: 'idle',
                    source: null,
                    sink: null,
                    model: null,
                    data_collection_policies: [],
                });
            })
        );

        await page.reload();

        await expect(page.getByText('Enable pipeline')).toBeVisible();
    });

    test('updates data collection policy', async ({ page, network }) => {
        network.use(
            http.get('/api/projects/{project_id}/pipeline', ({ response }) => {
                return response(200).json({
                    project_id: 'id-1',
                    status: 'idle',
                    source: null,
                    sink: null,
                    model: null,
                    data_collection_policies: [
                        {
                            type: 'fixed_rate',
                            enabled: false,
                            rate: 12,
                        },
                    ],
                });
            })
        );

        await page.goto('/projects/id-1/inference');

        // Open both tabs just to make sure everything works
        await page.getByRole('button', { name: 'Toggle Model statistics tab' }).click();
        await expect(page.getByText('Model statistics')).toBeVisible();

        await page.getByRole('button', { name: 'Toggle Data collection policy' }).click();
        await expect(page.getByRole('heading', { name: 'Data collection' })).toBeVisible();

        // Update values
        await expect(page.getByRole('switch', { name: 'Toggle auto capturing' })).not.toBeChecked();

        network.use(
            http.patch('/api/projects/{project_id}/pipeline', () => {
                return HttpResponse.json({});
            }),
            http.get('/api/projects/{project_id}/pipeline', ({ response }) => {
                return response(200).json({
                    project_id: 'id-1',
                    status: 'idle',
                    source: null,
                    sink: null,
                    model: null,
                    data_collection_policies: [
                        {
                            type: 'fixed_rate',
                            enabled: true,
                            rate: 12,
                        },
                    ],
                });
            })
        );

        await page.getByRole('switch', { name: 'Toggle auto capturing' }).click();
        await expect(page.getByRole('switch', { name: 'Toggle auto capturing' })).toBeChecked();

        network.use(
            http.get('/api/projects/{project_id}/pipeline', ({ response }) => {
                return response(200).json({
                    project_id: 'id-1',
                    status: 'idle',
                    source: null,
                    sink: null,
                    model: null,
                    data_collection_policies: [
                        {
                            type: 'fixed_rate',
                            enabled: true,
                            rate: 20,
                        },
                    ],
                });
            })
        );

        const sliderInput = page.locator('input[type="range"]');

        await expect(sliderInput).toBeVisible();
        await sliderInput.fill('20');

        await expect(sliderInput).toHaveValue('20');
    });

    test('updates input and output source', async ({ page, network }) => {
        network.use(
            http.patch('/api/sources/{source_id}', () => {
                return HttpResponse.json({});
            }),
            http.patch('/api/sinks/{sink_id}', () => {
                return HttpResponse.json({});
            })
        );
        await page.goto('/projects/id-1/inference');

        await page.getByRole('button', { name: 'Input source' }).click();
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

        await page.getByRole('button', { name: 'Input source' }).click();

        await expect(page.locator('input[name="name"]')).toHaveValue('New Webcam');
        await expect(page.locator('input[name="device_id"]')).toHaveValue('1');

        // Go to output tab
        await page.getByLabel('Dataset import tabs').getByText('Output setup').click();

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

        await page.getByRole('button', { name: 'Input source' }).click();
        await page.getByLabel('Dataset import tabs').getByText('Output setup').click();

        await expect(page.locator('input[name="name"]')).toHaveValue('New Folder');

        await expect(page.locator('input[aria-roledescription="Number field"]')).toHaveValue('5');
        await expect(page.locator('input[name="folder_path"]')).toHaveValue('some/path');
        await expect(page.locator('input[name="output_formats"][value="predictions"]')).toBeChecked();
    });
});
