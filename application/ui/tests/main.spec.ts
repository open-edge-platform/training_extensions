// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { expect, http, test } from './fixtures';

test.describe('Inference', () => {
    test('starts stream', async ({ page }) => {
        await page.goto('/projects/id-1/inference');

        await expect(page.getByLabel('Idle')).toBeVisible();

        await page.getByLabel('Start stream').click();

        // TODO: fix the stream mock and update this to "Connected"
        await expect(page.getByLabel('Connecting')).toBeVisible();
    });

    test('activates a pipeline', async ({ page, network }) => {
        let activePipelineProjectId = 'id-1';

        // Only project 1 has an active pipeline
        network.use(
            http.get('/api/projects/{project_id}/pipeline', ({ params, response }) => {
                return response(200).json({
                    project_id: params.project_id,
                    status: params.project_id === activePipelineProjectId ? 'running' : 'idle',
                    source: null,
                    sink: null,
                    model: null,
                    data_collection_policies: [],
                });
            })
        );

        await page.goto('/projects');

        await expect(page.getByText('Project 1')).toBeVisible();
        await expect(page.getByLabel(/Active/)).toBeVisible();

        await expect(page.getByText('Project 2')).toBeVisible();
        await expect(page.getByText('Project 3')).toBeVisible();

        // Open menu options
        await page.getByTestId('id-3').click();
        await page.getByText(/Activate/).click();

        activePipelineProjectId = 'id-3';

        await expect(page.getByText('Project enabled successfully')).toBeVisible();
    });
});
