// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { expect } from '@playwright/test';
import { getMockedProject } from 'mocks/mock-project';
import { HttpResponse } from 'msw';

import { http, test } from '../../fixtures';
import { candyBinaryHandler, redLabel } from '../annotator-fixtures';

const mockedProject = getMockedProject({
    id: '123e4567-e89b-12d3-a456-426614174002',
    task: {
        exclusive_labels: true,
        task_type: 'instance_segmentation',
        labels: [redLabel],
    },
});

test.describe('SSIM tool', () => {
    test.beforeEach(async ({ network }) => {
        network.use(
            http.get('/api/projects/{project_id}', () => {
                return HttpResponse.json(mockedProject);
            }),
            candyBinaryHandler
        );
    });

    test('Draw a template region and adds polygon annotations', async ({ page, ssimTool, annotatorPage }) => {
        await page.goto(`/projects/${mockedProject.id}/dataset`);
        await page.getByRole('img', { name: 'item-1.jpg' }).dblclick();

        await test.step('Select SSIM tool', async () => {
            await ssimTool.selectTool();
        });

        await test.step('Wait for SSIM worker to be ready', async () => {
            await expect(page.getByLabel('ssim preview')).toHaveAttribute('data-loading', 'false', { timeout: 30000 });
        });

        await test.step('Draw a template region', async () => {
            await ssimTool.drawTemplate({ x: 100, y: 100, width: 150, height: 150 });
        });

        await test.step('Expect at least one polygon annotation', async () => {
            await expect(async () => {
                const items = await annotatorPage.getAnnotationsListItems('annotation polygon');

                expect(items.length).toBeGreaterThanOrEqual(1);
            }).toPass({ timeout: 15000 });
        });
    });
});
