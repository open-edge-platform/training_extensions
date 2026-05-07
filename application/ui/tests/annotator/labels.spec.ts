// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { expect } from '@playwright/test';
import { getMockedProject } from 'mocks/mock-project';
import { HttpResponse } from 'msw';

import { http, test } from '../fixtures';
import { blueLabel, candyBinaryHandler, redLabel } from './annotator-fixtures';

const mockedDetectionProject = getMockedProject({
    id: '123e4567-e89b-12d3-a456-426614174000',
    task: {
        exclusive_labels: true,
        task_type: 'detection',
        labels: [redLabel, blueLabel],
    },
});

test.describe('Annotator', () => {
    test.beforeEach(async ({ network }) => {
        network.use(
            http.get('/api/projects/{project_id}', () => {
                return HttpResponse.json(mockedDetectionProject);
            }),
            candyBinaryHandler,
            http.get('/api/projects/{project_id}/dataset/media/{media_id}/annotations', async () => {
                return HttpResponse.json({
                    annotations: [],
                    user_reviewed: true,
                    subset: 'training',
                });
            })
        );
    });

    test('Update label name with spaces', async ({ page, annotatorPage }) => {
        const nameWithSpaces = 'Label with spaces';
        await annotatorPage.goto(mockedDetectionProject.id, 'item-1');

        await test.step('Open labels editor', async () => {
            await page.getByRole('button', { name: 'Edit labels' }).click();
        });

        const filterLabel = page.getByLabel('Label name').first();

        await test.step('Update label name with empty space', async () => {
            await filterLabel.clear();
            await filterLabel.pressSequentially(nameWithSpaces);
        });

        await expect(filterLabel).toHaveValue(nameWithSpaces);
    });
});
