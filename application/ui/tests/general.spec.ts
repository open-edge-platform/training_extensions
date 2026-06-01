// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Page } from '@playwright/test';
import { HttpResponse } from 'msw';

import { getMockedProject } from '../mocks/mock-project';
import { expect, http, test } from './fixtures';

const expectToastToBeInteractive = async (page: Page, message: string) => {
    const toast = page.getByLabel('toast').filter({ hasText: message });
    const dismissButton = toast.getByRole('button').first();

    await expect(toast).toBeVisible();
    await expect(dismissButton).toBeVisible();
    await dismissButton.click();
};

test.describe('Toast Rendering', () => {
    test('shows toast from create-project form context', async ({ page }) => {
        await page.goto('/projects/new');

        await page.getByLabel('detection', { exact: true }).click();
        await page.getByRole('textbox', { name: 'Create label input' }).fill('Object');
        await page.getByRole('button', { name: /Create label/i }).click();
        await page.getByRole('button', { name: /delete label object/i }).click();

        await expectToastToBeInteractive(page, 'At least one object is required');
    });

    test('shows toast from dialog context', async ({ page, network }) => {
        network.use(
            http.get('/api/projects', () => {
                return HttpResponse.json([
                    getMockedProject({ id: 'id-1', name: 'Project 1' }),
                    getMockedProject({ id: 'id-2', name: 'Project 2' }),
                ]);
            })
        );

        await page.goto('/projects');

        await page.getByTestId('id-1').click();
        await page.getByText(/Delete/).click();
        await page.getByRole('button', { name: 'Delete' }).click();

        await expectToastToBeInteractive(page, 'Project deleted successfully');
    });

    test('shows toast from inference page context', async ({ page, network }) => {
        network.use(
            http.get('/api/sources', () => {
                return HttpResponse.json([]);
            }),
            http.get('/api/sinks', () => {
                return HttpResponse.json([]);
            }),
            http.get('/api/projects/{project_id}/pipeline', () => {
                return HttpResponse.json({
                    project_id: 'id-1',
                    status: 'running',
                    source: null,
                    sink: null,
                    model: null,
                    device: 'cpu',
                });
            })
        );

        await page.goto('/projects/id-1/inference');
        await page.getByRole('switch', { name: /Disable pipeline/i }).click();

        await expectToastToBeInteractive(page, 'Pipeline disabled successfully');
    });
});
