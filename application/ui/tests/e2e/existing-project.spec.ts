// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { expect, test } from '@playwright/test';

test('[E2E] Existing project', async ({ page }) => {
    await test.step('Navigate to root page', async () => {
        await page.goto('/projects');
    });

    await test.step('Go to seeded project', async () => {
        await page.getByText('Test Project').click();
    });

    await test.step('Start stream to begin capture', async () => {
        await page.getByLabel('Start stream').click();

        await expect(page.getByLabel('Connected')).toBeVisible({ timeout: 60000 });

        // Wait for 3 seconds
        await page.waitForTimeout(3000);
    });

    await test.step('Confirm data was collected', async () => {
        await page.getByLabel('Start stream').click();

        await expect(page.getByLabel('Connected')).toBeVisible();

        // Wait for 3 seconds and stop the stream
        await page.waitForTimeout(3000);
        await page.getByRole('button', { name: 'Stop' }).click();

        await page.getByLabel('Header navigation').getByText('Dataset').click();

        // Confirm media was uploaded
        const listbox = page.getByRole('listbox', { name: 'data-collection-grid' });
        const options = listbox.getByRole('option');

        expect(options.count).toBeGreaterThanOrEqual(40);
    });

    await test.step('Cleanup', async () => {
        await page.getByText('Geti Tune').click(); // Go back to /projects
        await expect(page.getByText('Test Project')).toBeVisible();

        await page.getByRole('button', { name: /open project options/i }).click();
        await page.getByText(/Delete/).click();
        await expect(page.getByText('Project deleted successfully')).toBeVisible();
        await expect(page.getByText('Test Project')).toBeHidden();
    });
});
