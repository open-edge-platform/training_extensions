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

    await test.step('Increase capture rate', async () => {
        await page.getByRole('button', { name: 'Toggle Data collection policy' }).click();

        const sliderInput = page.locator('input[type="range"]');

        await expect(sliderInput).toBeVisible();
        // Rate of 5 images per second
        await sliderInput.fill('5');
        await expect(sliderInput).toHaveValue('5');
    });

    await test.step('Start stream to begin capture', async () => {
        await page.getByLabel('Start stream').click();

        await expect(page.getByLabel('Connected')).toBeVisible({ timeout: 120000 });

        // Wait a bit for the stream to capture some data
        await page.waitForTimeout(5000);
    });

    await test.step('Confirm data was collected', async () => {
        await page.getByLabel('Header navigation').getByText('Dataset').click();

        await expect(page.getByText(/\d+ images?/)).toBeVisible({ timeout: 10000 });

        const listbox = page.getByRole('listbox', { name: 'data-collection-grid' });
        const options = listbox.getByRole('option');

        const count = await options.count();
        expect(count).toBeGreaterThanOrEqual(1);
    });
});
