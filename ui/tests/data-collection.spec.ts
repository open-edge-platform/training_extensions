// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { expect, test } from './fixtures';

test.describe('Dataset', () => {
    test('list items', async ({ page }) => {
        await page.goto('/dataset');

        await expect(page.getByText('30 images')).toBeVisible();

        await page.getByLabel('select all').click();

        await expect(page.getByText('30 selected')).toBeVisible();
    });

    test('select multiple images', async ({ page }) => {
        const selectedElements = 5;

        await page.goto('/dataset');
        const elements = await page.getByRole('option').all();

        for await (const element of elements.slice(0, selectedElements)) {
            await element.click();
        }

        await expect(page.getByText(`${selectedElements} selected`)).toBeVisible();
    });
});
