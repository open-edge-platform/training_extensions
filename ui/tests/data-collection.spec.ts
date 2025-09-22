// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { HttpResponse } from 'msw';

import { expect, http, test } from './fixtures';
import { getMultipleMockedMediaItems } from './test-utils/mocks';

const mockedItems = getMultipleMockedMediaItems(20, '1');
const mockedItems2 = getMultipleMockedMediaItems(20, '2');
const mockedItems3 = getMultipleMockedMediaItems(20, '3');
const totalElements = mockedItems.length + mockedItems2.length + mockedItems3.length;

test.describe('Dataset', () => {
    test.beforeEach(({ network }) => {
        network.use(
            http.get('/api/projects/{project_id}/dataset/items', ({ query }) => {
                const offset = Number(query.get('offset') ?? 0);
                const limit = Number(query.get('limit'));
                const items = offset === 0 ? mockedItems : offset === 20 ? mockedItems2 : mockedItems3;

                return HttpResponse.json({
                    items,
                    pagination: {
                        offset,
                        limit,
                        count: items.length,
                        total: totalElements,
                    },
                });
            })
        );
    });

    test('list items', async ({ page }) => {
        await page.goto('projects/some-id/dataset');
        const loadedItems = 40;

        await expect(page.getByText(`${loadedItems} images`)).toBeVisible();

        await page.getByLabel('select all').click();

        await expect(page.getByText(`${loadedItems} selected`)).toBeVisible();
    });

    test('select multiple images', async ({ page }) => {
        const selectedElements = 5;

        await page.goto('projects/some-id/dataset');
        const elements = await page.getByRole('option').all();

        for await (const element of elements.slice(0, selectedElements)) {
            await element.click();
        }

        await expect(page.getByText(`${selectedElements} selected`)).toBeVisible();
    });

    test('loads additional items when scrolling to the end of the container', async ({ page }) => {
        await page.goto('projects/some-id/dataset');

        await expect(page.getByText('40 images')).toBeVisible();

        await page.getByRole('listbox', { name: 'data-collection-grid' }).press('End');

        await expect(page.getByText(`${totalElements} images`)).toBeVisible();
    });
});
