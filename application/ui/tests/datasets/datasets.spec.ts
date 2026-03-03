// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

import { getMultipleMockedMediaImage } from 'mocks/mock-media';
import { HttpResponse } from 'msw';

import { expect, http, test } from '../fixtures';

const mockedItems = getMultipleMockedMediaImage(20, '1');
const mockedItems2 = getMultipleMockedMediaImage(20, '2');
const mockedItems3 = getMultipleMockedMediaImage(20, '3');
const totalElements = mockedItems.length + mockedItems2.length + mockedItems3.length;

const dirname = path.dirname(fileURLToPath(import.meta.url));
const sampleImagePath = path.resolve(dirname, '../assets/candy-thumbnail.png');
const sampleImageBuffer = fs.readFileSync(sampleImagePath);

test.describe('Dataset', () => {
    test.beforeEach(({ network }) => {
        network.use(
            http.get('/api/projects/{project_id}/dataset/media/{media_id}/binary', ({}) => {
                return HttpResponse.arrayBuffer(sampleImageBuffer.buffer, {
                    headers: { 'Content-Type': 'image/png' },
                });
            }),
            http.get('/api/projects/{project_id}/dataset/media', ({ query }) => {
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
        await page.goto('projects/id-1/dataset');
        const loadedItems = 40;

        await expect(page.getByText(`${loadedItems} images`)).toBeVisible();

        await page.getByLabel('select all').click();

        await expect(page.getByText(`${loadedItems} selected`)).toBeVisible();
    });

    test('select multiple images', async ({ page }) => {
        const selectedElements = 5;

        await page.goto('projects/id-1/dataset');

        await expect(page.getByText('40 images')).toBeVisible();

        const listbox = page.getByRole('listbox', { name: 'data-collection-grid' });
        const options = listbox.getByRole('option');

        for (let i = 0; i < selectedElements; i++) {
            await options.nth(i).click();
        }

        await expect(page.getByText(`${selectedElements} selected`)).toBeVisible();
    });

    test('loads additional items when scrolling to the end of the container', async ({ page }) => {
        await page.goto('projects/id-1/dataset');

        await expect(page.getByText('40 images')).toBeVisible();

        await page.getByRole('listbox', { name: 'data-collection-grid' }).press('End');

        await expect(page.getByText(`${totalElements} images`)).toBeVisible();
    });

    test('selected media item is saved in the URL', async ({ page, annotatorPage }) => {
        const [firstElement] = mockedItems;
        await page.goto('projects/id-1/dataset');
        await expect(annotatorPage.getAnnotationsList()).not.toBeInViewport();

        await page.getByRole('img', { name: firstElement.name, exact: true }).dblclick();

        await expect(annotatorPage.getAnnotationsList()).toBeInViewport();

        expect(page.url()).toContain(`/dataset/${firstElement.id}`);
    });
});
