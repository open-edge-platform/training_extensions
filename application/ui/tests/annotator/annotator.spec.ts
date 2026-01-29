// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

import { expect } from '@playwright/test';
import { getMockedLabel } from 'mocks/mock-labels';
import { mockedMedia } from 'mocks/mock-media';
import { getMockedProject } from 'mocks/mock-project';
import { HttpResponse } from 'msw';

import { http, test } from '../fixtures';

const filename = fileURLToPath(import.meta.url);
const dirname = path.dirname(filename);
const candyPngPath = path.resolve(dirname, '../assets/candy.png');
const candyPngBuffer = fs.readFileSync(candyPngPath);

const redLabel = getMockedLabel({ id: 'red-label', name: 'red-label', color: '#ad2323' });
const blueLabel = getMockedLabel({ id: 'blue-label', name: 'blue-label', color: '#2424a0' });

const mockedProject = getMockedProject({
    id: 'candy-id',
    task: {
        exclusive_labels: true,
        task_type: 'detection',
        labels: [redLabel, blueLabel],
    },
});

test.beforeEach(async ({ network, page }) => {
    network.use(
        http.get('/api/projects/{project_id}', () => {
            return HttpResponse.json(mockedProject);
        }),
        http.get('/api/projects/{project_id}/dataset/media', () => {
            return HttpResponse.json({
                items: [mockedMedia({ width: 1000, height: 750 })],
                pagination: { offset: 0, limit: 20, count: 1, total: 1 },
            });
        }),
        http.get('/api/projects/{project_id}/dataset/media/{media_id}/binary', async () => {
            return HttpResponse.arrayBuffer(candyPngBuffer.buffer, {
                headers: { 'Content-Type': 'image/png' },
            });
        }),
        http.get('/api/projects/{project_id}/dataset/items/{dataset_item_id}/annotations', async () => {
            return HttpResponse.json({
                annotations: [],
                user_reviewed: false,
            });
        })
    );

    await page.goto(`/projects/${mockedProject.id}/dataset`);
    await page.getByRole('img', { name: 'item-1.jpg' }).dblclick();
});

test.describe('Annotator', () => {
    test('Add and change annotations labels', async ({ page, boundingBoxTool }) => {
        await test.step('Draw an annotation', async () => {
            await boundingBoxTool.selectTool();
            await boundingBoxTool.drawBoundingBox({ x: 100, y: 100, width: 150, height: 150 });
            await expect(page.getByLabel(`label ${redLabel.name}`).nth(1)).toBeInViewport();
        });

        await test.step('Change annotation label', async () => {
            await page.getByRole('button', { name: 'selection tool' }).click();

            await page.getByLabel('annotation rect').nth(1).click();
            await page.getByRole('button', { name: `${redLabel.name} Label Picker` }).click();
            await page.getByTestId('popover').getByText(blueLabel.name).click();

            await expect(page.getByLabel(`label ${blueLabel.name}`).nth(1)).toBeInViewport();
        });

        await test.step('Change label selection', async () => {
            await page.getByRole('img', { name: 'annotations' }).click();
            await page.getByRole('button', { name: `${blueLabel.name} Label Picker` }).click();
            await page.getByTestId('popover').getByText(redLabel.name).click();
        });

        await test.step('Draw a second annotation', async () => {
            await boundingBoxTool.selectTool();
            await boundingBoxTool.drawBoundingBox({ x: 300, y: 200, width: 150, height: 150 });

            await expect(page.getByLabel(`label ${redLabel.name}`).nth(1)).toBeInViewport();
        });

        await test.step('label picker updates based on selected annotation', async () => {
            await page.getByRole('button', { name: 'selection tool' }).click();

            await page.getByLabel('annotation rect').nth(3).click();
            await expect(page.getByRole('button', { name: `${redLabel.name} Label Picker` })).toBeInViewport();

            await page.getByLabel('annotation rect').nth(2).click();
            await expect(page.getByRole('button', { name: `${blueLabel.name} Label Picker` })).toBeInViewport();
        });
    });

    test('change multiple labels at once', async ({ page, boundingBoxTool }) => {
        const annotations = [
            { x: 100, y: 100, width: 150, height: 150 },
            { x: 300, y: 200, width: 150, height: 150 },
            { x: 600, y: 300, width: 150, height: 150 },
        ];

        await test.step('Draw annotations', async () => {
            await boundingBoxTool.selectTool();

            for await (const annotation of annotations) {
                await boundingBoxTool.drawBoundingBox(annotation);
            }

            expect(await page.getByLabel(`label ${redLabel.name} background`).count()).toBe(annotations.length);
        });

        await test.step('Remove labels', async () => {
            await page.getByRole('button', { name: 'selection tool' }).click();
            const labels = page.getByLabel('Remove red-label');

            await labels.nth(0).click();
            await labels.nth(1).click();
        });

        await test.step('change select annotation label', async () => {
            const container = page.getByLabel('annotation rect');

            await container.nth(5).click({ modifiers: ['Shift'] });
            await container.nth(4).click({ modifiers: ['Shift'] });
            await container.nth(3).click({ modifiers: ['Shift'] });

            await page.getByRole('button', { name: `${redLabel.name} Label Picker` }).click();
            await page.getByTestId('popover').getByText(blueLabel.name).click();

            expect(await page.getByLabel(`label ${blueLabel.name} background`).count()).toBe(annotations.length);
        });
    });
});
