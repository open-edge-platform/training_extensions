// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

import { expect, Page } from '@playwright/test';
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
const greenLabel = getMockedLabel({ id: 'green-label', name: 'green-label', color: '#33b74bff' });
const yellowLabel = getMockedLabel({ id: 'yellow-label', name: 'yellow-label', color: '#ffff00' });

const mockedClassificationProject = getMockedProject({
    id: 'candy-id',
    task: {
        exclusive_labels: true,
        task_type: 'classification',
        labels: [redLabel, blueLabel, greenLabel, yellowLabel],
    },
});

test.beforeEach(async ({ network }) => {
    network.use(
        http.get('/api/projects/{project_id}', () => {
            return HttpResponse.json(mockedClassificationProject);
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
        })
    );
});

test.describe('Annotator Classification', () => {
    test.beforeEach(async ({ page }) => {
        await page.goto(`/projects/${mockedClassificationProject.id}/dataset`);
        await page.getByRole('img', { name: 'item-1.jpg' }).dblclick();
    });

    const getAnnotationShape = (page: Page) => {
        // The annotation consists of two elements: a mask (first) and the annotation shape itself (second)
        return page.getByLabel('annotation full image').last();
    };

    test.describe('Single label', () => {
        test('select multiple labels update the current annotation', async ({ page }) => {
            await test.step('selecting a label adds a new labeled annotation', async () => {
                await page.getByRole('button', { name: 'Select label Label Picker' }).click();
                await page.getByTestId('popover').getByText(redLabel.name).click();

                const annotation = getAnnotationShape(page);
                await expect(annotation).toBeVisible();
                await expect(annotation).toHaveAttribute('stroke', redLabel.color);
            });

            await test.step('selecting another label changes the annotation label', async () => {
                const annotation = getAnnotationShape(page);

                await page.getByRole('button', { name: `${redLabel.name} Label Picker` }).click();
                await page.getByTestId('popover').getByText(greenLabel.name).click();
                await expect(annotation).toHaveAttribute('stroke', greenLabel.color);

                await page.getByRole('button', { name: `${greenLabel.name} Label Picker` }).click();
                await page.getByTestId('popover').getByText(yellowLabel.name).click();
                await page.getByLabel(`label ${yellowLabel.name}`).nth(3).click();
                await expect(annotation).toHaveAttribute('stroke', yellowLabel.color);
            });

            await test.step('count the total annotations, including the mask', async () => {
                expect(await page.getByLabel('annotation full image').count()).toBe(2);
            });
        });

        test('remove the annotations when label is removed', async ({ page }) => {
            await test.step('add initial labeled annotation', async () => {
                await page.getByRole('button', { name: 'Select label Label Picker' }).click();
                await page.getByTestId('popover').getByText(redLabel.name).click();

                const annotation = getAnnotationShape(page);
                await expect(annotation).toBeVisible();
                await expect(annotation).toHaveAttribute('stroke', redLabel.color);
            });

            await test.step('remove the annotation label', async () => {
                await page.getByLabel(`Remove ${redLabel.name}`).nth(1).click();
                expect(await page.getByLabel('annotation full image').count()).toBe(0);
            });
        });
    });

    test.describe('Multiple labels', () => {
        test.beforeEach(async ({ network, page }) => {
            network.use(
                http.get('/api/projects/{project_id}', () => {
                    return HttpResponse.json({
                        ...mockedClassificationProject,
                        task: { ...mockedClassificationProject.task, exclusive_labels: false },
                    });
                })
            );

            await page.goto(`/projects/${mockedClassificationProject.id}/dataset`);
            await page.getByRole('img', { name: 'item-1.jpg' }).dblclick();
        });

        test('add multiple labels', async ({ page }) => {
            await test.step('add initial labeled annotation', async () => {
                await page.getByRole('button', { name: 'Select label Label Picker' }).click();
                await page.getByTestId('popover').getByText(redLabel.name).click();

                const annotation = getAnnotationShape(page);
                await expect(annotation).toBeVisible();
                await expect(annotation).toHaveAttribute('stroke', redLabel.color);
            });

            await test.step('add more labels and keep the initial stroke color', async () => {
                await page.getByRole('button', { name: `${redLabel.name} Label Picker` }).click();
                await page.getByTestId('popover').getByText(greenLabel.name).click();

                await page.getByRole('button', { name: `${greenLabel.name} Label Picker` }).click();
                await page.getByTestId('popover').getByText(yellowLabel.name).click();

                await expect(page.getByLabel(`label ${redLabel.name} background`).nth(1)).toBeVisible();
                await expect(page.getByLabel(`label ${greenLabel.name} background`).nth(1)).toBeVisible();
                await expect(page.getByLabel(`label ${yellowLabel.name} background`).nth(1)).toBeVisible();

                const annotation = getAnnotationShape(page);
                await expect(annotation).toBeVisible();
                await expect(annotation).toHaveAttribute('stroke', redLabel.color);
            });
        });

        test('remove the annotations when all labels are removed', async ({ page }) => {
            await test.step('add annotation with multiple labels', async () => {
                await page.getByRole('button', { name: 'Select label Label Picker' }).click();
                await page.getByTestId('popover').getByText(redLabel.name).click();

                await page.getByRole('button', { name: `${redLabel.name} Label Picker` }).click();
                await page.getByTestId('popover').getByText(greenLabel.name).click();

                await page.getByRole('button', { name: `${greenLabel.name} Label Picker` }).click();
                await page.getByTestId('popover').getByText(yellowLabel.name).click();
            });

            await test.step('removing all labels', async () => {
                await page.getByLabel(`Remove ${redLabel.name}`).nth(1).click();
                await page.getByLabel(`Remove ${greenLabel.name}`).nth(1).click();
                await page.getByLabel(`Remove ${yellowLabel.name}`).nth(1).click();

                expect(await page.getByLabel('annotation full image').count()).toBe(0);
            });
        });

        test(`hide/show annotation's label using setting`, async ({ page }) => {
            await test.step('add multi-label annotation', async () => {
                await page.getByRole('button', { name: 'Select label Label Picker' }).click();
                await page.getByTestId('popover').getByText(redLabel.name).click();

                await page.getByRole('button', { name: `${redLabel.name} Label Picker` }).click();
                await page.getByTestId('popover').getByText(greenLabel.name).click();

                await page.getByRole('button', { name: `${greenLabel.name} Label Picker` }).click();
                await page.getByTestId('popover').getByText(yellowLabel.name).click();

                await expect(page.getByLabel(`label ${redLabel.name} background`).nth(1)).toBeVisible();
                await expect(page.getByLabel(`label ${greenLabel.name} background`).nth(1)).toBeVisible();
                await expect(page.getByLabel(`label ${yellowLabel.name} background`).nth(1)).toBeVisible();
            });

            await test.step('hide labels', async () => {
                await page.getByRole('button', { name: 'Settings' }).click();
                await page.getByRole('switch', { name: 'Hide labels' }).click();

                await expect(page.getByLabel(`label ${redLabel.name} background`).nth(1)).toBeHidden();
                await expect(page.getByLabel(`label ${greenLabel.name} background`).nth(1)).toBeHidden();
                await expect(page.getByLabel(`label ${yellowLabel.name} background`).nth(1)).toBeHidden();

                await expect(getAnnotationShape(page)).toBeVisible();
            });

            await test.step('show labels', async () => {
                await page.getByRole('switch', { name: 'Hide labels' }).click();

                await expect(page.getByLabel(`label ${redLabel.name} background`).nth(1)).toBeVisible();
                await expect(page.getByLabel(`label ${greenLabel.name} background`).nth(1)).toBeVisible();
                await expect(page.getByLabel(`label ${yellowLabel.name} background`).nth(1)).toBeVisible();

                await expect(getAnnotationShape(page)).toBeVisible();
            });
        });

        test('hide/show annotation', async ({ page }) => {
            await test.step('add multi-label annotation', async () => {
                await page.getByRole('button', { name: 'Select label Label Picker' }).click();
                await page.getByTestId('popover').getByText(redLabel.name).click();

                await page.getByRole('button', { name: `${redLabel.name} Label Picker` }).click();
                await page.getByTestId('popover').getByText(greenLabel.name).click();

                await expect(page.getByLabel(`label ${redLabel.name} background`).nth(1)).toBeVisible();
                await expect(page.getByLabel(`label ${greenLabel.name} background`).nth(1)).toBeVisible();
            });

            await test.step('hide annotation', async () => {
                await page.getByRole('button', { name: 'Hide annotations' }).click();

                await expect(page.getByLabel(`label ${redLabel.name} background`).nth(1)).toBeHidden();
                await expect(page.getByLabel(`label ${greenLabel.name} background`).nth(1)).toBeHidden();

                await expect(getAnnotationShape(page)).toBeHidden();
            });

            await test.step('show annotation', async () => {
                await page.getByRole('button', { name: 'Show annotations' }).click();

                await expect(page.getByLabel(`label ${redLabel.name} background`).nth(1)).toBeVisible();
                await expect(page.getByLabel(`label ${greenLabel.name} background`).nth(1)).toBeVisible();

                await expect(getAnnotationShape(page)).toBeVisible();
            });
        });
    });
});
