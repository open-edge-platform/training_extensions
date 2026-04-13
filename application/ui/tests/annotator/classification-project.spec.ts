// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { expect, Page } from '@playwright/test';
import { getMockedLabel } from 'mocks/mock-labels';
import { getMockedProject } from 'mocks/mock-project';
import { HttpResponse } from 'msw';

import { http, test } from '../fixtures';
import { blueLabel, candyBinaryHandler, redLabel } from './annotator-fixtures';

const greenLabel = getMockedLabel({ id: 'green-label', name: 'green-label', color: '#33b74bff' });
const yellowLabel = getMockedLabel({ id: 'yellow-label', name: 'yellow-label', color: '#ffff00' });

const mockedClassificationProject = getMockedProject({
    id: '123e4567-e89b-12d3-a456-426614174000',
    task: {
        exclusive_labels: true,
        task_type: 'classification',
        labels: [redLabel, blueLabel, greenLabel, yellowLabel],
    },
});

test.describe('Annotator Classification', () => {
    test.beforeEach(async ({ network }) => {
        network.use(
            http.get('/api/projects/{project_id}', () => {
                return HttpResponse.json(mockedClassificationProject);
            }),
            candyBinaryHandler
        );
    });

    const getAnnotationShape = (page: Page) => {
        // The annotation consists of two elements: a mask (first) and the annotation shape itself (second)
        return page.getByLabel('annotation full image').last();
    };

    test.describe('Single label', () => {
        test('select multiple labels update the current annotation', async ({ page }) => {
            await page.goto(`/projects/${mockedClassificationProject.id}/dataset`);
            await page.getByRole('img', { name: 'item-1.jpg' }).dblclick();

            await test.step('selecting a label adds a new labeled annotation', async () => {
                await page.getByRole('button', { name: `Label ${redLabel.name}` }).click();

                const annotation = getAnnotationShape(page);
                await expect(annotation).toBeVisible();
                await expect(annotation).toHaveAttribute('stroke', redLabel.color);
            });

            await test.step('selecting another label changes the annotation label', async () => {
                const annotation = getAnnotationShape(page);
                await page.getByRole('button', { name: `Label ${greenLabel.name}` }).click();

                await expect(annotation).toHaveAttribute('stroke', greenLabel.color);

                await page.getByRole('button', { name: `Label ${yellowLabel.name}` }).click();

                await expect(page.getByLabel(`label ${yellowLabel.name} background`)).toBeVisible();
                await expect(annotation).toHaveAttribute('stroke', yellowLabel.color);
            });

            await test.step('count the total annotations, including the mask', async () => {
                expect(await page.getByLabel('annotation full image').count()).toBe(2);
            });
        });

        test('remove the annotations when label is removed', async ({ page }) => {
            await page.goto(`/projects/${mockedClassificationProject.id}/dataset`);
            await page.getByRole('img', { name: 'item-1.jpg' }).dblclick();

            await test.step('add initial labeled annotation', async () => {
                await page.getByRole('button', { name: `Label ${redLabel.name}` }).click();

                const annotation = getAnnotationShape(page);
                await expect(annotation).toBeVisible();
                await expect(annotation).toHaveAttribute('stroke', redLabel.color);
            });

            await test.step('remove the annotation label', async () => {
                await page.getByLabel(`Remove ${redLabel.name}`).click();
                expect(await page.getByLabel('annotation full image').count()).toBe(0);
            });
        });

        test('does not support empty label', async ({ page }) => {
            await page.goto(`/projects/${mockedClassificationProject.id}/dataset`);
            await page.getByRole('img', { name: 'item-1.jpg' }).dblclick();

            await expect(page.getByRole('button', { name: 'Label No label' })).toBeHidden();
            await expect(page.getByRole('button', { name: `Label ${redLabel.name}` })).toBeVisible();
            await expect(page.getByRole('button', { name: `Label ${blueLabel.name}` })).toBeVisible();
            await expect(page.getByRole('button', { name: `Label ${greenLabel.name}` })).toBeVisible();
            await expect(page.getByRole('button', { name: `Label ${yellowLabel.name}` })).toBeVisible();
        });
    });

    test.describe('Multiple labels', () => {
        test.beforeEach(async ({ network }) => {
            network.use(
                http.get('/api/projects/{project_id}', () => {
                    return HttpResponse.json({
                        ...mockedClassificationProject,
                        task: { ...mockedClassificationProject.task, exclusive_labels: false },
                    });
                })
            );
        });

        test('add multiple labels', async ({ page }) => {
            await page.goto(`/projects/${mockedClassificationProject.id}/dataset`);
            await page.getByRole('img', { name: 'item-1.jpg' }).dblclick();

            await test.step('add initial labeled annotation', async () => {
                await page.getByRole('button', { name: `Label ${redLabel.name}` }).click();

                const annotation = getAnnotationShape(page);
                await expect(annotation).toBeVisible();
                await expect(annotation).toHaveAttribute('stroke', redLabel.color);
            });

            await test.step('add more labels and use white stroke color for multiple labels', async () => {
                await page.getByRole('button', { name: `Label ${greenLabel.name}` }).click();
                await page.getByRole('button', { name: `Label ${yellowLabel.name}` }).click();

                await expect(page.getByLabel(`label ${redLabel.name} background`)).toBeVisible();
                await expect(page.getByLabel(`label ${greenLabel.name} background`)).toBeVisible();
                await expect(page.getByLabel(`label ${yellowLabel.name} background`)).toBeVisible();

                const annotation = getAnnotationShape(page);
                await expect(annotation).toBeVisible();
                await expect(annotation).toHaveAttribute('stroke', 'white');
            });
        });

        test('remove the annotations when all labels are removed', async ({ page }) => {
            await page.goto(`/projects/${mockedClassificationProject.id}/dataset`);
            await page.getByRole('img', { name: 'item-1.jpg' }).dblclick();

            await test.step('add annotation with multiple labels', async () => {
                await page.getByRole('button', { name: `Label ${redLabel.name}` }).click();
                await page.getByRole('button', { name: `Label ${greenLabel.name}` }).click();
                await page.getByRole('button', { name: `Label ${yellowLabel.name}` }).click();
            });

            await test.step('removing all labels', async () => {
                await page.getByLabel(`Remove ${redLabel.name}`).click();
                await page.getByLabel(`Remove ${greenLabel.name}`).click();
                await page.getByLabel(`Remove ${yellowLabel.name}`).click();

                expect(await page.getByLabel('annotation full image').count()).toBe(0);
            });
        });

        test(`hide/show annotation's label using setting`, async ({ page }) => {
            await page.goto(`/projects/${mockedClassificationProject.id}/dataset`);
            await page.getByRole('img', { name: 'item-1.jpg' }).dblclick();

            await test.step('add multi-label annotation', async () => {
                await page.getByRole('button', { name: `Label ${redLabel.name}` }).click();
                await page.getByRole('button', { name: `Label ${greenLabel.name}` }).click();
                await page.getByRole('button', { name: `Label ${yellowLabel.name}` }).click();

                await expect(page.getByLabel(`label ${redLabel.name} background`)).toBeVisible();
                await expect(page.getByLabel(`label ${greenLabel.name} background`)).toBeVisible();
                await expect(page.getByLabel(`label ${yellowLabel.name} background`)).toBeVisible();
            });

            await test.step('hide labels', async () => {
                await page.getByRole('button', { name: 'Settings' }).click();
                await page.getByRole('switch', { name: 'Hide labels' }).click();

                await expect(page.getByLabel(`label ${redLabel.name} background`)).toBeHidden();
                await expect(page.getByLabel(`label ${greenLabel.name} background`)).toBeHidden();
                await expect(page.getByLabel(`label ${yellowLabel.name} background`)).toBeHidden();

                await expect(getAnnotationShape(page)).toBeVisible();
            });

            await test.step('show labels', async () => {
                await page.getByRole('switch', { name: 'Hide labels' }).click();

                await expect(page.getByLabel(`label ${redLabel.name} background`)).toBeVisible();
                await expect(page.getByLabel(`label ${greenLabel.name} background`)).toBeVisible();
                await expect(page.getByLabel(`label ${yellowLabel.name} background`)).toBeVisible();

                await expect(getAnnotationShape(page)).toBeVisible();
            });
        });

        test('hide/show annotation', async ({ page }) => {
            await page.goto(`/projects/${mockedClassificationProject.id}/dataset`);
            await page.getByRole('img', { name: 'item-1.jpg' }).dblclick();

            await test.step('add multi-label annotation', async () => {
                await page.getByRole('button', { name: `Label ${redLabel.name}` }).click();
                await page.getByRole('button', { name: `Label ${greenLabel.name}` }).click();

                await expect(page.getByLabel(`label ${redLabel.name} background`)).toBeVisible();
                await expect(page.getByLabel(`label ${greenLabel.name} background`)).toBeVisible();
            });

            await test.step('hide annotation', async () => {
                await page.getByRole('button', { name: 'Hide annotations' }).click();

                await expect(page.getByLabel(`label ${redLabel.name} background`)).toBeHidden();
                await expect(page.getByLabel(`label ${greenLabel.name} background`)).toBeHidden();

                await expect(getAnnotationShape(page)).toBeHidden();
            });

            await test.step('show annotation', async () => {
                await page.getByRole('button', { name: 'Show annotations' }).click();

                await expect(page.getByLabel(`label ${redLabel.name} background`)).toBeVisible();
                await expect(page.getByLabel(`label ${greenLabel.name} background`)).toBeVisible();

                await expect(getAnnotationShape(page)).toBeVisible();
            });
        });

        test.describe('Handles empty label', () => {
            test('label assignment', async ({ page }) => {
                await page.goto(`/projects/${mockedClassificationProject.id}/dataset`);
                await page.getByRole('img', { name: 'item-1.jpg' }).dblclick();

                await test.step('add multi-label annotation', async () => {
                    await page.getByRole('button', { name: `Label ${redLabel.name}` }).click();
                    await page.getByRole('button', { name: `Label ${greenLabel.name}` }).click();

                    await expect(page.getByLabel(`label ${redLabel.name} background`)).toBeVisible();
                    await expect(page.getByLabel(`label ${greenLabel.name} background`)).toBeVisible();
                });

                await test.step('assigning "no label" removes other labels', async () => {
                    await page.getByRole('button', { name: `Label No label` }).click();

                    await expect(page.getByLabel(`label No label background`)).toBeVisible();
                    await expect(page.getByLabel(`label ${redLabel.name} background`)).toBeHidden();
                    await expect(page.getByLabel(`label ${greenLabel.name} background`)).toBeHidden();
                });

                await test.step('assigning other label than "no label" removes "no label"', async () => {
                    await page.getByRole('button', { name: `Label ${redLabel.name}` }).click();

                    await expect(page.getByLabel(`label No label background`)).toBeHidden();
                    await expect(page.getByLabel(`label ${redLabel.name} background`)).toBeVisible();
                });
            });

            test('renders "No label" when server returns empty annotations list', async ({ page, network }) => {
                network.use(
                    http.get('/api/projects/{project_id}/dataset/media/{media_id}/annotations', () => {
                        return HttpResponse.json({
                            annotations: [],
                            user_reviewed: true,
                            subset: 'training',
                        });
                    })
                );

                await page.goto(`/projects/${mockedClassificationProject.id}/dataset`);
                await page.getByRole('img', { name: 'item-1.jpg' }).dblclick();

                await expect(page.getByLabel(`label No label background`)).toBeVisible();
            });
        });
    });
});
