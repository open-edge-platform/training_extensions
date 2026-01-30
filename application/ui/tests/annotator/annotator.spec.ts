// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

import { expect } from '@playwright/test';
import { getMockedLabel } from 'mocks/mock-labels';
import { getMockedProject } from 'mocks/mock-project';
import { HttpResponse } from 'msw';

import { AnnotationDTO } from '../../src/constants/shared-types';
import { http, test } from '../fixtures';

const filename = fileURLToPath(import.meta.url);
const dirname = path.dirname(filename);
const candyPngPath = path.resolve(dirname, '../assets/candy.png');
const candyPngBuffer = fs.readFileSync(candyPngPath);

const redLabel = getMockedLabel({ id: 'red-label', name: 'red-label', color: '#ad2323' });
const blueLabel = getMockedLabel({ id: 'blue-label', name: 'blue-label', color: '#2424a0' });

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
            http.get('/api/projects/{project_id}/dataset/media/{media_id}/binary', async () => {
                return HttpResponse.arrayBuffer(candyPngBuffer.buffer, {
                    headers: { 'Content-Type': 'image/png' },
                });
            }),
            http.get('/api/projects/{project_id}/dataset/items/{dataset_item_id}/annotations', async () => {
                return HttpResponse.json({
                    annotations: [],
                    user_reviewed: true,
                });
            })
        );
    });

    test('Add and change annotations labels', async ({ page, boundingBoxTool }) => {
        await page.goto(`/projects/${mockedDetectionProject.id}/dataset`);
        await page.getByRole('img', { name: 'item-1.jpg' }).dblclick();

        await test.step('Draw an annotation', async () => {
            await boundingBoxTool.selectTool();
            await boundingBoxTool.drawBoundingBox({ x: 100, y: 100, width: 150, height: 150 });
            await expect(page.getByLabel(`label ${redLabel.name}`).nth(1)).toBeInViewport();
        });

        await test.step('Change annotation label by clicking label badge', async () => {
            await page.getByRole('button', { name: 'selection tool' }).click();
            await page.getByLabel('annotation rect').nth(1).click();

            await expect(page.getByRole('button', { name: `Label ${redLabel.name}` })).toHaveAttribute(
                'aria-pressed',
                'true'
            );

            await page.getByRole('button', { name: `Label ${blueLabel.name}` }).click();

            await expect(page.getByLabel(`label ${blueLabel.name}`).nth(1)).toBeInViewport();
            await expect(page.getByRole('button', { name: `Label ${blueLabel.name}` })).toHaveAttribute(
                'aria-pressed',
                'true'
            );
        });

        await test.step('Draw a second annotation', async () => {
            await boundingBoxTool.selectTool();
            await boundingBoxTool.drawBoundingBox({ x: 300, y: 200, width: 150, height: 150 });

            await expect(page.getByLabel(`label ${blueLabel.name}`).nth(1)).toBeInViewport();
        });

        await test.step('Change second annotation to red label', async () => {
            await page.getByRole('button', { name: 'selection tool' }).click();
            await page.getByLabel('annotation rect').nth(3).click();
            await page.getByRole('button', { name: `Label ${redLabel.name}` }).click();

            await expect(page.getByLabel(`label ${redLabel.name}`).nth(1)).toBeInViewport();
        });

        await test.step('Verify both annotations have correct labels', async () => {
            await page.getByLabel('annotation rect').nth(2).click();
            await expect(page.getByLabel(`label ${blueLabel.name}`).nth(1)).toBeInViewport();

            await page.getByLabel('annotation rect').nth(3).click();
            await expect(page.getByLabel(`label ${redLabel.name}`).nth(1)).toBeInViewport();
        });
    });

    test('change multiple labels at once', async ({ page, boundingBoxTool }) => {
        await page.goto(`/projects/${mockedDetectionProject.id}/dataset`);
        await page.getByRole('img', { name: 'item-1.jpg' }).dblclick();

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

        await test.step('Change selected annotations label using label badge', async () => {
            const container = page.getByLabel('annotation rect');

            await container.nth(5).click({ modifiers: ['Shift'] });
            await container.nth(4).click({ modifiers: ['Shift'] });
            await container.nth(3).click({ modifiers: ['Shift'] });

            await page.getByRole('button', { name: `Label ${blueLabel.name}` }).click();

            expect(await page.getByLabel(`label ${blueLabel.name} background`).count()).toBe(annotations.length);
        });
    });

    test('Annotation vs Prediction', async ({ page, annotatorPage, boundingBoxTool, network }) => {
        const predictions = [
            {
                shape: {
                    type: 'rectangle',
                    x: 3,
                    y: 0,
                    width: 780,
                    height: 421,
                },
                labels: [
                    {
                        id: blueLabel.id,
                    },
                ],
                confidences: [0.9619140625],
            },
            {
                shape: {
                    type: 'rectangle',
                    x: 1007,
                    y: 624,
                    width: 909,
                    height: 456,
                },
                labels: [
                    {
                        id: blueLabel.id,
                    },
                ],
                confidences: [0.9599609375],
            },
            {
                shape: {
                    type: 'rectangle',
                    x: 291,
                    y: 0,
                    width: 1553,
                    height: 889,
                },
                labels: [
                    {
                        id: blueLabel.id,
                    },
                ],
                confidences: [0.904296875],
            },
        ] satisfies AnnotationDTO[];

        network.use(
            http.get('/api/projects/{project_id}/dataset/items/{dataset_item_id}/annotations', async () => {
                return HttpResponse.json({
                    annotations: predictions,
                    user_reviewed: false,
                    prediction_model_id: '123',
                });
            })
        );

        await page.goto(`/projects/${mockedDetectionProject.id}/dataset`);
        await page.getByRole('img', { name: 'item-1.jpg' }).dblclick();

        await test.step('Draws an annotation in annotation mode', async () => {
            await expect(annotatorPage.getAnnotationMode('Annotation')).toHaveAttribute('aria-pressed', 'true');
            await expect(annotatorPage.getAnnotationMode('Prediction')).toHaveAttribute('aria-pressed', 'false');

            const annotation = { x: 100, y: 100, width: 150, height: 150 };

            await expect(annotatorPage.getPrimaryToolbar()).toBeVisible();

            await boundingBoxTool.selectTool();
            await boundingBoxTool.drawBoundingBox(annotation);

            expect(await annotatorPage.getAnnotationsListItems('annotation rect')).toHaveLength(1);

            await expect(page.getByLabel(`label ${redLabel.name} background`)).toHaveCount(1);

            expect(await annotatorPage.getAnnotationsListItems('prediction rect')).toHaveLength(0);
        });

        await test.step('Displays server prediction in prediction mode', async () => {
            await annotatorPage.openPredictionMode();

            await expect(annotatorPage.getAnnotationMode('Prediction')).toHaveAttribute('aria-pressed', 'true');
            await expect(annotatorPage.getAnnotationMode('Annotation')).toHaveAttribute('aria-pressed', 'false');

            await expect(annotatorPage.getPrimaryToolbar()).toBeHidden();
            await expect(page.getByLabel('Labels')).toHaveAttribute('aria-disabled', 'true');

            await expect(page.getByLabel(`label ${blueLabel.name} background`)).toHaveCount(predictions.length);
            expect(await annotatorPage.getAnnotationsListItems('prediction rect')).toHaveLength(predictions.length);

            expect(await annotatorPage.getAnnotationsListItems('annotation rect')).toHaveLength(0);
            await expect(page.getByLabel(`label ${redLabel.name} background`)).toHaveCount(0);
        });

        await test.step('Hides predictions in annotation mode, restores annotation', async () => {
            await annotatorPage.openAnnotationMode();

            expect(await annotatorPage.getAnnotationsListItems('prediction rect')).toHaveLength(0);
            expect(await annotatorPage.getAnnotationsListItems('annotation rect')).toHaveLength(1);
        });
    });
});
