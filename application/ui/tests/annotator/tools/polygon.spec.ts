// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

import { expect } from '@playwright/test';
import { getMockedLabel } from 'mocks/mock-labels';
import { getMockedProject } from 'mocks/mock-project';
import { HttpResponse } from 'msw';

import { Polygon } from '../../../src/shared/types';
import { http, test } from '../../fixtures';
import { withRelative } from '../../utils/mouse';

const filename = fileURLToPath(import.meta.url);
const dirname = path.dirname(filename);
const candyPngPath = path.resolve(dirname, '../../assets/candy.png');
const candyPngBuffer = fs.readFileSync(candyPngPath);

const redLabel = getMockedLabel({ id: 'red-label', name: 'red-label', color: '#ad2323' });
const blueLabel = getMockedLabel({ id: 'blue-label', name: 'blue-label', color: '#2424a0' });

const mockedDetectionProject = getMockedProject({
    id: '123e4567-e89b-12d3-a456-426614174000',
    task: {
        exclusive_labels: true,
        task_type: 'instance_segmentation',
        labels: [redLabel, blueLabel],
    },
});

const polygonShape: Polygon = {
    type: 'polygon',
    points: [
        { x: 100, y: 100 },
        { x: 250, y: 100 },
        { x: 250, y: 250 },
        { x: 180, y: 300 },
        { x: 100, y: 250 },
    ],
};

test.describe('Polygon', () => {
    test.beforeEach(async ({ network }) => {
        network.use(
            http.get('/api/projects/{project_id}', () => {
                return HttpResponse.json(mockedDetectionProject);
            }),
            http.get('/api/projects/{project_id}/dataset/media/{media_id}/binary', async () => {
                return HttpResponse.arrayBuffer(candyPngBuffer.buffer, {
                    headers: { 'Content-Type': 'image/png' },
                });
            })
        );
    });

    test('Remove points', async ({ page, polygonTool }) => {
        await page.goto(`/projects/${mockedDetectionProject.id}/dataset`);
        await page.getByRole('img', { name: 'item-1.jpg' }).dblclick();

        await test.step('Draw an annotation', async () => {
            await polygonTool.selectPolygonTool();
            await polygonTool.drawPolygon(polygonShape);
        });

        await test.step('Remove point with context menu', async ({}) => {
            const relative = await withRelative(page);
            const currentPoint = polygonShape.points[3];
            const relativePoint = relative(currentPoint.x, currentPoint.y);

            await page.getByRole('button', { name: 'selection tool' }).click();

            await expect(
                page.getByLabel(`Resize polygon (${currentPoint.x}, ${currentPoint.y}) anchor`)
            ).toBeInViewport();
            await polygonTool.deletePointByRightClick(relativePoint);

            await expect(
                page.getByLabel(`Resize polygon (${currentPoint.x}, ${currentPoint.y}) anchor`)
            ).not.toBeInViewport();
            await expect(page.getByRole('button', { name: 'delete point' })).not.toBeInViewport();
        });
    });
});
