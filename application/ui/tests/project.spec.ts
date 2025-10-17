// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Page } from '@playwright/test';
import { HttpResponse } from 'msw';

import { getMockedProject } from '../mocks/mock-project';
import { expect, http, test } from './fixtures';

const fillProjectForm = async ({
    page,
    name,
    task,
    labelNames,
}: {
    page: Page;
    name: string;
    task: string;
    labelNames: string[];
}) => {
    // Edit name
    await page.getByRole('button', { name: /Project #1/ }).click();
    await page.getByLabel(/edit project name/).fill(name);
    await page.getByRole('button', { name: /Confirm/ }).click();

    // Edit task
    await page.getByLabel(task, { exact: true }).click();

    // Edit first label
    await page.getByLabel('Label input for Object').fill(labelNames[0]);

    // Add the rest of the labels
    for (let i = 1; i < labelNames.length; i++) {
        await page.getByRole('button', { name: /add next object/i }).click();

        await page.getByLabel('Label input for Object').fill(labelNames[i]);
    }
};

test.describe('Project', () => {
    test.beforeEach(({ network }) => {
        network.use(
            http.get('/api/projects', () => {
                return HttpResponse.json([
                    getMockedProject({
                        id: 'id-1',
                        name: 'Project 1',
                    }),
                    getMockedProject({
                        id: 'id-2',
                        name: 'Project 2',
                    }),
                    getMockedProject({
                        id: 'id-3',
                        name: 'Project 3',
                    }),
                ]);
            })
        );
    });

    test('displays the list of projects', async ({ page }) => {
        await page.goto('/projects');

        await expect(page.getByText('Project 1')).toBeVisible();
        await expect(page.getByText('Project 2')).toBeVisible();
        await expect(page.getByText('Project 3')).toBeVisible();
    });

    test('creates a project', async ({ page, network }) => {
        await page.goto('/projects/new');

        await fillProjectForm({
            page,
            name: 'New Project',
            task: 'instance_segmentation',
            labelNames: ['Person', 'Animal'],
        });

        network.use(
            http.post('/api/projects', ({ response }) => {
                return response(201).json(
                    getMockedProject({
                        id: 'new project id',
                        name: 'New Project',
                        task: {
                            task_type: 'instance_segmentation',
                            exclusive_labels: false,
                            labels: [
                                { id: '1', color: 'red', name: 'Person' },
                                { id: '2', color: 'blue', name: 'Animal' },
                            ],
                        },
                    })
                );
            })
        );

        network.use(
            http.get('/api/projects', () => {
                return HttpResponse.json([
                    getMockedProject({
                        id: '1',
                        name: 'Project 1',
                    }),
                    getMockedProject({
                        id: '2',
                        name: 'Project 2',
                    }),
                    getMockedProject({
                        id: 'id-3',
                        name: 'Project 3',
                    }),
                    getMockedProject({
                        id: 'new project id',
                        name: 'New Project',
                        task: {
                            task_type: 'instance_segmentation',
                            exclusive_labels: false,
                            labels: [
                                { id: '1', color: 'red', name: 'Person' },
                                { id: '2', color: 'blue', name: 'Animal' },
                            ],
                        },
                    }),
                ]);
            })
        );

        await page.getByRole('button', { name: /Create project/ }).click();

        // Correctly navigated to inference page
        await page.waitForURL(/inference/);
        expect(page.url()).toContain('/inference');

        // Go back to project list and confirm the project was created
        await page.goto('/projects');

        await expect(page.getByText('New Project')).toBeVisible();
    });

    test('deletes a project', async ({ page, network }) => {
        await page.goto('/projects');

        // Open menu options
        await page.getByTestId('id-3').click();

        network.use(
            http.get('/api/projects', () => {
                return HttpResponse.json([
                    getMockedProject({
                        id: '1',
                        name: 'Project 1',
                    }),
                    getMockedProject({
                        id: '2',
                        name: 'Project 2',
                    }),
                ]);
            })
        );

        await page.getByText(/Delete/).click();

        await expect(page.getByText('Project deleted successfully')).toBeVisible();

        await expect(page.getByText('Project 3')).toBeHidden();
        await expect(page.getByText('Project 1')).toBeVisible();
        await expect(page.getByText('Project 2')).toBeVisible();
    });
});
