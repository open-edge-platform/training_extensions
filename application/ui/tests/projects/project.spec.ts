// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { HttpResponse } from 'msw';

import { getMockedProject } from '../../mocks/mock-project';
import type { ProjectCreate } from '../../src/constants/shared-types';
import { expect, http, test } from '../fixtures';
import { stepCreateProject } from '../workflows/workflow-steps';
import { ProjectPage } from './project-page';

test.describe('Project', () => {
    test.beforeEach(({ network }) => {
        network.use(
            http.get('/api/projects', () =>
                HttpResponse.json([
                    getMockedProject({ id: 'id-1', name: 'Project 1' }),
                    getMockedProject({ id: 'id-2', name: 'Project 2' }),
                    getMockedProject({ id: 'id-3', name: 'Project 3' }),
                ])
            )
        );
    });

    test('displays the list of projects', async ({ page }) => {
        const projectPage = new ProjectPage(page);

        await projectPage.gotoList();

        await expect(page.getByText('Project 1')).toBeVisible();
        await expect(page.getByText('Project 2')).toBeVisible();
        await expect(page.getByText('Project 3')).toBeVisible();
    });

    test('creates a project', async ({ page, network }) => {
        const projectPage = new ProjectPage(page);
        const projectName = 'New Project';

        network.use(
            http.post('/api/projects', async ({ request, response }) => {
                const body: ProjectCreate = await request.json();

                return response(201).json(
                    getMockedProject({
                        id: 'new project id',
                        name: body.name,
                        task: {
                            task_type: body.task.task_type,
                            exclusive_labels: false,
                            labels: (body.task.labels ?? []).map((label, index) => ({
                                id: (index + 1).toString(),
                                color: index % 2 === 0 ? 'red' : 'blue',
                                name: label.name,
                            })),
                        },
                    })
                );
            })
        );

        network.use(
            http.get('/api/projects', () => {
                return HttpResponse.json([
                    getMockedProject({ id: '1', name: 'Project 1' }),
                    getMockedProject({ id: '2', name: 'Project 2' }),
                    getMockedProject({ id: 'id-3', name: 'Project 3' }),
                    getMockedProject({
                        id: 'new project id',
                        name: projectName,
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

        await stepCreateProject(page, {
            projectName,
            task: 'instance_segmentation',
            labels: ['Person', 'Animal'],
        });

        // Go back to project list and confirm the project was created
        await projectPage.gotoList();

        await expect(page.getByText(projectName, { exact: true })).toBeVisible();
    });

    test('disables create button for single-label classification based if there are no at least two labels', async ({
        page,
        network,
    }) => {
        const projectPage = new ProjectPage(page);

        await projectPage.gotoCreate();

        await projectPage.fillProjectForm({
            name: 'Single-label project labels check',
            task: 'classification',
            classificationType: 'Single-label',
            labelNames: ['Person'],
        });

        await expect(projectPage.getCreateProjectButton()).toBeDisabled();

        await projectPage.addLabel('Plane');

        await expect(projectPage.getCreateProjectButton()).toBeEnabled();

        network.use(
            http.post('/api/projects', ({ response }) => {
                return response(201).json(
                    getMockedProject({
                        id: 'single-label-project-id',
                        name: 'Single-label project labels check',
                        task: {
                            task_type: 'classification',
                            exclusive_labels: true,
                            labels: [
                                { id: '1', color: 'red', name: 'Person' },
                                { id: '2', color: 'blue', name: 'Plane' },
                            ],
                        },
                    })
                );
            })
        );

        await projectPage.getCreateProjectButton().click();
        await page.waitForURL(/dataset/);
    });

    test('deletes a project', async ({ page, network }) => {
        const projectPage = new ProjectPage(page);

        await projectPage.gotoList();

        // Open menu options
        await projectPage.openProjectMenu('id-3');

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

        await projectPage.clickDeleteMenuAction();
        await projectPage.confirmDeleteProject();

        await expect(page.getByText('Project deleted successfully')).toBeVisible();

        await expect(page.getByText('Project 3')).toBeHidden();
        await expect(page.getByText('Project 1')).toBeVisible();
        await expect(page.getByText('Project 2')).toBeVisible();
    });
});
