// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { HttpResponse } from 'msw';

import { getMockedProject } from '../../mocks/mock-project';
import { expect, http, test } from '../fixtures';
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

        await projectPage.gotoCreate();

        await projectPage.fillProjectForm({
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
                    getMockedProject({ id: '1', name: 'Project 1' }),
                    getMockedProject({ id: '2', name: 'Project 2' }),
                    getMockedProject({ id: 'id-3', name: 'Project 3' }),
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

        await expect(page.getByText('Person')).toBeVisible();
        await expect(page.getByText('Animal')).toBeVisible();

        await projectPage.getCreateProjectButton().click();

        // Correctly navigated to dataset page
        await page.waitForURL(/dataset/);

        // Go back to project list and confirm the project was created
        await projectPage.gotoList();

        await expect(page.getByText('New Project')).toBeVisible();
    });

    test('toggles create button for multi-label classification based on label count', async ({ page }) => {
        const projectPage = new ProjectPage(page);

        await projectPage.gotoCreate();

        await projectPage.fillProjectForm({
            name: 'Multi-label project labels check',
            task: 'classification',
            classificationType: 'Multi-label',
            labelNames: ['Person'],
        });

        await expect(projectPage.getCreateProjectButton()).toBeDisabled();
        await expect(projectPage.getMultiLabelValidationMessage()).toBeVisible();

        await projectPage.addLabel('Car');

        await expect(projectPage.getCreateProjectButton()).toBeEnabled();
        await expect(projectPage.getMultiLabelValidationMessage()).toBeHidden();
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
