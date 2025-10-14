// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { expect, Page, test } from '@playwright/test';

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
    await page.getByLabel(task).click();

    // Edit first label
    await page.getByLabel('Label input for Object').fill(labelNames[0]);

    // Add the rest of the labels
    for (let i = 1; i < labelNames.length; i++) {
        await page.getByRole('button', { name: /add next object/i }).click();

        await page.getByLabel('Label input for Object').fill(labelNames[i]);
    }
};

test.skip('Project creation', async ({ page }) => {
    await test.step('Navigate to projects page', async () => {
        await page.goto('/projects');
    });

    await test.step('Create new project', async () => {
        await page.getByText('Create project').click();

        await fillProjectForm({
            page,
            name: 'New Project',
            task: 'instance_segmentation',
            labelNames: ['Person', 'Animal'],
        });

        await page.getByRole('button', { name: /Create project/ }).scrollIntoViewIfNeeded();
        await page.getByRole('button', { name: /Create project/ }).click();

        await page.waitForURL(/inference/);
    });

    await test.step('Verify project appears in project list', async () => {
        await page.getByText('Geti Tune').click(); // Go back to /projects
        await expect(page.getByText('New Project', { exact: true })).toBeVisible();
    });

    await test.step('Delete created project', async () => {
        await page.getByRole('button', { name: /open project options/i }).click();
        await page.getByText(/Delete/).click();
        await expect(page.getByText('Project deleted successfully')).toBeVisible();
        await expect(page.getByText('New Project')).toBeHidden();
    });
});
