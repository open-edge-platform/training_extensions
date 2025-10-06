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
    await page.getByLabel(task, { exact: true }).click();

    // Edit first label
    await page.getByLabel('Label input for Object').fill(labelNames[0]);

    // Add the rest of the labels
    for (let i = 1; i < labelNames.length; i++) {
        await page.getByRole('button', { name: /add next object/i }).click();

        await page.getByLabel('Label input for Object').fill(labelNames[i]);
    }
};

test('[E2E] Minimum workflow', async ({ page }) => {
    await test.step('Navigate to root page', async () => {
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

    await test.step('Setup source', async () => {
        await page.getByRole('button', { name: 'Pipeline configuration' }).click();
        await page.getByRole('button', { name: 'Webcam' }).click();

        await page.locator('input[name="name"]').fill('New Webcam');
        await page.getByLabel('webcam device id').fill('1');

        await page.getByRole('button', { name: 'Apply' }).click();
    });

    await test.step('Setup sink', async () => {
        await page.getByLabel('Dataset import tabs').getByText('Output').click();

        await page.getByRole('button', { name: 'Video file' }).click();
        await page.locator('input[name="name"]').fill('Video source');
        await page.locator('input[name="video_path"]').fill('data/media/test_video.mp4');

        await page.getByRole('button', { name: 'Apply' }).click();

        // Click outside the dialog to close it
        await page.click('body', { position: { x: 10, y: 10 } });
    });

    await test.step('Update data collection policy', async () => {
        await page.getByRole('button', { name: 'Toggle Data collection policy' }).click();
        await expect(page.getByRole('heading', { name: 'Data collection' })).toBeVisible();

        await expect(page.getByRole('switch', { name: 'Toggle auto capturing' })).not.toBeChecked();

        await page.getByRole('switch', { name: 'Toggle auto capturing' }).click();
        await expect(page.getByRole('switch', { name: 'Toggle auto capturing' })).toBeChecked();
    });

    await test.step('Start stream to begin capture', async () => {
        await page.getByLabel('Start stream').click();

        await expect(page.getByLabel('Connected')).toBeVisible({ timeout: 60000 });

        // Wait for 3 seconds
        await page.waitForTimeout(3000);
    });

    await test.step('Confirm data was collected', async () => {
        await page.getByLabel('Start stream').click();

        await expect(page.getByLabel('Connected')).toBeVisible();

        // Wait for 3 seconds and stop the stream
        await page.waitForTimeout(3000);
        await page.getByRole('button', { name: 'Stop' }).click();

        await page.getByLabel('Header navigation').getByText('Dataset').click();

        // Confirm media was uploaded
        const listbox = page.getByRole('listbox', { name: 'data-collection-grid' });
        const options = listbox.getByRole('option');

        expect(options.count).toBeGreaterThanOrEqual(40);
    });

    await test.step('Cleanup', async () => {
        await page.getByText('Geti Tune').click(); // Go back to /projects
        await expect(page.getByText('New Project')).toBeVisible();

        await page.getByRole('button', { name: /open project options/i }).click();
        await page.getByText(/Delete/).click();
        await expect(page.getByText('Project deleted successfully')).toBeVisible();
        await expect(page.getByText('New Project')).toBeHidden();
    });
});
