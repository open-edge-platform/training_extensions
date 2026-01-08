// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { expect, type Page } from '@playwright/test';

export class AnnotatorPage {
    constructor(private readonly page: Page) {}

    getMediaItemImage() {
        return this.page.getByLabel('media item image');
    }

    getProcessingImage() {
        return this.page.getByText('Processing image, please wait...');
    }

    async annotateAt(x: number, y: number) {
        const image = this.getMediaItemImage();
        const box = await image.boundingBox();

        if (box) {
            const hoverX = x;
            const hoverY = y;

            // Hover to trigger preview
            await this.page.mouse.move(hoverX, hoverY);

            // Wait for preview to appear
            await expect(this.page.getByLabel('Segment anything preview')).toBeVisible({ timeout: 10000 });

            await this.page.mouse.click(hoverX, hoverY);
        }
    }

    async addAnnotation() {
        const image = this.getMediaItemImage();
        const box = await image.boundingBox();

        if (box) {
            // Position: middle horizontally, 20% from the bottom vertically
            const hoverX = box.x + box.width / 2;
            const hoverY = box.y + box.height * 0.8;

            await this.annotateAt(hoverX, hoverY);
        }
    }

    getAnnotation() {
        return this.page.getByLabel('annotation list').getByLabel('annotation polygon');
    }

    async hideAnnotations() {
        await this.page.getByRole('button', { name: 'Hide annotations' }).click();
    }

    async showAnnotations() {
        await this.page.getByRole('button', { name: 'Show annotations' }).click();
    }

    async undoAnnotation() {
        await this.page.getByRole('button', { name: 'undo' }).click();
    }

    async redoAnnotation() {
        await this.page.getByRole('button', { name: 'redo' }).click();
    }

    async openSettings() {
        await this.page.getByRole('button', { name: 'Settings' }).click();
    }

    async closeSettings() {
        await this.page.getByRole('button', { name: 'Close settings' }).click();
    }

    async zoomIn() {
        await this.page.getByRole('button', { name: 'Zoom in' }).click();
    }

    async zoomOut() {
        await this.page.getByRole('button', { name: 'Zoom out' }).click();
    }

    async fitToScreen() {
        await this.page.getByRole('button', { name: 'Fit image to screen' }).click();
    }

    async getZoomValue() {
        return this.page.getByTestId('zoom-level');
    }
}
