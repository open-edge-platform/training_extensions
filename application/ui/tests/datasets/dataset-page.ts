// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { type Page } from '@playwright/test';

export class DatasetPage {
    constructor(private readonly page: Page) {}

    async goto(projectId = 'id-1') {
        await this.page.goto(`projects/${projectId}/dataset`);
    }

    getMediaGrid() {
        return this.page.getByRole('listbox', { name: 'data-collection-grid' });
    }

    getMediaGridOptions() {
        return this.getMediaGrid().getByRole('option');
    }

    getMediaItemByName(name: string) {
        return this.page.getByRole('img', { name, exact: true });
    }

    async clickMediaItem(name: string) {
        await this.getMediaItemByName(name).click();
    }

    async dblClickMediaItem(name: string) {
        await this.getMediaItemByName(name).dblclick();
    }

    getSelectAllCheckbox() {
        return this.page.getByLabel('select all');
    }

    async selectAll() {
        await this.getSelectAllCheckbox().click();
    }

    getSelectedCountText(count: number) {
        return this.page.getByText(`${count} selected`);
    }

    getImagesCountText(count: number) {
        return this.page.getByText(`${count} media item`);
    }

    getUploadInput() {
        return this.page.getByLabel('Upload media files');
    }

    getUploadButton() {
        return this.page.getByRole('button', { name: 'Upload media' });
    }

    async uploadFiles(files: { name: string; mimeType: string; buffer: Buffer }[]) {
        await this.getUploadInput().setInputFiles(files);
    }

    getUploadProgressText(total: number, succeeded: number, failed = 0) {
        const parts = [succeeded > 0 ? `${succeeded} succeeded` : null, failed > 0 ? `${failed} failed` : null]
            .filter(Boolean)
            .join(', ');
        const text = `Uploading ${total} item(s)...${parts ? ` (${parts})` : ''}`;

        return this.page.getByText(text);
    }

    getUploadFinishedText(total: number) {
        return this.page.getByText(`Uploaded ${total} item(s)`);
    }

    getAssignLabelButton() {
        return this.page.getByRole('button', { name: 'Assign label' });
    }

    async clickAssignLabel() {
        await this.getAssignLabelButton().click();
    }

    getLabelAssignmentDialog() {
        return this.page.getByRole('dialog');
    }

    getLabelAssignmentHeading() {
        return this.page.getByRole('heading', { name: 'Label assignment' });
    }

    getLabelCheckbox(labelName: string) {
        return this.page.getByRole('checkbox', { name: `Select ${labelName}` });
    }

    async selectLabel(labelName: string) {
        await this.getLabelCheckbox(labelName).click();
    }

    getContinueButton() {
        return this.page.getByRole('button', { name: 'Continue' });
    }

    async clickContinue() {
        await this.getContinueButton().click();
    }

    getBulkDialogAssignButton() {
        return this.getLabelAssignmentDialog().getByRole('button', { name: 'Assign' });
    }

    async clickDialogAssign() {
        await this.getBulkDialogAssignButton().click();
    }
}
