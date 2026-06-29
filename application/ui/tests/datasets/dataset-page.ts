// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { type Page } from '@playwright/test';

export const pluralizeItems = (count: number) => {
    const pluralRules = new Intl.PluralRules('en');

    return pluralRules.select(count) === 'one' ? 'item' : 'items';
};

export class DatasetPage {
    constructor(private readonly page: Page) {}

    goto(projectId = 'id-1') {
        return this.page.goto(`projects/${projectId}/dataset`);
    }

    getMediaGrid() {
        return this.page.getByRole('listbox', { name: 'data-collection-grid' });
    }

    async selectMediaItem(mediaId: string) {
        await this.getMediaGrid()
            .getByRole('checkbox', {
                name: `Select media item ${mediaId}`,
                exact: true,
            })
            .click();
    }

    getMediaItemByName(name: string) {
        return this.page.getByRole('img', { name, exact: true });
    }

    dblClickMediaItem(name: string) {
        return this.getMediaItemByName(name).dblclick();
    }

    getSelectAllCheckbox() {
        return this.page.getByLabel('select all');
    }

    selectAll() {
        return this.getSelectAllCheckbox().click();
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

    uploadFiles(files: { name: string; mimeType: string; buffer: Buffer }[]) {
        return this.getUploadInput().setInputFiles(files);
    }

    getUploadProgressText(total: number) {
        return this.page.getByText(`Uploading ${total} ${pluralizeItems(total)}...`);
    }

    getUploadProgressDetailText(succeeded: number, failed = 0) {
        const parts = [succeeded > 0 ? `${succeeded} succeeded` : null, failed > 0 ? `${failed} failed` : null]
            .filter(Boolean)
            .join(', ');
        return this.page.getByText(`(${parts})`);
    }

    getUploadFinishedText(total: number) {
        return this.page.getByText(`Uploaded ${total} ${pluralizeItems(total)}`);
    }

    getShowDetailsButton() {
        return this.page.getByRole('button', { name: 'Show details' });
    }

    clickShowDetails() {
        // Sonner stacks/re-renders toasts during upload progress updates, which can briefly cause
        // the toast container to intercept clicks on the button. Forcing the click bypasses that race.
        // eslint-disable-next-line playwright/no-force-option
        return this.getShowDetailsButton().click({ force: true });
    }

    getUploadDetailsDialog() {
        return this.page
            .getByRole('dialog')
            .filter({ has: this.page.getByRole('heading', { name: 'Upload details' }) });
    }

    getUploadDetailsRows() {
        return this.getUploadDetailsDialog().getByRole('row');
    }

    closeUploadDetailsDialog() {
        return this.getUploadDetailsDialog().getByRole('button', { name: 'Close' }).click();
    }

    getAssignLabelButton() {
        return this.page.getByRole('button', { name: 'Assign label' });
    }

    clickAssignLabel() {
        return this.getAssignLabelButton().click();
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

    selectLabel(labelName: string) {
        return this.getLabelCheckbox(labelName).click();
    }

    getContinueButton() {
        return this.page.getByRole('button', { name: 'Continue' });
    }

    clickContinue() {
        return this.getContinueButton().click();
    }

    getSkipButton() {
        return this.page.getByRole('button', { name: 'Skip' });
    }

    clickSkip() {
        return this.getSkipButton().click();
    }

    getBulkDialogAssignButton() {
        return this.getLabelAssignmentDialog().getByRole('button', { name: 'Assign' });
    }

    clickDialogAssign() {
        return this.getBulkDialogAssignButton().click();
    }
}
