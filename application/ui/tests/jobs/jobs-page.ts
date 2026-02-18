// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Page } from '@playwright/test';

import { paths } from '../../src/constants/paths';

export class JobsPage {
    constructor(private page: Page) {}

    async goto(projectId: string = 'id-1') {
        await this.page.goto(paths.project.models({ projectId }));
    }

    getCurrentTrainingSection() {
        return this.page.getByRole('heading', { name: 'Current training' });
    }

    getTrainingTag() {
        return this.page.getByRole('button', { name: 'Training', exact: true });
    }

    getStatusTag() {
        return this.page.getByRole('button', { name: 'Training in progress...' });
    }

    getCancelButton() {
        return this.page.getByRole('button', { name: 'Cancel training job' });
    }

    getConfirmCancelDialog() {
        return this.page.getByRole('alertdialog', { name: 'Cancel training' });
    }

    async cancelTrainingJob() {
        await this.getCancelButton().click();
        await this.getConfirmCancelDialog().getByRole('button', { name: 'Cancel' }).click();
    }

    getArchitectureText(architecture: string) {
        return this.page.getByText(architecture);
    }

    getViewLogsButton() {
        return this.page.getByRole('button', { name: 'View logs' });
    }

    getLogsDialog() {
        return this.page.getByRole('dialog', { name: 'Training logs' });
    }

    getCloseLogsDialogButton() {
        return this.page.getByRole('button', { name: 'Close dialog' });
    }

    async openLogsDialog() {
        await this.getViewLogsButton().click();
    }

    async closeLogsDialog() {
        await this.getCloseLogsDialogButton().click();
    }
}
