// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Page } from '@playwright/test';

import { paths } from '../../src/constants/paths';

export class JobsPage {
    constructor(private page: Page) {}

    async goto(projectId: string = 'id-1') {
        await this.page.goto(paths.project.models({ projectId }));
    }

    getCurrentRunningSection() {
        return this.page.getByRole('heading', { name: 'Currently running' });
    }

    getRunningTag() {
        return this.page.getByRole('button', { name: 'Running', exact: true });
    }

    getStatusTag() {
        return this.page.getByRole('button', { name: 'Training in progress...' });
    }

    getCancelButton() {
        return this.page.getByRole('button', { name: 'Cancel job' });
    }

    getConfirmCancelDialog() {
        return this.page.getByRole('alertdialog', { name: 'Stop job' });
    }

    async cancelRunningJob() {
        await this.getCancelButton().click();
        await this.getConfirmCancelDialog().getByRole('button', { name: 'Cancel' }).click();
    }

    getArchitectureText(architecture: string) {
        return this.page.getByText(architecture);
    }
}
