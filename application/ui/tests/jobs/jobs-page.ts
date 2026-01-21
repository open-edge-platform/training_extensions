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

    async cancelTrainingJob() {
        await this.getCancelButton().click();
    }

    getArchitectureText(architecture: string) {
        return this.page.getByText(architecture);
    }
}
