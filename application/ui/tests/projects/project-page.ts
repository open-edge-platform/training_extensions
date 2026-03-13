// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Page } from '@playwright/test';

import { paths } from '../../src/constants/paths';

type ProjectFormOptions = {
    name: string;
    task: string;
    classificationType?: 'Single-label' | 'Multi-label';
    labelNames: string[];
};

export class ProjectPage {
    constructor(private page: Page) {}

    async gotoList() {
        await this.page.goto(paths.project.index({}));
    }

    async gotoCreate() {
        await this.page.goto(paths.project.new({}));
    }

    getCreateProjectButton() {
        return this.page.getByRole('button', { name: /Create project/ });
    }

    getMultiLabelValidationMessage() {
        return this.page.getByText('At least 2 labels are required for multi-label classification');
    }

    async setProjectName(name: string) {
        await this.page.getByRole('textbox', { name: 'Project name input' }).fill(name);
    }

    async selectTask(task: string) {
        await this.page.getByLabel(task, { exact: true }).click();
    }

    async selectClassificationType(type: 'Single-label' | 'Multi-label') {
        await this.page.getByRole('radio', { name: type }).click();
    }

    async addLabel(labelName: string) {
        await this.page.getByRole('textbox', { name: 'Create label input' }).fill(labelName);
        await this.page.getByRole('button', { name: /Create label/ }).click();
    }

    async fillProjectForm({ name, task, classificationType, labelNames }: ProjectFormOptions) {
        await this.setProjectName(name);
        await this.selectTask(task);

        if (classificationType !== undefined) {
            await this.selectClassificationType(classificationType);
        }

        for (const labelName of labelNames) {
            await this.addLabel(labelName);
        }
    }

    async openProjectMenu(projectId: string) {
        await this.page.getByTestId(projectId).click();
    }

    async clickDeleteMenuAction() {
        await this.page.getByText(/Delete/).click();
    }

    async confirmDeleteProject() {
        await this.page.getByRole('button', { name: /Delete/ }).click();
    }
}
