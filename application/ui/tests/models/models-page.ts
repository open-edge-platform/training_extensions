// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Page } from '@playwright/test';

import { paths } from '../../src/constants/paths';

export class ModelsPage {
    constructor(private page: Page) {}

    async goto(projectId: string = 'id-1') {
        await this.page.goto(paths.project.models({ projectId }));
    }

    getGroupByPicker() {
        return this.page.getByRole('button', { name: 'Group models' });
    }

    getSortByPicker() {
        return this.page.getByRole('button', { name: 'Sort models' });
    }

    async selectGroupBy(option: 'dataset' | 'architecture') {
        await this.getGroupByPicker().click();
        await this.page.getByRole('option', { name: option }).click();
    }

    async selectSortBy(option: 'name' | 'trained' | 'architecture' | 'size' | 'score') {
        await this.getSortByPicker().click();
        await this.page.getByRole('option', { name: option }).click();
    }

    async togglePinActiveModel() {
        await this.page.getByRole('button', { name: 'Pin active model on top' }).click();
    }

    getSearchInput() {
        return this.page.getByLabel('Search models');
    }

    async expandSearch() {
        await this.page.getByRole('button', { name: 'Search models' }).click();
    }

    async searchModels(query: string) {
        await this.expandSearch();
        await this.getSearchInput().fill(query);
    }

    getModelRows() {
        return this.page.locator('[data-testid^="model-disclosure-"]');
    }

    getModelByName(name: string) {
        return this.page.getByTestId('model-name').filter({ hasText: name });
    }

    async expandModel(name: string) {
        await this.getModelByName(name).click();
    }

    async openModelMenu() {
        await this.page.getByLabel('Model actions').first().click();
    }

    async clickRenameAction() {
        await this.page.getByRole('menuitem', { name: 'Rename' }).click();
    }

    async clickDeleteAction() {
        await this.page.getByRole('menuitem', { name: 'Delete' }).click();
    }

    async renameModel(newName: string) {
        const textbox = this.page.getByRole('textbox', { name: 'Model name' });

        await textbox.fill(newName);
        await textbox.press('Enter');
    }

    async confirmDelete() {
        await this.page.getByRole('button', { name: 'Delete' }).click();
    }

    async getModelNamesInOrder() {
        const rows = this.getModelRows();
        const names = await rows.locator('[class*="modelName"]').allTextContents();

        return names;
    }
}
