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

    async selectPickerOption(label: string, optionName: string) {
        await this.page.getByLabel(label, { exact: true }).last().click();
        await this.page.getByRole('option', { name: optionName, exact: true }).click();
    }

    async openTrainModelDialog() {
        await this.page.getByRole('button', { name: 'Train model' }).click();
    }

    async selectModelArchitecture(architectureName: string) {
        await this.page.getByRole('radio', { name: architectureName, exact: true }).click();
    }

    async startTraining() {
        await this.page.getByRole('button', { name: 'Start' }).click();
    }

    async openModelListingOptionsMenu() {
        await this.page.getByRole('button', { name: 'Model listing options' }).click();
    }

    async togglePinActiveModel() {
        await this.openModelListingOptionsMenu();
        await this.page.getByRole('menuitem', { name: /Pin active model on top|Unpin active model from top/ }).click();
    }

    async toggleShowHideFailedModels() {
        await this.openModelListingOptionsMenu();
        await this.page.getByRole('menuitem', { name: /Show failed models|Hide failed models/ }).click();
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

    async clickSetActiveAction() {
        await this.page.getByRole('menuitem', { name: 'Set as active' }).click();
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

    async openDatasetMenu() {
        await this.page.getByLabel('Dataset actions').first().click();
    }

    async clickRenameDatasetAction() {
        await this.page.getByRole('menuitem', { name: 'Rename' }).click();
    }

    async clickDeleteDatasetAction() {
        await this.page.getByRole('menuitem', { name: 'Delete' }).click();
    }

    async renameDatasetRevision(newName: string) {
        const textbox = this.page.getByRole('textbox', { name: 'Dataset revision name' });

        await textbox.fill(newName);
        await textbox.press('Enter');
    }

    getDatasetHeaderByName(name: string) {
        return this.page.getByRole('heading').filter({ hasText: name });
    }

    getThreeSectionRange(datasetId: string) {
        return this.page.getByTestId(`dataset-range-${datasetId}`);
    }

    async clickTrainingDatasetsTab() {
        await this.page.getByRole('tab', { name: 'Training datasets' }).click();
    }
}
