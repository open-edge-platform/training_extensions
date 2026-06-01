// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { expect, Page } from '@playwright/test';

import { BoundingBoxToolPage } from '../annotator/bounding-box-tool-page';
import { AnnotatorPage } from '../datasets/annotator-page';
import { DatasetPage } from '../datasets/dataset-page';
import { StreamPage } from '../inference/stream-page';
import { ModelsPage } from '../models/models-page';
import { ProjectPage } from '../projects/project-page';
import { CreatedProject, CreateProjectInput, FlowInput, InferenceSourceSinkConfig, UploadMediaItem } from './types';

const filename = fileURLToPath(import.meta.url);
const dirname = path.dirname(filename);
const assetsDirectory = path.resolve(dirname, '../assets');

const DEFAULT_TASK: CreateProjectInput['task'] = 'detection';
const DEFAULT_LABELS = ['Object'];
const DEFAULT_INFERENCE_CONFIG: InferenceSourceSinkConfig = {
    sourceName: 'E2E Source',
    sinkName: 'E2E Sink',
    sinkFolderPath: 'e2e-output',
    rateLimitSamples: 5,
    rateLimitSeconds: 1,
};

const TIMEOUT = {
    mediaVisible: 60000,
    architectureVisible: 30000,
    trainingStarted: 120000,
    connected: 120000,
    capturedMedia: 180000,
} as const;

const getProjectIdFromUrl = (url: string): string => {
    const match = url.match(/\/projects\/([^/]+)\//);

    if (match === null || match[1] === undefined) {
        throw new Error(`Could not extract project id from URL: ${url}`);
    }

    return match[1];
};

const openPipelineTab = async (page: Page, tabName: 'Input' | 'Output'): Promise<void> => {
    await page.getByLabel('Pipeline configuration tabs').getByText(tabName).click();
};

const selectPickerOptionIfVisible = async (page: Page, label: string): Promise<void> => {
    const picker = page.getByLabel(label, { exact: true }).last();

    if (!(await picker.isVisible())) {
        return;
    }

    await picker.click();

    const option = page.getByRole('option').first();
    await expect(option).toBeVisible();
    await option.click();
};

export const stepCreateProject = async (page: Page, input: CreateProjectInput): Promise<CreatedProject> => {
    const projectPage = new ProjectPage(page);
    const projectName = input.projectName ?? `${input.projectNamePrefix ?? 'project'}-${input.task}`;

    await projectPage.gotoCreate();
    await projectPage.fillProjectForm({
        name: projectName,
        task: input.task,
        labelNames: input.labels,
    });

    await projectPage.getCreateProjectButton().scrollIntoViewIfNeeded();
    await projectPage.getCreateProjectButton().click();

    await page.waitForURL(/\/projects\/[^/]+\/dataset/);

    return {
        id: getProjectIdFromUrl(page.url()),
        name: projectName,
    };
};

export const getDefaultUploadMedia = (): UploadMediaItem[] => [
    {
        name: 'candy.png',
        mimeType: 'image/png',
        buffer: fs.readFileSync(path.resolve(assetsDirectory, 'candy.png')),
    },
];

export const stepUploadMedia = async (
    datasetPage: DatasetPage,
    media: UploadMediaItem[] = getDefaultUploadMedia()
): Promise<void> => {
    await datasetPage.uploadFiles(media);

    const mediaGrid = datasetPage.getMediaGrid();
    await expect(mediaGrid).toBeVisible();
    await expect(mediaGrid.getByRole('option').first()).toBeVisible({ timeout: TIMEOUT.mediaVisible });
};

export const stepOpenFirstItemInAnnotator = async (page: Page, datasetPage: DatasetPage): Promise<void> => {
    const firstItem = datasetPage.getMediaGrid().getByRole('option').first();

    await expect(firstItem).toBeVisible({ timeout: TIMEOUT.mediaVisible });
    await firstItem.dblclick();

    await expect(page).toHaveURL(/\/projects\/[^/]+\/dataset\/items\//);
};

export const stepAnnotateSingleBox = async (
    page: Page,
    annotatorPage: AnnotatorPage,
    boundingBoxTool: BoundingBoxToolPage
): Promise<void> => {
    await expect(annotatorPage.getMediaItemImage()).toBeVisible({ timeout: TIMEOUT.mediaVisible });

    await boundingBoxTool.selectTool();
    await boundingBoxTool.drawBoundingBox({ x: 150, y: 150, width: 220, height: 180 });

    await expect(page.getByLabel('annotation rect').first()).toBeVisible();

    await annotatorPage.submit();
};

export const stepTrainModel = async (page: Page, modelsPage: ModelsPage, projectId: string): Promise<void> => {
    await modelsPage.goto(projectId);
    await modelsPage.openTrainModelDialog();

    const firstArchitecture = page.getByRole('radio').first();
    await expect(firstArchitecture).toBeVisible({ timeout: TIMEOUT.architectureVisible });
    await firstArchitecture.click();

    await selectPickerOptionIfVisible(page, 'Select dataset');
    await selectPickerOptionIfVisible(page, 'Select model revision');

    await modelsPage.startTraining();

    await expect(async () => {
        const hasRunningHeading = await page.getByRole('heading', { name: 'Currently running' }).isVisible();
        const hasTrainingMessage = await page
            .getByText(/Training in progress|running/i)
            .first()
            .isVisible();

        expect(hasRunningHeading || hasTrainingMessage).toBe(true);
    }).toPass({ timeout: TIMEOUT.trainingStarted });
};

export const cleanupProject = async (page: Page, projectId: string): Promise<void> => {
    if (process.env.E2E_KEEP_RESOURCES === 'true') {
        return;
    }

    const response = await page.request.delete(`/api/projects/${projectId}`);

    if (![200, 202, 204, 404].includes(response.status())) {
        throw new Error(`Failed to cleanup project ${projectId}. Status: ${response.status()}`);
    }
};

export const stepConfigureInferenceSourceAndSink = async (
    page: Page,
    config: InferenceSourceSinkConfig = DEFAULT_INFERENCE_CONFIG
): Promise<void> => {
    await expect(page.getByRole('button', { name: 'Add new source' })).toBeVisible({ timeout: TIMEOUT.mediaVisible });

    await page.getByRole('button', { name: 'Add new source' }).click();
    await page.getByRole('button', { name: 'USB Camera' }).click();

    await page.getByRole('textbox', { name: 'Name' }).fill(config.sourceName);
    await page.getByRole('button', { name: 'Camera list' }).click();
    await page.getByRole('option').first().click();
    await page.getByRole('button', { name: 'Add & Connect' }).click();

    await openPipelineTab(page, 'Output');
    await page.getByRole('button', { name: 'Add new sink' }).click();
    await page.getByRole('button', { name: 'Folder' }).click();

    await page.locator('input[name="name"]').fill(config.sinkName);
    await page.locator('input[aria-roledescription="Number field"]').first().fill(String(config.rateLimitSamples));
    await page.locator('input[aria-roledescription="Number field"]').nth(1).fill(String(config.rateLimitSeconds));
    await page.locator('input[name="folder_path"]').fill(config.sinkFolderPath);
    await page.locator('input[name="output_formats"][value="predictions"]').check();
    await page.getByRole('button', { name: 'Add & Connect' }).click();
};

export const stepStartStreamWithAutoCapture = async (page: Page, streamPage: StreamPage): Promise<void> => {
    const pipelineSwitch = page.getByRole('switch', { name: /Enable pipeline|Disable Pipeline/i });

    await expect(pipelineSwitch).toBeVisible({ timeout: TIMEOUT.mediaVisible });

    if ((await pipelineSwitch.getAttribute('aria-label'))?.toLowerCase().includes('enable') === true) {
        await pipelineSwitch.click();
    }

    await streamPage.startStream();
    await expect(page.getByLabel('Connected')).toBeVisible({ timeout: TIMEOUT.connected });

    await page.getByRole('button', { name: 'Toggle Data collection policy' }).click();

    const autoCaptureSwitch = page.getByRole('switch', { name: 'Toggle auto capturing' });
    await expect(autoCaptureSwitch).toBeVisible();

    if ((await autoCaptureSwitch.isChecked()) === false) {
        await autoCaptureSwitch.click();
    }

    await expect(autoCaptureSwitch).toBeChecked();
};

export const stepWaitForCapturedMediaAndOpenAnnotator = async (page: Page, datasetPage: DatasetPage): Promise<void> => {
    await page.getByRole('tab', { name: 'Dataset' }).click();

    const mediaOptions = datasetPage.getMediaGrid().getByRole('option');
    await expect(async () => {
        expect(await mediaOptions.count()).toBeGreaterThan(0);
    }).toPass({ timeout: TIMEOUT.capturedMedia });

    await mediaOptions.first().dblclick();
    await expect(page).toHaveURL(/\/projects\/[^/]+\/dataset\/items\//);
};

// Flow 1: create project -> add media -> annotate -> train
export const runFlowCreateAnnotateTrain = async ({ page, projectNamePrefix }: FlowInput): Promise<CreatedProject> => {
    const datasetPage = new DatasetPage(page);
    const annotatorPage = new AnnotatorPage(page);
    const boundingBoxTool = new BoundingBoxToolPage(page);
    const modelsPage = new ModelsPage(page);

    const project = await stepCreateProject(page, {
        projectNamePrefix,
        task: DEFAULT_TASK,
        labels: DEFAULT_LABELS,
    });

    await stepUploadMedia(datasetPage);
    await stepOpenFirstItemInAnnotator(page, datasetPage);
    await stepAnnotateSingleBox(page, annotatorPage, boundingBoxTool);
    await stepTrainModel(page, modelsPage, project.id);

    return project;
};

// Flow 2: create project -> configure inference -> stream auto-capture -> annotate -> train
export const runFlowStreamAutoCaptureAnnotateTrain = async ({
    page,
    projectNamePrefix,
}: FlowInput): Promise<CreatedProject> => {
    const datasetPage = new DatasetPage(page);
    const annotatorPage = new AnnotatorPage(page);
    const boundingBoxTool = new BoundingBoxToolPage(page);
    const modelsPage = new ModelsPage(page);
    const streamPage = new StreamPage(page);

    const project = await stepCreateProject(page, {
        projectNamePrefix,
        task: DEFAULT_TASK,
        labels: DEFAULT_LABELS,
    });

    await page.goto(`/projects/${project.id}/inference`);

    await stepConfigureInferenceSourceAndSink(page);
    await stepStartStreamWithAutoCapture(page, streamPage);
    await stepWaitForCapturedMediaAndOpenAnnotator(page, datasetPage);
    await stepAnnotateSingleBox(page, annotatorPage, boundingBoxTool);
    await stepTrainModel(page, modelsPage, project.id);

    return project;
};
