// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { getMockedJob } from 'mocks/mock-job';
import { getMockedLogEntryJson } from 'mocks/mock-log-entry';
import { getMockedModel } from 'mocks/mock-model';
import { HttpResponse } from 'msw';

import { expect, http, test } from '../fixtures';

const mockedJob = getMockedJob({
    job_id: 'job-logs-test',
    job_type: 'train',
    status: 'RUNNING',
    progress: 50,
    message: 'Running: Train Model',
    metadata: {
        project: { id: 'id-1' },
        model: {
            id: 'model-logs-test',
            architecture: 'Custom_Object_Detection_Gen3_ATSS',
            parent_revision_id: null,
            dataset_revision_id: 'dataset-1',
        },
    },
    started_at: '2026-02-06T10:00:00.000000+00:00',
    finished_at: null,
});

const mockedSuccessfulModel = getMockedModel({
    id: 'model-logs-test',
    name: 'ATSS Model v1',
    architecture: 'Custom_Object_Detection_Gen3_ATSS',
    training_info: {
        status: 'successful',
        label_schema_revision: { labels: [] },
        start_time: '2026-02-06T10:00:00.000000+00:00',
        end_time: '2026-02-06T12:00:00.000000+00:00',
        dataset_revision_id: 'dataset-1',
    },
});

test.describe('TrainingLogsDialog - Active Job (SSE streaming)', () => {
    test.beforeEach(({ network }) => {
        network.use(
            http.get('/api/jobs', () => {
                return HttpResponse.json([mockedJob]);
            }),
            http.get('/api/projects/{project_id}/models/{model_id}', () => {
                return HttpResponse.json(mockedSuccessfulModel);
            }),
            http.get('/api/jobs/{job_id}/logs', () => {
                return new HttpResponse(':ok\n\n', {
                    status: 200,
                    headers: {
                        'Content-Type': 'text/event-stream',
                        'Cache-Control': 'no-cache',
                    },
                });
            })
        );
    });

    test('renders the dialog with heading and toolbar, then closes on button click', async ({ jobsPage }) => {
        await jobsPage.goto();

        await jobsPage.openLogsDialog();
        const dialog = jobsPage.getLogsDialog();

        await expect(dialog).toBeVisible();
        await expect(dialog.getByRole('heading', { name: 'Training Logs' })).toBeVisible();
        await expect(dialog.getByRole('button', { name: /minimum log level/i })).toBeVisible();
        await expect(dialog.getByLabel('Search logs')).toBeVisible();
        await expect(dialog.getByRole('button', { name: 'Scroll to bottom' })).toBeHidden();

        await jobsPage.closeLogsDialog();
        await expect(dialog).toBeHidden();
    });

    test('renders log entries received via SSE stream', async ({ jobsPage, network }) => {
        const logMessage1 = 'Preparing model weights';
        const logMessage2 = 'Starting training loop';

        network.use(
            http.get('/api/jobs/{job_id}/logs', () => {
                const sseBody =
                    `data: ${getMockedLogEntryJson({ message: logMessage1 })}\n\n` +
                    `data: ${getMockedLogEntryJson({ message: logMessage2 })}\n\n`;

                return new HttpResponse(sseBody, {
                    status: 200,
                    headers: {
                        'Content-Type': 'text/event-stream',
                        'Cache-Control': 'no-cache',
                    },
                });
            })
        );

        await jobsPage.goto();
        await jobsPage.openLogsDialog();

        const dialog = jobsPage.getLogsDialog();
        await expect(dialog.getByText(logMessage1)).toBeVisible();
        await expect(dialog.getByText(logMessage2)).toBeVisible();
    });
});

test.describe('TrainingLogsDialog - Historical Model Logs (REST fetch)', () => {
    test.beforeEach(({ network }) => {
        network.use(
            http.get('/api/jobs', () => {
                return HttpResponse.json([]);
            }),
            http.get('/api/projects/{project_id}/models', () => {
                return HttpResponse.json([mockedSuccessfulModel]);
            }),
            http.get('/api/projects/{project_id}/models/{model_id}', () => {
                return HttpResponse.json(mockedSuccessfulModel);
            })
        );
    });

    test('renders historical model logs after loading', async ({ modelsPage, network }) => {
        const logMessage = 'Training completed successfully';

        network.use(
            http.get('/api/projects/{project_id}/models/{model_id}/logs', () => {
                return new HttpResponse(getMockedLogEntryJson({ message: logMessage }), {
                    status: 200,
                    headers: { 'Content-Type': 'text/plain' },
                });
            })
        );

        await modelsPage.goto();
        await modelsPage.openModelMenu();
        await modelsPage.clickViewTrainingLogsAction();

        const dialog = modelsPage.getLogsDialog();
        await expect(dialog).toBeVisible();
        await expect(dialog.getByText(logMessage)).toBeVisible();
    });

    test('renders toolbar without auto-scroll toggle for historical logs and dismisses on close', async ({
        modelsPage,
        network,
    }) => {
        network.use(
            http.get('/api/projects/{project_id}/models/{model_id}/logs', () => {
                return new HttpResponse('', {
                    status: 200,
                    headers: { 'Content-Type': 'text/plain' },
                });
            })
        );

        await modelsPage.goto();
        await modelsPage.openModelMenu();
        await modelsPage.clickViewTrainingLogsAction();

        const dialog = modelsPage.getLogsDialog();
        await expect(dialog).toBeVisible();
        await expect(dialog.getByRole('switch', { name: 'Auto-scroll' })).toBeHidden();
        await expect(dialog.getByRole('button', { name: /minimum log level/i })).toBeVisible();
        await expect(dialog.getByLabel('Search logs')).toBeVisible();

        await modelsPage.closeLogsDialog();
        await expect(dialog).toBeHidden();
    });

    test('shows error message when model logs endpoint fails', async ({ modelsPage, network }) => {
        network.use(
            http.get('/api/projects/{project_id}/models/{model_id}/logs', () => {
                return new HttpResponse(null, { status: 500, statusText: 'Internal Server Error' });
            })
        );

        await modelsPage.goto();
        await modelsPage.openModelMenu();
        await modelsPage.clickViewTrainingLogsAction();

        const dialog = modelsPage.getLogsDialog();
        await expect(dialog).toBeVisible();
        await expect(dialog.getByText(/Failed to load logs/)).toBeVisible();
    });

    test('shows loading state before log data is returned', async ({ modelsPage, network }) => {
        let resolveResponse: ((value: Response) => void) | undefined;
        const pendingResponse = new Promise<Response>((resolve) => {
            resolveResponse = resolve;
        });

        network.use(
            http.get('/api/projects/{project_id}/models/{model_id}/logs', async () => {
                await pendingResponse;

                return new HttpResponse('', {
                    status: 200,
                    headers: { 'Content-Type': 'text/plain' },
                });
            })
        );

        await modelsPage.goto();
        await modelsPage.openModelMenu();
        await modelsPage.clickViewTrainingLogsAction();

        const dialog = modelsPage.getLogsDialog();
        await expect(dialog).toBeVisible();

        await expect(dialog.getByRole('progressbar')).toBeVisible();

        resolveResponse?.(new Response());
        await expect(dialog.getByRole('progressbar')).toBeHidden();
    });

    test('filters log entries by level inside the dialog', async ({ modelsPage, network, page }) => {
        const debugLine = getMockedLogEntryJson({
            message: 'Debug step',
            level: { icon: '🐛', name: 'DEBUG', no: 10 },
        });
        const infoLine = getMockedLogEntryJson({ message: 'Info step', level: { icon: 'ℹ️', name: 'INFO', no: 20 } });
        const errorLine = getMockedLogEntryJson({
            message: 'Error occurred',
            level: { icon: '❌', name: 'ERROR', no: 40 },
        });

        network.use(
            http.get('/api/projects/{project_id}/models/{model_id}/logs', () => {
                return new HttpResponse([debugLine, infoLine, errorLine].join('\n'), {
                    status: 200,
                    headers: { 'Content-Type': 'text/plain' },
                });
            })
        );

        await modelsPage.goto();
        await modelsPage.openModelMenu();
        await modelsPage.clickViewTrainingLogsAction();

        const dialog = modelsPage.getLogsDialog();
        await expect(dialog).toBeVisible();

        // Default level is INFO: DEBUG should be filtered out
        await expect(dialog.getByText('Info step')).toBeVisible();
        await expect(dialog.getByText('Error occurred')).toBeVisible();
        await expect(dialog.getByText('Debug step')).toBeHidden();

        await dialog.getByRole('button', { name: /minimum log level/i }).click();
        await page.getByRole('option', { name: 'ERROR' }).click();

        await expect(dialog.getByText('Error occurred')).toBeVisible();
        await expect(dialog.getByText('Info step')).toBeHidden();
        await expect(dialog.getByText('Debug step')).toBeHidden();
    });

    test('searches log entries by message text inside the dialog', async ({ modelsPage, network }) => {
        const lines = [
            getMockedLogEntryJson({ message: 'Preparing model weights' }),
            getMockedLogEntryJson({ message: 'Starting training loop' }),
            getMockedLogEntryJson({ message: 'Epoch 1/30 complete' }),
        ].join('\n');

        network.use(
            http.get('/api/projects/{project_id}/models/{model_id}/logs', () => {
                return new HttpResponse(lines, {
                    status: 200,
                    headers: { 'Content-Type': 'text/plain' },
                });
            })
        );

        await modelsPage.goto();
        await modelsPage.openModelMenu();
        await modelsPage.clickViewTrainingLogsAction();

        const dialog = modelsPage.getLogsDialog();
        await expect(dialog).toBeVisible();

        await dialog.getByLabel('Search logs').fill('Epoch');

        await expect(dialog.getByText('Epoch 1/30 complete')).toBeVisible();
        await expect(dialog.getByText('Preparing model weights')).toBeHidden();
        await expect(dialog.getByText('Starting training loop')).toBeHidden();
    });
});
