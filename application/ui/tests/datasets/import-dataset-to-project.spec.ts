// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { getMockedJob } from 'mocks/mock-job';
import { getMockedLabel } from 'mocks/mock-labels';
import { getMockedProject } from 'mocks/mock-project';
import { getMockedStagedDataset } from 'mocks/mock-staged-dataset';
import { HttpResponse } from 'msw';

import { expect, http, test } from '../fixtures';
import {
    DATASET_FILENAME,
    deleteStagedDatasetHandler,
    getMockedImportJob,
    getMockedPrepareJob,
    IMPORT_JOB_ID,
    jobPollHandler,
    PREPARE_JOB_ID,
    STAGED_DATASET_ID,
    stagedDatasetWithMetadata,
} from './utils';

const mockedProject = getMockedProject({
    task: {
        task_type: 'detection',
        exclusive_labels: true,
        labels: [getMockedLabel({ name: 'cat' }), getMockedLabel({ name: 'dog' })],
    },
});

test.describe('Import dataset to project', () => {
    test.beforeEach(({ network }) => {
        network.use(
            http.get('/api/projects/{project_id}', () => {
                return HttpResponse.json(mockedProject, { status: 200 });
            }),
            http.get('/api/staged_datasets/{staged_dataset_id}', () => {
                return HttpResponse.json(stagedDatasetWithMetadata, { status: 200 });
            }),
            http.get('/api/projects/{project_id}/dataset/items', () => {
                return HttpResponse.json({
                    items: [],
                    pagination: { total: 0, count: 0, limit: 10, offset: 0 },
                });
            }),
            http.post('/api/staged_datasets', () => {
                return HttpResponse.json(getMockedStagedDataset({ id: STAGED_DATASET_ID }), { status: 201 });
            }),
            http.post('/api/jobs', async ({ request }) => {
                const body = await request.json();
                if (body.job_type === 'import_dataset_to_project') {
                    return HttpResponse.json(getMockedImportJob('import_dataset_to_project'), { status: 201 });
                }
                return HttpResponse.json(getMockedPrepareJob(), { status: 201 });
            })
        );
    });

    test('import dataset with default label mapping', async ({ page, network, importDatasetPage }) => {
        const importingJob = getMockedImportJob('import_dataset_to_project', {
            progress: 60,
            message: 'Importing progress...',
        });

        const preparePoll = jobPollHandler({
            jobId: PREPARE_JOB_ID,
            whileRunning: getMockedPrepareJob(),
            whenDone: getMockedPrepareJob({ status: 'DONE', progress: 100, message: 'Preparation completed' }),
        });
        const importPoll = jobPollHandler({
            jobId: IMPORT_JOB_ID,
            whileRunning: importingJob,
            whenDone: getMockedImportJob('import_dataset_to_project', {
                status: 'DONE',
                progress: 100,
                message: 'Import completed',
            }),
        });
        const { handler: deleteHandler, getDeletedId } = deleteStagedDatasetHandler();

        network.use(
            http.get('/api/jobs/{job_id}', ({ params }) => {
                const jobId = params.job_id as string;
                const job = importPoll(jobId) ?? preparePoll(jobId);
                return HttpResponse.json(job, { status: 200 });
            }),
            deleteHandler
        );

        await page.goto(`projects/${mockedProject.id}/dataset`);

        await importDatasetPage.openImportDialog();

        await test.step('Upload dataset zip file', async () => {
            await importDatasetPage.loadZipFile(DATASET_FILENAME);
        });

        await test.step('Verify preparation progress', async () => {
            await expect(importDatasetPage.getPreparingStatus()).toBeVisible();
        });

        await test.step('Submit import with default mapping', async () => {
            await expect(importDatasetPage.getStatisticsHeading()).toBeVisible();
            await expect(importDatasetPage.getLabelMappingHeading()).toBeVisible();

            for (const label of stagedDatasetWithMetadata.metadata?.labels ?? []) {
                await expect(importDatasetPage.getLabelMappingButton(label)).toBeVisible();
            }

            await expect(importDatasetPage.getIncludeUnannotatedCheckbox()).toBeChecked();
            await importDatasetPage.submit();
            await expect(importDatasetPage.getDialog()).toBeHidden();
        });

        await test.step('Verify import job progress', async () => {
            await expect(importDatasetPage.getImportStatusText(DATASET_FILENAME, 'processing')).toBeVisible();
            await expect(page.getByText(String(importingJob.message))).toBeVisible();
        });

        await test.step('Show success toast and removes the staged dataset', async () => {
            await expect(page.getByText(`Dataset ${DATASET_FILENAME} 16 B`)).toBeVisible();
            await expect.poll(() => getDeletedId()).toBe(STAGED_DATASET_ID);
        });
    });

    test('cancel import job removes staged files', async ({ page, network, importDatasetPage }) => {
        const importingJob = getMockedImportJob('import_dataset_to_project', {
            progress: 60,
            message: 'Importing progress...',
        });

        const preparePoll = jobPollHandler({
            jobId: PREPARE_JOB_ID,
            whileRunning: getMockedPrepareJob(),
            whenDone: getMockedPrepareJob({ status: 'DONE', progress: 100, message: 'Preparation completed' }),
        });

        network.use(
            http.get('/api/jobs/{job_id}', ({ params }) => {
                const jobId = params.job_id as string;
                const job = jobId === IMPORT_JOB_ID ? importingJob : preparePoll(jobId);
                return HttpResponse.json(job, { status: 200 });
            }),
            http.post('/api/jobs/{job_id}:cancel', () => {
                return HttpResponse.json(getMockedJob({ ...importingJob, status: 'CANCELLED' }), { status: 200 });
            }),
            http.delete('/api/staged_datasets/{staged_dataset_id}', () => {
                return new HttpResponse(null, { status: 204 });
            })
        );

        await page.goto(`projects/${mockedProject.id}/dataset`);

        await importDatasetPage.openImportDialog();

        await test.step('Upload and complete preparation', async () => {
            await importDatasetPage.loadZipFile(DATASET_FILENAME);
            await expect(importDatasetPage.getPreparingStatus()).toBeVisible();
            await expect(importDatasetPage.getStatisticsHeading()).toBeVisible();
        });

        await test.step('Submit import', async () => {
            await importDatasetPage.submit();
            await expect(importDatasetPage.getDialog()).toBeHidden();
        });

        await test.step('Cancel the running import job', async () => {
            await expect(importDatasetPage.getImportStatusText(DATASET_FILENAME, 'processing')).toBeVisible();
            await importDatasetPage.cancelJobFromStatusCard();

            await expect(importDatasetPage.getImportStatusText(DATASET_FILENAME, 'processing')).toBeHidden();
        });
    });

    test('cancel prepare job removes staged files', async ({ page, network, importDatasetPage }) => {
        const runningPrepareJob = getMockedPrepareJob();

        network.use(
            http.post('/api/jobs', async () => {
                return HttpResponse.json(getMockedPrepareJob(), { status: 201 });
            }),
            http.get('/api/jobs/{job_id}', () => {
                return HttpResponse.json(runningPrepareJob, { status: 200 });
            }),
            http.post('/api/jobs/{job_id}:cancel', () => {
                return HttpResponse.json(getMockedJob({ ...runningPrepareJob, status: 'CANCELLED' }), { status: 200 });
            }),
            http.delete('/api/staged_datasets/{staged_dataset_id}', () => {
                return new HttpResponse(null, { status: 204 });
            })
        );

        await page.goto(`projects/${mockedProject.id}/dataset`);

        await importDatasetPage.openImportDialog();

        await test.step('Upload dataset zip file', async () => {
            await importDatasetPage.loadZipFile(DATASET_FILENAME);
        });

        await test.step('Cancel the prepare job from dialog', async () => {
            await expect(importDatasetPage.getPreparingStatus()).toBeVisible();
            await importDatasetPage.cancelPrepareJobInDialog();
            await expect(importDatasetPage.getDialog()).toBeHidden();
        });
    });
});
