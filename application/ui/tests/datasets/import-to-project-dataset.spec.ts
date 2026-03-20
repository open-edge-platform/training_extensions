// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { getMockedJob } from 'mocks/mock-job';
import { getMockedLabel } from 'mocks/mock-labels';
import { getMockedProject } from 'mocks/mock-project';
import { getMockedStagedDataset } from 'mocks/mock-staged-dataset';
import { HttpResponse } from 'msw';

import { expect, http, test } from '../fixtures';

const STAGED_DATASET_ID = 'staged-dataset-789';
const PREPARE_JOB_ID = 'prepare-job-123';
const IMPORT_JOB_ID = 'import-job-456';

const mockedProject = getMockedProject({
    task: {
        task_type: 'detection',
        exclusive_labels: true,
        labels: [getMockedLabel({ name: 'cat' }), getMockedLabel({ name: 'dog' })],
    },
});

const prepareJob = getMockedJob({
    job_id: PREPARE_JOB_ID,
    job_type: 'prepare_dataset_for_import',
    status: 'RUNNING',
    progress: 50,
    message: 'Analyzing dataset archive...',
});

const prepareDoneJob = getMockedJob({
    ...prepareJob,
    status: 'DONE',
    progress: 100,
    message: 'Preparation completed',
});

const importJob = getMockedJob({
    job_id: IMPORT_JOB_ID,
    job_type: 'import_dataset_to_project',
    status: 'RUNNING',
    progress: 0,
    message: 'Importing dataset...',
});

const importingJob = getMockedJob({
    job_id: IMPORT_JOB_ID,
    job_type: 'import_dataset_to_project',
    status: 'RUNNING',
    progress: 60,
    message: 'Importing progress...',
});

const importDoneJob = getMockedJob({
    job_id: IMPORT_JOB_ID,
    job_type: 'import_dataset_to_project',
    status: 'DONE',
    progress: 100,
    message: 'Import completed',
});

const stagedDatasetWithMetadata = getMockedStagedDataset({
    id: STAGED_DATASET_ID,
    ready_for_import: true,
    metadata: {
        labels: ['cat', 'dog'],
        num_images: 100,
        num_annotated_images: 80,
        num_frames: 0,
        num_annotated_frames: 0,
        num_annotations: 200,
        annotation_type: 'bounding_box',
        num_videos: 0,
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
                    return HttpResponse.json(importJob, { status: 201 });
                }
                return HttpResponse.json(
                    getMockedJob({ job_id: PREPARE_JOB_ID, job_type: 'prepare_dataset_for_import' }),
                    { status: 201 }
                );
            })
        );
    });

    test('import dataset with default label mapping', async ({ page, network, importDatasetPage }) => {
        let prepareJobPollCount = 0;
        let importJobPollCount = 0;
        let deletedStagedDatasetId: string | undefined;

        network.use(
            http.get('/api/jobs/{job_id}', ({ params }) => {
                const jobId = params.job_id as string;

                if (jobId === IMPORT_JOB_ID) {
                    importJobPollCount += 1;
                    const job = importJobPollCount <= 2 ? importingJob : importDoneJob;
                    return HttpResponse.json(job, { status: 200 });
                }

                prepareJobPollCount += 1;
                const job = prepareJobPollCount <= 2 ? prepareJob : prepareDoneJob;
                return HttpResponse.json(job, { status: 200 });
            }),
            http.delete('/api/staged_datasets/{staged_dataset_id}', ({ params }) => {
                deletedStagedDatasetId = params.staged_dataset_id as string;
                return new HttpResponse(null, { status: 204 });
            })
        );

        await page.goto(`projects/${mockedProject.id}/dataset`);

        await importDatasetPage.openImportDialog();

        await test.step('Upload dataset zip file', async () => {
            await importDatasetPage.loadZipFile('my-dataset.zip');
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
            await expect(importDatasetPage.getImportStatusText('my-dataset.zip', 'processing')).toBeVisible();
            await expect(page.getByText(String(importingJob.message))).toBeVisible();

            await expect(importDatasetPage.getImportStatusText('my-dataset.zip', 'success')).toBeVisible();
            await expect(page.getByText('Ready')).toBeVisible();
        });

        await test.step('Close removes the staged dataset', async () => {
            await importDatasetPage.closeImportStatus();

            await expect(importDatasetPage.getImportStatusText('my-dataset.zip', 'success')).toBeHidden();
            await expect.poll(() => deletedStagedDatasetId).toBe(STAGED_DATASET_ID);
        });
    });

    test('cancel import job removes staged files', async ({ page, network, importDatasetPage }) => {
        let prepareJobPollCount = 0;

        network.use(
            http.get('/api/jobs/{job_id}', ({ params }) => {
                const jobId = params.job_id as string;

                if (jobId === IMPORT_JOB_ID) {
                    return HttpResponse.json(importingJob, { status: 200 });
                }

                prepareJobPollCount += 1;
                const job = prepareJobPollCount <= 2 ? prepareJob : prepareDoneJob;
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
            await importDatasetPage.loadZipFile('my-dataset.zip');
            await expect(importDatasetPage.getPreparingStatus()).toBeVisible();
            await expect(importDatasetPage.getStatisticsHeading()).toBeVisible();
        });

        await test.step('Submit import', async () => {
            await importDatasetPage.submit();
            await expect(importDatasetPage.getDialog()).toBeHidden();
        });

        await test.step('Cancel the running import job', async () => {
            await expect(importDatasetPage.getImportStatusText('my-dataset.zip', 'processing')).toBeVisible();
            await importDatasetPage.cancelJobFromStatusCard();

            await expect(importDatasetPage.getImportStatusText('my-dataset.zip', 'processing')).toBeHidden();
        });
    });

    test('cancel prepare job removes staged files', async ({ page, network, importDatasetPage }) => {
        network.use(
            http.post('/api/jobs', async () => {
                return HttpResponse.json(
                    getMockedJob({ job_id: PREPARE_JOB_ID, job_type: 'prepare_dataset_for_import' }),
                    { status: 201 }
                );
            }),
            http.get('/api/jobs/{job_id}', () => {
                return HttpResponse.json(prepareJob, { status: 200 });
            }),
            http.post('/api/jobs/{job_id}:cancel', () => {
                return HttpResponse.json(getMockedJob({ ...prepareJob, status: 'CANCELLED' }), { status: 200 });
            }),
            http.delete('/api/staged_datasets/{staged_dataset_id}', () => {
                return new HttpResponse(null, { status: 204 });
            })
        );

        await page.goto(`projects/${mockedProject.id}/dataset`);

        await importDatasetPage.openImportDialog();

        await test.step('Upload dataset zip file', async () => {
            await importDatasetPage.loadZipFile('my-dataset.zip');
        });

        await test.step('Cancel the prepare job from dialog', async () => {
            await expect(importDatasetPage.getPreparingStatus()).toBeVisible();
            await importDatasetPage.cancelPrepareJobInDialog();
            await expect(importDatasetPage.getDialog()).toBeHidden();
        });
    });
});
