// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { getMockedJob } from 'mocks/mock-job';
import { HttpResponse } from 'msw';

import { expect, http, test } from '../fixtures';

const mockedTrainingJob = getMockedJob({
    job_id: 'job-1',
    job_type: 'train',
    status: 'RUNNING',
    progress: 45,
    message: 'Training in progress...',
    metadata: {
        project: { id: 'id-1' },
        model: {
            id: 'ef3983f1-cef0-4ebe-91db-7330f1dd6e27',
            name: 'ATSS (ef3983f1)',
            architecture: 'Custom_Object_Detection_Gen3_ATSS',
            parent_revision_id: null,
            dataset_revision_id: 'dataset-1',
        },
        device: {
            type: 'cpu',
            name: 'CPU',
        },
    },
    started_at: '2026-01-19T08:15:00.000000+00:00',
    finished_at: null,
});

test.describe('Jobs - Current Running', () => {
    test('displays current running section when a job is running', async ({ jobsPage, network }) => {
        network.use(
            http.get('/api/jobs', () => {
                return HttpResponse.json([mockedTrainingJob]);
            }),
            http.post('/api/jobs/{job_id}:cancel', () => {
                return HttpResponse.json(null, { status: 204 });
            })
        );

        await jobsPage.goto();

        await expect(jobsPage.getCurrentRunningSection()).toBeVisible();
        await expect(jobsPage.getRunningTag()).toBeVisible();
        await expect(jobsPage.getStatusTag()).toBeVisible();
    });

    test('shows model architecture in training row', async ({ jobsPage, network }) => {
        network.use(
            http.get('/api/jobs', () => {
                return HttpResponse.json([mockedTrainingJob]);
            }),
            http.post('/api/jobs/{job_id}:cancel', () => {
                return HttpResponse.json(null, { status: 204 });
            })
        );

        await jobsPage.goto();

        await expect(jobsPage.getArchitectureText('Custom_Object_Detection_Gen3_ATSS')).toBeVisible();
    });

    test('can cancel a running training job', async ({ jobsPage, network }) => {
        let hasCancelledTraining = false;

        network.use(
            http.get('/api/jobs', () => {
                return HttpResponse.json(hasCancelledTraining ? [] : [mockedTrainingJob]);
            }),
            http.post('/api/jobs/{job_id}:cancel', () => {
                hasCancelledTraining = true;
                return HttpResponse.json(null, { status: 204 });
            })
        );

        await jobsPage.goto();

        await expect(jobsPage.getCurrentRunningSection()).toBeVisible();

        await jobsPage.cancelRunningJob();

        await expect(jobsPage.getCurrentRunningSection()).toBeHidden();
    });

    test('hides current running section when no jobs are running', async ({ jobsPage }) => {
        await jobsPage.goto('id-2');

        await expect(jobsPage.getCurrentRunningSection()).toBeHidden();
    });
});
