// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { getMockedJob } from 'mocks/mock-job';
import { HttpResponse } from 'msw';

import { expect, http, test } from '../fixtures';

const mockedTrainingJob = getMockedJob({
    job_id: 'job-1',
    job_type: 'train',
    status: 'running',
    progress: 45,
    message: 'Training in progress...',
    metadata: {
        project: { id: 'id-1' },
        model: {
            id: 'model-123',
            architecture: 'Custom_Object_Detection_Gen3_ATSS',
            parent_revision_id: null,
            dataset_revision_id: 'dataset-1',
        },
    },
    started_at: '2026-01-19T08:15:00.000000+00:00',
    finished_at: null,
});

test.describe('Jobs - Current Training', () => {
    test.beforeEach(({ network }) => {
        network.use(
            http.get('/api/jobs', () => {
                return HttpResponse.json([mockedTrainingJob]);
            }),
            http.post('/api/jobs/{job_id}:cancel', () => {
                return HttpResponse.json(null, { status: 204 });
            })
        );
    });

    test('displays current training section when a job is running', async ({ jobsPage }) => {
        await jobsPage.goto();

        await expect(jobsPage.getCurrentTrainingSection()).toBeVisible();
        await expect(jobsPage.getTrainingTag()).toBeVisible();
        await expect(jobsPage.getStatusTag()).toBeVisible();
    });

    test('shows model architecture in training row', async ({ jobsPage }) => {
        await jobsPage.goto();

        await expect(jobsPage.getArchitectureText('Custom_Object_Detection_Gen3_ATSS')).toBeVisible();
    });

    test('can cancel a running training job', async ({ jobsPage, network }) => {
        await jobsPage.goto();

        await expect(jobsPage.getCurrentTrainingSection()).toBeVisible();

        // After cancel, return empty jobs list
        network.use(
            http.get('/api/jobs', () => {
                return HttpResponse.json([]);
            })
        );

        await jobsPage.cancelTrainingJob();

        await expect(jobsPage.getCurrentTrainingSection()).toBeHidden();
    });

    test('hides current training section when no jobs are running', async ({ jobsPage, network }) => {
        network.use(
            http.get('/api/jobs', () => {
                return HttpResponse.json([]);
            })
        );

        await jobsPage.goto();

        await expect(jobsPage.getCurrentTrainingSection()).toBeHidden();
    });
});
