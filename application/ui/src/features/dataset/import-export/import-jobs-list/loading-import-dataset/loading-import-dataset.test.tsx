// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { QueryClient } from '@tanstack/react-query';
import { screen, waitFor } from '@testing-library/react';
import { HttpResponse } from 'msw';
import { render } from 'test-utils/render';

import { getMockedPrepareImportDatasetJob } from '../../../../../../mocks/mock-job';
import { http } from '../../../../../api/utils';
import { Job } from '../../../../../constants/shared-types';
import { server } from '../../../../../msw-node-setup';
import { createQueryClient, getQueryKey } from '../../../../../query-client/query-client';
import { LoadingImportDataset } from './loading-import-dataset.component';

const stagedDatasetId = 'staged-dataset-123';

let queryClient: QueryClient;
const mockedDeleteImportEntry = vi.fn();

vi.mock('../../../../../hooks/localStorage/use-import-dataset-to-project.hook', () => ({
    useImportDatasetToProject: () => ({
        getAllImportEntries: () => [],
        getImportEntry: () => ({
            stagedDatasetId,
            fileName: 'dataset.zip',
            size: 1024,
            importJobId: 'job-123',
            prepareJobId: null,
            step: 'importing',
        }),
        deleteImportEntry: mockedDeleteImportEntry,
        appendImportEntry: vi.fn(),
    }),
}));

describe('LoadingImportDataset', () => {
    const renderApp = (job: Job) => {
        server.use(
            http.get('/api/jobs/{job_id}', () => {
                return HttpResponse.json(job);
            })
        );

        return render(<LoadingImportDataset stagedDatasetId={stagedDatasetId} />, { queryClient });
    };

    beforeEach(() => {
        vi.clearAllMocks();
        queryClient = createQueryClient();
    });

    it('renders failed job state and displays job message and error', async () => {
        const job = getMockedPrepareImportDatasetJob({
            status: 'FAILED',
            message: 'Import failed due to validation error',
            error: 'Dataset validation failed: missing required fields',
        });

        renderApp(job);

        expect(await screen.findByText('Import failed due to validation error')).toBeVisible();
        expect(await screen.findByText('Dataset validation failed: missing required fields')).toBeVisible();
        expect(mockedDeleteImportEntry).not.toHaveBeenCalled();
    });

    it('renders active job state when job is running', async () => {
        const job = getMockedPrepareImportDatasetJob({
            status: 'RUNNING',
            progress: 55.8,
            message: 'Preparing dataset for import',
        });

        renderApp(job);

        expect(await screen.findByText('dataset.zip file is being processed for import')).toBeVisible();
        expect(await screen.findByText('56%')).toBeVisible();
    });

    it('invalidates dataset media query when import job status query succeeds', async () => {
        const invalidateQueriesSpy = vi.spyOn(queryClient, 'invalidateQueries');
        const job = getMockedPrepareImportDatasetJob({ status: 'DONE' });

        renderApp(job);

        expect(await screen.findByText('dataset.zip file has been imported successfully')).toBeVisible();

        await waitFor(() => {
            expect(invalidateQueriesSpy).toHaveBeenCalledWith({
                queryKey: getQueryKey([
                    'get',
                    '/api/projects/{project_id}/dataset/media',
                    { params: { path: { project_id: '123' } } },
                ]),
            });
        });
        expect(mockedDeleteImportEntry).not.toHaveBeenCalled();
    });

    it('renders failed state when job status query fails with error', async () => {
        const invalidateQueriesSpy = vi.spyOn(queryClient, 'invalidateQueries');

        server.use(
            http.get('/api/jobs/{job_id}', () => {
                // eslint-disable-next-line @typescript-eslint/ban-ts-comment
                // @ts-expect-error
                return HttpResponse.json({ detail: 'Job not found' }, { status: 404 });
            })
        );

        render(<LoadingImportDataset stagedDatasetId={stagedDatasetId} />, { queryClient });

        expect(await screen.findByText('An error occurred during import.')).toBeVisible();
        expect(await screen.findByText('Job not found')).toBeVisible();
        expect(invalidateQueriesSpy).not.toHaveBeenCalled();
        expect(mockedDeleteImportEntry).not.toHaveBeenCalled();
    });
});
