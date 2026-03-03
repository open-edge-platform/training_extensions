// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { screen, waitFor } from '@testing-library/react';
import { userEvent } from '@testing-library/user-event';
import { getMockedProject } from 'mocks/mock-project';
import { HttpResponse } from 'msw';
import { render } from 'test-utils/render';

import { getMockedPrepareImportDatasetJob } from '../../../../../../mocks/mock-job';
import { http } from '../../../../../api/utils';
import { PrepareImportDatasetJob } from '../../../../../constants/shared-types';
import { server } from '../../../../../msw-node-setup';
import { ImportActiveJob } from './import-active-job.component';

const mockedDeleteImportEntry = vi.fn();

vi.mock('../../../../../hooks/localStorage/use-import-dataset-to-project.hook', () => ({
    useImportDatasetToProject: () => ({
        deleteImportEntry: mockedDeleteImportEntry,
    }),
}));

describe('ImportActiveJob', () => {
    const renderApp = (job: PrepareImportDatasetJob) => {
        server.use(
            http.get('/api/projects/{project_id}', () => {
                return HttpResponse.json(getMockedProject({ id: 'project-123' }));
            }),
            http.delete('/api/staged_datasets/{staged_dataset_id}', () => {
                return HttpResponse.json(null, { status: 200 });
            })
        );

        render(
            <ImportActiveJob
                job={job}
                fileName='dataset.zip'
                size={1024}
                stagedDatasetId={job.metadata.staged_dataset_id}
            />
        );
    };

    beforeEach(() => {
        vi.clearAllMocks();
    });

    it('shows message and progress for running jobs', async () => {
        renderApp(
            getMockedPrepareImportDatasetJob({
                status: 'RUNNING',
                progress: 55.8,
                message: 'Preparing dataset for import',
            })
        );

        expect(await screen.findByText('Preparing dataset for import')).toBeVisible();
        expect(await screen.findByText('56%')).toBeVisible();
    });

    it('does not show progress for pending jobs and falls back to status text when message is missing', async () => {
        renderApp(getMockedPrepareImportDatasetJob({ status: 'PENDING', progress: 42, message: null }));

        expect(await screen.findByText('pending')).toBeVisible();
        expect(screen.queryByText(/%/)).not.toBeInTheDocument();
    });

    it('removes from local storage when job is cancelled successfully', async () => {
        server.use(
            http.post('/api/jobs/{job_id}:cancel', () => {
                return HttpResponse.json(null, { status: 204 });
            })
        );

        renderApp(getMockedPrepareImportDatasetJob({ status: 'RUNNING', progress: 10 }));

        await userEvent.click(await screen.findByRole('button', { name: /cancel job dialog/i }));
        await userEvent.click(await screen.findByRole('button', { name: /cancel job/i }));

        await waitFor(() => {
            expect(mockedDeleteImportEntry).toHaveBeenCalledTimes(1);
        });
    });
});
