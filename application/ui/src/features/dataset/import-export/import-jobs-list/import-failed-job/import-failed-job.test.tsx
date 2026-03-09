// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { screen } from '@testing-library/react';
import { userEvent } from '@testing-library/user-event';
import { getMockedProject } from 'mocks/mock-project';
import { HttpResponse } from 'msw';
import { render } from 'test-utils/render';

import { getMockedPrepareImportDatasetJob } from '../../../../../../mocks/mock-job';
import { http } from '../../../../../api/utils';
import { PrepareImportDatasetJob } from '../../../../../constants/shared-types';
import { server } from '../../../../../msw-node-setup';
import { ImportFailedJob } from './import-failed-job.component';

const mockedDeleteImportEntry = vi.fn();

vi.mock('../../../../../hooks/localStorage/use-import-dataset-to-project.hook', () => ({
    useImportDatasetToProject: () => ({
        deleteImportEntry: mockedDeleteImportEntry,
    }),
}));

describe('ImportFailedJob', () => {
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
            <ImportFailedJob
                error={job.error ?? ''}
                message={job.message ?? ''}
                size={1024}
                fileName='dataset.zip'
                stagedDatasetId={job.metadata.staged_dataset_id}
            />
        );
    };

    beforeEach(() => {
        vi.clearAllMocks();
    });

    it('removes from local storage when close button is clicked', async () => {
        renderApp(getMockedPrepareImportDatasetJob({ status: 'FAILED' }));

        const closeButton = await screen.findByRole('button', { name: 'close import dataset status' });
        await userEvent.click(closeButton);

        expect(mockedDeleteImportEntry).toHaveBeenCalledTimes(1);
    });

    it('displays job message and job error', async () => {
        const mockImportJob = getMockedPrepareImportDatasetJob({
            status: 'FAILED',
            message: 'Import failed due to validation error',
            error: 'Dataset validation failed: missing required fields',
        });
        renderApp(mockImportJob);

        expect(await screen.findByText('Import failed due to validation error')).toBeVisible();
        expect(await screen.findByText('Dataset validation failed: missing required fields')).toBeVisible();
    });
});
