// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { screen } from '@testing-library/react';
import { userEvent } from '@testing-library/user-event';
import { getMockedProject } from 'mocks/mock-project';
import { HttpResponse } from 'msw';
import { render } from 'test-utils/render';

import { getMockedJobExportJob } from '../../../../../../../mocks/mock-job';
import { http } from '../../../../../../api/utils';
import { ExportDatasetJob } from '../../../../../../constants/shared-types';
import { server } from '../../../../../../msw-node-setup';
import { downloadFile } from '../../../../../../shared/util';
import { ExportCompletedJob } from './export-completed-job.component';

vi.mock('../../../../../../shared/util', async (importActual) => {
    const actual = await importActual<typeof import('../../../../../../shared/util')>();
    return {
        ...actual,
        downloadFile: vi.fn(),
    };
});

const mockedRemoveLsExportId = vi.fn();

vi.mock('../../../../../../hooks/storage/use-export-dataset.hook', () => ({
    useExportDataset: () => ({
        removeLsExportId: mockedRemoveLsExportId,
        getLsExportIds: vi.fn(() => []),
        addLsExportId: vi.fn(),
    }),
}));

describe('ExportCompletedJob', () => {
    const renderApp = (job: ExportDatasetJob) => {
        server.use(
            http.get('/api/projects/{project_id}', () => {
                return HttpResponse.json(getMockedProject({ id: 'project-123' }));
            }),
            http.get('/api/staged_datasets/{staged_dataset_id}', () => {
                return HttpResponse.json({});
            })
        );

        render(<ExportCompletedJob job={job} />);
    };

    beforeEach(() => {
        vi.clearAllMocks();
    });

    it('downloads dataset when download button is clicked', async () => {
        const mockExportJob = getMockedJobExportJob({ status: 'FINISHED' });
        renderApp(mockExportJob);

        const downloadButton = await screen.findByRole('button', { name: 'download dataset' });
        await userEvent.click(downloadButton);

        expect(downloadFile).toHaveBeenCalledWith(
            expect.stringContaining(`/api/staged_datasets/${mockExportJob.metadata.dataset_id}/zip`),
            `dataset_${mockExportJob.metadata.dataset_id}.zip`
        );
    });

    it('deletes staged dataset and removes from local storage when close button is clicked', async () => {
        const deleteStageFileSpy = vi.fn();

        server.use(
            http.delete('/api/staged_datasets/{staged_dataset_id}', () => {
                deleteStageFileSpy();
                return HttpResponse.json(null, { status: 204 });
            })
        );

        const mockExportJob = getMockedJobExportJob({ status: 'FINISHED' });
        renderApp(mockExportJob);

        const closeButton = await screen.findByRole('button', { name: 'close export dataset status' });
        await userEvent.click(closeButton);

        expect(deleteStageFileSpy).toHaveBeenCalled();
        expect(mockedRemoveLsExportId).toHaveBeenCalledWith(mockExportJob.job_id);
    });
});
