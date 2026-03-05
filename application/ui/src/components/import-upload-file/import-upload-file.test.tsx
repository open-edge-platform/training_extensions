// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { HttpResponse } from 'msw';
import { render } from 'test-utils/render';

import { getMockedPrepareImportDatasetJob } from '../../../mocks/mock-job';
import { http } from '../../api/utils';
import { ImportDatasetDialogStateProvider } from '../../features/dataset/providers/export-import-dataset-dialog-provider.component';
import { server } from '../../msw-node-setup';
import { ImportUploadFile } from './import-upload-file.component';

describe('ImportUploadFile', () => {
    const validFile = new File(['file content'], 'test.zip', { type: 'application/zip' });
    const inValidFiles = new File(['foo'], 'video.mov', { type: 'video/quicktime' });
    const mockedStagedDatasetId = 'staged-dataset-123';
    const mockedPrepareImportDatasetJob = getMockedPrepareImportDatasetJob({});

    const renderApp = () => {
        server.use(
            http.post('/api/staged_datasets', () => {
                return HttpResponse.json(
                    {
                        id: mockedStagedDatasetId,
                        format: 'geti',
                        size: 123,
                        metadata: null,
                        compressed: true,
                        ready_for_export: false,
                        ready_for_import: true,
                    },
                    { status: 201 }
                );
            }),
            http.post('/api/jobs', () => {
                return HttpResponse.json(mockedPrepareImportDatasetJob, { status: 202 });
            })
        );
        const mockedOnFileUploaded = vi.fn();

        render(
            <ImportDatasetDialogStateProvider>
                <ImportUploadFile onFileUploaded={mockedOnFileUploaded} />
            </ImportDatasetDialogStateProvider>
        );

        return mockedOnFileUploaded;
    };

    beforeEach(() => {
        vi.clearAllMocks();
    });

    it('invalid file extension', async () => {
        const mockedOnFileUploaded = renderApp();

        const uploadFileElement = screen.getByTestId(/upload-zip-file/i);

        await userEvent.upload(uploadFileElement, [inValidFiles]);

        expect(screen.getByText(/Unsupported file format. Please upload a valid .zip file./i)).toBeVisible();

        await waitFor(() => {
            expect(mockedOnFileUploaded).not.toHaveBeenCalled();
        });
    });

    it('valid file extension', async () => {
        const mockedOnFileUploaded = renderApp();

        const uploadFileElement = screen.getByTestId(/upload-zip-file/i);

        await userEvent.upload(uploadFileElement, [validFile]);

        await waitFor(() => {
            expect(mockedOnFileUploaded).toHaveBeenCalledWith(
                expect.objectContaining({
                    stagedDatasetId: mockedStagedDatasetId,
                    prepareJobId: mockedPrepareImportDatasetJob.job_id,
                })
            );
        });
    });
});
