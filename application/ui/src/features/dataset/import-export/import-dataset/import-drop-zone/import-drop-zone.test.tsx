// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { HttpResponse } from 'msw';
import { render } from 'test-utils/render';

import { getMockedPrepareImportDatasetJob } from '../../../../../../mocks/mock-job';
import { http } from '../../../../../api/utils';
import { server } from '../../../../../msw-node-setup';
import { ImportDatasetDialogStateProvider } from '../../../providers/export-import-dataset-dialog-provider.component';
import { ImportDropZone } from './import-drop-zone.component';

const mockedAppendImportEntry = vi.fn();

vi.mock('../../../../../hooks/localStorage/use-import-dataset-to-project.hook', () => ({
    useImportDatasetToProject: () => ({
        appendImportEntry: mockedAppendImportEntry,
    }),
}));

describe('ImportDropZone', () => {
    const validFile = new File(['file content'], 'test.zip', { type: 'application/zip' });
    const inValidFiles = new File(['foo'], 'video.mov', { type: 'video/quicktime' });

    const renderApp = () => {
        server.use(
            http.post('/api/staged_datasets', () => {
                return HttpResponse.json(
                    {
                        id: 'staged-dataset-123',
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
                return HttpResponse.json(getMockedPrepareImportDatasetJob({}), { status: 202 });
            })
        );

        render(
            <ImportDatasetDialogStateProvider>
                <ImportDropZone />
            </ImportDatasetDialogStateProvider>
        );
    };

    beforeEach(() => {
        vi.clearAllMocks();
    });

    it('invalid file extension', async () => {
        renderApp();

        const uploadFileElement = screen.getByTestId(/upload-zip-file/i);

        await userEvent.upload(uploadFileElement, [inValidFiles]);

        expect(screen.getByText(/Unsupported file format. Please upload a valid .zip file./i)).toBeVisible();

        await waitFor(() => {
            expect(mockedAppendImportEntry).not.toHaveBeenCalled();
        });
    });

    it('valid file extension', async () => {
        renderApp();

        const uploadFileElement = screen.getByTestId(/upload-zip-file/i);

        await userEvent.upload(uploadFileElement, [validFile]);

        await waitFor(() => {
            expect(mockedAppendImportEntry).toHaveBeenCalledWith(expect.objectContaining({ step: 'preparing' }));
        });
    });
});
