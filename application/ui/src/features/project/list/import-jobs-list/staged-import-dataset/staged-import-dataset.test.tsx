// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { screen } from '@testing-library/react';
import { userEvent } from '@testing-library/user-event';
import { HttpResponse } from 'msw';
import { render } from 'test-utils/render';

import { http } from '../../../../../api/utils';
import { server } from '../../../../../msw-node-setup';
import { formatBytes } from '../../../../../shared/util';
import { ImportDatasetDialogProvider } from '../../../providers/import-dataset-dialog-provider.component';
import { StagedImportDataset } from './staged-import-dataset.component';

const openDialogSpy = vi.fn();
const setCurrentStepSpy = vi.fn();
const setCurrentStagedIdSpy = vi.fn();

vi.mock('../../../providers/import-dataset-dialog-provider.component', async (importActual) => {
    const actual = await importActual<typeof import('../../../providers/import-dataset-dialog-provider.component')>();
    return {
        ...actual,
        useImportDatasetDialog: () => ({
            datasetImportDialogState: { open: openDialogSpy },
            setCurrentStep: setCurrentStepSpy,
            setCurrentStagedId: setCurrentStagedIdSpy,
        }),
    };
});

describe('StagedImportDataset', () => {
    const stagedDatasetId = 'test-staged-dataset-id';
    const fileName = 'dataset.zip';

    beforeEach(() => {
        vi.clearAllMocks();
    });

    it('opens the import dialog and sets current staged id on Continue', async () => {
        const stagedDatasetSize = 1024;
        const message = 'Map labels for the uploaded dataset';

        server.use(
            http.get('/api/staged_datasets/{staged_dataset_id}', () => {
                return HttpResponse.json({
                    id: stagedDatasetId,
                    format: 'unknown',
                    compressed: true,
                    ready_for_export: false,
                    ready_for_import: true,
                    size: stagedDatasetSize,
                });
            })
        );

        render(
            <ImportDatasetDialogProvider>
                <StagedImportDataset
                    message={message}
                    fileName={fileName}
                    stagedDatasetId={stagedDatasetId}
                    openState={'labelMapping'}
                    primaryButtonLabel={'continue'}
                />
            </ImportDatasetDialogProvider>
        );

        expect(
            await screen.findByText(`Import dataset - ${fileName} - ${formatBytes(stagedDatasetSize)}`)
        ).toBeVisible();

        expect(await screen.findByText(message)).toBeVisible();

        await userEvent.click(await screen.findByRole('button', { name: /continue/i }));

        expect(setCurrentStepSpy).toHaveBeenCalledWith('labelMapping');
        expect(setCurrentStagedIdSpy).toHaveBeenCalledWith(stagedDatasetId);
        expect(openDialogSpy).toHaveBeenCalledTimes(1);
    });

    it('shows an error and disables Continue when staged dataset fetch fails', async () => {
        const message = 'Map labels for the uploaded dataset';
        server.use(
            http.get('/api/staged_datasets/{staged_dataset_id}', () => {
                // eslint-disable-next-line @typescript-eslint/ban-ts-comment
                // @ts-expect-error
                return HttpResponse.json({ detail: 'Something went wrong' }, { status: 500 });
            })
        );

        render(
            <StagedImportDataset
                message={message}
                fileName={fileName}
                stagedDatasetId={stagedDatasetId}
                primaryButtonLabel={'continue'}
                openState={'labelMapping'}
            />
        );

        expect(await screen.findByText('Error: Something went wrong')).toBeVisible();
        expect(await screen.findByRole('button', { name: /continue/i })).toBeDisabled();
        expect(screen.queryByText(message)).not.toBeInTheDocument();
    });
});
