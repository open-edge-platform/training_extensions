// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { screen } from '@testing-library/react';
import { userEvent } from '@testing-library/user-event';
import { HttpResponse } from 'msw';
import { render } from 'test-utils/render';

import { http } from '../../../api/utils';
import { server } from '../../../msw-node-setup';
import { formatBytes } from '../../../shared/util';
import { StagedImportDataset } from './staged-import-dataset.component';

describe('StagedImportDataset', () => {
    const stagedDatasetId = 'test-staged-dataset-id';
    const fileName = 'dataset.zip';

    it('opens the import dialog and sets current staged id on Continue', async () => {
        const stagedDatasetSize = 1024;
        const message = 'Map labels for the uploaded dataset';
        const openDialogSpy = vi.fn();

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
            <StagedImportDataset
                message={message}
                fileName={fileName}
                stagedDatasetId={stagedDatasetId}
                primaryButtonLabel={'continue'}
                onOpen={openDialogSpy}
                deleteEntry={vi.fn()}
            />
        );

        expect(
            await screen.findByText(`Import dataset - ${fileName} - ${formatBytes(stagedDatasetSize)}`)
        ).toBeVisible();

        expect(await screen.findByText(message)).toBeVisible();

        await userEvent.click(await screen.findByRole('button', { name: /continue/i }));

        expect(openDialogSpy).toHaveBeenCalledTimes(1);
    });
});
