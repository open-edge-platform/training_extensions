// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { act, waitFor } from '@testing-library/react';
import { HttpResponse } from 'msw';
import { renderHook } from 'test-utils/render';

import { http } from '../../api/utils';
import { server } from '../../msw-node-setup';
import { useDeleteStagedDataset } from './staged-file.hook';

const deleteImportEntryMock = vi.fn();

vi.mock('../localStorage/use-import-dataset-to-project.hook', () => ({
    useImportDatasetToProject: () => ({
        deleteImportEntry: deleteImportEntryMock,
    }),
}));

describe('useDeleteStagedDataset', () => {
    beforeEach(() => {
        deleteImportEntryMock.mockReset();
    });

    it('deletes a staged dataset successfully', async () => {
        const stagedDatasetId = 'staged-dataset-1';
        const requestSpy = vi.fn();

        server.use(
            http.delete('/api/staged_datasets/{staged_dataset_id}', ({ params }) => {
                requestSpy(params.staged_dataset_id);
                return new HttpResponse(null, { status: 204 });
            })
        );

        const { result } = renderHook(() => useDeleteStagedDataset({ stagedDatasetId }));

        await act(async () => {
            await result.current.mutateAsync();
        });

        await waitFor(() => {
            expect(result.current.isSuccess).toBe(true);
            expect(requestSpy).toHaveBeenCalledWith(stagedDatasetId);
            expect(deleteImportEntryMock).toHaveBeenCalledWith(stagedDatasetId);
        });
    });

    it('removes import entry when server returns not found', async () => {
        const stagedDatasetId = 'staged-dataset-1';

        server.use(
            http.delete('/api/staged_datasets/{staged_dataset_id}', () =>
                // eslint-disable-next-line @typescript-eslint/ban-ts-comment
                // @ts-expect-error
                HttpResponse.json({ detail: 'staged dataset not found' }, { status: 404 })
            )
        );

        const { result } = renderHook(() => useDeleteStagedDataset({ stagedDatasetId }));

        await act(async () => {
            await expect(result.current.mutateAsync()).rejects.toBeDefined();
        });

        await waitFor(() => {
            expect(result.current.isError).toBe(true);
            expect(deleteImportEntryMock).toHaveBeenCalledWith(stagedDatasetId);
        });
    });

    it('does not remove import entry for other errors', async () => {
        const stagedDatasetId = 'staged-dataset-1';

        server.use(
            http.delete('/api/staged_datasets/{staged_dataset_id}', () =>
                // eslint-disable-next-line @typescript-eslint/ban-ts-comment
                // @ts-expect-error
                HttpResponse.json({ detail: 'permission denied' }, { status: 403 })
            )
        );

        const { result } = renderHook(() => useDeleteStagedDataset({ stagedDatasetId }));

        await act(async () => {
            await expect(result.current.mutateAsync()).rejects.toBeDefined();
        });

        await waitFor(() => {
            expect(result.current.isError).toBe(true);
            expect(deleteImportEntryMock).not.toHaveBeenCalled();
        });
    });
});
