// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { $api } from '../../api/client';
import { isNonEmptyString } from '../../shared/util';
import { isInvalidStagedFile } from './util';

type useDeleteStagedDatasetProps = {
    stagedDatasetId: string | null | undefined;
    onError?: (error: unknown) => void;
    onSuccess?: () => void;
    deleteEntry?: () => void;
};

export const useStagedDataset = (stagedDatasetId: string | null | undefined) => {
    return $api.useQuery(
        'get',
        '/api/staged_datasets/{staged_dataset_id}',
        {
            params: { path: { staged_dataset_id: stagedDatasetId } },
        },
        { enabled: isNonEmptyString(stagedDatasetId) }
    );
};

export const useStagedDatasetSuspense = (stagedDatasetId: string | null | undefined) => {
    return $api.useSuspenseQuery('get', '/api/staged_datasets/{staged_dataset_id}', {
        params: { path: { staged_dataset_id: stagedDatasetId } },
    });
};

export const useDeleteStagedDataset = ({
    stagedDatasetId,
    onError,
    onSuccess,
    deleteEntry,
}: useDeleteStagedDatasetProps) => {
    const params = { params: { path: { staged_dataset_id: stagedDatasetId } } };

    const deleteMutation = $api.useMutation('delete', '/api/staged_datasets/{staged_dataset_id}', {
        onSuccess: () => {
            deleteEntry?.();
            onSuccess?.();
        },
        onError: (error) => {
            isInvalidStagedFile(error) && deleteEntry?.();
            onError?.(error);
        },
    });

    return {
        ...deleteMutation,
        mutate: () => deleteMutation.mutate(params),
        mutateAsync: () => deleteMutation.mutateAsync(params),
    };
};
