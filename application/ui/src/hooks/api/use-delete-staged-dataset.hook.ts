// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { $api } from '../../api/client';
import { isInvalidStagedFile } from './util';

type useDeleteStagedDatasetProps = {
    stagedDatasetId: string | null | undefined;
    deleteEntry: () => void;
    onSuccess?: () => void;
    onError?: (error: unknown) => void;
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
            deleteEntry();
            onSuccess?.();
        },
        onError: (error) => {
            isInvalidStagedFile(error) && deleteEntry();
            onError?.(error);
        },
    });

    return {
        ...deleteMutation,
        mutate: () => deleteMutation.mutate(params),
        mutateAsync: () => deleteMutation.mutateAsync(params),
    };
};
