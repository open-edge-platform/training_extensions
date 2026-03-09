// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { $api } from '../../api/client';
import { useImportDatasetToProject } from '../localStorage/use-import-dataset-to-project.hook';
import { isInvalidStagedFile } from './util';

type useDeleteStagedDatasetProps = {
    stagedDatasetId: string;
    onSuccess?: () => void;
    onError?: (error: unknown) => void;
};

export const useDeleteStagedDataset = ({ stagedDatasetId, onSuccess, onError }: useDeleteStagedDatasetProps) => {
    const { deleteImportEntry } = useImportDatasetToProject();
    const params = { params: { path: { staged_dataset_id: stagedDatasetId } } };

    const deleteMutation = $api.useMutation('delete', '/api/staged_datasets/{staged_dataset_id}', {
        onSuccess: () => {
            deleteImportEntry(stagedDatasetId);
            onSuccess?.();
        },
        onError: (error) => {
            isInvalidStagedFile(error) && deleteImportEntry(stagedDatasetId);
            onError?.(error);
        },
    });

    return {
        ...deleteMutation,
        mutate: () => deleteMutation.mutate(params),
        mutateAsync: () => deleteMutation.mutateAsync(params),
    };
};
