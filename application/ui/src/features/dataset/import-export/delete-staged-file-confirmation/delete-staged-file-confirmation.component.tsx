// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { AlertDialog, Button, DialogTrigger } from '@geti/ui';

import { $api } from '../../../../api/client';
import { useImportDatasetToProject } from '../../../../hooks/localStorage/use-import-dataset-to-project.hook';
import { isInvalidStagedFile } from '../util';

type DeleteStagedFileConfirmationProps = {
    stagedDatasetId: string;
};

export const DeleteStagedFileConfirmation = ({ stagedDatasetId }: DeleteStagedFileConfirmationProps) => {
    const { deleteImportEntry } = useImportDatasetToProject();

    const removeStagedDatasetMutation = $api.useMutation('delete', '/api/staged_datasets/{staged_dataset_id}');

    const handleCancel = () => {
        removeStagedDatasetMutation.mutate(
            { params: { path: { staged_dataset_id: stagedDatasetId } } },
            {
                onSuccess: () => deleteImportEntry(stagedDatasetId),
                onError: (error) => {
                    isInvalidStagedFile(error) && deleteImportEntry(stagedDatasetId);
                },
            }
        );
    };

    return (
        <DialogTrigger>
            <Button variant='secondary' style='fill' aria-label='delete import dataset status'>
                Delete
            </Button>
            <AlertDialog
                title='Delete Staged File'
                variant='destructive'
                cancelLabel='Cancel'
                autoFocusButton='primary'
                primaryActionLabel='Delete'
                onPrimaryAction={handleCancel}
                isPrimaryActionDisabled={removeStagedDatasetMutation.isPending}
            >
                {`Are you sure you want to delete the dataset file "${stagedDatasetId}"?`}
            </AlertDialog>
        </DialogTrigger>
    );
};
