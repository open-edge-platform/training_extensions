// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { AlertDialog, Button, DialogTrigger } from '@geti/ui';
import { useImportDatasetToProject } from 'hooks/localStorage/use-import-dataset-to-project.hook';

import { useDeleteStagedDataset } from '../../../../hooks/api/use-delete-staged-dataset.hook';

type DeleteStagedFileConfirmationProps = {
    stagedDatasetId: string;
};

export const DeleteStagedFileConfirmation = ({ stagedDatasetId }: DeleteStagedFileConfirmationProps) => {
    const { deleteImportEntry } = useImportDatasetToProject();
    const deleteFileMutation = useDeleteStagedDataset({
        stagedDatasetId,
        deleteEntry: () => deleteImportEntry(stagedDatasetId),
    });

    const handleCancel = () => {
        deleteFileMutation.mutate();
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
                isPrimaryActionDisabled={deleteFileMutation.isPending}
            >
                {`Are you sure you want to delete the dataset file "${stagedDatasetId}"?`}
            </AlertDialog>
        </DialogTrigger>
    );
};
