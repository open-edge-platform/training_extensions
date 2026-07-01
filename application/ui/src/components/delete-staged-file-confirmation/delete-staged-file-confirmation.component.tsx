// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { AlertDialog, Button, DialogTrigger } from '@geti/ui';
import { useDeleteStagedDataset } from 'hooks/api/staged-dataset.hook';

type DeleteStagedFileConfirmationProps = {
    stagedDatasetId: string;
    deleteEntry: () => void;
};

export const DeleteStagedFileConfirmation = ({ stagedDatasetId, deleteEntry }: DeleteStagedFileConfirmationProps) => {
    const deleteFileMutation = useDeleteStagedDataset({ stagedDatasetId, deleteEntry });

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
