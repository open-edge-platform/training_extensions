// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Button, ButtonGroup } from '@geti/ui';
import { useDeleteStagedDataset } from 'hooks/api/staged-dataset.hook';

import { TASK_SELECTION_FORM_ID } from './util';

type ImportTaskSelectionButtonsProps = {
    stagedDatasetId: string;
    onClose: () => void;
    deleteEntry: () => void;
};

export const ImportTaskSelectionButtons = ({
    stagedDatasetId,
    onClose,
    deleteEntry,
}: ImportTaskSelectionButtonsProps) => {
    const deleteFileMutation = useDeleteStagedDataset({ stagedDatasetId, onSuccess: onClose, deleteEntry });

    const isPending = deleteFileMutation.isPending;

    const handleDeleteJob = () => {
        deleteFileMutation.mutate();
    };

    return (
        <ButtonGroup>
            <Button variant='negative' isPending={isPending} isDisabled={isPending} onPress={handleDeleteJob}>
                Delete
            </Button>

            <Button onPress={onClose} isPending={isPending} isDisabled={isPending} variant='secondary'>
                Hide
            </Button>

            <Button type='submit' form={TASK_SELECTION_FORM_ID} variant='primary'>
                Next
            </Button>
        </ButtonGroup>
    );
};
