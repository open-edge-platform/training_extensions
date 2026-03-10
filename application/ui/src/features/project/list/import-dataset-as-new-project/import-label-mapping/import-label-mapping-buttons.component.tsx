// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Button, ButtonGroup } from '@geti/ui';
import { useDeleteStagedDataset } from 'hooks/api/staged-dataset.hook';

import { useImportDatasetDialog } from '../../../providers/import-dataset-dialog-provider.component';

type ImportLabelMappingButtonsProps = {
    stagedDatasetId: string;
    onClose: () => void;
    deleteEntry: () => void;
};

export const ImportLabelMappingButtons = ({
    stagedDatasetId,
    onClose,
    deleteEntry,
}: ImportLabelMappingButtonsProps) => {
    const { setCurrentStep } = useImportDatasetDialog();
    const deleteFileMutation = useDeleteStagedDataset({ stagedDatasetId, onSuccess: onClose, deleteEntry });

    const isPending = deleteFileMutation.isPending;

    const handleCancelJob = () => {
        deleteFileMutation.mutate();
    };

    const handleBack = () => {
        setCurrentStep('taskTypeSelection');
    };

    return (
        <ButtonGroup>
            <Button variant='negative' isPending={isPending} isDisabled={isPending} onPress={handleCancelJob}>
                Delete
            </Button>

            <Button onPress={onClose} isPending={isPending} isDisabled={isPending} variant='secondary'>
                Hide
            </Button>

            <Button onPress={handleBack} isPending={isPending} isDisabled={isPending} variant='secondary'>
                Back
            </Button>

            <Button type='submit' variant='primary'>
                Create
            </Button>
        </ButtonGroup>
    );
};
