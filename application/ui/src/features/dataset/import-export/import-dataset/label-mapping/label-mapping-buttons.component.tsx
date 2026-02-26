// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Button, ButtonGroup } from '@geti/ui';

import { $api } from '../../../../../api/client';
import { useImportDatasetToProject } from '../../../../../hooks/localStorage/use-import-dataset-to-project.hook';
import { IMPORT_DATASET_FORM_ID } from './util';

interface LabelMappingButtonsProps {
    stagedDatasetId: string;
    onClose: () => void;
}

export const LabelMappingButtons = ({ stagedDatasetId, onClose }: LabelMappingButtonsProps) => {
    const { deleteImportEntry } = useImportDatasetToProject();

    const deleteFileMutation = $api.useMutation('delete', '/api/staged_datasets/{staged_dataset_id}');

    const handleDelete = () => {
        deleteFileMutation.mutateAsync(
            { params: { path: { staged_dataset_id: String(stagedDatasetId) } } },
            {
                onSuccess: () => {
                    onClose();
                    deleteImportEntry({ stagedDatasetId });
                },
            }
        );
    };
    return (
        <ButtonGroup>
            <Button
                onPress={handleDelete}
                variant='negative'
                isPending={deleteFileMutation.isPending}
                isDisabled={deleteFileMutation.isPending}
            >
                Delete
            </Button>

            <Button onPress={onClose} variant='secondary'>
                Hide
            </Button>

            <Button type='submit' form={IMPORT_DATASET_FORM_ID} variant='accent'>
                Submit
            </Button>
        </ButtonGroup>
    );
};
