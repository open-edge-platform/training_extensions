// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Button, ButtonGroup } from '@geti/ui';

import { useDeleteStagedDataset } from '../../../../../hooks/api/staged-file.hook';
import { IMPORT_DATASET_FORM_ID } from './util';

type LabelMappingButtonsProps = {
    stagedDatasetId: string;
    onClose: () => void;
};

export const LabelMappingButtons = ({ stagedDatasetId, onClose }: LabelMappingButtonsProps) => {
    const deleteFileMutation = useDeleteStagedDataset({ stagedDatasetId, onSuccess: onClose });

    const handleDelete = () => {
        deleteFileMutation.mutate();
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
