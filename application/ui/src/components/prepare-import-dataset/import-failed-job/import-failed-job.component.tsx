// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Button, dimensionValue, Divider, Flex, Text, View } from '@geti/ui';
import { useDeleteStagedDataset } from 'hooks/api/use-delete-staged-dataset.hook';

import { formatBytes } from '../../../shared/util';

type ImportFailedJobProps = {
    size: number;
    fileName: string;
    stagedDatasetId: string;
    message?: string;
    error?: string;
    deleteEntry: () => void;
};

export const ImportFailedJob = ({
    size,
    error,
    message,
    fileName,
    stagedDatasetId,
    deleteEntry,
}: ImportFailedJobProps) => {
    const deleteFileMutation = useDeleteStagedDataset({ stagedDatasetId, deleteEntry });

    const handleClose = () => {
        deleteFileMutation.mutate();
    };

    return (
        <View padding='size-150'>
            <Flex justifyContent='space-between' alignItems='center' gap='size-250'>
                <Text UNSAFE_style={{ fontWeight: 500, fontSize: dimensionValue('size-200') }}>
                    Import dataset - {fileName} - {formatBytes(size)}
                </Text>

                <Flex justifyContent='space-between' alignItems='center' gap='size-250'>
                    <Button
                        variant='secondary'
                        style='fill'
                        aria-label='close import dataset status'
                        onPress={handleClose}
                    >
                        Close
                    </Button>
                </Flex>
            </Flex>

            <Text>{message}</Text>
            <Divider size='S' marginY='size-150' />
            <Text>{error}</Text>
        </View>
    );
};
