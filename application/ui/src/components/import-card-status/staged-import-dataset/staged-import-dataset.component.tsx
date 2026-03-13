// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Button, dimensionValue, Divider, Flex, Text, View } from '@geti/ui';
import { InfoOutline } from '@geti/ui/icons';
import { useStagedDataset } from 'hooks/api/staged-dataset.hook';

import { formatBytes } from '../../../shared/util';
import { DeleteStagedFileConfirmation } from '../../delete-staged-file-confirmation/delete-staged-file-confirmation.component';

type StagedImportDatasetProps = {
    message: string;
    fileName: string;
    stagedDatasetId: string;
    primaryButtonLabel: string;
    onOpen: () => void;
    deleteEntry: () => void;
};

export const StagedImportDataset = ({
    message,
    fileName,
    stagedDatasetId,
    primaryButtonLabel,
    onOpen,
    deleteEntry,
}: StagedImportDatasetProps) => {
    const { error, isError, data: stagedDataset } = useStagedDataset(stagedDatasetId);

    return (
        <View
            position='relative'
            borderColor='gray-200'
            borderRadius='regular'
            backgroundColor='gray-75'
            borderWidth='thin'
        >
            <View padding='size-150'>
                <Flex justifyContent='space-between' alignItems='center' gap='size-250'>
                    <Text UNSAFE_style={{ fontWeight: 500, fontSize: dimensionValue('size-200') }}>
                        Import dataset - {fileName} - {formatBytes(stagedDataset?.size ?? 0)}
                    </Text>

                    <Flex justifyContent='space-between' alignItems='center' gap='size-250'>
                        <DeleteStagedFileConfirmation stagedDatasetId={stagedDatasetId} deleteEntry={deleteEntry} />

                        <Button onPress={onOpen} isDisabled={isError}>
                            {primaryButtonLabel}
                        </Button>
                    </Flex>
                </Flex>

                <Divider size='S' marginY='size-150' />

                <Flex alignItems='center' gap='size-100'>
                    <InfoOutline width={16} height={16} />

                    <Text>{isError ? `Error: ${error?.detail}` : message}</Text>
                </Flex>
            </View>
        </View>
    );
};
