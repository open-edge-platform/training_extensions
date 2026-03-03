// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Button, dimensionValue, Divider, Flex, Text, View } from '@geti/ui';
import { InfoOutline } from '@geti/ui/icons';

import { $api } from '../../../../../api/client';
import { formatBytes } from '../../../../../shared/util';
import { useImportDatasetDialogState } from '../../../providers/export-import-dataset-dialog-provider.component';
import { DeleteStagedFileConfirmation } from '../../delete-staged-file-confirmation/delete-staged-file-confirmation.component';

type StagedImportDatasetProps = {
    fileName: string;
    stagedDatasetId: string;
};

export const StagedImportDataset = ({ stagedDatasetId, fileName }: StagedImportDatasetProps) => {
    const { datasetImportDialogState, setCurrentStep, setCurrentStagedId } = useImportDatasetDialogState();

    const {
        error,
        isError,
        isFetching,
        data: stagedDataset,
    } = $api.useQuery('get', '/api/staged_datasets/{staged_dataset_id}', {
        params: { path: { staged_dataset_id: String(stagedDatasetId) } },
    });

    const handleOpen = () => {
        setCurrentStep('labelMapping');
        setCurrentStagedId(stagedDatasetId);
        datasetImportDialogState.open();
    };

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

                    <Divider size='S' marginY='size-150' />

                    <Flex justifyContent='space-between' alignItems='center' gap='size-250'>
                        <DeleteStagedFileConfirmation stagedDatasetId={stagedDatasetId} />

                        <Button
                            aria-label='continue dataset import'
                            onPress={handleOpen}
                            isDisabled={isError || isFetching}
                        >
                            Continue
                        </Button>
                    </Flex>
                </Flex>

                <Divider size='S' marginY='size-150' />

                <Flex alignItems='center' gap='size-100'>
                    <InfoOutline width={16} height={16} />

                    <Text>{isError ? `Error: ${error?.detail}` : 'Map labels for the uploaded dataset'}</Text>
                </Flex>
            </View>
        </View>
    );
};
