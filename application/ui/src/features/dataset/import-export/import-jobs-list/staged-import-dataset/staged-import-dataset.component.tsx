// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Button, dimensionValue, Divider, Flex, Text, View } from '@geti/ui';
import { InfoOutline } from '@geti/ui/icons';

import { $api } from '../../../../../api/client';
import { useImportDatasetToProject } from '../../../../../hooks/localStorage/use-import-dataset-to-project.hook';
import { formatBytes } from '../../../../../shared/util';
import { useImportDatasetDialogState } from '../../../providers/export-import-dataset-dialog-provider.component';

type StagedImportDatasetProps = {
    fileName: string;
    stagedDatasetId: string;
};

export const StagedImportDataset = ({ stagedDatasetId, fileName }: StagedImportDatasetProps) => {
    const { deleteImportEntry } = useImportDatasetToProject();
    const { datasetImportDialogState, setCurrentStep, setCurrentStagedId } = useImportDatasetDialogState();

    const { data: stagedDataset } = $api.useQuery('get', '/api/staged_datasets/{staged_dataset_id}', {
        params: { path: { staged_dataset_id: String(stagedDatasetId) } },
    });

    const deleteFileMutation = $api.useMutation('delete', '/api/staged_datasets/{staged_dataset_id}');

    const handleOpen = () => {
        setCurrentStep('labelMapping');
        setCurrentStagedId(stagedDatasetId);
        datasetImportDialogState.open();
    };

    const handleDelete = () => {
        deleteFileMutation.mutateAsync(
            { params: { path: { staged_dataset_id: String(stagedDatasetId) } } },
            {
                onSuccess: () => {
                    datasetImportDialogState.close();
                    setCurrentStep('dropzone');
                    setCurrentStagedId(null);
                    deleteImportEntry({ stagedDatasetId });
                },
            }
        );
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
                        <Button
                            variant='secondary'
                            style='fill'
                            aria-label='delete import dataset status'
                            onPress={handleDelete}
                        >
                            Delete
                        </Button>
                        <Button aria-label='continue dataset import' onPress={handleOpen}>
                            Continue
                        </Button>
                    </Flex>
                </Flex>

                <Divider size='S' marginY='size-150' />

                <Flex alignItems='center' gap='size-100'>
                    <InfoOutline width={16} height={16} />

                    <Text>Map labels for the uploaded dataset</Text>
                </Flex>
            </View>
        </View>
    );
};
