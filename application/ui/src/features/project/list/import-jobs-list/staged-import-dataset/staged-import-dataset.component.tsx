// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Button, dimensionValue, Divider, Flex, Text, View } from '@geti/ui';
import { InfoOutline } from '@geti/ui/icons';
import { useImportDatasetAsNewProject } from 'hooks/localStorage/use-import-dataset-as-new-project.hook';

import { $api } from '../../../../../api/client';
import { DeleteStagedFileConfirmation } from '../../../../../components/delete-staged-file-confirmation/delete-staged-file-confirmation.component';
import { formatBytes } from '../../../../../shared/util';
import { ImportDatasetAsNewProjectState } from '../../../../dataset/import-export/import-dataset/util';
import { useImportDatasetDialog } from '../../../providers/import-dataset-dialog-provider.component';

type StagedImportDatasetProps = {
    message: string;
    fileName: string;
    stagedDatasetId: string;
    primaryButtonLabel: string;
    openState: ImportDatasetAsNewProjectState;
};

export const StagedImportDataset = ({
    message,
    fileName,
    openState,
    stagedDatasetId,
    primaryButtonLabel,
}: StagedImportDatasetProps) => {
    const { deleteImportEntry } = useImportDatasetAsNewProject();
    const { datasetImportDialogState, setCurrentStep, setCurrentStagedId } = useImportDatasetDialog();

    const {
        error,
        isError,
        isFetching,
        data: stagedDataset,
    } = $api.useQuery('get', '/api/staged_datasets/{staged_dataset_id}', {
        params: { path: { staged_dataset_id: String(stagedDatasetId) } },
    });

    const handleOpen = () => {
        setCurrentStep(openState);
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
                        <DeleteStagedFileConfirmation
                            stagedDatasetId={stagedDatasetId}
                            deleteEntry={() => deleteImportEntry(stagedDatasetId)}
                        />

                        <Button onPress={handleOpen} isDisabled={isError || isFetching}>
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
