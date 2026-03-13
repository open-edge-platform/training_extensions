// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Button, dimensionValue, Divider, Flex, Loading, Text, View } from '@geti/ui';
import { InfoOutline } from '@geti/ui/icons';
import { useStagedDataset } from 'hooks/api/staged-dataset.hook';

import { formatBytes } from '../../../shared/util';
import { DeleteStagedFileConfirmation } from '../../delete-staged-file-confirmation/delete-staged-file-confirmation.component';
import { getErrorMessage } from '../../util';
import { ImportFailedJob } from '../import-failed-job/import-failed-job.component';

type StagedImportDatasetProps = {
    message: string;
    fileName: string;
    stagedDatasetId: string;
    primaryButtonLabel: string;
    onOpen: () => void;
    deleteEntry: () => void;
};

const Container = ({ children }: { children: React.ReactNode }) => (
    <View
        position={'relative'}
        borderColor={'gray-200'}
        borderRadius={'regular'}
        backgroundColor={'gray-75'}
        borderWidth={'thin'}
        minHeight={'size-1600'}
    >
        {children}
    </View>
);

export const StagedImportDataset = ({
    message,
    fileName,
    stagedDatasetId,
    primaryButtonLabel,
    onOpen,
    deleteEntry,
}: StagedImportDatasetProps) => {
    const { error, isError, isFetching, data: stagedDataset } = useStagedDataset(stagedDatasetId);

    if (isFetching) {
        return (
            <Container>
                <Loading mode='inline' size='S' style={{ height: '100%', alignItems: 'center' }} />
            </Container>
        );
    }

    if (isError) {
        return (
            <Container>
                <ImportFailedJob
                    size={0}
                    fileName={fileName}
                    error={getErrorMessage(error)}
                    message={'An error occurred during staged file reading'}
                    stagedDatasetId={stagedDatasetId}
                    deleteEntry={deleteEntry}
                />
            </Container>
        );
    }

    return (
        <Container>
            <View padding='size-150'>
                <Flex justifyContent='space-between' alignItems='center' gap='size-250'>
                    <Text UNSAFE_style={{ fontWeight: 500, fontSize: dimensionValue('size-200') }}>
                        Import dataset - {fileName} - {formatBytes(stagedDataset?.size ?? 0)}
                    </Text>

                    <Flex justifyContent='space-between' alignItems='center' gap='size-250'>
                        <DeleteStagedFileConfirmation stagedDatasetId={stagedDatasetId} deleteEntry={deleteEntry} />

                        <Button onPress={onOpen}>{primaryButtonLabel}</Button>
                    </Flex>
                </Flex>

                <Divider size='S' marginY='size-150' />

                <Flex alignItems='center' gap='size-100'>
                    <InfoOutline width={16} height={16} />

                    <Text>{message}</Text>
                </Flex>
            </View>
        </Container>
    );
};
