// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Button, dimensionValue, Divider, Flex, Text, View } from '@geti/ui';

import { $api } from '../../../../../api/client';
import { PrepareImportDatasetJob } from '../../../../../constants/shared-types';
import { useImportDatasetToProject } from '../../../../../hooks/localStorage/use-import-dataset-to-project.hook';
import { formatBytes } from '../../../../../shared/util';

type ImportFailedJobProps = {
    size: number;
    fileName: string;
    stagedDatasetId: string;
    job: PrepareImportDatasetJob;
};

export const ImportFailedJob = ({ job, fileName, size, stagedDatasetId }: ImportFailedJobProps) => {
    const { deleteImportEntry } = useImportDatasetToProject();
    const deleteFileMutation = $api.useMutation('delete', '/api/staged_datasets/{staged_dataset_id}');

    const handleClose = () => {
        deleteFileMutation.mutateAsync(
            { params: { path: { staged_dataset_id: stagedDatasetId } } },
            {
                onSuccess: () => {
                    deleteImportEntry(stagedDatasetId);
                },
            }
        );
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

            <Text>{job.message}</Text>
            <Divider size='S' marginY='size-150' />
            <Text>{job.error}</Text>
        </View>
    );
};
